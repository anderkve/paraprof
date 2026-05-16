"""
Driver for the continuation-hooks benchmark.

Subprocesses ``mpiexec`` invocations of ``run_continuation_benchmark.py`` for
every (target, config, seed) combination, then aggregates per-projection
target-call counts, achieved-max-logL deltas, and the continuation diagnostic
counters into a single comparison report.

Modes:
  * ``baseline``  — both hooks off
  * ``secant``    — secant predictor warm-start only
  * ``basin``     — online basin-switch detection only
  * ``both``      — both on

Reports:
  * Per-projection target-call counts vs baseline (speedup factor).
  * Per-projection grid quality: mean and worst-case logL deficit vs
    the best per-projection result across all configs (the "best
    observed" grid is taken as the per-cell pointwise maximum across
    all configs; deficit = best - this_config).
  * Hook diagnostic totals (secant win rate, basin-switch tests / hits).

Run with:
    python examples/run_continuation_benchmark_driver.py \\
        [--ranks N] [--seeds K] [--targets ...]

This script itself does not need MPI.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

DEFAULT_TARGETS = ['himmelblau_4d', 'rosenbrock_4d', 'rastrigin_4d']
CONFIGS = ['baseline', 'secant', 'basin', 'both']

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'run_continuation_benchmark.py')


def run_one(target, config, seed, ranks, tmpdir, allow_root):
    out = os.path.join(tmpdir, f"{target}_{config}_seed{seed}.json")
    cmd = ['mpiexec']
    if allow_root:
        cmd.append('--allow-run-as-root')
    cmd += ['--oversubscribe',
            '-n', str(ranks),
            sys.executable, RUNNER,
            '--target', target,
            '--config', config,
            '--seed', str(seed),
            '--out', out]
    print(f"-> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def per_projection_calls(summary):
    """Convert cumulative snapshots into per-projection deltas."""
    cum = [p['cumulative_target_calls'] for p in summary['projections']]
    per = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return per, cum


def aggregate_runs(target_results, target):
    """
    Collapse per-seed results into per-config stats.

    Returns a dict { config -> {'per_proj_calls': [seed][proj] -> int,
                                'grids': [seed][proj] -> np.ndarray,
                                'elapsed_s': [seed] -> float,
                                'diagnostics': [seed][proj] -> dict} }.
    """
    out = {}
    for config in CONFIGS:
        runs = [r for r in target_results if r['config'] == config]
        per_proj = []
        grids = []
        diags = []
        elapsed = []
        for r in runs:
            calls, _ = per_projection_calls(r)
            per_proj.append(calls)
            grids.append([np.asarray(p['coarse_grid_values'])
                          for p in r['projections']])
            diags.append([p['diagnostics'] for p in r['projections']])
            elapsed.append(r['elapsed_s'])
        out[config] = {
            'per_proj_calls': per_proj,
            'grids': grids,
            'diagnostics': diags,
            'elapsed_s': elapsed,
        }
    return out


def grid_pointwise_best(agg):
    """
    Per-projection, per-cell pointwise max of logL across all (config, seed)
    runs.  NaN cells count as missing and are skipped.  The result is the
    "best observed" reference profile we measure each run against.
    """
    n_proj = len(next(iter(agg.values()))['grids'][0])
    best = []
    for p in range(n_proj):
        # Collect every observed grid for this projection.
        stacks = []
        for config in CONFIGS:
            for g in agg[config]['grids']:
                stacks.append(g[p])
        stack = np.stack(stacks, axis=0)
        # nanmax over the run-axis. NaNs that exist across *all* runs stay NaN.
        with np.errstate(invalid='ignore'):
            ref = np.nanmax(stack, axis=0)
        best.append(ref)
    return best


def grid_deficit(grid, ref, bad_threshold=1e-2):
    """Deficit = ref - grid, with NaN cells dropped.

    Returns
    -------
    (mean, max, n_cells, n_bad)
        ``n_bad`` is the number of cells with deficit > ``bad_threshold``.
        ``n_bad`` is much more robust than ``max`` to single-cell outliers.
    """
    g = np.asarray(grid)
    r = np.asarray(ref)
    mask = ~(np.isnan(g) | np.isnan(r))
    if not mask.any():
        return float('nan'), float('nan'), 0, 0
    diff = r[mask] - g[mask]
    # Numerical noise can push diff slightly negative; clip at zero.
    diff = np.clip(diff, 0.0, None)
    n_bad = int(np.sum(diff > bad_threshold))
    return float(diff.mean()), float(diff.max()), int(mask.sum()), n_bad


def summarize_diagnostics(per_proj_diag_list):
    """
    Sum diagnostic counters across all projections and all seeds for a
    given config.  ``per_proj_diag_list`` is shape [n_seeds][n_proj].
    """
    totals = {
        'secant_tested': 0,
        'secant_won': 0,
        'basin_switch_tests': 0,
        'basin_switch_improvements': 0,
        'patching_tests_total': 0,
        'patching_improvements_total': 0,
    }
    for per_seed in per_proj_diag_list:
        for d in per_seed:
            for k in totals:
                totals[k] += d.get(k, 0)
    return totals


def print_target_report(target, agg):
    """Print a per-target table comparing the four configs."""
    print(f"\n{'='*78}")
    print(f" Target: {target}")
    print(f"{'='*78}")

    baseline_calls = np.array(agg['baseline']['per_proj_calls'])  # [seed, proj]
    if baseline_calls.size == 0:
        print(" (no baseline runs)")
        return
    n_proj = baseline_calls.shape[1]

    # Pointwise best-observed grid per projection (across all configs+seeds).
    ref_grids = grid_pointwise_best(agg)

    print(f"\nTotal target-function calls (mean across seeds, summed over projections):")
    print(f"  {'config':<10s}  {'total':>10s}  {'speedup':>9s}  {'elapsed_s':>10s}")
    base_total = baseline_calls.sum(axis=1).mean()
    for config in CONFIGS:
        calls = np.array(agg[config]['per_proj_calls'])
        if calls.size == 0:
            continue
        tot = calls.sum(axis=1).mean()
        sp = base_total / tot if tot > 0 else float('inf')
        el = float(np.mean(agg[config]['elapsed_s']))
        print(f"  {config:<10s}  {tot:>10.0f}  {sp:>8.2f}x  {el:>10.2f}")

    print(f"\nPer-projection mean target-function calls (avg over seeds):")
    print(f"  {'proj':>4s}  " +
          "  ".join(f"{c:>10s}" for c in CONFIGS) +
          f"  {'best/base':>10s}")
    for p in range(n_proj):
        row_calls = []
        for c in CONFIGS:
            calls = np.array(agg[c]['per_proj_calls'])
            row_calls.append(calls[:, p].mean() if calls.size else float('nan'))
        best = min(row_calls)
        sp = row_calls[0] / best if best > 0 else float('inf')
        print(f"  {p:>4d}  " +
              "  ".join(f"{v:>10.0f}" for v in row_calls) +
              f"  {sp:>9.2f}x")

    print(f"\nGrid quality (deficit vs pointwise best, avg over seeds):")
    print(f"  {'config':<10s}  {'mean|Δ|':>10s}  {'max|Δ|':>10s}  "
          f"{'bad_cells/seed':>14s}  cells")
    for config in CONFIGS:
        grids = agg[config]['grids']
        if not grids:
            continue
        means = []
        maxes = []
        bad_counts_per_seed = []  # one entry per seed: total bad cells across projs
        cell_count = 0
        for s, per_seed_grids in enumerate(grids):
            bad_this_seed = 0
            for p, g in enumerate(per_seed_grids):
                m, mx, n, nb = grid_deficit(g, ref_grids[p])
                means.append(m)
                maxes.append(mx)
                bad_this_seed += nb
                cell_count = max(cell_count, n)
            bad_counts_per_seed.append(bad_this_seed)
        mean_m = float(np.nanmean(means)) if means else float('nan')
        max_m = float(np.nanmax(maxes)) if maxes else float('nan')
        bad_avg = float(np.mean(bad_counts_per_seed)) if bad_counts_per_seed else 0.0
        print(f"  {config:<10s}  {mean_m:>10.4e}  {max_m:>10.4e}  "
              f"{bad_avg:>14.1f}  {cell_count}")

    print(f"\nHook diagnostics (summed over all projections and seeds):")
    print(f"  {'config':<10s}  {'secant_tested':>14s}  {'secant_won':>11s}  "
          f"{'win_rate':>9s}  {'bsw_tests':>10s}  {'bsw_improvements':>17s}  "
          f"{'patch_tests':>12s}  {'patch_improvements':>19s}")
    for config in CONFIGS:
        tots = summarize_diagnostics(agg[config]['diagnostics'])
        wr = (tots['secant_won'] / tots['secant_tested']
              if tots['secant_tested'] else float('nan'))
        print(f"  {config:<10s}  {tots['secant_tested']:>14d}  "
              f"{tots['secant_won']:>11d}  "
              f"{wr:>8.1%}  "
              f"{tots['basin_switch_tests']:>10d}  "
              f"{tots['basin_switch_improvements']:>17d}  "
              f"{tots['patching_tests_total']:>12d}  "
              f"{tots['patching_improvements_total']:>19d}")


def print_overall_summary(target_results):
    """Aggregate the speedup numbers across all targets."""
    print(f"\n{'='*78}")
    print(" Overall summary (target-call speedup vs baseline)")
    print(f"{'='*78}")
    print(f"  {'target':<20s}  " + "  ".join(f"{c:>10s}" for c in CONFIGS))
    for target, agg in target_results.items():
        baseline_calls = np.array(agg['baseline']['per_proj_calls'])
        if baseline_calls.size == 0:
            continue
        base_total = baseline_calls.sum(axis=1).mean()
        speedups = []
        for c in CONFIGS:
            calls = np.array(agg[c]['per_proj_calls'])
            tot = calls.sum(axis=1).mean() if calls.size else float('nan')
            sp = base_total / tot if tot > 0 else float('inf')
            speedups.append(sp)
        print(f"  {target:<20s}  " +
              "  ".join(f"{v:>9.2f}x" for v in speedups))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks', type=int, default=4,
                        help='MPI ranks per child invocation (default: 4)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of RNG seeds per config (default: 3)')
    parser.add_argument('--allow-root', action='store_true',
                        help='Pass --allow-run-as-root to mpiexec '
                             '(required when running as root in a container).')
    parser.add_argument('--keep-tmp', action='store_true',
                        help='Do not delete per-run JSON files when done.')
    parser.add_argument('--targets', nargs='+', default=DEFAULT_TARGETS,
                        help='Test targets to benchmark (default: %(default)s).')
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix='paraprof_contbench_')
    try:
        target_results = {target: [] for target in args.targets}
        for target in args.targets:
            for config in CONFIGS:
                for seed in range(args.seeds):
                    r = run_one(target, config, seed, args.ranks, tmpdir,
                                args.allow_root)
                    target_results[target].append(r)

        all_aggs = {}
        for target in args.targets:
            agg = aggregate_runs(target_results[target], target)
            all_aggs[target] = agg
            print_target_report(target, agg)

        print_overall_summary(all_aggs)

        if args.keep_tmp:
            print(f"\nPer-run JSON files kept in {tmpdir}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
