"""
Comprehensive driver for the secant-predictor benchmark.

Sweeps a matrix of (target, grid_size, config, seed) combinations via
``mpiexec`` invocations of ``run_secant_benchmark.py`` and reports:

  * Per (target, grid_size): total target-call counts and grid-quality
    deficit vs the pointwise-best observed grid, for baseline vs secant.
  * Predictor diagnostics: win rate, number of cells where the predictor
    actually fired.
  * A grand summary table showing the average effect of the secant
    predictor across the whole matrix.

The matrix is configurable via CLI flags; default sweep is:

  Targets:      rosenbrock_4d, himmelblau_4d, rastrigin_4d, levy_4d,
                styblinski_tang_4d, ackley_4d, griewank_4d,
                rosenbrock_6d, rastrigin_6d
  Grids:        15, 25 for 4D; 12 for 6D (size auto-trimmed)
  Configs:      baseline (secant OFF), secant (secant ON)
  Seeds:        3

Run with:
    python examples/run_secant_benchmark_driver.py [--ranks N] [--seeds K]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


CONFIGS = ['baseline', 'secant']

# Default sweep: medium-cost matrix covering smooth, multimodal, and
# rugged-multimodal targets at two grid resolutions for 4D, one for 6D.
DEFAULT_4D_TARGETS = [
    'rosenbrock_4d', 'himmelblau_4d', 'rastrigin_4d',
    'levy_4d', 'styblinski_tang_4d', 'ackley_4d', 'griewank_4d',
]
DEFAULT_6D_TARGETS = ['rosenbrock_6d', 'rastrigin_6d']
DEFAULT_4D_GRIDS = [15, 25]
DEFAULT_6D_GRIDS = [12]

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'run_secant_benchmark.py')


def run_one(target, config, grid, seed, ranks, tmpdir, allow_root):
    out = os.path.join(tmpdir,
                       f"{target}_g{grid}_{config}_seed{seed}.json")
    cmd = ['mpiexec']
    if allow_root:
        cmd.append('--allow-run-as-root')
    cmd += ['--oversubscribe',
            '-n', str(ranks),
            sys.executable, RUNNER,
            '--target', target,
            '--config', config,
            '--grid', str(grid),
            '--seed', str(seed),
            '--out', out]
    print(f"-> {target}/g{grid}/{config}/seed{seed} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mpiexec failed for {target}/g{grid}/{config}/seed{seed}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    with open(out) as f:
        return json.load(f)


def per_projection_calls(summary):
    """Cumulative -> per-projection deltas."""
    cum = [p['cumulative_target_calls'] for p in summary['projections']]
    per = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return per


def aggregate_per_setting(runs):
    """Group ``runs`` by config and seed.

    Returns dict {config -> {'calls': [seed][proj] -> int,
                             'grids': [seed][proj] -> array,
                             'diag':  [seed][proj] -> dict,
                             'elapsed': [seed] -> float}}.
    """
    out = {c: {'calls': [], 'grids': [], 'diag': [], 'elapsed': []}
           for c in CONFIGS}
    for r in runs:
        cfg = r['config']
        out[cfg]['calls'].append(per_projection_calls(r))
        out[cfg]['grids'].append([np.asarray(p['coarse_grid_values'])
                                  for p in r['projections']])
        out[cfg]['diag'].append([p['diagnostics'] for p in r['projections']])
        out[cfg]['elapsed'].append(r['elapsed_s'])
    return out


def pointwise_best_per_proj(agg):
    """Per-projection per-cell max over all (config, seed) runs."""
    n_proj = len(agg['baseline']['grids'][0])
    best = []
    for p in range(n_proj):
        stack = []
        for c in CONFIGS:
            for g in agg[c]['grids']:
                stack.append(g[p])
        s = np.stack(stack, axis=0)
        with np.errstate(invalid='ignore'):
            best.append(np.nanmax(s, axis=0))
    return best


def deficit(grid, ref, bad_threshold=1e-2):
    """Per-cell deficit = ref - grid with NaN cells dropped."""
    g = np.asarray(grid); r = np.asarray(ref)
    mask = ~(np.isnan(g) | np.isnan(r))
    if not mask.any():
        return float('nan'), float('nan'), 0, 0
    d = np.clip(r[mask] - g[mask], 0.0, None)
    return float(d.mean()), float(d.max()), int(mask.sum()), int(np.sum(d > bad_threshold))


def summarize_setting(agg, target, grid):
    """Compute a per-(target, grid) summary line: returns dict ready to
    print and to feed into the grand summary."""
    base_calls = np.array(agg['baseline']['calls'])
    sec_calls = np.array(agg['secant']['calls'])
    if base_calls.size == 0 or sec_calls.size == 0:
        return None

    ref = pointwise_best_per_proj(agg)

    # Cost.
    base_total = base_calls.sum(axis=1).mean()
    sec_total = sec_calls.sum(axis=1).mean()
    cost_ratio = sec_total / base_total if base_total > 0 else float('nan')

    # Quality.
    def _quality(cfg):
        means, maxes, bads, ncells = [], [], [], 0
        for per_seed_grids in agg[cfg]['grids']:
            for p, g in enumerate(per_seed_grids):
                m, mx, n, nb = deficit(g, ref[p])
                means.append(m); maxes.append(mx); bads.append(nb)
                ncells = max(ncells, n)
        per_seed_bads = []
        n_proj = len(agg[cfg]['grids'][0])
        for s in range(len(agg[cfg]['grids'])):
            per_seed_bads.append(
                sum(bads[s * n_proj + p] for p in range(n_proj))
            )
        return {
            'mean_deficit': float(np.nanmean(means)),
            'max_deficit': float(np.nanmax(maxes)),
            'bad_per_seed': float(np.mean(per_seed_bads)),
            'cells_per_proj': ncells,
        }

    qb = _quality('baseline')
    qs = _quality('secant')

    # Diagnostics (predictor).
    diag_totals = {'tested': 0, 'won': 0}
    for per_seed in agg['secant']['diag']:
        for d in per_seed:
            diag_totals['tested'] += d.get('secant_tested', 0)
            diag_totals['won'] += d.get('secant_won', 0)
    win_rate = (diag_totals['won'] / diag_totals['tested']
                if diag_totals['tested'] else float('nan'))

    return {
        'target': target,
        'grid': grid,
        'base_total_calls': base_total,
        'secant_total_calls': sec_total,
        'cost_ratio': cost_ratio,
        'baseline_mean_deficit': qb['mean_deficit'],
        'secant_mean_deficit': qs['mean_deficit'],
        'baseline_max_deficit': qb['max_deficit'],
        'secant_max_deficit': qs['max_deficit'],
        'baseline_bad_per_seed': qb['bad_per_seed'],
        'secant_bad_per_seed': qs['bad_per_seed'],
        'cells_per_proj': qb['cells_per_proj'],
        'predictor_tested': diag_totals['tested'],
        'predictor_won': diag_totals['won'],
        'win_rate': win_rate,
        'n_seeds': len(agg['baseline']['calls']),
        'n_projections': len(agg['baseline']['calls'][0]),
    }


def print_per_setting(summary):
    print(f"\n--- {summary['target']} @ grid={summary['grid']} "
          f"({summary['n_projections']} 2D projections, "
          f"{summary['n_seeds']} seeds) ---")
    print(f"  Total target calls (mean over seeds):")
    print(f"    baseline:   {summary['base_total_calls']:>10,.0f}")
    print(f"    secant:     {summary['secant_total_calls']:>10,.0f}  "
          f"({summary['cost_ratio']:.3f}x baseline)")
    print(f"  Grid deficit vs pointwise best:")
    md_ratio = (summary['secant_mean_deficit'] / summary['baseline_mean_deficit']
                if summary['baseline_mean_deficit'] > 0 else float('nan'))
    bd_ratio = (summary['secant_bad_per_seed'] / summary['baseline_bad_per_seed']
                if summary['baseline_bad_per_seed'] > 0 else float('nan'))
    print(f"    {'metric':<14s}  {'baseline':>10s}  {'secant':>10s}  {'ratio':>8s}")
    print(f"    {'mean deficit':<14s}  "
          f"{summary['baseline_mean_deficit']:>10.3e}  "
          f"{summary['secant_mean_deficit']:>10.3e}  "
          f"{md_ratio:>8.3f}")
    print(f"    {'max deficit':<14s}  "
          f"{summary['baseline_max_deficit']:>10.3e}  "
          f"{summary['secant_max_deficit']:>10.3e}  "
          f"{'':>8s}")
    print(f"    {'bad cells/seed':<14s}  "
          f"{summary['baseline_bad_per_seed']:>10.1f}  "
          f"{summary['secant_bad_per_seed']:>10.1f}  "
          f"{bd_ratio:>8.3f}")
    print(f"  Predictor diagnostics (secant config only):")
    print(f"    candidates_tested = {summary['predictor_tested']:,}, "
          f"won = {summary['predictor_won']:,}, "
          f"win_rate = {summary['win_rate']:.1%}")


def print_grand_summary(summaries):
    """One-line-per-setting summary at the end."""
    if not summaries:
        return
    print("\n" + "=" * 100)
    print(" Grand summary: secant predictor vs baseline (lower deficit ratio = secant better)")
    print("=" * 100)
    header = (f"  {'target':<24s} {'grid':>5s} {'cost x':>7s} "
              f"{'meanΔ_base':>12s} {'meanΔ_sec':>12s} {'meanΔ ratio':>13s} "
              f"{'badΔ_base':>11s} {'badΔ_sec':>10s} {'badΔ ratio':>12s} "
              f"{'win_rate':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in summaries:
        md_ratio = (s['secant_mean_deficit'] / s['baseline_mean_deficit']
                    if s['baseline_mean_deficit'] > 0 else float('nan'))
        bd_ratio = (s['secant_bad_per_seed'] / s['baseline_bad_per_seed']
                    if s['baseline_bad_per_seed'] > 0 else float('nan'))
        print(f"  {s['target']:<24s} {s['grid']:>5d} {s['cost_ratio']:>6.3f}x "
              f"{s['baseline_mean_deficit']:>12.3e} {s['secant_mean_deficit']:>12.3e} "
              f"{md_ratio:>12.3f}x "
              f"{s['baseline_bad_per_seed']:>11.1f} {s['secant_bad_per_seed']:>10.1f} "
              f"{bd_ratio:>11.3f}x "
              f"{s['win_rate']:>8.1%}")

    # Geometric-mean aggregates across the matrix (geometric mean of
    # ratios; treats halving and doubling symmetrically).
    cost_ratios = [s['cost_ratio'] for s in summaries
                   if np.isfinite(s['cost_ratio']) and s['cost_ratio'] > 0]
    md_ratios = [s['secant_mean_deficit'] / s['baseline_mean_deficit']
                 for s in summaries
                 if s['baseline_mean_deficit'] > 0
                 and np.isfinite(s['secant_mean_deficit'])
                 and (s['secant_mean_deficit'] / s['baseline_mean_deficit']) > 0]
    bd_ratios = [s['secant_bad_per_seed'] / s['baseline_bad_per_seed']
                 for s in summaries
                 if s['baseline_bad_per_seed'] > 0
                 and (s['secant_bad_per_seed'] / s['baseline_bad_per_seed']) > 0]
    win_rates = [s['win_rate'] for s in summaries if np.isfinite(s['win_rate'])]

    def _geomean(xs):
        if not xs:
            return float('nan')
        return float(np.exp(np.mean(np.log(xs))))

    print()
    print(f"  Geometric means over {len(summaries)} (target, grid) settings:")
    print(f"    cost ratio (secant/baseline): {_geomean(cost_ratios):.3f}x")
    print(f"    mean-deficit ratio:           {_geomean(md_ratios):.3f}x")
    print(f"    bad-cells ratio:              {_geomean(bd_ratios):.3f}x")
    if win_rates:
        print(f"    mean predictor win rate:      "
              f"{float(np.mean(win_rates)):.1%}")


def write_csv(summaries, path):
    """Long-form CSV: one row per (target, grid) summary."""
    import csv
    fields = ['target', 'grid', 'n_seeds', 'n_projections', 'cells_per_proj',
              'base_total_calls', 'secant_total_calls', 'cost_ratio',
              'baseline_mean_deficit', 'secant_mean_deficit',
              'baseline_max_deficit', 'secant_max_deficit',
              'baseline_bad_per_seed', 'secant_bad_per_seed',
              'predictor_tested', 'predictor_won', 'win_rate']
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--allow-root', action='store_true')
    parser.add_argument('--targets-4d', nargs='*', default=DEFAULT_4D_TARGETS,
                        help='4D test targets (pass empty list to skip all 4D)')
    parser.add_argument('--targets-6d', nargs='*', default=DEFAULT_6D_TARGETS,
                        help='6D test targets (pass empty list to skip all 6D)')
    parser.add_argument('--grids-4d', nargs='*', type=int,
                        default=DEFAULT_4D_GRIDS,
                        help='Grid sizes for 4D targets')
    parser.add_argument('--grids-6d', nargs='*', type=int,
                        default=DEFAULT_6D_GRIDS,
                        help='Grid sizes for 6D targets')
    parser.add_argument('--csv', default=None,
                        help='Optional CSV path to write the grand-summary table.')
    parser.add_argument('--keep-tmp', action='store_true')
    args = parser.parse_args()

    settings = []
    for t in args.targets_4d:
        for g in args.grids_4d:
            settings.append((t, g))
    for t in args.targets_6d:
        for g in args.grids_6d:
            settings.append((t, g))

    print(f"=== Secant-predictor benchmark ===")
    print(f"Settings:  {len(settings)} (target, grid) combinations")
    print(f"Configs:   {CONFIGS}")
    print(f"Seeds:     {args.seeds}")
    print(f"Total runs: {len(settings) * len(CONFIGS) * args.seeds}")
    print()

    tmpdir = tempfile.mkdtemp(prefix='paraprof_secant_')
    summaries = []
    try:
        for (target, grid) in settings:
            runs = []
            for config in CONFIGS:
                for seed in range(args.seeds):
                    try:
                        r = run_one(target, config, grid, seed, args.ranks,
                                    tmpdir, args.allow_root)
                        runs.append(r)
                    except RuntimeError as e:
                        print(f"  !! skipped: {e}", file=sys.stderr)
            if not runs:
                continue
            agg = aggregate_per_setting(runs)
            s = summarize_setting(agg, target, grid)
            if s is not None:
                print_per_setting(s)
                summaries.append(s)

        print_grand_summary(summaries)

        if args.csv:
            write_csv(summaries, args.csv)
            print(f"\nCSV written to {args.csv}")
        if args.keep_tmp:
            print(f"\nPer-run JSON files kept in {tmpdir}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
