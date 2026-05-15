"""
Driver for the proximity-warm-start benchmark.

Subprocesses four ``mpiexec`` invocations of
``run_proximity_warm_start_benchmark.py`` -- one for each (target, mode)
combination -- then prints a side-by-side report of per-projection
target-call counts and grid accuracy.

Run with:
    python examples/run_proximity_warm_start_benchmark_driver.py [--ranks N]

This script does *not* itself need MPI (it only spawns mpiexec children).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

DEFAULT_TARGETS = ['himmelblau_4d', 'rosenbrock_4d',
                   'rosenbrock_6d', 'rastrigin_4d']
MODES = ['baseline', 'proximity']

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'run_proximity_warm_start_benchmark.py')


def run_one(target, mode, ranks, tmpdir, allow_root):
    out = os.path.join(tmpdir, f"{target}_{mode}.json")
    cmd = ['mpiexec']
    if allow_root:
        cmd.append('--allow-run-as-root')
    cmd += ['-n', str(ranks),
            sys.executable, RUNNER,
            '--target', target,
            '--mode', mode,
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


def grid_diffs(base_summary, prox_summary):
    """Per-projection (max, mean) absolute differences over cells covered in
    both runs (NaN cells are excluded from the comparison)."""
    out = []
    for pb, pp in zip(base_summary['projections'], prox_summary['projections']):
        gb = np.asarray(pb['coarse_grid_values'])
        gp = np.asarray(pp['coarse_grid_values'])
        mask = ~(np.isnan(gb) | np.isnan(gp))
        if not mask.any():
            out.append((float('nan'), float('nan'), 0))
            continue
        diff = np.abs(gb[mask] - gp[mask])
        out.append((float(diff.max()), float(diff.mean()), int(mask.sum())))
    return out


def print_report(target, base, prox, accs):
    bp, _ = per_projection_calls(base)
    pp, _ = per_projection_calls(prox)
    print(f"\n=== {target} ===")
    print(f"Wall time: baseline={base['elapsed_s']:.1f}s, "
          f"proximity={prox['elapsed_s']:.1f}s")
    print(f"{'Proj':>4s}  {'dims':>10s}  {'baseline':>10s}  {'proximity':>10s}  "
          f"{'saved':>10s}  {'speedup':>8s}  {'max|Δ|':>10s}  {'mean|Δ|':>10s}  cells")
    for i, p in enumerate(base['projections']):
        diff = bp[i] - pp[i]
        sp = (bp[i] / pp[i]) if pp[i] > 0 else float('inf')
        mx, mn, ncells = accs[i]
        print(f"  {i+1:2d}  {str(p['dims']):>10s}  "
              f"{bp[i]:>10d}  {pp[i]:>10d}  {diff:>10d}  "
              f"{sp:>7.2f}x  {mx:>10.3e}  {mn:>10.3e}  {ncells}")
    tot_b, tot_p = sum(bp), sum(pp)
    sp_tot = (tot_b / tot_p) if tot_p > 0 else float('inf')
    print(f"  TOT  {'':>10s}  {tot_b:>10d}  {tot_p:>10d}  "
          f"{tot_b - tot_p:>10d}  {sp_tot:>7.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ranks', type=int, default=4,
                        help='MPI ranks per child invocation (default: 4)')
    parser.add_argument('--allow-root', action='store_true',
                        help='Pass --allow-run-as-root to mpiexec '
                             '(required when running as root in a container).')
    parser.add_argument('--keep-tmp', action='store_true',
                        help='Do not delete per-run JSON files when done.')
    parser.add_argument('--targets', nargs='+', default=DEFAULT_TARGETS,
                        help='Test targets to benchmark (default: %(default)s).')
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix='paraprof_proxbench_')
    try:
        results = {}
        for target in args.targets:
            for mode in MODES:
                results[(target, mode)] = run_one(
                    target, mode, args.ranks, tmpdir, args.allow_root)

        for target in args.targets:
            base = results[(target, 'baseline')]
            prox = results[(target, 'proximity')]
            accs = grid_diffs(base, prox)
            print_report(target, base, prox, accs)
        if args.keep_tmp:
            print(f"\nPer-run JSON files kept in {tmpdir}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
