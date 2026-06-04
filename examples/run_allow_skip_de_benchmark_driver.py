"""
Driver for the neighbour-certified DE-skip A/B benchmark.

For each requested target it subprocesses ``run_allow_skip_de_benchmark.py``
twice (baseline = skip off, certify = skip on) under mpiexec, then prints a
per-target report:

  * final cumulative target_calls for each mode and the percentage change;
  * how many cells took the DE skip;
  * a ROI grid-quality check -- max and mean |Delta logL| between the two
    modes' final coarse grids, restricted to cells inside the ROI in either
    mode. A genuine win is a call reduction with ROI grid error ~ 0.

Usage:
    python examples/run_allow_skip_de_benchmark_driver.py \\
        --targets himmelblau_4d rosenbrock_4d --ncores 4
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def _run(target, mode, ncores, out_path):
    env = dict(os.environ)
    env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    script = os.path.join(os.path.dirname(__file__),
                          'run_allow_skip_de_benchmark.py')
    cmd = ['mpiexec', '--oversubscribe', '-n', str(ncores),
           sys.executable, script,
           '--target', target, '--mode', mode, '--out', out_path]
    subprocess.run(cmd, check=True, env=env)
    with open(out_path) as f:
        return json.load(f)


def _roi_grid_error(base_proj, cert_proj, roi_threshold):
    """Max/mean |Delta logL| over cells inside the ROI in either mode."""
    b = np.array(base_proj['coarse_grid_values'], dtype=float)
    c = np.array(cert_proj['coarse_grid_values'], dtype=float)
    finite = np.isfinite(b) & np.isfinite(c)
    if not finite.any():
        return float('nan'), float('nan')
    gmax = max(np.nanmax(b[finite]), np.nanmax(c[finite]))
    roi = finite & ((b > gmax - roi_threshold) | (c > gmax - roi_threshold))
    if not roi.any():
        return 0.0, 0.0
    diff = np.abs(b[roi] - c[roi])
    return float(diff.max()), float(diff.mean())


# ROI thresholds mirror run_allow_skip_de_benchmark.TARGET_KWARGS.
ROI_THRESHOLD = {
    'himmelblau_4d': 4.0,
    'rosenbrock_4d': 8.0,
    'rosenbrock_6d': 8.0,
    'rastrigin_4d': 8.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--ncores', type=int, default=4)
    args = parser.parse_args()

    print("=" * 78)
    print("Neighbour-certified DE-skip A/B benchmark")
    print("=" * 78)

    for target in args.targets:
        with tempfile.TemporaryDirectory() as td:
            base = _run(target, 'baseline', args.ncores,
                        os.path.join(td, 'base.json'))
            cert = _run(target, 'certify', args.ncores,
                        os.path.join(td, 'cert.json'))

        base_calls = base['projections'][-1]['cumulative_target_calls']
        cert_calls = cert['projections'][-1]['cumulative_target_calls']
        pct = 100.0 * (cert_calls - base_calls) / base_calls

        roi_thr = ROI_THRESHOLD[target]
        max_errs, mean_errs = [], []
        for bp, cp in zip(base['projections'], cert['projections']):
            me, ae = _roi_grid_error(bp, cp, roi_thr)
            max_errs.append(me)
            mean_errs.append(ae)

        print(f"\n### {target}")
        print(f"  baseline calls : {base_calls:>10,}")
        print(f"  certify  calls : {cert_calls:>10,}  ({pct:+.1f}%)")
        print(f"  cells certified: {cert['cells_skipped']:>10,}")
        print(f"  global_max     : base={base['projections'][-1]['global_max']:.6e}"
              f"  cert={cert['projections'][-1]['global_max']:.6e}")
        print(f"  ROI grid error : max |dlogL| = {np.nanmax(max_errs):.3e},"
              f"  mean |dlogL| = {np.nanmean(mean_errs):.3e}")

    print("\n" + "=" * 78)


if __name__ == '__main__':
    main()
