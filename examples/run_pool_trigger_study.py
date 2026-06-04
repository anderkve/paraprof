"""
Replicate study for the cross-projection second trigger of smooth-certify
(idea 3 flavor b). For each target it runs N seeds in both arms (neighbour
trigger only vs neighbour + cross-projection trigger), both with
``de.smooth_certify`` on, and reports:

  * the marginal call savings the second trigger adds (mean over seeds, plus a
    Mann-Whitney U test);
  * cells certified total and the subset only the cross-projection trigger
    caught (mean per scan) -- does it fire at all on top of the neighbour one;
  * ROI deficit vs the elementwise-max reference grid, to confirm the extra
    skips do not cost accuracy.

Usage:
    python examples/run_pool_trigger_study.py \\
        --targets himmelblau_4d rosenbrock_4d rastrigin_4d --ncores 4 --reps 5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from run_pool_trigger_benchmark import TARGET_KWARGS  # noqa: F401 (path check)
from run_smooth_certify_replicate_study import (
    BASE_SEED, _reference_grids, _run_quality, _mwu, _fmt, ROI_THRESHOLD,
)


def _run(target, mode, seed, ncores, out_path):
    env = dict(os.environ)
    env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    script = os.path.join(os.path.dirname(__file__), 'run_pool_trigger_benchmark.py')
    cmd = ['mpiexec', '--oversubscribe', '-n', str(ncores), sys.executable, script,
           '--target', target, '--mode', mode, '--seed', str(seed), '--out', out_path]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--reps', type=int, default=5)
    args = parser.parse_args()

    seeds = [BASE_SEED + 1000 * k for k in range(args.reps)]
    print("=" * 78)
    print(f"Cross-projection second-trigger study  (reps={args.reps})")
    print("smooth_certify ON in both arms; toggling only the pool trigger")
    print("=" * 78)

    for target in args.targets:
        roi_thr = ROI_THRESHOLD[target]
        runs = {'neighbour': [], 'both': []}
        with tempfile.TemporaryDirectory() as td:
            for mode in ('neighbour', 'both'):
                for seed in seeds:
                    runs[mode].append(_run(target, mode, seed, args.ncores,
                                           os.path.join(td, f'{mode}_{seed}.json')))
            refs = _reference_grids(runs['neighbour'] + runs['both'])
            defs = {m: [_run_quality(r, refs, roi_thr)[0] for r in runs[m]]
                    for m in runs}

        calls = {m: [r['projections'][-1]['cumulative_target_calls']
                     for r in runs[m]] for m in runs}
        cert = {m: [r['cells_certified'] for r in runs[m]] for m in runs}
        pool_only = [r['cells_certified_pool_only'] for r in runs['both']]

        nm = np.mean(calls['neighbour'])
        bm = np.mean(calls['both'])
        pct = 100.0 * (bm - nm) / nm

        print(f"\n### {target}")
        print(f"  calls  neighbour-only {_fmt(calls['neighbour'])}")
        print(f"         both triggers  {_fmt(calls['both'])}   "
              f"({pct:+.1f}% marginal)   MWU p={_mwu(calls['neighbour'], calls['both']):.3f}")
        print(f"  cells certified: neighbour {_fmt(cert['neighbour'])}, "
              f"both {_fmt(cert['both'])}")
        print(f"  cells only via cross-projection trigger: {_fmt(pool_only)}/scan")
        print(f"  ROI mean deficit: neighbour {_fmt(defs['neighbour'])}, "
              f"both {_fmt(defs['both'])}   MWU p={_mwu(defs['neighbour'], defs['both']):.3f}")

    print("\n" + "=" * 78)


if __name__ == '__main__':
    main()
