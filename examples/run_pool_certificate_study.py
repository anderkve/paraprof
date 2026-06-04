"""
Replicate study for the cross-projection pool-certificate pass (idea 3, flavor a).

Question: how often does a cross-projection certificate actually raise a grid
value, and at what evaluation cost? For each target it runs N seeds in both
modes (pass off / on) and reports, per target:

  * pool-certificate activity: cells tested, cells raised, and total logL gained
    (mean over seeds) -- the direct measurement;
  * call overhead: mean target_calls off vs on;
  * ROI quality: mean deficit of each mode against the elementwise-max
    reference grid (one-sided lower bound), to confirm whether the raises
    translate into a measurable accuracy gain.

Usage:
    python examples/run_pool_certificate_study.py \\
        --targets himmelblau_4d rosenbrock_4d rastrigin_4d --ncores 4 --reps 4
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from run_pool_certificate_benchmark import TARGET_KWARGS  # noqa: F401  (validates import path)
from run_smooth_certify_replicate_study import (
    BASE_SEED, _reference_grids, _run_quality, ROI_THRESHOLD,
)


def _run(target, mode, seed, ncores, out_path):
    env = dict(os.environ)
    env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    script = os.path.join(os.path.dirname(__file__),
                          'run_pool_certificate_benchmark.py')
    cmd = ['mpiexec', '--oversubscribe', '-n', str(ncores), sys.executable,
           script, '--target', target, '--mode', mode, '--seed', str(seed),
           '--out', out_path]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out_path) as f:
        return json.load(f)


def _m(xs):
    xs = np.array([x for x in xs if np.isfinite(x)], dtype=float)
    return (xs.mean(), xs.std(ddof=1) if xs.size > 1 else 0.0) if xs.size else (float('nan'), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--reps', type=int, default=4)
    args = parser.parse_args()

    seeds = [BASE_SEED + 1000 * k for k in range(args.reps)]
    print("=" * 78)
    print(f"Pool-certificate replicate study  (reps={args.reps})")
    print("=" * 78)

    for target in args.targets:
        roi_thr = ROI_THRESHOLD[target]
        runs = {'baseline': [], 'certify': []}
        with tempfile.TemporaryDirectory() as td:
            for mode in ('baseline', 'certify'):
                for seed in seeds:
                    runs[mode].append(_run(target, mode, seed, args.ncores,
                                           os.path.join(td, f'{mode}_{seed}.json')))
            refs = _reference_grids(runs['baseline'] + runs['certify'])
            defs = {m: [_run_quality(r, refs, roi_thr)[0] for r in runs[m]]
                    for m in runs}

        calls = {m: [r['projections'][-1]['cumulative_target_calls']
                     for r in runs[m]] for m in runs}
        tests = [r['pool_cert_tests'] for r in runs['certify']]
        raises = [r['pool_cert_raises'] for r in runs['certify']]
        gain = [r['pool_cert_gain'] for r in runs['certify']]

        bm, _ = _m(calls['baseline'])
        cm, _ = _m(calls['certify'])
        tm, _ = _m(tests)
        rm, _ = _m(raises)
        gm, _ = _m(gain)
        dbm, dbs = _m(defs['baseline'])
        dcm, dcs = _m(defs['certify'])

        print(f"\n### {target}")
        print(f"  pool-cert (on runs): tested {tm:.0f}/scan, raised {rm:.1f}/scan, "
              f"total logL gain {gm:.3g}")
        print(f"  calls: baseline {bm:,.0f}   certify {cm:,.0f}  ({100*(cm-bm)/bm:+.1f}%)")
        print(f"  ROI mean deficit vs reference: baseline {dbm:.4g} +/- {dbs:.2g}   "
              f"certify {dcm:.4g} +/- {dcs:.2g}")

    print("\n" + "=" * 78)


if __name__ == '__main__':
    main()
