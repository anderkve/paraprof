"""
Single-configuration runner for the cross-projection pool-certificate study
(``cross_projection.pool_certificate``, idea 3 flavor a).

Runs every 2D projection of one N-D test target at 50x50 grids with the
pool-certificate pass either off or on, and writes a JSON summary -- per
projection cumulative target_calls + the final coarse grid, plus the cumulative
pool-certificate diagnostics (cells tested / raised / total logL gained) -- to
``--out``. Driven from ``run_pool_certificate_study.py``.

Run with:
    mpiexec -n <ncores> python examples/run_pool_certificate_benchmark.py \\
        --target himmelblau_4d --mode certify --seed 750123 --out /tmp/h.json
"""
import argparse
import itertools
import json
import time
import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_scan, worker_main, get_test_function, set_log_level,
)

# Reuse the smooth-certify benchmark's per-target settings verbatim.
from run_smooth_certify_benchmark import TARGET_KWARGS, _make_projections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--mode', required=True, choices=['baseline', 'certify'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=750123)
    args = parser.parse_args()

    set_log_level('WARNING')
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    np.random.seed(args.seed)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]
    projections = _make_projections(cfg['dims'])

    if myrank == 0:
        advanced_config = dict(cfg['advanced_config'] or {})
        advanced_config['cross_projection'] = {
            'pool_certificate': (args.mode == 'certify'),
        }
        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood, bounds=bounds, projections=projections,
            advanced_config=advanced_config, **cfg['kwargs'],
        ) as sampler:
            results = run_scan(comm=comm, sampler=sampler,
                               projections=projections, save_plots=False,
                               myrank=myrank)
            pc = (sampler.pool_cert_tests, sampler.pool_cert_raises,
                  sampler.pool_cert_gain)
        elapsed = time.time() - t0

        summary = {
            'target': args.target, 'mode': args.mode, 'seed': args.seed,
            'elapsed_s': elapsed, 'n_ranks': comm.Get_size(),
            'pool_cert_tests': int(pc[0]), 'pool_cert_raises': int(pc[1]),
            'pool_cert_gain': float(pc[2]), 'projections': [],
        }
        for i, r in enumerate(results):
            cfg_i = projections[i]
            grid_pts = [n + 1 for n in cfg_i['grid_points']]
            grid = np.full(grid_pts, np.nan)
            for idx, sol in r['coarse_solution'].get('solutions', {}).items():
                grid[tuple(idx)] = sol['likelihood']
            summary['projections'].append({
                'dims': cfg_i['dims'],
                'cumulative_target_calls': int(r['metrics']['total_target_calls']),
                'coarse_grid_values': grid.tolist(),
            })
        with open(args.out, 'w') as f:
            json.dump(summary, f)
        print(f"[{args.target}/{args.mode}] calls={results[-1]['metrics']['total_target_calls']}, "
              f"pool_cert tests/raises/gain = {pc[0]}/{pc[1]}/{pc[2]:.4g}")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
