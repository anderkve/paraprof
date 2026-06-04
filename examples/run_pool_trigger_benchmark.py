"""
A/B runner for the cross-projection second trigger of smooth-certify
(idea 3 flavor b, folded into ``de.smooth_certify``).

``de.smooth_certify`` is ON in both arms; the only difference is whether the
cross-projection trigger (``sampler.de_pool_certify_trigger``) is allowed to
fire in addition to the neighbour-agreement trigger:

  * mode ``neighbour`` -- neighbour trigger only (the existing behaviour);
  * mode ``both``      -- neighbour trigger + cross-projection trigger.

So the call-count difference isolates the *marginal* savings the second trigger
adds. The summary also records how many cells were certified in total and how
many *only* the cross-projection trigger caught.

Run with:
    mpiexec -n <ncores> python examples/run_pool_trigger_benchmark.py \\
        --target himmelblau_4d --mode both --seed 750123 --out /tmp/h.json
"""
import argparse
import json
import time
import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_scan, worker_main, get_test_function, set_log_level,
)
from run_smooth_certify_benchmark import TARGET_KWARGS, _make_projections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--mode', required=True, choices=['neighbour', 'both'])
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
        advanced_config['de'] = {'smooth_certify': True}  # ON in both arms

        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood, bounds=bounds, projections=projections,
            advanced_config=advanced_config, **cfg['kwargs'],
        ) as sampler:
            # Toggle only the second (cross-projection) trigger.
            sampler.de_pool_certify_trigger = (args.mode == 'both')
            results = run_scan(comm=comm, sampler=sampler,
                               projections=projections, save_plots=False,
                               myrank=myrank)
            certified = sampler.de_cells_smooth_certified
            pool_only = sampler.de_cells_certified_pool_only
        elapsed = time.time() - t0

        summary = {
            'target': args.target, 'mode': args.mode, 'seed': args.seed,
            'elapsed_s': elapsed, 'n_ranks': comm.Get_size(),
            'cells_certified': int(certified),
            'cells_certified_pool_only': int(pool_only),
            'projections': [],
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
        print(f"[{args.target}/{args.mode}] "
              f"calls={results[-1]['metrics']['total_target_calls']}, "
              f"certified={certified} (pool-only={pool_only})")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
