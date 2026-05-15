"""
Single-configuration runner for the proximity-warm-start benchmark.

Runs all six 2D projections of one 4D test target at 50x50 grids, with the
proximity warm-start feature either enabled or disabled, and writes a JSON
summary (per-projection cumulative target_calls + the final coarse-grid
profile values per projection) to ``--out``.

Intended to be driven from ``run_proximity_warm_start_benchmark_driver.py``,
which subprocesses four invocations of this script (himmelblau/rosenbrock x
baseline/proximity) and prints a single comparison report. You can also run
this script directly to inspect a single configuration.

Run with:
    mpiexec -n <ncores> python examples/run_proximity_warm_start_benchmark.py \\
        --target himmelblau_4d --mode proximity --out /tmp/h_p.json

Required: at least 2 MPI ranks.
"""
import argparse
import json
import time
import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_scan, worker_main, get_test_function, set_log_level,
)


TARGET_KWARGS = {
    'himmelblau_4d': {
        'kwargs': dict(
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=80,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
        ),
        'advanced_config': None,
    },
    'rosenbrock_4d': {
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=100,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
        ),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
}

# All six 2D projections of the four parameters.
PROJECTIONS = [{'dims': [i, j], 'grid_points': [50, 50]}
               for i in range(4) for j in range(i + 1, 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--mode', required=True, choices=['baseline', 'proximity'])
    parser.add_argument('--out', required=True,
                        help='Path to write the JSON summary on rank 0.')
    args = parser.parse_args()

    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    # Same RNG seed for both modes -> the only difference is one LHS slot
    # being replaced by a proximity warm-start sample when --mode proximity.
    np.random.seed(750123)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]

    if myrank == 0:
        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=PROJECTIONS,
            advanced_config=cfg['advanced_config'],
            **cfg['kwargs'],
        ) as sampler:
            on = (args.mode == 'proximity')
            sampler._proximity_warm_start = on
            sampler._pool_seeded_initial_maxima = on
            results = run_scan(
                comm=comm, sampler=sampler, projections=PROJECTIONS,
                save_plots=False, myrank=myrank,
            )
        elapsed = time.time() - t0

        summary = {
            'target': args.target,
            'mode': args.mode,
            'elapsed_s': elapsed,
            'n_ranks': comm.Get_size(),
            'projections': [],
        }
        for i, r in enumerate(results):
            cfg_i = PROJECTIONS[i]
            grid_pts = [n + 1 for n in cfg_i['grid_points']]  # +1 for endpoints
            grid = np.full(grid_pts, np.nan)
            for idx, sol in r['coarse_solution'].get('solutions', {}).items():
                grid[tuple(idx)] = sol['likelihood']
            summary['projections'].append({
                'dims': cfg_i['dims'],
                'cumulative_target_calls': int(r['metrics']['total_target_calls']),
                'global_max': float(r['metrics']['global_max']),
                'coarse_grid_shape': list(grid.shape),
                'coarse_grid_values': grid.tolist(),
            })
        with open(args.out, 'w') as f:
            json.dump(summary, f)
        print(f"[{args.target}/{args.mode}] elapsed={elapsed:.1f}s, "
              f"final calls={results[-1]['metrics']['total_target_calls']}")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
