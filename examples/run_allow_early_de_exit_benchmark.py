"""
A/B runner for the neighbour-certified DE-skip feature (``de.allow_early_DE_exit``).

Runs every 2D projection of one N-D test target at 50x50 grids twice -- once
with the DE-skip on, once off -- and writes a JSON summary
(per-projection cumulative target_calls + the final coarse-grid profile values)
to ``--out``. The driver subprocesses the two modes and reports the
target-call delta together with a ROI grid-quality comparison, so a call-count
win is only counted if grid quality is preserved.

Run one configuration directly with:
    mpiexec -n <ncores> python examples/run_allow_early_de_exit_benchmark.py \\
        --target himmelblau_4d --mode certify --out /tmp/h_certify.json

Required: at least 2 MPI ranks.
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


# Same targets/settings as the proximity-warm-start benchmark, so the two
# features are measured on comparable ground. Cross-projection transfer is left
# at its default (on) -- allow_early_DE_exit should compose with it.
TARGET_KWARGS = {
    'himmelblau_4d': {
        'dims': 4,
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
        'dims': 4,
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=100,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
        ),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
    'rosenbrock_6d': {
        'dims': 6,
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=120,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
        ),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
    'rastrigin_4d': {
        'dims': 4,
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=100,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
        ),
        'advanced_config': None,
    },
}


def _make_projections(n_dims):
    """All C(n_dims, 2) 2D projections at a 50x50 grid each."""
    return [{'dims': [i, j], 'grid_points': [50, 50]}
            for i, j in itertools.combinations(range(n_dims), 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--mode', required=True, choices=['baseline', 'certify'])
    parser.add_argument('--out', required=True,
                        help='Path to write the JSON summary on rank 0.')
    parser.add_argument('--seed', type=int, default=750123,
                        help='Master RNG seed (vary it for replicate runs).')
    args = parser.parse_args()

    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    # Master RNG seed. For a single A/B the same seed is used for both modes so
    # the only intended difference is whether the certified DE skip fires; the
    # replicate study varies it to sample independent scans.
    np.random.seed(args.seed)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]
    projections = _make_projections(cfg['dims'])

    if myrank == 0:
        advanced_config = dict(cfg['advanced_config'] or {})
        advanced_config['de'] = {'allow_early_DE_exit': (args.mode == 'certify')}

        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            advanced_config=advanced_config,
            **cfg['kwargs'],
        ) as sampler:
            results = run_scan(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False, myrank=myrank,
            )
            certified = sampler.de_cells_skipped
        elapsed = time.time() - t0

        summary = {
            'target': args.target,
            'mode': args.mode,
            'seed': args.seed,
            'elapsed_s': elapsed,
            'n_ranks': comm.Get_size(),
            'cells_skipped': int(certified),
            'projections': [],
        }
        for i, r in enumerate(results):
            cfg_i = projections[i]
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
              f"final calls={results[-1]['metrics']['total_target_calls']}, "
              f"certified={certified}")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
