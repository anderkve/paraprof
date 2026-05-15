"""
Run a 1D + 2D profile-likelihood scan for a chosen test function and dump
the resulting profile-likelihood grids (plus the cumulative target-function
evaluation count across the scan) to disk, ready to be turned into the
publication-quality README showcase plots by ``make_showcase_plots.py``.

Run with MPI for a single test function, e.g.::

    mpiexec -n 4 python examples/run_showcase_scan.py --function himmelblau_4d
"""
import argparse
import json
import os
import sys

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_scan, worker_main,
    get_test_function, set_log_level,
)

set_log_level('INFO')

# Per-function tuning. Each entry picks the projection dims for the 1D and
# 2D showcase plots and any sampler/projection overrides that materially
# improve the quality of the resulting plot for that landscape.
SHOWCASE_FUNCTIONS = {
    'himmelblau_4d': {
        'dim_1d': 0,
        'dims_2d': [0, 1],
        'grid_1d': 120,
        'grid_2d': 60,
        'roi_threshold': 5.0,
        'n_initial_optimizations': 80,
        'max_patching_waves': 20,
        'lbfgsb_max_iter': 25,
        'advanced_config': {},
    },
    'rosenbrock_4d': {
        'dim_1d': 0,
        'dims_2d': [0, 1],
        'grid_1d': 120,
        'grid_2d': 60,
        'roi_threshold': 8.0,
        'n_initial_optimizations': 100,
        'max_patching_waves': 25,
        'lbfgsb_max_iter': 25,
        'advanced_config': {
            'convergence_threshold': 1e-7,
        },
    },
    'ackley_4d': {
        'dim_1d': 0,
        'dims_2d': [0, 1],
        'grid_1d': 120,
        'grid_2d': 60,
        'roi_threshold': 8.0,
        'n_initial_optimizations': 100,
        'max_patching_waves': 25,
        'lbfgsb_max_iter': 25,
        'advanced_config': {
            'convergence_threshold': 1e-7,
        },
    },
}


def grid_dict_to_array(grid_dict, grid_shape):
    """Convert a {grid_idx: fitness} dict to a dense ndarray with NaN for missing cells."""
    arr = np.full(grid_shape, np.nan)
    for idx, fitness in grid_dict.items():
        arr[idx] = fitness
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--function', required=True,
                        choices=sorted(SHOWCASE_FUNCTIONS.keys()),
                        help='Name of the registered test function to scan.')
    parser.add_argument('--output-dir', default='examples/example_plots/showcase/data',
                        help='Directory to write .npz grid data and the per-function JSON summary.')
    parser.add_argument('--seed', type=int, default=750123)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    np.random.seed(args.seed)

    cfg = SHOWCASE_FUNCTIONS[args.function]
    log_likelihood, param_bounds, _ = get_test_function(args.function)

    projections = [
        # 1D profile, refined 2x with patching on both grids.
        {'dims': [cfg['dim_1d']], 'grid_points': [cfg['grid_1d']],
         'grid_refinement_factor': 2, 'patch_refined_grid': True},
        # 2D profile, refined 2x with patching on both grids.
        {'dims': list(cfg['dims_2d']), 'grid_points': [cfg['grid_2d'], cfg['grid_2d']],
         'grid_refinement_factor': 2, 'patch_refined_grid': True},
    ]

    if myrank != 0:
        worker_main(comm, myrank)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    with ProfileProjector(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=projections,
        roi_threshold=cfg['roi_threshold'],
        pop_per_grid_point=3,
        n_initial_optimizations=cfg['n_initial_optimizations'],
        max_patching_waves=cfg['max_patching_waves'],
        lbfgsb_max_iter=cfg['lbfgsb_max_iter'],
        advanced_config=cfg['advanced_config'],
    ) as sampler:
        results = run_scan(
            comm=comm,
            sampler=sampler,
            projections=projections,
            save_plots=False,
            myrank=myrank,
        )

    # Both per-projection metric snapshots hold the *cumulative* call count
    # at the end of that projection, so the totals across the scan come from
    # the last projection.
    total_target_calls = int(results[-1]['metrics']['total_target_calls'])

    # ``run_projection`` stores the refined solution if refinement ran, else
    # ``coarse_solution``. Use whichever is available for the final grid.
    def best_solution(res):
        return res['refined_solution'] if res['refined_solution'] is not None else res['coarse_solution']

    sol_1d = best_solution(results[0])
    sol_2d = best_solution(results[1])

    # Pack 1D data
    grid_1d_axes = sol_1d['grid_axes']
    grid_1d_shape = sol_1d['grid_shape']
    likelihood_1d = grid_dict_to_array(
        {idx: s['likelihood'] for idx, s in sol_1d['solutions'].items()},
        grid_1d_shape,
    )

    # Pack 2D data
    grid_2d_axes = sol_2d['grid_axes']
    grid_2d_shape = sol_2d['grid_shape']
    likelihood_2d = grid_dict_to_array(
        {idx: s['likelihood'] for idx, s in sol_2d['solutions'].items()},
        grid_2d_shape,
    )

    out_npz = os.path.join(args.output_dir, f'{args.function}.npz')
    np.savez_compressed(
        out_npz,
        function=args.function,
        total_target_calls=total_target_calls,
        # 1D
        proj_dim_1d=cfg['dim_1d'],
        axis_1d=grid_1d_axes[0],
        likelihood_1d=likelihood_1d,
        # 2D
        proj_dims_2d=np.array(cfg['dims_2d'], dtype=int),
        axis_2d_x=grid_2d_axes[0],
        axis_2d_y=grid_2d_axes[1],
        likelihood_2d=likelihood_2d,
    )

    out_json = os.path.join(args.output_dir, f'{args.function}.json')
    with open(out_json, 'w') as f:
        json.dump({
            'function': args.function,
            'total_target_calls': total_target_calls,
            'projections': [
                {
                    'dims': r['projection_config']['dims'],
                    'grid_points': r['projection_config']['grid_points'],
                    'coarse_target_calls': int(r['metrics']['coarse_target_calls']),
                    'refined_target_calls': int(r['metrics'].get('refined_target_calls', r['metrics']['coarse_target_calls'])),
                    'global_max': float(r['metrics']['global_max']),
                }
                for r in results
            ],
        }, f, indent=2)

    print(f"[{args.function}] total target-function evaluations: {total_target_calls}")
    print(f"[{args.function}] grid data: {out_npz}")
    print(f"[{args.function}] summary:   {out_json}")


if __name__ == '__main__':
    sys.exit(main())
