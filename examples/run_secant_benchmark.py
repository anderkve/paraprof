"""
Single-configuration runner for the secant-predictor benchmark.

Runs every 2D projection of one N-D test target at a configurable grid
resolution, with the secant-predictor warm-start either enabled or disabled.
Writes a JSON summary on rank 0 containing per-projection target-call
counts, the coarse-grid result, and the secant diagnostic counters.

This is a leaner, secant-only variant of
``run_continuation_benchmark.py``; it does not exercise the
``online_basin_switch`` hook and exposes the grid resolution as a CLI
argument so the driver can sweep multiple grid sizes per target.

Run with:
    mpiexec -n <ncores> python examples/run_secant_benchmark.py \\
        --target rosenbrock_4d --config secant --grid 30 --seed 0 \\
        --out /tmp/run.json
"""
import argparse
import itertools
import json
import time

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, get_test_function, set_log_level, terminate_workers,
    worker_main,
)
from paraprof.master import run_projection


CONFIGS = ['baseline', 'secant']


# Per-target sampler kwargs. The grid resolution is supplied on the
# command line so a single target can be benchmarked at multiple
# resolutions without editing this table.
TARGET_KWARGS = {
    'rosenbrock_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=8.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
    'rosenbrock_6d': {
        'dims': 6,
        'kwargs': dict(roi_threshold=8.0, pop_per_grid_point=3,
                       n_initial_optimizations=60, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
    'himmelblau_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=4.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'rastrigin_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=8.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'rastrigin_6d': {
        'dims': 6,
        'kwargs': dict(roi_threshold=8.0, pop_per_grid_point=3,
                       n_initial_optimizations=60, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'levy_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=4.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'styblinski_tang_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=4.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'ackley_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=4.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
    'griewank_4d': {
        'dims': 4,
        'kwargs': dict(roi_threshold=4.0, pop_per_grid_point=3,
                       n_initial_optimizations=40, max_patching_waves=10,
                       lbfgsb_max_iter=15),
        'advanced_config': None,
    },
}


def _make_projections(n_dims, grid):
    """All C(n_dims, 2) 2D projections at the given grid resolution."""
    return [{'dims': [i, j], 'grid_points': [grid, grid]}
            for i, j in itertools.combinations(range(n_dims), 2)]


def _advanced_config_for(config_name, base):
    """Apply the secant switch on top of any target-specific advanced cfg.

    Both runs explicitly disable ``online_basin_switch`` so the only
    independent variable across the two configs is the secant predictor.
    """
    out = dict(base or {})
    out['continuation'] = {
        'secant_predictor_warm_start': (config_name == 'secant'),
        'online_basin_switch': False,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--config', required=True, choices=CONFIGS)
    parser.add_argument('--grid', type=int, required=True,
                        help='Grid resolution (per axis) for every 2D projection')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True,
                        help='Path to write the JSON summary on rank 0.')
    args = parser.parse_args()

    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    np.random.seed(args.seed)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]
    projections = _make_projections(cfg['dims'], args.grid)

    if myrank == 0:
        advanced_config = _advanced_config_for(args.config, cfg['advanced_config'])

        per_projection_diag = []

        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            advanced_config=advanced_config,
            **cfg['kwargs'],
        ) as sampler:
            results = []
            comm.bcast(sampler.target_func, root=0)
            for idx, proj in enumerate(projections):
                if idx > 0:
                    sampler._reset_for_new_projection(proj)
                rs = run_projection(
                    comm=comm,
                    sampler=sampler,
                    projection_config=proj,
                    save_plots=False,
                    plot_settings=None,
                    skip_init_opt_on_warm_start=(idx > 0),
                    myrank=myrank,
                )
                rs['projection_config'] = proj
                results.append(rs)
                per_projection_diag.append({
                    'secant_tested': int(sampler._secant_predictor_candidates_tested),
                    'secant_won': int(sampler._secant_predictor_candidates_won),
                    'patching_tests_total': int(sampler._patching_tests_total),
                    'patching_improvements_total': int(sampler._patching_improvements_total),
                })
            terminate_workers(comm, myrank)
        elapsed = time.time() - t0

        summary = {
            'target': args.target,
            'config': args.config,
            'grid': args.grid,
            'seed': args.seed,
            'elapsed_s': elapsed,
            'n_ranks': comm.Get_size(),
            'projections': [],
        }
        for i, r in enumerate(results):
            cfg_i = projections[i]
            grid_pts = [n + 1 for n in cfg_i['grid_points']]
            grid_arr = np.full(grid_pts, np.nan)
            for idx, sol in r['coarse_solution'].get('solutions', {}).items():
                grid_arr[tuple(idx)] = sol['likelihood']
            summary['projections'].append({
                'dims': cfg_i['dims'],
                'cumulative_target_calls': int(r['metrics']['total_target_calls']),
                'global_max': float(r['metrics']['global_max']),
                'coarse_grid_shape': list(grid_arr.shape),
                'coarse_grid_values': grid_arr.tolist(),
                'diagnostics': per_projection_diag[i],
            })
        with open(args.out, 'w') as f:
            json.dump(summary, f)
        print(f"[{args.target}/grid={args.grid}/{args.config}/seed={args.seed}] "
              f"elapsed={elapsed:.1f}s, "
              f"final calls={results[-1]['metrics']['total_target_calls']}")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
