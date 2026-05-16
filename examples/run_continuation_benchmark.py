"""
Single-configuration runner for the continuation-hooks benchmark.

Runs every 2D projection of one N-D test target at a fixed grid resolution,
with one of four combinations of the new continuation hooks enabled:

  * ``baseline``  — both off (current default behavior reproduced)
  * ``secant``    — secant predictor warm-start only
  * ``basin``     — online basin-switch detection only
  * ``both``      — both on (the default since the hooks were added)

Writes a JSON summary on rank 0 with per-projection target-call counts,
the post-run coarse grid, and the diagnostic counters
(``_secant_predictor_candidates_*``, ``_online_basin_switch_*``,
``_patching_tests_total``, ``_patching_improvements_total``) summed across
all projections in the scan.

Intended to be driven by ``run_continuation_benchmark_driver.py``; can be
invoked directly for a single configuration.

Run with:
    mpiexec -n <ncores> python examples/run_continuation_benchmark.py \\
        --target himmelblau_4d --config both --seed 1 --out /tmp/h_both.json
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


CONFIGS = ['baseline', 'secant', 'basin', 'both']


# Per-target sampler kwargs. Kept small / fast so the full 4-config × N-seed
# sweep can complete in a single benchmarking session.
TARGET_KWARGS = {
    'himmelblau_4d': {
        'dims': 4,
        'grid': 30,
        'kwargs': dict(
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=40,
            max_patching_waves=10,
            lbfgsb_max_iter=15,
        ),
        'advanced_config': None,
    },
    'rosenbrock_4d': {
        'dims': 4,
        'grid': 25,
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=40,
            max_patching_waves=10,
            lbfgsb_max_iter=15,
        ),
        'advanced_config': {'convergence_threshold': 1e-7},
    },
    'rastrigin_4d': {
        'dims': 4,
        'grid': 25,
        'kwargs': dict(
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=40,
            max_patching_waves=10,
            lbfgsb_max_iter=15,
        ),
        'advanced_config': None,
    },
}


def _make_projections(n_dims, grid):
    """All C(n_dims, 2) 2D projections at the given grid resolution."""
    return [{'dims': [i, j], 'grid_points': [grid, grid]}
            for i, j in itertools.combinations(range(n_dims), 2)]


def _advanced_config_for(config_name, base):
    """Apply the secant / basin switches on top of any target-specific config."""
    secant_on = config_name in ('secant', 'both')
    basin_on = config_name in ('basin', 'both')
    out = dict(base or {})
    out['continuation'] = {
        'secant_predictor_warm_start': secant_on,
        'online_basin_switch': basin_on,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=sorted(TARGET_KWARGS))
    parser.add_argument('--config', required=True, choices=CONFIGS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True,
                        help='Path to write the JSON summary on rank 0.')
    args = parser.parse_args()

    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    # All four configs share the seed so the only differences across configs
    # come from the continuation hooks (not the activation RNG, LHS, etc.).
    np.random.seed(args.seed)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]
    projections = _make_projections(cfg['dims'], cfg['grid'])

    if myrank == 0:
        advanced_config = _advanced_config_for(args.config, cfg['advanced_config'])

        # Per-projection diagnostic accumulators. The sampler counters are
        # reset at every _reset_for_new_projection call, so we snapshot them
        # at the end of each projection (inside the run_scan loop is awkward;
        # we instead post-process per-projection metrics via the returned
        # results plus rank-0 callbacks below).
        per_projection_diag = []

        t0 = time.time()
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            advanced_config=advanced_config,
            **cfg['kwargs'],
        ) as sampler:
            # We run the projections one at a time so we can snapshot the
            # diagnostic counters before they're reset for the next one.
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
                    'basin_switch_tests': int(sampler._online_basin_switch_tests),
                    'basin_switch_improvements': int(sampler._online_basin_switch_improvements),
                    'patching_tests_total': int(sampler._patching_tests_total),
                    'patching_improvements_total': int(sampler._patching_improvements_total),
                })
            terminate_workers(comm, myrank)
        elapsed = time.time() - t0

        summary = {
            'target': args.target,
            'config': args.config,
            'seed': args.seed,
            'elapsed_s': elapsed,
            'n_ranks': comm.Get_size(),
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
                'global_max': float(r['metrics']['global_max']),
                'coarse_grid_shape': list(grid.shape),
                'coarse_grid_values': grid.tolist(),
                'diagnostics': per_projection_diag[i],
            })
        with open(args.out, 'w') as f:
            json.dump(summary, f)
        print(f"[{args.target}/{args.config}/seed={args.seed}] "
              f"elapsed={elapsed:.1f}s, "
              f"final calls={results[-1]['metrics']['total_target_calls']}")
    else:
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
