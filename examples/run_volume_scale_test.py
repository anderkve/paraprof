"""
Large-scale volume-sampling test driver.

Runs all six 2D projections of a 4D test function, then a volume-sampling
stage whose anchor count (`n_points`) equals the number of target
evaluations spent on the projections, with the stage `roi_threshold` set
to the projection `roi_threshold` + 2 (reaching into the shell).

The phase-tagged sample log (`samples_<func>.csv`) accumulates every
evaluation from both stages, which is the "total sample set" the
companion plotting script visualises.

    mpiexec -n <ncores> python run_volume_scale_test.py <func> <roi> [grid]

e.g.  mpiexec -n 4 python run_volume_scale_test.py himmelblau_4d 4.0 26
"""
import json
import sys
import time

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, get_test_function, run_all_projections,
    run_volume_sampling, set_log_level, terminate_workers, worker_main,
)
from paraprof.volume import normalize_volume_config

set_log_level('INFO')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

func_name = sys.argv[1]
roi_threshold = float(sys.argv[2])
grid = int(sys.argv[3]) if len(sys.argv) > 3 else 26

np.random.seed(20260613)

log_likelihood, bounds, _ = get_test_function(func_name)

# All six 2D projections of the 4D space.
PROJECTIONS = [
    {'dims': [i, j], 'grid_points': [grid, grid]}
    for i in range(4) for j in range(i + 1, 4)
]

samples_file = f"samples_{func_name}.csv"
volume_file = f"volume_{func_name}.csv"

if rank == 0:
    comm.bcast((log_likelihood, None), root=0)
    t0 = time.time()

    with ProfileProjector(
        target_func=log_likelihood,
        bounds=bounds,
        projections=PROJECTIONS,
        roi_threshold=roi_threshold,
        pop_per_grid_point=3,
        n_initial_optimizations=60,
        samples_output_file=samples_file,
    ) as sampler:

        results = run_all_projections(
            comm=comm, sampler=sampler, projections=PROJECTIONS,
            save_plots=False, myrank=rank,
        )
        n_projection_evals = sampler.target_calls
        t_proj = time.time()

        # Volume sampling: anchors == projection evaluations; band depth
        # = projection roi_threshold + 2 (reach into the shell).
        sampler.volume_sampling_config = normalize_volume_config(
            {
                'roi_threshold': roi_threshold + 2.0,
                'n_points': n_projection_evals,
                'output_file': volume_file,
            },
            roi_threshold=roi_threshold,
        )
        vol = run_volume_sampling(comm, sampler, results, myrank=rank)
        sampler._flush_samples_buffer()
        t_vol = time.time()

        total_evals = sampler.target_calls
        summary = {
            'function': func_name,
            'projection_roi_threshold': roi_threshold,
            'volume_roi_threshold': roi_threshold + 2.0,
            'grid': grid,
            'n_projection_evals': int(n_projection_evals),
            'volume_n_points': int(n_projection_evals),
            'n_volume_evals': int(total_evals - n_projection_evals),
            'total_evals': int(total_evals),
            'global_max': float(sampler.global_max_target_val),
            'projection_seconds': round(t_proj - t0, 1),
            'volume_seconds': round(t_vol - t_proj, 1),
        }
        if not vol.get('skipped'):
            stats = vol['stats']
            summary['volume_stats'] = {
                'n_anchors': int(stats['n_anchors']),
                'n_covered': int(stats['n_covered']),
                'n_projected': int(stats['n_projected']),
                'n_holes': int(stats['n_holes']),
                'prefilter_acceptance': stats['prefilter_acceptance'],
                'probe_acceptance': stats['probe_acceptance'],
                'volume_estimate': stats['volume_estimate'],
            }
        else:
            summary['volume_skipped_reason'] = vol.get('reason')

        with open(f"scale_summary_{func_name}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("SCALE_SUMMARY", json.dumps(summary), flush=True)

    terminate_workers(comm, myrank=rank)
else:
    worker_main(comm, rank)
