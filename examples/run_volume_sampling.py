"""
Example: ROI/shell volume sampling after a profile-likelihood scan.

After scanning two 2D projections of the 4D Himmelblau log-likelihood,
the volume-sampling stage collects a stratified, well-spread set of
samples in the full 4D good-fit region (``mode='roi'``) or in a band just
outside it (``mode='shell'``) — the parts of parameter space the profile
surfaces themselves never cover. See docs/volume_sampling_plan.md for the
design. Run with MPI:

    mpiexec -n <ncores> python run_volume_sampling.py [roi|shell]

Required: at least 2 MPI ranks (1 master + 1+ workers).

Outputs (in the working directory):
- ``volume_samples.csv``       tagged samples: [params..., logL, tag]
- ``volume_samples_summary.json``  stage statistics incl. volume estimate
- ``volume_plot_*_volume_*.png``   samples over the 2D profile maps
"""
import sys

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, plot_volume_samples, run_scan, worker_main,
    get_test_function, set_log_level,
)

set_log_level('INFO')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

np.random.seed(750123)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'roi'

log_likelihood, param_bounds, _ = get_test_function('himmelblau_4d')

PROJECTIONS = [
    {'dims': [0, 1], 'grid_points': [40, 40]},
    {'dims': [2, 3], 'grid_points': [40, 40]},
]

if myrank == 0:
    with ProfileProjector(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS,
        roi_threshold=4.0,
        pop_per_grid_point=3,

        # The harvest tier streams this file, so the scan's own
        # evaluations seed the volume stage for free.
        samples_output_file="samples.csv",

        volume_sampling={
            'mode': MODE,                # 'roi' or 'shell'
            'shell_threshold': 25.0,     # outer edge (shell mode only)
            'n_points': 500,             # anchors = target sample count
            'output_file': "volume_samples.csv",
        },
    ) as sampler:
        results = run_scan(
            comm=comm, sampler=sampler, projections=PROJECTIONS,
            save_plots=True, myrank=myrank,
        )

        vol = sampler.volume_stage_result
        if not vol['skipped']:
            stats = vol['stats']
            print(f"\nVolume sampling ({MODE}):")
            print(f"  covered {stats['n_covered']}, projected "
                  f"{stats['n_projected']}, holes {stats['n_holes']} "
                  f"of {stats['n_anchors']} anchors "
                  f"({stats['evals_used']} stage evaluations)")
            if stats['volume_estimate'] is not None:
                print(f"  band volume estimate: "
                      f"{stats['volume_estimate']:.4g} "
                      f"+/- {stats['volume_estimate_err']:.2g}")

            # Scatter the tagged samples over both 2D profile maps.
            for res in results:
                grid = (res['refined_solution']
                        or res['coarse_solution'])
                plot_volume_samples(
                    vol, dims=tuple(grid['projection_dims']),
                    filename=f"volume_plot_{MODE}",
                    grid_solution=grid,
                )
else:
    worker_main(comm, myrank)
