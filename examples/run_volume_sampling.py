"""
Example: volume sampling after a profile-likelihood scan.

After scanning two 2D projections of the 4D Himmelblau log-likelihood, the
volume-sampling stage populates the full 4D good-fit region with a set of
samples that balances space-filling and lnL-filling — using an
affine-invariant ensemble of umbrella walkers seeded inside the projection
envelope (see docs/volume_sampling_plan.md). Passing a threshold on the
command line widens the stage's own ROI beyond the projection's
``roi_threshold``, so the sampling reaches into the shell outside the
good-fit region. Run with MPI:

    mpiexec -n <ncores> python run_volume_sampling.py [roi_threshold]

Required: at least 2 MPI ranks (1 master + 1+ workers).

Outputs (in the working directory):
- ``volume_samples.csv``            in-band samples: [params..., logL]
- ``volume_samples_summary.json``   stage statistics incl. the lnL histogram
- ``volume_plot_*_volume_*.png``    samples (coloured by logL) over the maps
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

# Optional: the volume stage's own ROI threshold (ΔlnL band depth). Defaults
# to the projection's roi_threshold; a larger value also explores the shell.
VOLUME_ROI_THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else None

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

        # The walkers warm-start from this file's scan samples near each
        # home level, so the scan's work seeds the volume stage for free.
        samples_output_file="samples.csv",

        volume_sampling={
            # None = use the projection's roi_threshold; larger reaches into
            # the shell outside the good-fit region.
            'roi_threshold': VOLUME_ROI_THRESHOLD,
            'n_walkers': 1000,           # ensemble size
            'n_steps': 30,               # stretch sweeps per walker
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
            print("\nVolume sampling:")
            print(f"  {stats['n_in_band']} in-band samples from "
                  f"{stats['n_evals']} evaluations "
                  f"(acceptance {stats['mean_acceptance']:.2f})")

            # Scatter the samples (coloured by logL) over both 2D profile maps.
            for res in results:
                grid = res['refined_solution'] or res['coarse_solution']
                plot_volume_samples(
                    vol, dims=tuple(grid['projection_dims']),
                    filename="volume_plot", grid_solution=grid,
                )
else:
    worker_main(comm, myrank)
