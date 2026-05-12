"""
Minimal example: profile-likelihood scan of the Rosenbrock-4D test function.

Demonstrates the core ProfileProjector API on a stiff narrow-valley problem.
For Rosenbrock the true peak is at (1, 1, 1, 1) with log L = 0; the narrow
valley benefits from a tight `convergence_threshold` and (if accuracy
matters) `lbfgsb.gradient_method='central'`.

Run with MPI:

    mpiexec -n <ncores> python run_rosenbrock_4d.py

Required: at least 2 MPI ranks (1 master + 1+ workers).
"""
import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_scan, worker_main,
    get_test_function, set_log_level,
)

set_log_level('INFO')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

np.random.seed(750123)

log_likelihood, param_bounds, _ = get_test_function('rosenbrock_4d')

# All six 2D projections, with a 2x grid refinement. Patching is on by
# default for the coarse grid; we also enable it on the refined grid here.
PROJECTIONS = [
    {'dims': [i, j], 'grid_points': [50, 50],
     'grid_refinement_factor': 2, 'patch_refined_grid': True}
    for i in range(4) for j in range(i + 1, 4)
]

if myrank == 0:
    output_file = f"samples_rank_{myrank}.csv"

    with ProfileProjector(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS,

        # Core tuning
        roi_threshold=8.0,            # Rosenbrock's deep valley benefits from a wider ROI
        pop_per_grid_point=3,
        n_initial_optimizations=100,
        max_patching_waves=20,
        lbfgsb_max_iter=20,

        # I/O
        samples_output_file=output_file,

        # Tighter DE convergence helps in the narrow Rosenbrock valley.
        # If you need the highest grid accuracy at the cost of ~50% more
        # target evaluations, also set 'lbfgsb': {'gradient_method': 'central'}.
        advanced_config={
            'convergence_threshold': 1e-7,
        },
    ) as sampler:

        # run_scan handles broadcasting target_func, running every projection
        # and terminating workers, so the main script doesn't have to.
        results = run_scan(
            comm=comm,
            sampler=sampler,
            projections=PROJECTIONS,
            save_plots=True,
            plot_settings={'dpi': 200, 'filetype': 'png'},
            myrank=myrank,
        )

        print("\n=== Summary ===")
        for i, res in enumerate(results):
            dims = res['projection_config']['dims']
            calls = res['metrics']['total_target_calls']
            max_ll = res['metrics']['global_max']
            print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
else:
    worker_main(comm, myrank)
