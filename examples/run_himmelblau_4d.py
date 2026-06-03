"""
Minimal example: profile-likelihood scan of the Himmelblau-4D test function.

Demonstrates the core ProfileProjector API, saving samples to an HDF5 file
(requires the optional dependency: ``pip install paraprof[hdf5]``). Run with MPI:

    mpiexec -n <ncores> python run_himmelblau_4d.py

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

# Test function: 4D Himmelblau (4 known peaks, max log L = 0).
log_likelihood, param_bounds, _ = get_test_function('himmelblau_4d')

# A few projections, all at the same resolution. Only `dims` and `grid_points`
# are required; the others are optional per-projection overrides.
PROJECTIONS = [
    # 1D profiles
    {'dims': [0], 'grid_points': [100]},
    {'dims': [1], 'grid_points': [100]},

    # 2D profiles, with grid refinement (2x denser refined grid + patching on both)
    {'dims': [0, 1], 'grid_points': [50, 50],
     'grid_refinement_factor': 2, 'patch_refined_grid': True},
    {'dims': [2, 3], 'grid_points': [50, 50],
     'grid_refinement_factor': 2, 'patch_refined_grid': True},
]

if myrank == 0:
    # HDF5 binary samples (the format follows the extension; use ".csv" for text).
    output_file = f"samples_rank_{myrank}.h5"
    # Warm-start from a *previous* run's samples, kept in a separate file from
    # the one we write now. It is absent on the first run, so warm-start is
    # skipped; to chain runs, rename a prior output_file to this path and re-run.
    warm_start_file = f"warm_start_rank_{myrank}.h5"

    with ProfileProjector(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS,

        # Core tuning
        roi_threshold=4.0,            # chi-squared units; cells within (max - this) are "in ROI"
        pop_per_grid_point=3,         # DE population size per grid cell
        n_initial_optimizations=80,   # global L-BFGS-B starts before grid optimization
        max_patching_waves=20,        # cap on patching iterations
        lbfgsb_max_iter=20,           # L-BFGS-B iterations per polish

        # I/O. Reading and writing use different files, so the same HDF5 file is
        # never open for read and write at once. (On some shared/HPC filesystems
        # HDF5 file locking can fail; export HDF5_USE_FILE_LOCKING=FALSE if you
        # hit an "unable to lock file" error.)
        samples_output_file=output_file,
        warm_start_file=warm_start_file,
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
