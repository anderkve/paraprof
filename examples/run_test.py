"""
Example script: Running the Grid-Anchored DE Sampler on a test function.

Usage:
    mpiexec -n <number_of_cores> python run_test.py
"""
import sys
import os
import numpy as np

# Add parent directory to path to import paraprof
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function

from paraprof import set_log_level
# set_log_level('DEBUG')
set_log_level('INFO')
# set_log_level('WARNING')


# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "rosenbrock_4d"
# TEST_FUNCTION = "rosenbrock_6d"
# TEST_FUNCTION = "sphere_6d"
# TEST_FUNCTION = "beale_2d"
# TEST_FUNCTION = "eggholder_2d"
# TEST_FUNCTION = "eggholder_4d"
# TEST_FUNCTION = "rastrigin_2d"
# TEST_FUNCTION = "ackley_4d"
# TEST_FUNCTION = "michalewicz_2d"
# TEST_FUNCTION = "michalewicz_4d"

# ============================================================================
# PROJECTION CONFIGURATION GUIDE
# ============================================================================
#
# ParaProf now supports two modes:
#
# 1. NORMAL MODE (n_continuous_dims > 0):
#    - Projects onto fewer dimensions than the function has
#    - Optimizes continuous dimensions at each grid point
#    - Example: 4D function with dims=[0,1] projects onto 2D, optimizes [2,3]
#
# 2. DIRECT EVALUATION MODE (n_continuous_dims == 0):
#    - NEW! Projects onto ALL dimensions of the function
#    - Evaluates directly at grid points (no optimization needed)
#    - Uses intelligent sparse grid activation for efficiency
#    - Example: 2D function with dims=[0,1] evaluates at 2D grid points
#
# ============================================================================

PROJECTIONS_TO_RUN = [
    # For 2D functions with DIRECT EVALUATION MODE (projects onto both dims)
    # Iteration 4: Higher resolution grid for better coverage
    {'dims': [0, 3], 'grid_points': [100, 100], 'enable_refinement': True, 'refinement_factor': 2},

    # Alternative: 1D projections with optimization (one continuous dim)
    # {'dims': [0], 'grid_points': [75], 'patching_coarse': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [1], 'grid_points': [75], 'patching_coarse': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
]

# For higher dimensional functions with NORMAL MODE (optimization at each grid point):
# Example for 4D function:
# PROJECTIONS_TO_RUN = [
#     {'dims': [0, 1], 'grid_points': [50, 50], 'patching_coarse': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
# ]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    # Use a single output file for all projections to enable warm start
    output_file = f"samples_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        # Core tuning
        roi_threshold=20,            # Large ROI for test
        pop_per_grid_point=3,
        max_patching_waves=20,
        lbfgsb_max_iter=10,
        # Enable emulator for speedup
        use_emulator=True,
        # I/O
        samples_output_file=output_file,
        # Advanced config
        advanced_config={
            'n_initial_optimizations': 100,
            'convergence_threshold': 1e-7,
            'de': {
                'num_generations': 100000,
                'max_num_to_evolve': None,
            },
            'emulator': {
                'max_neighbors': 200,
                'confidence_threshold': -1.0,
                'noise_level': 0.0001,
            }
        }
    )

    # Broadcast the target function to all workers (once, before all projections)
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # --- Run all projections with automatic refinement handling ---
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        save_plots=True,
        plot_settings={'dpi': 300, 'filetype': 'png'},
        myrank=myrank
    )

    # Print summary of all projections
    print("\n" + "="*80)
    print("=== Summary of All Projections ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
    print("="*80 + "\n")

    # Terminate all workers after all projections complete
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
