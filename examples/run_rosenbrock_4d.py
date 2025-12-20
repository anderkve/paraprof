"""
Example script: Running the Grid-Anchored DE Sampler on the Rosenbrock 4D test function.

Usage:
    mpiexec -n <number_of_cores> python run_rosenbrock_4d.py
"""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "rosenbrock_4d"

PROJECTIONS_TO_RUN = [

    {'dims': [0, 1], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [0, 2], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [0, 3], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [1, 2], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [1, 3], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [2, 3], 'grid_points': [150, 150], 'patching_coarse': True, 'patching_refined': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},

]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    # Use a single output file for all projections to enable warm start
    output_file = f"samples_rank_{myrank}.csv"

    # ===== SIMPLIFIED INTERFACE =====
    # Only specify the core tuning parameters - most defaults work well!
    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        # Core tuning
        roi_threshold=8.0,                   # Rosenbrock has deep valley -> larger ROI
        pop_per_grid_point=3,
        max_patching_waves=20,
        LBFGSB_max_iter=20,
        # I/O
        samples_output_file=output_file,
        # Optional: override auto-configured n_initial_optimizations
        advanced_config={
            'n_initial_optimizations': 100,  # Default would be min(100, 20*4) = 80
            'convergence_threshold': 1e-7,   # Default would be 8.0 / 1000 = 0.008
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
        num_generations=100000,
        max_num_to_evolve=None,
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
