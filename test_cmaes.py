"""
Test script for CMA-ES optimization method.

Usage:
    mpiexec -n 2 python test_cmaes.py
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
from paraprof import set_log_level

# Set logging level
set_log_level('INFO')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "himmelblau_4d"

PROJECTIONS_TO_RUN = [
    # 1D projection to test CMA-ES
    {
        'dims': [0],
        'grid_points': [20],
        'optimization_method': 'cmaes',
        'lbfgsb_refinement': False,  # Disable refinement for initial test
        'patching_coarse': False,     # Disable patching for initial test
    },
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---
    print("="*80)
    print("Testing CMA-ES optimization method")
    print("="*80)

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(np.prod(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    # Use a single output file for all projections
    output_file = f"samples_cmaes_test_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=1,  # Small for testing
        n_initial_optimizations=10,  # Reduced for testing
        roi_threshold=4.0,
        convergence_threshold=1e-6,
        convergence_window=3,
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,
        # CMA-ES specific settings
        cmaes_lambda=None,  # Will auto-set based on dimensionality
        cmaes_mu=None,      # Will auto-set based on lambda
        cmaes_max_generations=50,  # Limit generations for testing
        # Disable emulator for initial test
        use_de_prescreening=False,
    )

    # Broadcast the target function to all workers (once, before all projections)
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # --- Run all projections ---
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=50,  # Not used for CMA-ES, but required parameter
        max_num_to_evolve=None,
        save_plots=True,
        plot_settings={'dpi': 150, 'filetype': 'png'},
        myrank=myrank
    )

    # Print summary of all projections
    print("\n" + "="*80)
    print("=== CMA-ES Test Summary ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
    print("="*80 + "\n")

    print("CMA-ES test completed successfully!")

    # Terminate all workers after all projections complete
    print("Master: Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
