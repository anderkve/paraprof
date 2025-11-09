"""
Example script: Testing 1D profile likelihood projection on Himmelblau 4D function.

Usage:
    mpiexec -n <number_of_cores> python run_himmelblau_1d.py
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

from sampler import GridAnchoredDESampler
from master import run_all_projections, terminate_workers
from worker import worker_main
from test_functions import get_test_function

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "himmelblau_4d"

# Test 1D projections on each parameter
PROJECTIONS_TO_RUN = [
    {'dims': [0], 'grid_points': [100], 'patching_coarse': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
    {'dims': [1], 'grid_points': [100], 'patching_coarse': True, 'lbfgsb': True, 'enable_refinement': True, 'refinement_factor': 2},
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    output_file = f"samples_1d_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=50,
        roi_threshold=4.0,
        convergence_threshold=1e-7,
        convergence_window=3,
        neighbor_pull_probability=0.5,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=20,
        LBFGSB_gradient_method="forward",
        max_patching_waves=10,
        patching_n_neighbors=1,
        memory_size=25,
        samples_output_file=output_file,
    )

    # Broadcast the target function to all workers
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # --- Run all projections ---
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

    # Print summary
    print("\n" + "="*80)
    print("=== Summary of All 1D Projections ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"  Projection {i+1} (dim {dims[0]}): {calls} calls, max logL = {max_ll:.4e}")
    print("="*80 + "\n")

    # Terminate all workers
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
