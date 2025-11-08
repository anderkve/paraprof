"""
Quick test of wave-based patching implementation.
"""
import sys
import os

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed.")
    sys.exit(1)

from sampler import GridAnchoredDESampler
from master import master_main, terminate_workers
from worker import worker_main
from test_functions import get_test_function

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Simple test configuration
TEST_FUNCTION = "himmelblau_4d"

PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [10, 10], 'patching': True, 'lbfgsb': True},
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    print("="*80)
    print("Testing Wave-Based Patching Implementation")
    print("="*80)

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=2,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=10,
        roi_threshold=3.2,
        convergence_threshold=1e-3,
        convergence_window=3,
        neighbor_pull_probability=0.5,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=20,
        LBFGSB_gradient_method="forward",
        max_patching_waves=5,  # Test with max 5 waves
        patching_n_neighbors=1,  # Test single best neighbor
        memory_size=250,
    )

    # Broadcast target function
    print("Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # Run the workflow
    master_main(
        comm=comm,
        sampler=sampler,
        num_generations=100,
        max_num_to_evolve=None,
        plot_callback=None,
        plot_interval=1000,
        skip_init_opt_on_warm_start=False,
        fig=None,
        axes=None,
        myrank=myrank
    )

    print("\n" + "="*80)
    print("Test completed successfully!")
    print(f"Total function calls: {sampler.target_calls}")
    print(f"Global max likelihood: {sampler.global_max_target_val:.6e}")
    print("="*80)

    # Terminate workers
    terminate_workers(comm, myrank)

else:
    # Worker process
    worker_main(comm, myrank)

print(f"Rank {myrank}: Done.")
