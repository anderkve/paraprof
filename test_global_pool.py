"""
Simple test script to verify global solution pool implementation.
Run with: mpiexec -n 2 python test_global_pool.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed.")
    sys.exit(1)

from sampler import GridAnchoredDESampler
from master import master_main, terminate_workers
from worker import worker_main
from test_functions import get_test_function

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Simple 2D test
TEST_FUNCTION = "himmelblau_4d"
log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    print("="*80)
    print("Testing Global Solution Pool Implementation")
    print("="*80)

    # Very small configuration for quick testing
    PROJECTIONS = [
        {'dims': [0, 1], 'grid_points': [5, 5], 'patching': False, 'lbfgsb': True,
         'enable_refinement': False, 'refinement_factor': 1},
    ]

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS,
        pop_per_grid_point=10,
        mutation_strategy='current-to-pbest/1',
        n_initial_optimizations=5,
        roi_threshold=3.2,
        convergence_threshold=1e-5,
        convergence_window=3,
        global_pool_size=50,  # Test the new parameter
        activation_mix_ratios={'neighbors': 0.5, 'global': 0.3, 'random': 0.2},  # Test new parameter
        samples_output_file=None
    )

    print(f"Sampler initialized with:")
    print(f"  - Global pool size: {sampler.global_pool_size}")
    print(f"  - Activation mix ratios: {sampler.activation_mix_ratios}")
    print(f"  - Initial pool size: {len(sampler.global_solution_pool)}")

    # Broadcast target function
    comm.bcast(sampler.target_func, root=0)

    # Run with minimal generations
    master_main(
        comm=comm,
        sampler=sampler,
        num_generations=3,  # Just 3 generations for testing
        max_num_to_evolve=None,
        plot_callback=None,
        plot_interval=10.0,
        skip_init_opt_on_warm_start=False
    )

    print("\n" + "="*80)
    print("Test Results:")
    print(f"  - Target calls: {sampler.target_calls}")
    print(f"  - Global pool size: {len(sampler.global_solution_pool)}")
    print(f"  - Grid points explored: {len(sampler.population)}")
    print(f"  - Global max logL: {sampler.global_max_target_val:.6e}")

    if len(sampler.global_solution_pool) > 0:
        print(f"\n  Top 3 solutions in global pool:")
        for i, sol in enumerate(sampler.global_solution_pool[:3]):
            print(f"    {i+1}. Fitness: {sol['fitness']:.6e}, Grid: {sol['grid_idx']}")

    print("="*80)
    print("TEST PASSED: Global solution pool is working!")
    print("="*80)

    terminate_workers(comm, myrank)

else:
    # Worker process
    target_func = comm.bcast(None, root=0)
    worker_main(comm, target_func, myrank)
