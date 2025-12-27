#!/usr/bin/env python
"""
Example: Testing ParaProf with Simulated Large-Scale MPI Performance

This script demonstrates how to test paraprof's scalability on a laptop
by simulating realistic HPC network latency and bandwidth.

Usage:
    # Test with 128 processes simulating InfiniBand network
    mpiexec -n 128 --oversubscribe python run_with_simulation.py

    # Test different network types
    PARAPROF_NETWORK=10gbe mpiexec -n 256 --oversubscribe python run_with_simulation.py
    PARAPROF_NETWORK=cloud mpiexec -n 64 --oversubscribe python run_with_simulation.py

    # Disable simulation (normal run)
    PARAPROF_SIMULATE_MPI=0 mpiexec -n 8 python run_with_simulation.py
"""
import os
import sys
import numpy as np

# Import the MPI simulator
from paraprof.mpi_simulator import get_simulated_communicator

# Get network type from environment (default: infiniband)
network_type = os.environ.get('PARAPROF_NETWORK', 'infiniband')

# Create simulated communicator
comm = get_simulated_communicator(network_type=network_type, enable=True)

rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("="*70)
    print(f"ParaProf Scalability Test - Simulating {network_type.upper()} network")
    print(f"Running with {size} MPI processes (physical cores: ~{os.cpu_count()})")
    print("="*70)
    print()

# Your existing paraprof code here
from paraprof import run_all_projections

# Define your target function
def target_function(x):
    """Example likelihood function."""
    # Add some computation to make it non-trivial
    result = -np.sum(x**2)
    for _ in range(1000):  # Simulate computation
        result += np.sin(x[0]) * 0.0001
    return result

# Broadcast target function to all workers
target_func = comm.bcast(target_function if rank == 0 else None, root=0)

if rank == 0:
    # Master process
    from paraprof.sampler import ProfileProjector
    from paraprof.grids import RectGrid

    # Define parameter space
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    # Create grid
    grid = RectGrid(
        bounds=bounds,
        grid_size=20,  # Start with smaller grid for testing
        origin='center'
    )

    # Create projector
    projector = ProfileProjector(grid)

    # Run profile likelihood estimation
    try:
        results = run_all_projections(
            comm=comm,
            target_func=target_func,
            projector=projector,
            likelihood_threshold=2.0,
            max_iterations=100,
            n_mutations=10,
            verbose=True
        )

        # Print simulation statistics
        if hasattr(comm, 'print_stats'):
            comm.print_stats()

        print("\n" + "="*70)
        print("SCALABILITY TEST RESULTS")
        print("="*70)
        print(f"Network type: {network_type}")
        print(f"MPI processes: {size}")
        print(f"Total iterations: {results.get('total_iterations', 'N/A')}")
        print(f"Function evaluations: {results.get('n_evals', 'N/A')}")
        print(f"Best likelihood: {results.get('best_likelihood', 'N/A'):.6f}")
        print(f"Wall time: {results.get('wall_time', 'N/A'):.2f} seconds")
        if 'wall_time' in results and results['n_evals'] > 0:
            throughput = results['n_evals'] / results['wall_time']
            print(f"Throughput: {throughput:.1f} evaluations/second")
            efficiency = throughput / (size - 1)  # Exclude master
            print(f"Worker efficiency: {efficiency:.2f} evals/sec/worker")
        print("="*70)

    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        comm.Abort(1)
else:
    # Worker process
    from paraprof.worker import worker_main
    worker_main(comm, target_func)

if rank == 0:
    print("\nSimulation complete!")
