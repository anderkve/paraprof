"""
Example script: Running the ProfileProjector with MPI simulation for scalability testing.

This is a modified version of run_test.py that includes network simulation capabilities.
The ONLY difference is lines 31-37 below - everything else is identical!

Usage:
    # Normal run (8 cores, no simulation)
    mpiexec -n 8 python run_test_with_simulation.py

    # Scalability test (128 cores, simulated InfiniBand network)
    mpiexec -n 128 --oversubscribe python run_test_with_simulation.py

    # Test different networks
    PARAPROF_NETWORK=10gbe mpiexec -n 256 --oversubscribe python run_test_with_simulation.py
    PARAPROF_NETWORK=cloud mpiexec -n 128 --oversubscribe python run_test_with_simulation.py

    # Disable simulation (use real MPI)
    PARAPROF_SIMULATE_MPI=0 mpiexec -n 128 --oversubscribe python run_test_with_simulation.py
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

# ============================================================================
# NETWORK SIMULATION SETUP (Only change from run_test.py!)
# ============================================================================
from paraprof.mpi_simulator import get_simulated_communicator

# Get network type from environment (default: infiniband)
network = os.environ.get('PARAPROF_NETWORK', 'infiniband')
comm = get_simulated_communicator(network_type=network, enable=True)

# That's it! Just replace:
#   comm = MPI.COMM_WORLD
# with the lines above, and you can test with 100s of processes!
# ============================================================================

from paraprof import ProfileProjector, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function
from paraprof import set_log_level

set_log_level('INFO')

# Get MPI info
myrank = comm.Get_rank()
mysize = comm.Get_size()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "rosenbrock_4d"

# ============================================================================
# PROJECTION CONFIGURATION GUIDE
# ============================================================================
# Adjust these parameters based on your problem:
#
# For testing scalability on laptop:
# - Use smaller grid_size (10-20) to keep runtime reasonable
# - Reduce max_iterations (50-100)
# - Enable simulation to test realistic HPC conditions
#
# For production runs on HPC:
# - Use larger grid_size (30-50+)
# - Increase max_iterations (500-1000+)
# - Disable simulation (PARAPROF_SIMULATE_MPI=0)
# ============================================================================

config = {
    "grid_size": 15,  # Reduced for scalability testing
    "likelihood_threshold": 2.0,
    "max_iterations": 50,  # Reduced for faster tests
    "n_mutations": 10,
    "use_emulator": True,
}

if myrank == 0:
    print("\n" + "="*70)
    print(f"ParaProf Scalability Test")
    print("="*70)
    print(f"MPI processes: {mysize}")
    print(f"Physical cores: ~{os.cpu_count()}")
    print(f"Network simulation: {network}")
    print(f"Test function: {TEST_FUNCTION}")
    print(f"Grid size: {config['grid_size']}")
    print("="*70 + "\n")


def main():
    """Main execution function."""

    # Master broadcasts the test function to all workers
    if myrank == 0:
        target_func, bounds = get_test_function(TEST_FUNCTION)
        target_func = comm.bcast(target_func, root=0)
    else:
        target_func = comm.bcast(None, root=0)

    if myrank == 0:
        # ====================================================================
        # Master process
        # ====================================================================
        import time
        start_time = time.time()

        # Create ProfileProjector
        projector = ProfileProjector(
            bounds=bounds,
            grid_size=config["grid_size"]
        )

        # Run profile likelihood estimation
        results = run_all_projections(
            comm=comm,
            target_func=target_func,
            projector=projector,
            likelihood_threshold=config["likelihood_threshold"],
            max_iterations=config["max_iterations"],
            n_mutations=config["n_mutations"],
            use_emulator=config["use_emulator"],
            verbose=True
        )

        end_time = time.time()
        wall_time = end_time - start_time

        # Print simulation statistics
        if hasattr(comm, 'print_stats'):
            comm.print_stats()

        # Print results summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Test function: {TEST_FUNCTION}")
        print(f"MPI processes: {mysize}")
        print(f"Network simulation: {network}")
        print(f"Grid size: {config['grid_size']}")
        print(f"Total iterations: {results.get('total_iterations', 'N/A')}")
        print(f"Function evaluations: {results.get('n_evals', 'N/A')}")

        if results.get('n_evals', 0) > 0:
            throughput = results['n_evals'] / wall_time
            worker_efficiency = throughput / (mysize - 1)
            print(f"Wall time: {wall_time:.2f} seconds")
            print(f"Throughput: {throughput:.1f} evaluations/second")
            print(f"Worker efficiency: {worker_efficiency:.2f} evals/sec/worker")

        # Calculate parallel efficiency if we have baseline
        baseline_throughput = 100.0  # Approximate baseline for 4 processes
        if results.get('n_evals', 0) > 0 and mysize > 4:
            expected_throughput = baseline_throughput * (mysize - 1) / 3  # Scale from 3 workers
            actual_throughput = results['n_evals'] / wall_time
            efficiency = actual_throughput / expected_throughput
            print(f"Parallel efficiency: {efficiency:.1%} (estimated)")

        print("="*70 + "\n")

        # Terminate workers
        terminate_workers(comm)

    else:
        # ====================================================================
        # Worker processes
        # ====================================================================
        worker_main(comm, target_func)


if __name__ == "__main__":
    main()
