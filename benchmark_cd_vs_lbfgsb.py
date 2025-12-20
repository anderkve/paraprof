"""
Benchmark script comparing Coordinate Descent vs L-BFGS-B for refinement.

Runs the same projection twice - once with CD, once with L-BFGS-B.
"""
import sys
import numpy as np
import time

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function
from paraprof import set_log_level

set_log_level('WARNING')  # Reduce output for benchmarking

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(42)

TEST_FUNCTION = "himmelblau_4d"

# 2D projection with refinement
PROJECTION = {
    'dims': [0, 1],
    'grid_points': [15, 15],
    'patching_coarse': True,
    'patching_refined': False,
    'lbfgsb': True,
    'enable_refinement': True,
    'refinement_factor': 2
}

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

def run_benchmark(use_cd, method_name):
    """Run a single benchmark."""
    if myrank == 0:
        print(f"\n{'='*80}")
        print(f"=== Running with {method_name} ===")
        print(f"{'='*80}\n")

    max_grid_points = len(PROJECTION['grid_points'])
    output_file = f"benchmark_{method_name}_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=[PROJECTION],
        pop_per_grid_point=3,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=15,
        roi_threshold=4.0,
        convergence_threshold=1e-7,
        convergence_window=3,
        neighbor_pull_probability=0.5,
        lbfgsb_ftol=1e-9,
        lbfgsb_max_iter=10,
        lbfgsb_gradient_method="forward",
        max_patching_waves=5,
        patching_n_neighbors=1,
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,
        use_de_prescreening=False,
        # CD settings
        use_cd_refinement=use_cd,
        cd_max_cycles=3,
        cd_step_fraction=0.01,
    )

    if myrank == 0:
        comm.bcast(sampler.target_func, root=0)
        start_time = time.time()
    else:
        comm.bcast(None, root=0)

    if myrank == 0:
        results = run_all_projections(
            comm=comm,
            sampler=sampler,
            projections=[PROJECTION],
            num_generations=5000,
            max_num_to_evolve=None,
            save_plots=False,
            myrank=myrank
        )
        elapsed = time.time() - start_time
        return results[0], elapsed
    else:
        worker_main(comm, myrank)
        return None, None


if myrank == 0:
    print("\n" + "="*80)
    print("=== CD vs L-BFGS-B Refinement Benchmark ===")
    print("="*80)
    print(f"Test function: {TEST_FUNCTION}")
    print(f"Projection: dims {PROJECTION['dims']}, grid {PROJECTION['grid_points']}")
    print(f"Refinement factor: {PROJECTION['refinement_factor']}x")
    print("="*80 + "\n")

# Run CD benchmark
result_cd, time_cd = run_benchmark(use_cd=True, method_name="CD")

# Reset workers
if myrank == 0:
    print("\nResetting for next benchmark...\n")
comm.Barrier()

# Run L-BFGS-B benchmark
result_lbfgsb, time_lbfgsb = run_benchmark(use_cd=False, method_name="LBFGSB")

if myrank == 0:
    # Print comparison
    print("\n" + "="*80)
    print("=== BENCHMARK RESULTS ===")
    print("="*80)

    cd_calls = result_cd['metrics']['total_target_calls']
    cd_refinement = result_cd['metrics']['refined_target_calls'] - result_cd['metrics']['coarse_target_calls']
    cd_max = result_cd['metrics']['global_max']

    lbfgsb_calls = result_lbfgsb['metrics']['total_target_calls']
    lbfgsb_refinement = result_lbfgsb['metrics']['refined_target_calls'] - result_lbfgsb['metrics']['coarse_target_calls']
    lbfgsb_max = result_lbfgsb['metrics']['global_max']

    print(f"\nCoordinate Descent:")
    print(f"  Total evaluations: {cd_calls}")
    print(f"  Refinement evaluations: {cd_refinement}")
    print(f"  Time: {time_cd:.2f}s")
    print(f"  Max log-likelihood: {cd_max:.6e}")

    print(f"\nL-BFGS-B:")
    print(f"  Total evaluations: {lbfgsb_calls}")
    print(f"  Refinement evaluations: {lbfgsb_refinement}")
    print(f"  Time: {time_lbfgsb:.2f}s")
    print(f"  Max log-likelihood: {lbfgsb_max:.6e}")

    print(f"\nReduction:")
    total_reduction = (1 - cd_calls / lbfgsb_calls) * 100
    refinement_reduction = (1 - cd_refinement / lbfgsb_refinement) * 100
    time_reduction = (1 - time_cd / time_lbfgsb) * 100

    print(f"  Total evaluations: {total_reduction:+.1f}%")
    print(f"  Refinement evaluations: {refinement_reduction:+.1f}%")
    print(f"  Time: {time_reduction:+.1f}%")
    print(f"  Quality (logL diff): {cd_max - lbfgsb_max:.2e}")

    print("="*80 + "\n")

    # Terminate workers
    terminate_workers(comm, myrank)
else:
    pass  # Workers already terminated by worker_main calls

if myrank == 0:
    print(f"rank {myrank}: Benchmark complete.")
else:
    print(f"rank {myrank}: Done.")
