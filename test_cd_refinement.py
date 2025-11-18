"""
Test script for Coordinate Descent refinement optimization.

This script runs a simple 2D projection with refinement enabled,
comparing CD vs L-BFGS-B performance.
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

set_log_level('INFO')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(42)

TEST_FUNCTION = "himmelblau_4d"

# Simple 2D projection with refinement enabled
PROJECTIONS_TO_RUN = [
    {
        'dims': [0, 1],
        'grid_points': [20, 20],
        'patching_coarse': True,
        'patching_refined': False,
        'lbfgsb': True,
        'enable_refinement': True,
        'refinement_factor': 2
    },
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---
    print("\n" + "="*80)
    print("=== Testing Coordinate Descent Refinement ===")
    print("="*80 + "\n")

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    output_file = f"test_cd_samples_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=20,
        roi_threshold=4.0,
        convergence_threshold=1e-7,
        convergence_window=3,
        neighbor_pull_probability=0.5,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=10,
        LBFGSB_gradient_method="forward",
        max_patching_waves=10,
        patching_n_neighbors=1,
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,
        use_de_prescreening=False,
        # CD refinement settings (enabled by default)
        use_cd_refinement=True,
        cd_max_cycles=3,
        cd_step_fraction=0.01,
    )

    # Broadcast the target function to all workers
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # Run all projections
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=10000,
        max_num_to_evolve=None,
        save_plots=True,
        plot_settings={'dpi': 150, 'filetype': 'png'},
        myrank=myrank
    )

    # Print summary
    print("\n" + "="*80)
    print("=== CD Refinement Test Results ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']

        if 'refined_target_calls' in res['metrics']:
            coarse_calls = res['metrics']['coarse_target_calls']
            refined_calls = res['metrics']['refined_target_calls']
            refinement_calls = refined_calls - coarse_calls

            print(f"\nProjection {i+1} (dims {dims}):")
            print(f"  Total calls: {calls}")
            print(f"  Coarse grid calls: {coarse_calls}")
            print(f"  Refinement calls: {refinement_calls}")
            print(f"  Max logL: {max_ll:.4e}")
        else:
            print(f"\nProjection {i+1} (dims {dims}):")
            print(f"  Total calls: {calls}")
            print(f"  Max logL: {max_ll:.4e}")
    print("="*80 + "\n")

    # Terminate all workers
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
