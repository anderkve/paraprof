"""
Test script for multi-GP refinement method.

This script tests the new MultiGPInterpolator on the Himmelblau 4D test function
with a 2D projection and compares results against linear interpolation.

Usage:
    OMP_NUM_THREADS=1 mpiexec -n 2 python test_multi_gp_refinement.py
"""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function, set_log_level

set_log_level('INFO')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Configuration
np.random.seed(42)
TEST_FUNCTION = "himmelblau_4d"

# Two projections: one with linear refinement, one with multi-GP refinement
PROJECTIONS_TO_RUN = [
    {
        'dims': [0, 1],
        'grid_points': [20, 20],
        'optimization_method': 'lbfgsb',
        'enable_refinement': True,
        'refinement_factor': 2,
        'refinement_method': 'linear',  # Baseline
        'patching_coarse': False,
        'patching_refined': False,
    },
    {
        'dims': [0, 1],
        'grid_points': [20, 20],
        'optimization_method': 'lbfgsb',
        'enable_refinement': True,
        'refinement_factor': 2,
        'refinement_method': 'multi_gp',  # New method
        'patching_coarse': False,
        'patching_refined': False,
    },
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # Master process
    print("="*80)
    print("Testing Multi-GP Refinement vs Linear Interpolation")
    print("="*80)

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        n_initial_optimizations=50,
        roi_threshold=4.0,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=20,
        samples_output_file=None,  # Don't save samples for this test
    )

    # Broadcast target function
    comm.bcast(sampler.target_func, root=0)

    # Run all projections
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=100000,
        save_plots=True,
        plot_settings={'dpi': 150, 'filetype': 'png'},
        myrank=myrank
    )

    # Compare results
    print("\n" + "="*80)
    print("=== Comparison: Linear vs Multi-GP Refinement ===")
    print("="*80)

    linear_result = results[0]
    multigp_result = results[1]

    print(f"\nLinear Interpolation:")
    print(f"  Coarse grid calls: {linear_result['metrics']['coarse_target_calls']}")
    print(f"  Refined grid calls: {linear_result['metrics']['refined_target_calls']}")
    print(f"  Total calls: {linear_result['metrics']['total_target_calls']}")
    print(f"  Global max logL: {linear_result['metrics']['global_max']:.6f}")

    print(f"\nMulti-GP Interpolation:")
    print(f"  Coarse grid calls: {multigp_result['metrics']['coarse_target_calls']}")
    print(f"  Refined grid calls: {multigp_result['metrics']['refined_target_calls']}")
    print(f"  Total calls: {multigp_result['metrics']['total_target_calls']}")
    print(f"  Global max logL: {multigp_result['metrics']['global_max']:.6f}")

    # Calculate efficiency metrics
    refinement_calls_linear = (linear_result['metrics']['refined_target_calls'] -
                               linear_result['metrics']['coarse_target_calls'])
    refinement_calls_multigp = (multigp_result['metrics']['refined_target_calls'] -
                                multigp_result['metrics']['coarse_target_calls'])

    print(f"\nRefinement Stage Only:")
    print(f"  Linear: {refinement_calls_linear} calls")
    print(f"  Multi-GP: {refinement_calls_multigp} calls")

    if refinement_calls_linear > 0:
        efficiency_gain = (refinement_calls_linear - refinement_calls_multigp) / refinement_calls_linear * 100
        print(f"  Efficiency gain: {efficiency_gain:.1f}%")

    print("="*80)

    # Terminate workers
    terminate_workers(comm, myrank)

else:
    # Worker process
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
