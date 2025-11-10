"""
Test script for direct evaluation mode with 2D functions.

This tests that 2D functions can now use 2D projections (dims=[0,1])
by evaluating at grid points directly without optimization.
"""
import numpy as np
from test_functions import get_test_function
from sampler import GridAnchoredDESampler

print("="*80)
print("TESTING DIRECT EVALUATION MODE")
print("="*80)

# Test 1: beale_2d with 2D projection
print("\nTest 1: beale_2d with 2D projection (dims=[0, 1])")
print("-"*80)

func, bounds, peaks = get_test_function('beale_2d')
print(f"Function: beale_2d ({len(bounds)}D)")
print(f"Projection: dims=[0, 1] (2D - uses all dimensions)")
print(f"Expected: Direct evaluation mode should activate")
print()

try:
    sampler = GridAnchoredDESampler(
        target_func=func,
        bounds=bounds,
        projections=[
            {'dims': [0, 1], 'grid_points': [10, 10]}
        ],
        pop_per_grid_point=5,
        n_initial_optimizations=10
    )

    print("✓ Sampler created successfully!")
    print(f"  - Direct eval mode: {sampler.direct_eval_mode}")
    print(f"  - Projection dims: {sampler.projection_dims}")
    print(f"  - Continuous dims: {sampler.continuous_dims}")
    print(f"  - Grid shape: {sampler.grid_shape}")
    print(f"  - n_cont_dims: {sampler.n_cont_dims}")

    if not sampler.direct_eval_mode:
        print("✗ ERROR: Direct eval mode should be True!")
        exit(1)

    if sampler.n_cont_dims != 0:
        print(f"✗ ERROR: n_cont_dims should be 0, got {sampler.n_cont_dims}!")
        exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: eggholder_2d with 2D projection
print("\n" + "="*80)
print("\nTest 2: eggholder_2d with 2D projection (dims=[0, 1])")
print("-"*80)

func2, bounds2, peaks2 = get_test_function('eggholder_2d')
print(f"Function: eggholder_2d ({len(bounds2)}D)")
print(f"Projection: dims=[0, 1] (2D - uses all dimensions)")
print()

try:
    sampler2 = GridAnchoredDESampler(
        target_func=func2,
        bounds=bounds2,
        projections=[
            {'dims': [0, 1], 'grid_points': [8, 8]}
        ],
        pop_per_grid_point=3,
        n_initial_optimizations=5
    )

    print("✓ Sampler created successfully!")
    print(f"  - Direct eval mode: {sampler2.direct_eval_mode}")
    print(f"  - Projection dims: {sampler2.projection_dims}")
    print(f"  - Continuous dims: {sampler2.continuous_dims}")
    print(f"  - Grid shape: {sampler2.grid_shape}")

    if not sampler2.direct_eval_mode:
        print("✗ ERROR: Direct eval mode should be True!")
        exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Verify normal mode still works (4D function with 2D projection)
print("\n" + "="*80)
print("\nTest 3: sphere_4d with 2D projection (dims=[0, 1]) - Normal Mode")
print("-"*80)

func3, bounds3, peaks3 = get_test_function('sphere_4d')
print(f"Function: sphere_4d ({len(bounds3)}D)")
print(f"Projection: dims=[0, 1] (2D projection in 4D space)")
print(f"Expected: Normal mode (has continuous dims [2, 3])")
print()

try:
    sampler3 = GridAnchoredDESampler(
        target_func=func3,
        bounds=bounds3,
        projections=[
            {'dims': [0, 1], 'grid_points': [8, 8]}
        ],
        pop_per_grid_point=5,
        n_initial_optimizations=10
    )

    print("✓ Sampler created successfully!")
    print(f"  - Direct eval mode: {sampler3.direct_eval_mode}")
    print(f"  - Projection dims: {sampler3.projection_dims}")
    print(f"  - Continuous dims: {sampler3.continuous_dims}")
    print(f"  - Grid shape: {sampler3.grid_shape}")
    print(f"  - n_cont_dims: {sampler3.n_cont_dims}")

    if sampler3.direct_eval_mode:
        print("✗ ERROR: Direct eval mode should be False for 4D!")
        exit(1)

    if sampler3.n_cont_dims != 2:
        print(f"✗ ERROR: n_cont_dims should be 2, got {sampler3.n_cont_dims}!")
        exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test ActivationJob in direct eval mode
print("\n" + "="*80)
print("\nTest 4: ActivationJob in direct evaluation mode")
print("-"*80)

from jobs.activation_job import ActivationJob

try:
    # Use the beale_2d sampler from Test 1
    grid_idx = (0, 0)
    activation_job = ActivationJob(job_id=1, sampler=sampler, grid_idx=grid_idx)

    print("✓ ActivationJob created successfully!")
    print(f"  - Pop size: {activation_job.pop_size}")
    print(f"  - n_cont_dims: {activation_job.n_cont_dims}")
    print(f"  - continuous_params shape: {activation_job.all_continuous_params.shape}")
    print(f"  - Number of full params: {len(activation_job.all_full_params)}")
    print(f"  - Full params[0]: {activation_job.all_full_params[0]}")

    if activation_job.pop_size != 1:
        print(f"✗ ERROR: Pop size should be 1 in direct eval mode, got {activation_job.pop_size}!")
        exit(1)

    if activation_job.all_continuous_params.shape != (1, 0):
        print(f"✗ ERROR: continuous_params should be (1, 0), got {activation_job.all_continuous_params.shape}!")
        exit(1)

    # Test that tasks can be created
    tasks = activation_job.start()
    print(f"\n  - Number of tasks created: {len(tasks)}")
    print(f"  - Task params: {tasks[0]['params']}")

    if len(tasks) != 1:
        print(f"✗ ERROR: Should create 1 task, got {len(tasks)}!")
        exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print()
print("Summary:")
print("  ✓ Direct evaluation mode activates for 2D functions with 2D projections")
print("  ✓ Normal mode still works for higher-dimensional functions")
print("  ✓ ActivationJob creates correct single-evaluation tasks")
print("  ✓ State structures are handled correctly")
print()
print("Next: Test with MPI to verify full workflow")
print("="*80)
