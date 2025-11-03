"""
Test script to validate lbfgsb and patching flag functionality.
This tests that the flags are properly read, validated, and enforced.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sampler import GridAnchoredDESampler
from test_functions import get_test_function


def test_flag_combination_patching_without_lbfgsb():
    """Test that patching=True with lbfgsb=False is now valid."""
    print("="*80)
    print("TEST 1: Valid combination (patching=True, lbfgsb=False)")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    # This combination should now be valid
    projection = [
        {'dims': [0, 1], 'grid_points': [10, 10], 'patching': True, 'lbfgsb': False}
    ]

    try:
        sampler = GridAnchoredDESampler(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=projection,
            pop_per_grid_point=1,
            n_initial_optimizations=5,
        )
        if sampler.enable_patching and not sampler.enable_lbfgsb:
            print("  PASS: Combination is valid")
            print(f"        enable_lbfgsb={sampler.enable_lbfgsb}, enable_patching={sampler.enable_patching}")
            return True
        else:
            print("  FAIL: Flags not set correctly")
            return False
    except ValueError as e:
        print(f"  FAIL: Should not raise error, but got: {e}")
        return False


def test_flag_reading():
    """Test that flags are correctly read from projection config."""
    print("\n" + "="*80)
    print("TEST 2: Flag reading from projection config")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    test_cases = [
        ({'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': False},
         False, False, "Both disabled"),
        ({'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': True},
         True, False, "L-BFGS-B enabled, patching disabled"),
        ({'dims': [0, 1], 'grid_points': [10, 10], 'patching': True, 'lbfgsb': True},
         True, True, "Both enabled"),
        ({'dims': [0, 1], 'grid_points': [10, 10], 'patching': True, 'lbfgsb': False},
         False, True, "Patching enabled, L-BFGS-B disabled"),
        ({'dims': [0, 1], 'grid_points': [10, 10]},  # Defaults
         True, True, "Default (both enabled)"),
    ]

    all_passed = True
    for projection_config, expected_lbfgsb, expected_patching, description in test_cases:
        print(f"\nTest case: {description}")
        print(f"  Config: {projection_config}")

        sampler = GridAnchoredDESampler(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=[projection_config],
            pop_per_grid_point=1,
            n_initial_optimizations=5,
        )

        if sampler.enable_lbfgsb == expected_lbfgsb and sampler.enable_patching == expected_patching:
            print(f"  PASS: enable_lbfgsb={sampler.enable_lbfgsb}, enable_patching={sampler.enable_patching}")
        else:
            print(f"  FAIL: Got enable_lbfgsb={sampler.enable_lbfgsb}, enable_patching={sampler.enable_patching}")
            print(f"        Expected enable_lbfgsb={expected_lbfgsb}, enable_patching={expected_patching}")
            all_passed = False

    return all_passed


def test_create_patching_jobs_defensive():
    """Test that create_patching_LBFGSB_jobs returns empty list when patching disabled."""
    print("\n" + "="*80)
    print("TEST 3: Defensive check in create_patching_LBFGSB_jobs")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    projection_config = {'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': True}

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=[projection_config],
        pop_per_grid_point=1,
        n_initial_optimizations=5,
    )

    # Manually populate some grid points to test
    sampler.population[(5, 5)] = {
        'continuous_params': np.array([[0.0, 0.0]]),
        'fitnesses': np.array([100.0]),
        'best_fitness': 100.0,
        'status': 'converged',
        'improvement_history': [],
        'optimizer_state': None
    }
    sampler.profile_likelihood_grid[(5, 5)] = 100.0
    sampler.global_max_target_val = 100.0

    jobs, next_id = sampler.create_patching_LBFGSB_jobs(0)

    if len(jobs) == 0:
        print("  PASS: No patching jobs created when patching disabled")
        return True
    else:
        print(f"  FAIL: Created {len(jobs)} jobs when patching disabled")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("RUNNING L-BFGS-B FLAG TESTS")
    print("="*80 + "\n")

    results = []

    # Test 1: Valid flag combination (patching without lbfgsb)
    results.append(("Valid combination (patching=True, lbfgsb=False)", test_flag_combination_patching_without_lbfgsb()))

    # Test 2: Flag reading
    results.append(("Flag reading", test_flag_reading()))

    # Test 3: Defensive check
    results.append(("Defensive check in create_patching_LBFGSB_jobs", test_create_patching_jobs_defensive()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
