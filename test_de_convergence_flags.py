"""
Integration test for DE convergence behavior with lbfgsb flag.
Tests that converged DE jobs correctly spawn or skip L-BFGS-B jobs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sampler import GridAnchoredDESampler
from test_functions import get_test_function
from jobs.de_job import DEGridPointJob


def test_de_convergence_with_lbfgsb_enabled():
    """Test that DE convergence spawns L-BFGS-B job when lbfgsb=True."""
    print("="*80)
    print("TEST: DE convergence with lbfgsb=True")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    projection_config = {'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': True}

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=[projection_config],
        pop_per_grid_point=2,
        convergence_window=3,
        convergence_threshold=0.01,
    )

    # Setup a grid point with converged DE (small improvements)
    # Note: on_finish will append one more improvement, so we need window-1 values
    grid_idx = (5, 5)
    sampler.population[grid_idx] = {
        'continuous_params': np.array([[0.0, 0.0], [0.1, 0.1]]),
        'fitnesses': np.array([10.0, 9.5]),
        'best_fitness': 10.0,
        'status': 'active',
        'improvement_history': [0.001, 0.002],  # 2 values, on_finish adds 1 more = 3 total
        'optimizer_state': None,
        'last_update_gen': 4
    }
    sampler.active_grid_indices.add(grid_idx)
    sampler.current_generation = 5

    # Create a DE job and simulate it finishing
    de_job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=grid_idx,
        parent_pool=[],
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )
    de_job.success = True
    de_job._is_finished = True

    # Call on_finish - should spawn L-BFGS-B job
    result = de_job.on_finish(next_job_id=100)

    if result is not None:
        new_job, new_id = result
        if new_job.type == 'LBFGSB' and new_id == 101:
            print("  PASS: L-BFGS-B job spawned correctly")
            print(f"        Job type: {new_job.type}, Grid idx: {new_job.grid_idx}")
            print(f"        Grid point status: {sampler.population[grid_idx]['status']}")
            return True
        else:
            print(f"  FAIL: Wrong job spawned - type: {new_job.type}, new_id: {new_id}")
            return False
    else:
        print("  FAIL: No job spawned when L-BFGS-B enabled")
        return False


def test_de_convergence_with_lbfgsb_disabled():
    """Test that DE convergence marks as optimized when lbfgsb=False."""
    print("\n" + "="*80)
    print("TEST: DE convergence with lbfgsb=False")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    projection_config = {'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': False}

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=[projection_config],
        pop_per_grid_point=2,
        convergence_window=3,
        convergence_threshold=0.01,
    )

    # Setup a grid point with converged DE
    # Note: on_finish will append one more improvement, so we need window-1 values
    grid_idx = (5, 5)
    sampler.population[grid_idx] = {
        'continuous_params': np.array([[0.0, 0.0], [0.1, 0.1]]),
        'fitnesses': np.array([10.0, 9.5]),
        'best_fitness': 10.0,
        'status': 'active',
        'improvement_history': [0.001, 0.002],  # 2 values, on_finish adds 1 more = 3 total
        'optimizer_state': None,
        'last_update_gen': 4
    }
    sampler.active_grid_indices.add(grid_idx)
    sampler.current_generation = 5

    # Create a DE job and simulate it finishing
    de_job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=grid_idx,
        parent_pool=[],
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )
    de_job.success = True
    de_job._is_finished = True

    # Call on_finish - should NOT spawn L-BFGS-B job
    result = de_job.on_finish(next_job_id=100)

    if result is None:
        status = sampler.population[grid_idx]['status']
        if status == 'optimized':
            print("  PASS: No job spawned, status set to 'optimized'")
            print(f"        Grid point status: {status}")
            return True
        else:
            print(f"  FAIL: No job spawned but status is '{status}' instead of 'optimized'")
            return False
    else:
        print(f"  FAIL: Job spawned when L-BFGS-B disabled")
        print(f"        Job type: {result[0].type}")
        return False


def test_de_not_converged():
    """Test that non-converged DE doesn't spawn L-BFGS-B job regardless of flag."""
    print("\n" + "="*80)
    print("TEST: DE not converged (should not spawn L-BFGS-B)")
    print("="*80)

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    projection_config = {'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': True}

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=[projection_config],
        pop_per_grid_point=2,
        convergence_window=3,
        convergence_threshold=0.01,
    )

    # Setup a grid point that hasn't converged yet
    grid_idx = (5, 5)
    sampler.population[grid_idx] = {
        'continuous_params': np.array([[0.0, 0.0], [0.1, 0.1]]),
        'fitnesses': np.array([10.0, 9.5]),
        'best_fitness': 10.0,
        'status': 'active',
        'improvement_history': [0.5, 0.4],  # Still improving (> threshold)
        'optimizer_state': None,
        'last_update_gen': 4
    }
    sampler.active_grid_indices.add(grid_idx)
    sampler.current_generation = 5

    # Create a DE job and simulate it finishing
    de_job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=grid_idx,
        parent_pool=[],
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )
    de_job.success = True
    de_job._is_finished = True

    # Call on_finish - should NOT spawn L-BFGS-B job (not converged)
    result = de_job.on_finish(next_job_id=100)

    if result is None:
        status = sampler.population[grid_idx]['status']
        if status == 'active':
            print("  PASS: No job spawned, status remains 'active'")
            print(f"        Grid point status: {status}")
            return True
        else:
            print(f"  FAIL: Status changed to '{status}' when not converged")
            return False
    else:
        print(f"  FAIL: Job spawned when DE not converged")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("RUNNING DE CONVERGENCE INTEGRATION TESTS")
    print("="*80 + "\n")

    results = []

    results.append(("DE convergence with lbfgsb=True", test_de_convergence_with_lbfgsb_enabled()))
    results.append(("DE convergence with lbfgsb=False", test_de_convergence_with_lbfgsb_disabled()))
    results.append(("DE not converged", test_de_not_converged()))

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
