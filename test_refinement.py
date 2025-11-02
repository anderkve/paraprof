"""
Test script for grid refinement feature (no MPI required).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler import GridAnchoredDESampler
from interpolation import GridInterpolator


def simple_test_function(params):
    """Simple test function: negative squared distance from [0.5, 0.5, 0.5, 0.5]"""
    target = np.array([0.5, 0.5, 0.5, 0.5])
    return -np.sum((params - target)**2)


def test_refinement_setup():
    """Test refinement setup and grid transfer logic."""
    print("\n" + "="*80)
    print("Test: Grid Refinement Setup")
    print("="*80)

    # Create sampler with simple 4D function
    bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_test_function,
        bounds=bounds,
        projections=projections,
        n_initial_optimizations=5,
        memory_size=10
    )

    print(f"Initial grid shape: {sampler.grid_shape}")
    print(f"Projection dims: {sampler.projection_dims}")
    print(f"Continuous dims: {sampler.continuous_dims}")

    # Manually populate some grid points (simulate converged run)
    print("\nPopulating coarse grid with mock solutions...")
    for i in range(0, 6, 2):  # Every other point
        for j in range(0, 6, 2):
            grid_idx = (i, j)
            # Create mock solution
            continuous_params = np.array([0.5, 0.5])  # Mock optimal continuous params
            likelihood = -0.1  # Mock likelihood

            sampler.population[grid_idx] = {
                'continuous_params': np.array([continuous_params]),
                'fitnesses': np.array([likelihood]),
                'best_fitness': likelihood,
                'status': 'converged',
                'improvement_history': [],
                'optimizer_state': None
            }
            sampler.active_grid_indices.add(grid_idx)
            sampler.profile_likelihood_grid[grid_idx] = likelihood

    print(f"Populated {len(sampler.population)} coarse grid points")

    # Export coarse solution
    print("\nExporting coarse grid solution...")
    coarse_solution = sampler.export_grid_solution()
    print(f"Exported {len(coarse_solution['solutions'])} solutions")
    print(f"Coarse grid shape: {coarse_solution['grid_shape']}")

    # Setup refinement
    print("\nSetting up refinement run...")
    refinement_factor = 2
    sampler.setup_refinement_run(coarse_solution, refinement_factor)

    # Check refinement state
    assert sampler.is_refinement_run == True
    assert sampler.refinement_factor == 2
    assert sampler.refinement_interpolator is not None
    print(f"Refinement mode: {sampler.is_refinement_run}")
    print(f"Refinement factor: {sampler.refinement_factor}")
    print(f"Interpolator: {sampler.refinement_interpolator}")

    # Reset for refined projection
    print("\nResetting for refined projection...")
    refined_config = {'dims': [0, 1], 'grid_points': [10, 10]}
    sampler._reset_for_new_projection(refined_config)

    print(f"New grid shape: {sampler.grid_shape}")
    print(f"Transferred points: {len(sampler.population)}")

    # Verify transferred points
    expected_transfers = len(coarse_solution['solutions'])
    actual_transfers = len([idx for idx, state in sampler.population.items()
                           if state['status'] == 'refined'])

    print(f"\nVerification:")
    print(f"  Expected transfers: {expected_transfers}")
    print(f"  Actual transfers: {actual_transfers}")

    # Check specific transferred points
    print(f"\nTransferred grid points:")
    for idx, state in sampler.population.items():
        if state['status'] == 'refined':
            print(f"  {idx}: likelihood={state['best_fitness']:.4f}")

    # Verify grid index mapping
    print(f"\nGrid index mapping test:")
    for coarse_idx in [(0, 0), (1, 1), (2, 2)]:
        if coarse_idx in coarse_solution['solutions']:
            fine_idx = sampler._map_coarse_to_fine_index(coarse_idx, refinement_factor)
            is_transferred = fine_idx in sampler.population
            print(f"  Coarse {coarse_idx} -> Fine {fine_idx}: transferred={is_transferred}")

    # Test interpolation
    print(f"\nInterpolation test:")
    test_coords = sampler._get_grid_coords_from_indices((3, 3))  # New point
    interp_params = sampler.refinement_interpolator.interpolate(test_coords)
    print(f"  Grid coords {test_coords}: interpolated continuous params = {interp_params}")

    success = (actual_transfers == expected_transfers) and (actual_transfers > 0)
    print("\n" + "="*80)
    print(f"Test Result: {'PASSED' if success else 'FAILED'}")
    print("="*80)

    return success


def test_refinement_activation_jobs():
    """Test refinement activation job creation."""
    print("\n" + "="*80)
    print("Test: Refinement Activation Jobs")
    print("="*80)

    # Create sampler
    bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_test_function,
        bounds=bounds,
        projections=projections
    )

    # Populate coarse grid
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            grid_idx = (i, j)
            continuous_params = np.array([0.5, 0.5])
            likelihood = -0.1

            sampler.population[grid_idx] = {
                'continuous_params': np.array([continuous_params]),
                'fitnesses': np.array([likelihood]),
                'best_fitness': likelihood,
                'status': 'converged',
                'improvement_history': [],
                'optimizer_state': None
            }

    # Export and setup refinement
    coarse_solution = sampler.export_grid_solution()
    sampler.setup_refinement_run(coarse_solution, 2)
    refined_config = {'dims': [0, 1], 'grid_points': [10, 10]}
    sampler._reset_for_new_projection(refined_config)

    # Create refinement activation jobs
    print("\nCreating refinement activation jobs...")
    jobs, _ = sampler.create_refinement_activation_jobs(next_job_id=0)

    print(f"Created {len(jobs)} activation jobs")
    print(f"Jobs are for grid points neighboring the {len(sampler.population)} transferred points")

    # Count expected neighbors
    transferred_points = [idx for idx, state in sampler.population.items()
                         if state['status'] == 'refined']
    print(f"Transferred points: {len(transferred_points)}")

    success = len(jobs) > 0
    print("\n" + "="*80)
    print(f"Test Result: {'PASSED' if success else 'FAILED'}")
    print("="*80)

    return success


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Grid Refinement Feature Test Suite")
    print("#"*80)

    results = []
    results.append(("Refinement Setup", test_refinement_setup()))
    results.append(("Activation Jobs", test_refinement_activation_jobs()))

    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)
    print("="*80)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*80)

    sys.exit(0 if all_passed else 1)
