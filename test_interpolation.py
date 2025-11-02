"""
Test script for grid interpolation functionality.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interpolation import GridInterpolator


def test_1d_projection_interpolation():
    """Test interpolation for a 1D projection (1 projection dim, 1 continuous dim)."""
    print("\n" + "="*80)
    print("Test 1: 1D Projection Interpolation")
    print("="*80)

    # Create a coarse grid: x1 is projected, x2 is continuous
    # True function: x2_best(x1) = sin(x1)
    grid_axes = [np.linspace(0, 2*np.pi, 11)]  # Coarse: 11 points
    projection_dims = [0]
    continuous_dims = [1]
    grid_shape = (11,)

    solutions = {}
    for i in range(11):
        x1 = grid_axes[0][i]
        x2_best = np.sin(x1)  # True function
        likelihood = -((x2_best - np.sin(x1))**2)  # Should be 0 at optimum

        solutions[(i,)] = {
            'continuous_params': np.array([x2_best]),
            'likelihood': likelihood,
            'full_params': np.array([x1, x2_best])
        }

    coarse_solution = {
        'grid_axes': grid_axes,
        'projection_dims': projection_dims,
        'continuous_dims': continuous_dims,
        'solutions': solutions,
        'grid_shape': grid_shape
    }

    # Create interpolator
    interpolator = GridInterpolator(coarse_solution)
    print(f"Interpolator: {interpolator}")

    # Test interpolation at intermediate points
    test_x1_values = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
    print("\nInterpolation test:")
    print(f"{'x1':>10} {'True x2':>12} {'Interp x2':>12} {'Error':>12}")
    print("-" * 50)

    max_error = 0.0
    for x1 in test_x1_values:
        true_x2 = np.sin(x1)
        interp_x2 = interpolator.interpolate([x1])[0]
        error = abs(true_x2 - interp_x2)
        max_error = max(max_error, error)
        print(f"{x1:10.4f} {true_x2:12.6f} {interp_x2:12.6f} {error:12.6f}")

    print(f"\nMax interpolation error: {max_error:.6f}")
    print("PASSED" if max_error < 0.1 else "FAILED")

    return max_error < 0.1


def test_2d_projection_interpolation():
    """Test interpolation for a 2D projection (2 projection dims, 2 continuous dims)."""
    print("\n" + "="*80)
    print("Test 2: 2D Projection Interpolation")
    print("="*80)

    # Create a coarse grid: (x1, x2) projected, (x3, x4) continuous
    # True functions: x3_best(x1, x2) = x1 + x2, x4_best(x1, x2) = x1 * x2
    n_points = 6
    grid_axes = [
        np.linspace(0, 1, n_points),  # x1
        np.linspace(0, 1, n_points)   # x2
    ]
    projection_dims = [0, 1]
    continuous_dims = [2, 3]
    grid_shape = (n_points, n_points)

    solutions = {}
    for i in range(n_points):
        for j in range(n_points):
            x1 = grid_axes[0][i]
            x2 = grid_axes[1][j]
            x3_best = x1 + x2
            x4_best = x1 * x2
            likelihood = -(x3_best**2 + x4_best**2)  # Arbitrary

            solutions[(i, j)] = {
                'continuous_params': np.array([x3_best, x4_best]),
                'likelihood': likelihood,
                'full_params': np.array([x1, x2, x3_best, x4_best])
            }

    coarse_solution = {
        'grid_axes': grid_axes,
        'projection_dims': projection_dims,
        'continuous_dims': continuous_dims,
        'solutions': solutions,
        'grid_shape': grid_shape
    }

    # Create interpolator
    interpolator = GridInterpolator(coarse_solution)
    print(f"Interpolator: {interpolator}")

    # Test interpolation at intermediate points
    test_points = [
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.25),
        (0.3, 0.7)
    ]

    print("\nInterpolation test:")
    print(f"{'x1':>8} {'x2':>8} {'True x3':>10} {'Int x3':>10} {'True x4':>10} {'Int x4':>10} {'Error':>10}")
    print("-" * 70)

    max_error = 0.0
    for x1, x2 in test_points:
        true_x3 = x1 + x2
        true_x4 = x1 * x2
        interp_params = interpolator.interpolate([x1, x2])
        interp_x3, interp_x4 = interp_params

        error_x3 = abs(true_x3 - interp_x3)
        error_x4 = abs(true_x4 - interp_x4)
        total_error = max(error_x3, error_x4)
        max_error = max(max_error, total_error)

        print(f"{x1:8.3f} {x2:8.3f} {true_x3:10.5f} {interp_x3:10.5f} "
              f"{true_x4:10.5f} {interp_x4:10.5f} {total_error:10.6f}")

    print(f"\nMax interpolation error: {max_error:.6f}")
    print("PASSED" if max_error < 0.05 else "FAILED")

    return max_error < 0.05


def test_sparse_grid_interpolation():
    """Test interpolation with sparse (incomplete) coarse grid."""
    print("\n" + "="*80)
    print("Test 3: Sparse Grid Interpolation")
    print("="*80)

    # Create a sparse grid with only some points filled
    grid_axes = [np.linspace(0, 1, 10)]
    projection_dims = [0]
    continuous_dims = [1]
    grid_shape = (10,)

    # Only fill every other point
    solutions = {}
    for i in range(0, 10, 2):  # Only even indices
        x1 = grid_axes[0][i]
        x2_best = x1**2
        solutions[(i,)] = {
            'continuous_params': np.array([x2_best]),
            'likelihood': -x2_best**2,
            'full_params': np.array([x1, x2_best])
        }

    coarse_solution = {
        'grid_axes': grid_axes,
        'projection_dims': projection_dims,
        'continuous_dims': continuous_dims,
        'solutions': solutions,
        'grid_shape': grid_shape
    }

    # Create interpolator
    interpolator = GridInterpolator(coarse_solution)
    print(f"Interpolator: {interpolator}")
    print(f"Coverage: {interpolator.get_coverage_fraction():.1%}")

    # Test interpolation - should still work with nearest neighbor
    x1_test = 0.5
    interp_x2 = interpolator.interpolate([x1_test])[0]
    print(f"\nTest point: x1={x1_test:.2f}, interpolated x2={interp_x2:.4f}")
    print("PASSED - Sparse interpolation completed without errors")

    return True


def test_grid_index_mapping():
    """Test coarse-to-fine grid index mapping."""
    print("\n" + "="*80)
    print("Test 4: Grid Index Mapping")
    print("="*80)

    # Import sampler to test the mapping methods
    from sampler import GridAnchoredDESampler
    from test_functions import get_test_function

    log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

    projections = [{'dims': [0, 1], 'grid_points': [10, 10]}]
    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=projections
    )

    refinement_factor = 2

    # Test mapping
    # Format: (coarse_idx, fine_idx, should_align)
    aligning_cases = [
        ((0, 0), (0, 0)),
        ((1, 1), (2, 2)),
        ((5, 3), (10, 6)),
        ((1, 2), (2, 4)),
    ]

    non_aligning_cases = [
        (3, 5),  # Only fine_idx
        (1, 3),
        (7, 9),
    ]

    print("\nCoarse to Fine mapping tests:")
    print(f"{'Coarse Idx':>15} {'Expected Fine':>15} {'Actual Fine':>15} {'Status':>10}")
    print("-" * 60)

    all_passed = True
    for coarse_idx, expected_fine in aligning_cases:
        fine_idx = sampler._map_coarse_to_fine_index(coarse_idx, refinement_factor)
        passed = (fine_idx == expected_fine)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{str(coarse_idx):>15} {str(expected_fine):>15} {str(fine_idx):>15} {status:>10}")

    print("\nFine to Coarse mapping tests:")
    print(f"{'Fine Idx':>15} {'Expected Coarse':>18} {'Actual Coarse':>18} {'Status':>10}")
    print("-" * 65)

    for expected_coarse, fine_idx in aligning_cases:
        coarse_idx = sampler._map_fine_to_coarse_index(fine_idx, refinement_factor)
        passed = (coarse_idx == expected_coarse)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{str(fine_idx):>15} {str(expected_coarse):>18} {str(coarse_idx):>18} {status:>10}")

    # Test non-aligning points
    print("\nNon-aligning fine indices (should return None):")
    for fine_idx in non_aligning_cases:
        coarse_idx = sampler._map_fine_to_coarse_index(fine_idx, refinement_factor)
        is_coarse = sampler._is_coarse_grid_point(fine_idx, refinement_factor)
        passed = (coarse_idx is None) and (not is_coarse)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {fine_idx}: coarse_idx={coarse_idx}, is_coarse={is_coarse} - {status}")

    print("\n" + ("PASSED" if all_passed else "FAILED"))
    return all_passed


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Grid Interpolation Test Suite")
    print("#"*80)

    results = []
    results.append(("1D Interpolation", test_1d_projection_interpolation()))
    results.append(("2D Interpolation", test_2d_projection_interpolation()))
    results.append(("Sparse Grid", test_sparse_grid_interpolation()))
    results.append(("Index Mapping", test_grid_index_mapping()))

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
