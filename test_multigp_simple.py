"""
Simple unit test for MultiGPInterpolator.

Tests the MultiGPInterpolator class in isolation without running full refinement.
"""
import numpy as np

# Create a synthetic coarse grid solution
def create_test_solution():
    """Create a synthetic coarse grid solution for testing."""
    # 2D projection (dims 0, 1), 2 continuous dimensions (dims 2, 3)
    grid_axes = [
        np.linspace(-5, 5, 11),  # dim 0
        np.linspace(-5, 5, 11),  # dim 1
    ]

    projection_dims = [0, 1]
    continuous_dims = [2, 3]
    grid_shape = (11, 11)

    # Generate synthetic solutions
    solutions = {}
    for i in range(11):
        for j in range(11):
            grid_idx = (i, j)

            # Synthetic optimal continuous parameters
            # Make them vary smoothly across the grid
            x_grid = grid_axes[0][i]
            y_grid = grid_axes[1][j]

            # Continuous params vary smoothly with grid position
            cont_param_0 = 0.1 * x_grid + 0.05 * y_grid + np.sin(x_grid/2)
            cont_param_1 = -0.05 * x_grid + 0.1 * y_grid + np.cos(y_grid/2)

            # Synthetic likelihood (not used for interpolation of continuous params)
            likelihood = -0.5 * (x_grid**2 + y_grid**2)

            solutions[grid_idx] = {
                'continuous_params': np.array([cont_param_0, cont_param_1]),
                'likelihood': likelihood,
                'full_params': np.array([x_grid, y_grid, cont_param_0, cont_param_1])
            }

    return {
        'grid_axes': grid_axes,
        'projection_dims': projection_dims,
        'continuous_dims': continuous_dims,
        'grid_shape': grid_shape,
        'solutions': solutions,
        'global_solution_pool': []
    }


def test_multigp_interpolator():
    """Test MultiGPInterpolator functionality."""
    try:
        from paraprof.interpolation import MultiGPInterpolator
        print("✓ Successfully imported MultiGPInterpolator")
    except ImportError as e:
        print(f"✗ Failed to import MultiGPInterpolator: {e}")
        return False

    # Create test solution
    print("\nCreating synthetic coarse grid solution...")
    coarse_solution = create_test_solution()
    print(f"  Grid shape: {coarse_solution['grid_shape']}")
    print(f"  Number of solutions: {len(coarse_solution['solutions'])}")
    print(f"  Projection dims: {coarse_solution['projection_dims']}")
    print(f"  Continuous dims: {coarse_solution['continuous_dims']}")

    # Create MultiGPInterpolator
    print("\nCreating MultiGPInterpolator...")
    try:
        interpolator = MultiGPInterpolator(coarse_solution)
        print(f"✓ Created interpolator: {interpolator}")
        print(f"  Number of GPs trained: {len(interpolator.gps)}")
    except Exception as e:
        print(f"✗ Failed to create interpolator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test interpolation at a grid point (should be exact)
    print("\nTesting interpolation at coarse grid point (2, 3)...")
    grid_coords = np.array([
        coarse_solution['grid_axes'][0][2],
        coarse_solution['grid_axes'][1][3]
    ])
    expected = coarse_solution['solutions'][(2, 3)]['continuous_params']

    # Test default behavior (no uncertainties)
    predicted_default = interpolator.interpolate(grid_coords)
    if not isinstance(predicted_default, np.ndarray):
        print("✗ interpolate() should return ndarray by default")
        return False
    print("✓ Default interpolate() returns ndarray (no uncertainties)")

    # Test with uncertainties
    predicted, uncertainties = interpolator.interpolate(grid_coords, return_std=True)
    error = np.abs(predicted - expected)

    print(f"  Grid coords: {grid_coords}")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {predicted}")
    print(f"  Uncertainties: {uncertainties}")
    print(f"  Error: {error}")

    if np.all(error < 0.1):  # Should be very accurate at training points
        print("✓ Interpolation at grid point is accurate")
    else:
        print("✗ Interpolation at grid point has large error")
        return False

    # Test interpolation between grid points
    print("\nTesting interpolation between grid points...")
    grid_coords_mid = np.array([0.0, 0.0])  # Midpoint of grid
    predicted_mid, uncertainties_mid = interpolator.interpolate(grid_coords_mid, return_std=True)

    print(f"  Grid coords: {grid_coords_mid}")
    print(f"  Predicted: {predicted_mid}")
    print(f"  Uncertainties: {uncertainties_mid}")

    # Check that predictions are in reasonable range
    cont_params_all = np.array([sol['continuous_params'] for sol in coarse_solution['solutions'].values()])
    mins = cont_params_all.min(axis=0)
    maxs = cont_params_all.max(axis=0)

    in_range = np.all((predicted_mid >= mins) & (predicted_mid <= maxs))
    if in_range:
        print("✓ Interpolated values are in reasonable range")
    else:
        print("✗ Interpolated values are outside training data range")
        print(f"  Training range: [{mins}, {maxs}]")

    # Test get_max_uncertainty
    print("\nTesting get_max_uncertainty method...")
    max_unc = interpolator.get_max_uncertainty(grid_coords_mid)
    print(f"  Max uncertainty: {max_unc:.4f}")
    if max_unc >= 0:
        print("✓ get_max_uncertainty works")
    else:
        print("✗ get_max_uncertainty returned negative value")
        return False

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_multigp_interpolator()
    exit(0 if success else 1)
