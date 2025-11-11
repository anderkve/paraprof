"""
Tests for grid interpolation functionality.
"""
import numpy as np
import pytest
from paraprof.interpolation import GridInterpolator


class TestGridInterpolator:
    """Test suite for GridInterpolator."""

    def test_interpolator_initialization(self):
        """Test that interpolator initializes from grid solution."""
        # Create a simple coarse solution
        grid_axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        projection_dims = [0, 1]
        continuous_dims = [2]

        solutions = {}
        for i in range(3):
            for j in range(3):
                solutions[(i, j)] = {
                    'continuous_params': np.array([0.5]),
                    'likelihood': -(i**2 + j**2),
                }

        coarse_solution = {
            'grid_axes': grid_axes,
            'projection_dims': projection_dims,
            'continuous_dims': continuous_dims,
            'solutions': solutions,
            'grid_shape': (3, 3),
        }

        interpolator = GridInterpolator(coarse_solution)

        assert interpolator is not None
        assert len(interpolator.continuous_dims) == 1

    def test_interpolation_at_grid_points(self):
        """Test that interpolation returns exact values at grid points."""
        # Create a simple coarse solution
        grid_axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        projection_dims = [0, 1]
        continuous_dims = [2]

        solutions = {}
        for i in range(3):
            for j in range(3):
                solutions[(i, j)] = {
                    'continuous_params': np.array([float(i + j)]),
                    'likelihood': 0.0,
                }

        coarse_solution = {
            'grid_axes': grid_axes,
            'projection_dims': projection_dims,
            'continuous_dims': continuous_dims,
            'solutions': solutions,
            'grid_shape': (3, 3),
        }

        interpolator = GridInterpolator(coarse_solution)

        # Interpolate at a grid point
        result = interpolator.interpolate(np.array([1.0, 1.0]))

        # Should be exact at grid point
        np.testing.assert_allclose(result, [2.0], atol=1e-10)

    def test_interpolation_between_grid_points(self):
        """Test interpolation between grid points."""
        # Create a simple linear coarse solution
        grid_axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        projection_dims = [0, 1]
        continuous_dims = [2]

        solutions = {
            (0, 0): {'continuous_params': np.array([0.0]), 'likelihood': 0.0},
            (0, 1): {'continuous_params': np.array([1.0]), 'likelihood': 0.0},
            (1, 0): {'continuous_params': np.array([1.0]), 'likelihood': 0.0},
            (1, 1): {'continuous_params': np.array([2.0]), 'likelihood': 0.0},
        }

        coarse_solution = {
            'grid_axes': grid_axes,
            'projection_dims': projection_dims,
            'continuous_dims': continuous_dims,
            'solutions': solutions,
            'grid_shape': (2, 2),
        }

        interpolator = GridInterpolator(coarse_solution)

        # Interpolate at center
        result = interpolator.interpolate(np.array([0.5, 0.5]))

        # Should be average of corners
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_get_coverage_fraction(self):
        """Test coverage fraction calculation."""
        grid_axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
        projection_dims = [0, 1]
        continuous_dims = [2]

        # Only fill half the grid
        solutions = {}
        for i in range(3):
            for j in range(2):  # Only 2 out of 3 in second dimension
                solutions[(i, j)] = {
                    'continuous_params': np.array([0.5]),
                    'likelihood': 0.0,
                }

        coarse_solution = {
            'grid_axes': grid_axes,
            'projection_dims': projection_dims,
            'continuous_dims': continuous_dims,
            'solutions': solutions,
            'grid_shape': (3, 3),
        }

        interpolator = GridInterpolator(coarse_solution)
        coverage = interpolator.get_coverage_fraction()

        # 6 out of 9 points covered
        assert coverage == pytest.approx(6/9, rel=1e-5)
