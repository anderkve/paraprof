"""
Tests for the ProfileProjector class.
"""
import numpy as np
import pytest
from paraprof import ProfileProjector


class TestProfileProjector:
    """Test suite for ProfileProjector."""

    def test_initialization(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        """Test that sampler initializes correctly."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
            pop_per_grid_point=2,
        )

        assert sampler.dims == 2
        assert len(sampler.bounds) == 2
        assert sampler.pop_per_grid_point == 2
        assert sampler.target_calls == 0
        assert sampler.global_max_target_val == -np.inf

    def test_grid_index_conversion(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        """Test grid index to coordinate conversion."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
        )

        # Test index (0, 0) should map to bounds minimum
        coords = sampler._get_grid_coords_from_indices((0, 0))
        np.testing.assert_allclose(coords, [-5.0, -5.0])

        # Test index at grid maximum
        max_idx = tuple(np.array(sampler.grid_shape) - 1)
        coords = sampler._get_grid_coords_from_indices(max_idx)
        np.testing.assert_allclose(coords, [5.0, 5.0])

    def test_valid_neighbors(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        """Test neighbor generation."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
        )

        # Corner point should have 3 neighbors (in 2D grid)
        neighbors = list(sampler._get_valid_neighbors((0, 0)))
        assert len(neighbors) == 3

        # Center point should have 8 neighbors (in 2D grid)
        center = (2, 2)
        neighbors = list(sampler._get_valid_neighbors(center))
        assert len(neighbors) == 8

    def test_ensure_bounds(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        """Test parameter bounding."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
        )

        # Test clipping values outside bounds
        vec = np.array([-10.0, 10.0])
        clipped = sampler._ensure_bounds(vec, [0, 1])
        np.testing.assert_allclose(clipped, [-5.0, 5.0])

    def test_projection_configuration(self, simple_2d_function, simple_bounds_4d):
        """Test projection dimension configuration."""
        projection_1d = {'dims': [0], 'grid_points': [10]}
        projection_2d = {'dims': [0, 1], 'grid_points': [5, 5]}

        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_4d,
            projections=[projection_1d, projection_2d],
        )

        # Check that first projection is configured
        assert sampler.projection_dims == [0]
        assert sampler.continuous_dims == [1, 2, 3]
        assert sampler.n_proj_dims == 1
        assert sampler.n_cont_dims == 3

    def test_invalid_projection_raises_error(self, simple_2d_function, simple_bounds_2d):
        """Test that invalid projections raise errors."""
        from paraprof.exceptions import InvalidProjectionError

        # Dimension out of bounds
        invalid_projection = {'dims': [5], 'grid_points': [10]}

        with pytest.raises(InvalidProjectionError, match="invalid dimension index"):
            sampler = ProfileProjector(
                target_func=simple_2d_function,
                bounds=simple_bounds_2d,
                projections=[invalid_projection],
            )

    def test_mutation_strategy_validation(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        """Test that invalid mutation strategies are rejected."""
        with pytest.raises(ValueError, match="mutation_strategy"):
            sampler = ProfileProjector(
                target_func=simple_2d_function,
                bounds=simple_bounds_2d,
                projections=[basic_projection_2d],
                mutation_strategy='invalid-strategy',
            )
