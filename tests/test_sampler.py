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
        assert sampler.profiled_dims == [1, 2, 3]
        assert sampler.n_proj_dims == 1
        assert sampler.n_prof_dims == 3

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


class TestProximityWarmStart:
    """Test the proximity-aware sampling that powers cross-projection
    knowledge transfer in ActivationJob."""

    def _make_sampler(self, simple_2d_function, simple_bounds_4d):
        """Sampler in a 2D projection over a 4D space, ready to query the
        proximity sampler. Projection dims [0, 1] -> profiled dims [2, 3]."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_4d,
            projections=[{'dims': [0, 1], 'grid_points': [5, 5]}],
            pop_per_grid_point=2,
        )
        return sampler

    def test_proximity_returns_nearest_in_projection_dims(
            self, simple_2d_function, simple_bounds_4d):
        sampler = self._make_sampler(simple_2d_function, simple_bounds_4d)

        # Three pool entries with the same fitness; only their projection-dim
        # coords differ. The returned profiled coords must be those of the
        # entry whose projection-dim coords are closest to the target.
        sampler._update_global_pool(
            np.array([0.0, 0.0, 7.0, 8.0]), 0.0, grid_idx=None)
        sampler._update_global_pool(
            np.array([4.0, 4.0, 1.0, 2.0]), 0.0, grid_idx=None)
        sampler._update_global_pool(
            np.array([-3.0, -3.0, 5.0, 6.0]), 0.0, grid_idx=None)

        # Target near (4, 4) in projection-dim space -> expect profiled
        # coords [1.0, 2.0] back.
        out = sampler._sample_proximity_from_global_pool(1, np.array([4.0, 4.0]))
        assert out is not None
        np.testing.assert_allclose(out[0], [1.0, 2.0])

    def test_pool_seeding_populates_initial_maxima(
            self, simple_2d_function, simple_bounds_4d):
        """Seeding initial_maxima from the in-memory pool should pick the
        best entry per (current-projection) grid cell, filter by ROI cutoff,
        and skip the global L-BFGS-B initial optimization stage."""
        sampler = self._make_sampler(simple_2d_function, simple_bounds_4d)
        # Three entries: one strong (within ROI), one weaker (within ROI),
        # one poor (outside ROI even after global max set).
        sampler._update_global_pool(
            np.array([0.0, 0.0, 0.0, 0.0]), -0.5, grid_idx=None)
        sampler._update_global_pool(
            np.array([3.0, 3.0, 0.5, 0.5]), -1.5, grid_idx=None)
        sampler._update_global_pool(
            np.array([-3.0, -3.0, 0.0, 0.0]), -50.0, grid_idx=None)
        sampler.roi_threshold = 3.0  # ROI = global_max - 3.0

        sampler._initialize_from_global_pool()

        # The two within-ROI entries should be in initial_maxima, sorted
        # best-first; the poor entry should be filtered out.
        assert len(sampler.initial_maxima) == 2
        assert sampler.initial_maxima[0]['target_val'] == -0.5
        assert sampler.initial_maxima[1]['target_val'] == -1.5

    def test_global_pool_size_scales_with_dimensionality(
            self, simple_2d_function):
        """``global_pool_size`` floors at ``DEFAULT_GLOBAL_POOL_SIZE``,
        otherwise scales linearly with target dimensionality, and caps at
        ``DEFAULT_GLOBAL_POOL_MAX`` to bound master memory in very high-D
        scans."""
        from paraprof.sampler import (
            DEFAULT_GLOBAL_POOL_SIZE, DEFAULT_GLOBAL_POOL_PER_DIM,
            DEFAULT_GLOBAL_POOL_MAX,
        )

        # 4-D target sits at the floor (n_dims * per_dim == default).
        sampler_4d = ProfileProjector(
            target_func=simple_2d_function,
            bounds=np.array([[-1.0, 1.0]] * 4),
            projections=[{'dims': [0, 1], 'grid_points': [5, 5]}],
            pop_per_grid_point=2,
        )
        assert sampler_4d.global_pool_size == max(
            DEFAULT_GLOBAL_POOL_SIZE,
            4 * DEFAULT_GLOBAL_POOL_PER_DIM,
        )

        # 10-D target lifts the pool above the floor but well below the cap.
        sampler_10d = ProfileProjector(
            target_func=simple_2d_function,
            bounds=np.array([[-1.0, 1.0]] * 10),
            projections=[{'dims': [0, 1], 'grid_points': [5, 5]}],
            pop_per_grid_point=2,
        )
        assert sampler_10d.global_pool_size == 10 * DEFAULT_GLOBAL_POOL_PER_DIM
        assert sampler_10d.global_pool_size > sampler_4d.global_pool_size

        # 80-D target would scale to 200 000 entries; the cap pins it.
        sampler_80d = ProfileProjector(
            target_func=simple_2d_function,
            bounds=np.array([[-1.0, 1.0]] * 80),
            projections=[{'dims': [0, 1], 'grid_points': [5, 5]}],
            pop_per_grid_point=2,
        )
        assert sampler_80d.global_pool_size == DEFAULT_GLOBAL_POOL_MAX
        assert 80 * DEFAULT_GLOBAL_POOL_PER_DIM > DEFAULT_GLOBAL_POOL_MAX

    def test_proximity_normalises_by_bounds_extent(
            self, simple_2d_function):
        # First projection dim spans 1.0, second spans 1000.0. Without
        # normalisation, the "closest" entry is dominated by dim-1 distance.
        bounds = np.array([[0.0, 1.0], [0.0, 1000.0],
                           [-1.0, 1.0], [-1.0, 1.0]])
        sampler = ProfileProjector(
            target_func=simple_2d_function, bounds=bounds,
            projections=[{'dims': [0, 1], 'grid_points': [5, 5]}],
            pop_per_grid_point=2,
        )
        # Two candidates near target (0.5, 500). Candidate A is closer in the
        # *normalised* sense; candidate B is closer in raw Euclidean distance
        # because its dim-1 deviation is smaller in absolute terms.
        sampler._update_global_pool(
            np.array([0.5, 800.0, 0.1, 0.2]), 0.0, grid_idx=None)  # A: norm-near
        sampler._update_global_pool(
            np.array([0.95, 500.0, 0.7, 0.8]), 0.0, grid_idx=None)  # B: raw-near
        out = sampler._sample_proximity_from_global_pool(1, np.array([0.5, 500.0]))
        # In normalised distance, A is closer (delta = (0, 0.3)) than
        # B (delta = (0.45, 0)). So we should get A's profiled coords back.
        np.testing.assert_allclose(out[0], [0.1, 0.2])

