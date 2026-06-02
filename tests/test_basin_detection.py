"""
Tests for the initial-optimization basin-detection machinery: online
single-linkage clustering of optima (``register_initial_optimum``) and the
Boender-Rinnooy Kan Bayesian stopping rule (``basin_detection_should_stop``).
"""
import numpy as np
import pytest
from paraprof import ProfileProjector


def _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                  advanced_config=None, **kwargs):
    return ProfileProjector(
        target_func=simple_2d_function,
        bounds=simple_bounds_2d,
        projections=[basic_projection_2d],
        advanced_config=advanced_config,
        **kwargs,
    )


class TestConfig:
    def test_defaults(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.basin_detection_enabled is True
        assert s.basin_batch_size is None
        assert s.basin_merge_tol == pytest.approx(0.02)
        assert s.basin_undiscovered_threshold == pytest.approx(0.5)
        # min_starts auto: max(10, 3 * n_dims) = max(10, 6) = 10, capped at cap.
        assert s.basin_min_starts == 10
        assert s.initial_optima_registry == []

    def test_advanced_config_overrides(self, simple_2d_function, simple_bounds_2d,
                                       basic_projection_2d):
        s = _make_sampler(
            simple_2d_function, simple_bounds_2d, basic_projection_2d,
            advanced_config={'basin_detection': {
                'enabled': False, 'batch_size': 7, 'merge_tol': 0.1,
                'undiscovered_threshold': 1.0, 'min_starts': 5,
            }},
        )
        assert s.basin_detection_enabled is False
        assert s.basin_batch_size == 7
        assert s.basin_merge_tol == pytest.approx(0.1)
        assert s.basin_undiscovered_threshold == pytest.approx(1.0)
        assert s.basin_min_starts == 5

    def test_min_starts_capped_at_cap(self, simple_2d_function, simple_bounds_2d,
                                      basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_initial_optimizations=4)
        # auto min_starts would be 10 but is capped at the cap.
        assert s.basin_min_starts == 4


class TestRegistry:
    def test_distinct_optima_registered_separately(self, simple_2d_function,
                                                   simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.register_initial_optimum(np.array([3.0, 2.0]), -0.1) is True
        assert s.register_initial_optimum(np.array([-3.0, -3.0]), -0.2) is True
        assert len(s.initial_optima_registry) == 2

    def test_nearby_endpoints_merge(self, simple_2d_function, simple_bounds_2d,
                                    basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.register_initial_optimum(np.array([3.0, 2.0]), -0.5)
        # bounds span is 10 per dim; tol=0.02 -> RMS-normalized. A point ~0.05
        # away per dim is well within tolerance.
        merged = s.register_initial_optimum(np.array([3.05, 2.03]), -0.1)
        assert merged is False
        assert len(s.initial_optima_registry) == 1
        entry = s.initial_optima_registry[0]
        assert entry['count'] == 2
        # Better target value and its point are retained on merge.
        assert entry['target_val'] == pytest.approx(-0.1)
        np.testing.assert_allclose(entry['point'], [3.05, 2.03])

    def test_merge_keeps_existing_when_worse(self, simple_2d_function, simple_bounds_2d,
                                             basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.register_initial_optimum(np.array([3.0, 2.0]), -0.1)
        s.register_initial_optimum(np.array([3.02, 2.01]), -0.9)
        entry = s.initial_optima_registry[0]
        assert entry['count'] == 2
        assert entry['target_val'] == pytest.approx(-0.1)
        np.testing.assert_allclose(entry['point'], [3.0, 2.0])

    def test_merge_tol_boundary(self, simple_2d_function, simple_bounds_2d,
                                basic_projection_2d):
        # Tight tolerance: two points 0.5/dim apart (span 10) should stay distinct.
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          advanced_config={'basin_detection': {'merge_tol': 0.001}})
        s.register_initial_optimum(np.array([3.0, 2.0]), -0.1)
        s.register_initial_optimum(np.array([3.5, 2.5]), -0.1)
        assert len(s.initial_optima_registry) == 2


class TestROIStats:
    def test_roi_filtering(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          roi_threshold=4.0)
        s.global_max_target_val = 0.0
        # Two ROI optima (>= -4) and one sub-ROI optimum (< -4).
        s.register_initial_optimum(np.array([3.0, 2.0]), -0.0)
        s.register_initial_optimum(np.array([-3.0, -3.0]), -1.0)
        s.register_initial_optimum(np.array([0.0, 4.5]), -30.0)
        W, n_roi = s.basin_detection_roi_stats()
        assert W == 2
        assert n_roi == 2  # one count each for the two ROI optima

    def test_roi_counts_aggregate(self, simple_2d_function, simple_bounds_2d,
                                  basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          roi_threshold=4.0)
        s.global_max_target_val = 0.0
        for _ in range(5):
            s.register_initial_optimum(np.array([3.0, 2.0]), 0.0)
        W, n_roi = s.basin_detection_roi_stats()
        assert W == 1
        assert n_roi == 5


class TestStoppingRule:
    def test_no_stop_below_min_starts(self, simple_2d_function, simple_bounds_2d,
                                      basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.global_max_target_val = 0.0
        for _ in range(5):
            s.register_initial_optimum(np.array([3.0, 2.0]), 0.0)
        # min_starts is 10; 5 completed must not stop regardless of stats.
        assert s.basin_detection_should_stop(5) is False

    def test_single_basin_stops(self, simple_2d_function, simple_bounds_2d,
                                basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.global_max_target_val = 0.0
        # 10 starts all in one basin: W=1, N_roi=10 -> 1/(10-1-1)=0.125 < 0.5.
        for _ in range(10):
            s.register_initial_optimum(np.array([3.0, 2.0]), 0.0)
        assert s.basin_detection_should_stop(10) is True

    def test_keeps_going_while_finding_new(self, simple_2d_function, simple_bounds_2d,
                                           basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.global_max_target_val = 0.0
        # 10 distinct optima from 10 starts: W=N_roi=10 -> denom < 0 -> no stop.
        for i in range(10):
            s.register_initial_optimum(np.array([float(i) * 0.4 - 2.0, 0.0]), 0.0)
        assert s.basin_detection_should_stop(10) is False

    def test_threshold_controls_stop(self, simple_2d_function, simple_bounds_2d,
                                     basic_projection_2d):
        # W=2, N_roi=12 -> expected undiscovered = 4/(12-2-1) = 0.444.
        cfg_lo = {'basin_detection': {'undiscovered_threshold': 0.5, 'min_starts': 5}}
        cfg_hi = {'basin_detection': {'undiscovered_threshold': 0.4, 'min_starts': 5}}
        for cfg, expected in [(cfg_lo, True), (cfg_hi, False)]:
            s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                              advanced_config=cfg, roi_threshold=4.0)
            s.global_max_target_val = 0.0
            for _ in range(6):
                s.register_initial_optimum(np.array([3.0, 2.0]), 0.0)
            for _ in range(6):
                s.register_initial_optimum(np.array([-3.0, -3.0]), 0.0)
            assert s.basin_detection_should_stop(12) is expected


class TestLHSPool:
    def test_pool_consumption(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.init_initial_opt_lhs(5)
        assert s._initial_opt_start_points.shape == (5, 2)
        seen = []
        for jid in range(5):
            job, nxt = s.create_one_initial_optimization_job(jid)
            assert nxt == jid + 1
            assert job.type == 'INITIAL_OPTIMIZATION'
            seen.append(job.start_params_full)
        assert s._initial_opt_lhs_idx == 5
        # All start points lie within bounds.
        pts = np.array(seen)
        assert np.all(pts[:, 0] >= simple_bounds_2d[0, 0])
        assert np.all(pts[:, 0] <= simple_bounds_2d[0, 1])

    def test_pool_refills_when_exhausted(self, simple_2d_function, simple_bounds_2d,
                                         basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.init_initial_opt_lhs(2)
        # Consume more than the pool holds; should refill defensively.
        for jid in range(4):
            s.create_one_initial_optimization_job(jid)
        # No exception; pool was regenerated.
        assert s._initial_opt_start_points is not None
