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
        assert s.basin_batch_size is None
        # merge_tol is a fixed internal constant, not a config knob.
        assert s.basin_merge_tol == pytest.approx(0.02)
        assert s.basin_undiscovered_threshold == pytest.approx(0.5)
        # min_starts auto: max(10, 3 * n_dims) = max(10, 6) = 10, capped at cap.
        assert s.basin_min_starts == 10
        assert s.initial_optima_registry == []
        # Default: generous ceiling min(400, 50*n_dims).
        assert s.n_initial_optimizations == 100  # min(400, 50*2)

    def test_advanced_config_overrides(self, simple_2d_function, simple_bounds_2d,
                                       basic_projection_2d):
        s = _make_sampler(
            simple_2d_function, simple_bounds_2d, basic_projection_2d,
            advanced_config={'basin_detection': {
                'batch_size': 7,
                'undiscovered_threshold': 1.0, 'min_starts': 5,
            }},
        )
        assert s.basin_batch_size == 7
        assert s.basin_undiscovered_threshold == pytest.approx(1.0)
        assert s.basin_min_starts == 5


class TestBatchSize:
    def test_auto_fd_aware_default(self, simple_2d_function, simple_bounds_2d,
                                   basic_projection_2d):
        # 2-D, forward FD -> fd_width = 2, so auto batch ~= n_workers / 2,
        # floored at 2 and capped at n_workers and the cap (default 100).
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.resolve_initial_opt_batch_size(1) == 1   # capped at n_workers
        assert s.resolve_initial_opt_batch_size(2) == 2   # floor
        assert s.resolve_initial_opt_batch_size(8) == 4
        assert s.resolve_initial_opt_batch_size(32) == 16


class TestRegistry:
    def test_distinct_optima_registered_separately(self, simple_2d_function,
                                                   simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.register_initial_optimum(np.array([3.0, 2.0]), -0.1) is True
        assert s.register_initial_optimum(np.array([-3.0, -3.0]), -0.2) is True
        assert len(s.initial_optima_registry) == 2

    def test_nearby_endpoints_merge_keeping_better(self, simple_2d_function,
                                                   simple_bounds_2d, basic_projection_2d):
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


class TestGlobalOptimaPrior:
    """The ``n_optima`` (global optima count) prior steering
    ``basin_detection_should_stop``."""

    def _register_distinct(self, s, n, target_val=-0.1):
        """Register ``n`` well-separated distinct optima (count 1 each)."""
        for i in range(n):
            s.register_initial_optimum(np.array([float(i) * 1.0 - 2.0, 0.0]),
                                       target_val)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "x", {'min': 5, 'max': 2},
                                     {'foo': 1}, {'min': 0}])
    def test_parse_invalid_raises(self, bad):
        from paraprof.sampler import ProfileProjector as PP
        from paraprof.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            PP._parse_n_optima(bad)

    def test_upper_bound_stops_when_reached(self, simple_2d_function,
                                            simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima={'max': 3})
        s.global_max_target_val = 0.0
        self._register_distinct(s, 2)
        assert s.basin_detection_should_stop(10) is False
        self._register_distinct(s, 3)
        assert s.basin_detection_should_stop(10) is True


class TestConvergenceGating:
    """Only converged initial-optimization runs feed the distinct-optima
    registry; truncated runs still update the max / pool / initial_maxima."""

    def _initial_opt_job(self, s, converged, params, fitness):
        from paraprof.jobs.lbfgsb_job import LBFGSBJob
        params = np.asarray(params, dtype=float)
        job = LBFGSBJob(
            job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=s,
            opt_dims=tuple(range(s.dims)), start_params=params,
            grid_idx=None, start_params_full=params,
        )
        job.success = True
        job.converged = converged
        job.current_params = params
        job.current_fitness = fitness
        return job

    def test_converged_run_registers(self, simple_2d_function, simple_bounds_2d,
                                     basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        self._initial_opt_job(s, True, [1.0, 1.0], -0.1).on_finish(99)
        assert len(s.initial_optima_registry) == 1
        assert s.global_max_target_val == pytest.approx(-0.1)
        assert len(s.global_solution_pool) == 1
        assert len(s.initial_maxima) == 1

    def test_truncated_run_skips_registry(self, simple_2d_function, simple_bounds_2d,
                                          basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        self._initial_opt_job(s, False, [1.0, 1.0], -0.1).on_finish(99)
        # Not counted as a distinct optimum...
        assert len(s.initial_optima_registry) == 0
        # ...but still a valid evaluation everywhere else.
        assert s.global_max_target_val == pytest.approx(-0.1)
        assert len(s.global_solution_pool) == 1
        assert len(s.initial_maxima) == 1
