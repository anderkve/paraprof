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

    def test_min_starts_capped_at_cap(self, simple_2d_function, simple_bounds_2d,
                                      basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_initial_optimizations=4)
        # auto min_starts would be 10 but is capped at the cap.
        assert s.basin_min_starts == 4

    def test_default_cap(self, simple_2d_function, simple_bounds_2d,
                         basic_projection_2d):
        # Default: generous ceiling min(400, 50*n_dims), since the stopping rule
        # controls the actual spend.
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.n_initial_optimizations == 100  # min(400, 50*2)
        # An explicit value overrides the default.
        explicit = _make_sampler(simple_2d_function, simple_bounds_2d,
                                 basic_projection_2d, n_initial_optimizations=7)
        assert explicit.n_initial_optimizations == 7

    def test_threshold_zero_disables_early_stop(self, simple_2d_function,
                                                simple_bounds_2d, basic_projection_2d):
        # undiscovered_threshold=0 is the "off" switch: the rule never fires,
        # even with many distinct ROI optima recorded past min_starts.
        s = _make_sampler(
            simple_2d_function, simple_bounds_2d, basic_projection_2d,
            advanced_config={'basin_detection': {'undiscovered_threshold': 0.0}},
        )
        s.global_max_target_val = 0.0
        for i in range(20):
            s.register_initial_optimum(np.array([float(i) * 0.4 - 2.0, 0.0]), 0.0)
        assert s.basin_detection_should_stop(20) is False


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

    def test_auto_capped_at_n_initial_optimizations(self, simple_2d_function,
                                                    simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_initial_optimizations=3)
        assert s.resolve_initial_opt_batch_size(32) == 3

    def test_auto_uses_central_fd_width(self, simple_2d_function, simple_bounds_2d,
                                        basic_projection_2d):
        # central FD doubles fd_width (2 evals/dim) -> fewer concurrent runs.
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          advanced_config={'lbfgsb': {'gradient_method': 'central'}})
        assert s.resolve_initial_opt_batch_size(8) == 2   # ceil(8 / (2*2)) = 2

    def test_explicit_override(self, simple_2d_function, simple_bounds_2d,
                               basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          advanced_config={'basin_detection': {'batch_size': 7}})
        # Explicit value is honored (only capped at n_initial_optimizations).
        assert s.resolve_initial_opt_batch_size(2) == 7
        assert s.resolve_initial_opt_batch_size(100) == 7


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
        # merge_tol is a fixed internal constant; poke it directly to exercise
        # the clustering threshold. Tight tolerance: two points 0.5/dim apart
        # (span 10) should stay distinct.
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        s.basin_merge_tol = 0.001
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


class TestGlobalOptimaPrior:
    """The ``n_optima`` (global optima count) prior steering
    ``basin_detection_should_stop``."""

    def _register_distinct(self, s, n, target_val=-0.1):
        """Register ``n`` well-separated distinct optima (count 1 each)."""
        for i in range(n):
            s.register_initial_optimum(np.array([float(i) * 1.0 - 2.0, 0.0]),
                                       target_val)

    # --- parsing ---
    def test_parse_none(self):
        from paraprof.sampler import ProfileProjector as PP
        assert PP._parse_n_optima(None) == (None, None)

    def test_parse_int_is_exact(self):
        from paraprof.sampler import ProfileProjector as PP
        assert PP._parse_n_optima(3) == (3, 3)

    def test_parse_dict(self):
        from paraprof.sampler import ProfileProjector as PP
        assert PP._parse_n_optima({'min': 2, 'max': 5}) == (2, 5)
        assert PP._parse_n_optima({'max': 4}) == (None, 4)
        assert PP._parse_n_optima({'min': 2}) == (2, None)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "x", {'min': 5, 'max': 2},
                                     {'foo': 1}, {'min': 0}])
    def test_parse_invalid_raises(self, bad):
        from paraprof.sampler import ProfileProjector as PP
        from paraprof.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            PP._parse_n_optima(bad)

    def test_constructor_sets_bounds(self, simple_2d_function, simple_bounds_2d,
                                     basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima=3)
        assert s.basin_min_optima == 3
        assert s.basin_max_optima == 3
        s2 = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                           n_optima={'max': 4})
        assert s2.basin_min_optima is None
        assert s2.basin_max_optima == 4

    def test_default_no_prior(self, simple_2d_function, simple_bounds_2d,
                              basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        assert s.basin_min_optima is None
        assert s.basin_max_optima is None

    # --- upper bound ---
    def test_upper_bound_stops_when_reached(self, simple_2d_function,
                                            simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima={'max': 3})
        s.global_max_target_val = 0.0
        self._register_distinct(s, 3)
        assert s.basin_detection_should_stop(10) is True

    def test_upper_bound_bypasses_min_starts(self, simple_2d_function,
                                             simple_bounds_2d, basic_projection_2d):
        # n_optima=1: stop after the very first converged start, well below the
        # default min_starts floor (10) -- the global guarantee makes it moot.
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima=1)
        s.global_max_target_val = 0.0
        self._register_distinct(s, 1)
        assert s.basin_detection_should_stop(1) is True

    def test_upper_bound_not_yet_reached(self, simple_2d_function, simple_bounds_2d,
                                         basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima={'max': 4})
        s.global_max_target_val = 0.0
        self._register_distinct(s, 2)
        assert s.basin_detection_should_stop(10) is False

    def test_upper_bound_counts_non_roi_optima(self, simple_2d_function,
                                               simple_bounds_2d, basic_projection_2d):
        # Global count includes optima outside the ROI (unlike the Bayesian
        # rule's W, which is ROI-restricted).
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima={'max': 2}, roi_threshold=3.0)
        s.global_max_target_val = 0.0
        self._register_distinct(s, 2, target_val=-10.0)   # both below ROI cutoff
        assert s.basin_detection_should_stop(10) is True

    # --- lower bound ---
    def test_lower_bound_overrides_bayesian_stop(self, simple_2d_function,
                                                 simple_bounds_2d, basic_projection_2d):
        # One basin hit many times: the Bayesian rule would stop (W=1, big N).
        base = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d)
        base.global_max_target_val = 0.0
        for _ in range(15):
            base.register_initial_optimum(np.array([3.0, 2.0]), -0.1)
        assert base.basin_detection_should_stop(15) is True   # BRK fires

        guarded = _make_sampler(simple_2d_function, simple_bounds_2d,
                                basic_projection_2d, n_optima={'min': 2})
        guarded.global_max_target_val = 0.0
        for _ in range(15):
            guarded.register_initial_optimum(np.array([3.0, 2.0]), -0.1)
        assert guarded.basin_detection_should_stop(15) is False  # prior overrides

    def test_lower_bound_satisfied_allows_bayesian_stop(self, simple_2d_function,
                                                        simple_bounds_2d,
                                                        basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima={'min': 2})
        s.global_max_target_val = 0.0
        # Two distinct basins, each hit several times so BRK can fire.
        for _ in range(8):
            s.register_initial_optimum(np.array([3.0, 2.0]), -0.1)
        for _ in range(8):
            s.register_initial_optimum(np.array([-3.0, -2.0]), -0.1)
        # W=2, n_roi=16, denom=13, expected = 4/13 ~ 0.31 < 0.5 -> stop.
        assert s.basin_detection_should_stop(16) is True

    def test_exact_int_stops_exactly_at_count(self, simple_2d_function,
                                              simple_bounds_2d, basic_projection_2d):
        s = _make_sampler(simple_2d_function, simple_bounds_2d, basic_projection_2d,
                          n_optima=2)
        s.global_max_target_val = 0.0
        self._register_distinct(s, 1)
        assert s.basin_detection_should_stop(10) is False   # below min=2
        self._register_distinct(s, 2)                        # now 2 distinct
        assert s.basin_detection_should_stop(10) is True    # max=2 reached


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
