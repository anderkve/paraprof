"""Tests for the volume-sampling building blocks (paraprof.volume, Phases 1-2)."""

import logging

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import ConfigurationError
from paraprof.sample_io import write_samples
from paraprof.volume import (
    VOLUME_CONFIG_DEFAULTS,
    AnchorSet,
    ProjectionEnvelope,
    generate_anchors,
    harvest_existing_samples,
    normalize_volume_config,
    resolve_harvest_files,
    volume_band,
)


def make_record(projection_dims, grid_axes, cell_values):
    """An export_grid_solution()-style record from a {grid_idx: logL} dict."""
    return {
        'projection_dims': list(projection_dims),
        'grid_axes': [np.asarray(ax, dtype=float) for ax in grid_axes],
        'solutions': {idx: {'likelihood': val} for idx, val in cell_values.items()},
        'grid_shape': tuple(len(ax) for ax in grid_axes),
    }


def band_envelope_1d(dim, lo, hi, n_dims, n_cells=101, bounds_extent=(-5.0, 5.0),
                     global_max=0.0):
    """Envelope from one 1D projection whose ROI cells span [lo, hi] on `dim`."""
    axis = np.linspace(bounds_extent[0], bounds_extent[1], n_cells)
    cells = {(i,): global_max for i, x in enumerate(axis) if lo <= x <= hi}
    rec = make_record([dim], [axis], cells)
    return ProjectionEnvelope([rec], global_max=global_max, n_dims=n_dims)


# --------------------------------------------------------------------------- #
# Config normalization
# --------------------------------------------------------------------------- #
class TestNormalizeVolumeConfig:

    def test_defaults_filled(self):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        # roi_threshold defaults to None in the template but is filled from
        # the projection's roi_threshold at normalization.
        expected = dict(VOLUME_CONFIG_DEFAULTS, roi_threshold=4.0)
        assert cfg == expected

    def test_returns_new_dict(self):
        user = {'n_points': 10}
        cfg = normalize_volume_config(user, roi_threshold=4.0)
        assert cfg['n_points'] == 10
        assert user == {'n_points': 10}

    def test_not_a_dict(self):
        with pytest.raises(ConfigurationError, match="must be a dict"):
            normalize_volume_config("roi", roi_threshold=4.0)

    def test_unknown_key(self):
        with pytest.raises(ConfigurationError, match="unknown keys.*'mod'"):
            normalize_volume_config({'mod': 'roi'}, roi_threshold=4.0)

    def test_roi_threshold_defaults_to_projection(self):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        assert cfg['roi_threshold'] == 4.0

    def test_roi_threshold_override(self):
        # The stage can reach past the projection ROI (into the shell).
        cfg = normalize_volume_config({'roi_threshold': 25.0}, roi_threshold=4.0)
        assert cfg['roi_threshold'] == 25.0

    @pytest.mark.parametrize("bad", [0, -0.1, np.inf, True])
    def test_bad_roi_threshold(self, bad):
        with pytest.raises(ConfigurationError, match="roi_threshold"):
            normalize_volume_config({'roi_threshold': bad}, roi_threshold=4.0)

    @pytest.mark.parametrize("key", ['n_points', 'eval_budget', 'search_max_iter'])
    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "10"])
    def test_bad_positive_ints(self, key, bad):
        with pytest.raises(ConfigurationError, match=key):
            normalize_volume_config({key: bad}, roi_threshold=4.0)

    @pytest.mark.parametrize("key", ['eval_budget', 'search_max_iter', 'min_spacing'])
    def test_none_allowed(self, key):
        cfg = normalize_volume_config({key: None}, roi_threshold=4.0)
        assert cfg[key] is None

    @pytest.mark.parametrize("bad", [0, -0.1, np.inf, True])
    def test_bad_min_spacing(self, bad):
        with pytest.raises(ConfigurationError, match="min_spacing"):
            normalize_volume_config({'min_spacing': bad}, roi_threshold=4.0)

    def test_bad_search(self):
        with pytest.raises(ConfigurationError, match="'lbfgsb' or 'none'"):
            normalize_volume_config({'search': 'de'}, roi_threshold=4.0)

    def test_bad_probe_all_anchors(self):
        with pytest.raises(ConfigurationError, match="probe_all_anchors"):
            normalize_volume_config({'probe_all_anchors': 1}, roi_threshold=4.0)

    def test_harvest_files_normalized_to_list(self):
        cfg = normalize_volume_config({'harvest_files': 'a.csv'}, roi_threshold=4.0)
        assert cfg['harvest_files'] == ['a.csv']
        cfg = normalize_volume_config(
            {'harvest_files': ('a.csv', 'b.h5')}, roi_threshold=4.0)
        assert cfg['harvest_files'] == ['a.csv', 'b.h5']

    def test_bad_harvest_files(self):
        with pytest.raises(ConfigurationError, match="harvest_files"):
            normalize_volume_config({'harvest_files': [1]}, roi_threshold=4.0)

    def test_bad_output_file(self):
        with pytest.raises(ConfigurationError, match="output_file"):
            normalize_volume_config({'output_file': ''}, roi_threshold=4.0)


class TestVolumeBand:

    def test_roi_band(self):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        lo, delta = volume_band(cfg, global_max=10.0)
        assert lo == 6.0 and delta == 4.0

    def test_widened_band(self):
        # A larger stage threshold reaches deeper (into the shell).
        cfg = normalize_volume_config({'roi_threshold': 25.0}, roi_threshold=4.0)
        lo, delta = volume_band(cfg, global_max=10.0)
        assert lo == -15.0 and delta == 25.0


# --------------------------------------------------------------------------- #
# ProjectionEnvelope
# --------------------------------------------------------------------------- #
class TestProjectionEnvelope:

    def test_cylinder_intersection(self):
        # Two 1D projections of a 2D space: ROI is |x| <= 2 and y >= 0.
        axis = np.linspace(-5.0, 5.0, 11)
        rec_x = make_record([0], [axis], {(i,): 0.0 for i, x in enumerate(axis)
                                          if abs(x) <= 2.0})
        rec_y = make_record([1], [axis], {(i,): 0.0 for i, y in enumerate(axis)
                                          if y >= 0.0})
        env = ProjectionEnvelope([rec_x, rec_y], global_max=0.0, n_dims=2)

        points = np.array([
            [0.0, 0.0],    # passes both
            [0.0, 3.0],    # passes both
            [4.0, 3.0],    # fails x (never-activated cell)
            [0.0, -3.0],   # fails y
            [4.0, -3.0],   # fails both
        ])
        np.testing.assert_array_equal(
            env.test(points, threshold_delta=4.0),
            [True, True, False, False, False],
        )

    def test_single_point_input(self):
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        assert env.test([0.0, 4.0], threshold_delta=4.0)[0]
        assert not env.test([3.0, 0.0], threshold_delta=4.0)[0]

    def test_below_threshold_cell_rejects(self):
        axis = np.linspace(-5.0, 5.0, 11)
        # Activated cells everywhere, but only the center cell is in the ROI.
        cells = {(i,): (0.0 if x == 0.0 else -10.0) for i, x in enumerate(axis)}
        env = ProjectionEnvelope([make_record([0], [axis], cells)],
                                 global_max=0.0, n_dims=1)
        assert env.test([[0.0]], threshold_delta=4.0)[0]
        assert not env.test([[2.0]], threshold_delta=4.0)[0]
        # A looser threshold (a widened stage ROI) lets the low cells through.
        assert env.test([[2.0]], threshold_delta=25.0)[0]

    def test_final_global_max_recomputes_membership(self):
        # Cells stored at logL=0; a later projection found global max 10, so
        # none of them are in the ROI any more.
        env_old = band_envelope_1d(0, -2.0, 2.0, n_dims=1, global_max=0.0)
        assert env_old.test([[0.0]], threshold_delta=4.0)[0]

        axis = np.linspace(-5.0, 5.0, 101)
        cells = {(i,): 0.0 for i, x in enumerate(axis) if abs(x) <= 2.0}
        env_new = ProjectionEnvelope([make_record([0], [axis], cells)],
                                     global_max=10.0, n_dims=1)
        assert not env_new.test([[0.0]], threshold_delta=4.0)[0]

    def test_covers_full_space(self):
        axis = np.linspace(-5.0, 5.0, 6)
        rec_full = make_record([0, 1], [axis, axis], {(0, 0): 0.0})
        rec_partial = make_record([0], [axis], {(0,): 0.0})
        assert ProjectionEnvelope([rec_partial, rec_full],
                                  global_max=0.0, n_dims=2).covers_full_space
        assert not ProjectionEnvelope([rec_partial],
                                      global_max=0.0, n_dims=2).covers_full_space

    def test_from_projection_results_prefers_refined(self):
        axis_coarse = np.linspace(-5.0, 5.0, 6)
        axis_fine = np.linspace(-5.0, 5.0, 11)
        coarse = make_record([0], [axis_coarse], {(0,): 0.0})
        refined = make_record([0], [axis_fine], {(i,): 0.0 for i in range(11)})
        results = [
            {'coarse_solution': coarse, 'refined_solution': refined},
            {'coarse_solution': coarse, 'refined_solution': None},
        ]
        env = ProjectionEnvelope.from_projection_results(
            results, global_max=0.0, n_dims=1)
        assert env.n_projections == 2
        assert len(env._records[0]['axes'][0]) == 11
        assert len(env._records[1]['axes'][0]) == 6

    def test_from_projection_results_missing_solution(self):
        with pytest.raises(ValueError, match="no exported grid solution"):
            ProjectionEnvelope.from_projection_results(
                [{'coarse_solution': None, 'refined_solution': None}],
                global_max=0.0, n_dims=1)

    def test_invalid_records(self):
        axis = np.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError, match="out of range"):
            ProjectionEnvelope([make_record([3], [axis], {})],
                               global_max=0.0, n_dims=2)
        with pytest.raises(ValueError, match="grid axes"):
            ProjectionEnvelope([make_record([0, 1], [axis], {})],
                               global_max=0.0, n_dims=2)
        with pytest.raises(ValueError, match="at least 2 points"):
            ProjectionEnvelope([make_record([0], [axis[:1]], {})],
                               global_max=0.0, n_dims=2)

    def test_bad_points_shape(self):
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        with pytest.raises(ValueError, match="shape"):
            env.test(np.zeros((3, 3)), threshold_delta=4.0)

    def test_cell_mapping_matches_sampler(self, simple_2d_function,
                                          simple_bounds_2d):
        """The envelope must bin points exactly like the scan did."""
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[{'dims': [0], 'grid_points': [13]}],
        )
        rec = make_record(sampler.projection_dims, sampler.grid_axes, {})
        env = ProjectionEnvelope([rec], global_max=0.0, n_dims=2)

        rng = np.random.default_rng(7)
        points = rng.uniform(-5.0, 5.0, size=(200, 2))
        env_idx = env.cell_indices(0, points)[0]
        sampler_idx = np.array(
            [sampler._get_grid_indices_from_point(p)[0] for p in points])
        np.testing.assert_array_equal(env_idx, sampler_idx)


# --------------------------------------------------------------------------- #
# Anchor generation
# --------------------------------------------------------------------------- #
class TestGenerateAnchors:

    BOUNDS_2D = np.array([[-5.0, 5.0], [-5.0, 5.0]])

    def test_anchors_pass_envelope(self):
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        anchor_set = generate_anchors(env, self.BOUNDS_2D, n_points=100,
                                      threshold_delta=4.0, seed=1)
        assert anchor_set.n_anchors == 100
        assert env.test(anchor_set.anchors, threshold_delta=4.0).all()
        # Half-cell rounding slack at the ROI edge (cell width 0.1).
        assert np.all(np.abs(anchor_set.anchors[:, 0]) <= 1.0 + 0.05)
        # Unconstrained dim spans the box.
        assert anchor_set.anchors[:, 1].min() < -3.0
        assert anchor_set.anchors[:, 1].max() > 3.0

    def test_prefilter_acceptance_estimate(self):
        # ROI band |x| <= 1 of [-5, 5]: ~20% of the box volume.
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        anchor_set = generate_anchors(env, self.BOUNDS_2D, n_points=100,
                                      threshold_delta=4.0, seed=2)
        assert anchor_set.n_draws >= 4096
        assert anchor_set.prefilter_acceptance == pytest.approx(0.21, abs=0.03)

    def test_seed_reproducibility(self):
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        a = generate_anchors(env, self.BOUNDS_2D, 50, 4.0, seed=3)
        b = generate_anchors(env, self.BOUNDS_2D, 50, 4.0, seed=3)
        c = generate_anchors(env, self.BOUNDS_2D, 50, 4.0, seed=4)
        np.testing.assert_array_equal(a.anchors, b.anchors)
        assert not np.array_equal(a.anchors, c.anchors)

    def test_draw_cap_warning(self):
        # Envelope with no activated cells rejects everything.
        axis = np.linspace(-5.0, 5.0, 11)
        env = ProjectionEnvelope([make_record([0], [axis], {})],
                                 global_max=0.0, n_dims=2)

        # paraprof's logger does not propagate, so caplog can't see it;
        # listen with a handler attached to the logger itself.
        messages = []
        handler = logging.Handler()
        handler.emit = lambda record: messages.append(record.getMessage())
        logger = logging.getLogger("paraprof")
        logger.addHandler(handler)
        try:
            anchor_set = generate_anchors(env, self.BOUNDS_2D, n_points=10,
                                          threshold_delta=4.0, seed=5,
                                          draw_cap=8192)
        finally:
            logger.removeHandler(handler)

        assert anchor_set.n_anchors == 0
        assert anchor_set.n_draws >= 8192
        assert anchor_set.prefilter_acceptance == 0.0
        assert any("draw cap" in m for m in messages)

    def test_min_spacing_respected(self):
        env = band_envelope_1d(0, -5.0, 5.0, n_dims=2)  # whole box passes
        spacing = 0.15
        anchor_set = generate_anchors(env, self.BOUNDS_2D, n_points=30,
                                      threshold_delta=4.0,
                                      min_spacing=spacing, seed=6)
        assert anchor_set.n_anchors == 30
        assert anchor_set.coverage_radius == spacing
        scaled = anchor_set.scaled_anchors
        diff = scaled[:, None, :] - scaled[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        assert dists.min() >= spacing

    def test_default_coverage_radius_is_median_nn(self):
        env = band_envelope_1d(0, -5.0, 5.0, n_dims=2)
        anchor_set = generate_anchors(env, self.BOUNDS_2D, n_points=64,
                                      threshold_delta=4.0, seed=7)
        assert np.isfinite(anchor_set.coverage_radius)
        assert 0.0 < anchor_set.coverage_radius < 1.0


# --------------------------------------------------------------------------- #
# Harvest tier
# --------------------------------------------------------------------------- #
try:
    import h5py  # noqa: F401
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

FORMATS = [
    pytest.param(".csv", id="csv"),
    pytest.param(".h5", id="hdf5",
                 marks=pytest.mark.skipif(not HAS_H5PY, reason="needs h5py")),
]


def make_anchor_set():
    """Four anchors in the corners of [0, 10]^2, coverage radius 0.1 (scaled)."""
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    anchors = np.array([
        [2.0, 2.0],   # anchor 0
        [8.0, 2.0],   # anchor 1
        [2.0, 8.0],   # anchor 2
        [8.0, 8.0],   # anchor 3
    ])
    return AnchorSet(anchors, bounds, coverage_radius=0.1)


class TestHarvest:

    @pytest.mark.parametrize("ext", FORMATS)
    def test_harvest_basics(self, tmp_path, ext):
        anchor_set = make_anchor_set()
        band_lo = -4.0
        rows = np.array([
            # Covers anchor 0 (scaled dist 0.05); a farther in-band sample
            # near it must lose to the closer one.
            [2.5, 2.0, -1.0],
            [2.1, 2.0, -3.0],
            # Exact distance tie for anchor 1 (same point evaluated twice):
            # higher logL must win.
            [8.0, 2.5, -2.0],
            [8.0, 2.5, -1.0],
            # In-band but outside the coverage radius of anchor 2 (scaled
            # dist 0.25): warm-start hint only.
            [2.0, 5.5, -2.0],
            # Out-of-band sample near anchor 3: ignored entirely.
            [8.0, 8.0, -100.0],
            # Non-finite logL: ignored.
            [8.0, 8.0, np.nan],
        ])
        path = str(tmp_path / f"samples{ext}")
        write_samples(rows, path)

        stats = harvest_existing_samples(
            anchor_set, [path], band_lo)

        assert stats['n_samples'] == 7
        assert stats['n_in_band'] == 5
        assert stats['n_covered'] == 2
        assert stats['n_with_warm_start'] == 3

        np.testing.assert_array_equal(anchor_set.covered,
                                      [True, True, False, False])
        # Anchor 0: nearest sample wins despite lower logL.
        np.testing.assert_allclose(anchor_set.rep_points[0], [2.1, 2.0])
        assert anchor_set.rep_logls[0] == -3.0
        # Anchor 1: distance tie broken by higher logL.
        np.testing.assert_allclose(anchor_set.rep_points[1], [8.0, 2.5])
        assert anchor_set.rep_logls[1] == -1.0
        # Anchor 2: warm-start hint beyond the coverage radius.
        np.testing.assert_allclose(anchor_set.rep_points[2], [2.0, 5.5])
        assert anchor_set.rep_dists[2] == pytest.approx(0.25)
        # Anchor 3: nothing in band nearby.
        assert not np.isfinite(anchor_set.rep_dists[3])

    def test_widened_band_reaches_deeper(self, tmp_path):
        # A deeper band_lo (a larger stage roi_threshold) lets shell
        # samples through that a shallower band rejects.
        rows = np.array([
            [2.0, 2.0, -1.0],    # in-band for either threshold
            [2.05, 2.0, -10.0],  # only in-band once the band reaches -25
        ])
        path = str(tmp_path / "samples.csv")
        write_samples(rows, path)

        shallow = harvest_existing_samples(make_anchor_set(), [path], band_lo=-4.0)
        assert shallow['n_in_band'] == 1

        deep_set = make_anchor_set()
        deep = harvest_existing_samples(deep_set, [path], band_lo=-25.0)
        assert deep['n_in_band'] == 2

    def test_later_file_can_improve(self, tmp_path):
        anchor_set = make_anchor_set()
        path_a = str(tmp_path / "a.csv")
        path_b = str(tmp_path / "b.csv")
        write_samples(np.array([[2.5, 2.0, -1.0]]), path_a)
        write_samples(np.array([[2.1, 2.0, -2.0]]), path_b)

        harvest_existing_samples(anchor_set, [path_a, path_b], -4.0)
        np.testing.assert_allclose(anchor_set.rep_points[0], [2.1, 2.0])

    def test_cross_file_distance_tie_prefers_higher_logl(self, tmp_path):
        path_a = str(tmp_path / "a.csv")
        path_b = str(tmp_path / "b.csv")
        write_samples(np.array([[2.1, 2.0, -2.0]]), path_a)
        write_samples(np.array([[2.1, 2.0, -1.0]]), path_b)

        for files in ([path_a, path_b], [path_b, path_a]):
            anchor_set = make_anchor_set()
            harvest_existing_samples(anchor_set, files, -4.0)
            assert anchor_set.rep_logls[0] == -1.0

    def test_small_chunks_match_single_pass(self, tmp_path):
        rng = np.random.default_rng(11)
        rows = np.column_stack([
            rng.uniform(0.0, 10.0, size=(200, 2)),
            rng.uniform(-8.0, 0.0, size=200),
        ])
        path = str(tmp_path / "samples.csv")
        write_samples(rows, path)

        one_pass = make_anchor_set()
        chunked = make_anchor_set()
        harvest_existing_samples(one_pass, [path], -4.0)
        harvest_existing_samples(chunked, [path], -4.0, chunk_size=7)
        np.testing.assert_array_equal(one_pass.rep_dists,
                                      chunked.rep_dists)
        np.testing.assert_array_equal(one_pass.rep_points,
                                      chunked.rep_points)

    def test_width_mismatch_raises(self, tmp_path):
        anchor_set = make_anchor_set()
        path = str(tmp_path / "samples.csv")
        write_samples(np.zeros((3, 5)), path)  # 4 dims + logL, anchors are 2D
        with pytest.raises(ConfigurationError, match="width"):
            harvest_existing_samples(anchor_set, [path], -4.0)

    def test_no_files_is_noop(self):
        anchor_set = make_anchor_set()
        stats = harvest_existing_samples(anchor_set, [], -4.0)
        assert stats['n_samples'] == 0
        assert not anchor_set.covered.any()


class TestResolveHarvestFiles:

    def test_combines_and_dedupes(self, tmp_path):
        own = tmp_path / "samples.csv"
        extra = tmp_path / "extra.csv"
        own.write_text("0.0,0.0,-1.0\n")
        extra.write_text("0.0,0.0,-1.0\n")
        cfg = normalize_volume_config(
            {'harvest_files': [str(extra), str(own)]}, roi_threshold=4.0)
        files = resolve_harvest_files(cfg, samples_output_file=str(own))
        assert files == [str(own), str(extra)]

    def test_missing_own_file_skipped(self, tmp_path):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        files = resolve_harvest_files(
            cfg, samples_output_file=str(tmp_path / "nope.csv"))
        assert files == []

    def test_missing_explicit_file_raises(self, tmp_path):
        cfg = normalize_volume_config(
            {'harvest_files': str(tmp_path / "nope.csv")}, roi_threshold=4.0)
        with pytest.raises(ConfigurationError, match="not found"):
            resolve_harvest_files(cfg)


# --------------------------------------------------------------------------- #
# ProfileProjector integration
# --------------------------------------------------------------------------- #
class TestSamplerConfigIntegration:

    def test_volume_sampling_stored_normalized(self, simple_2d_function,
                                               simple_bounds_2d,
                                               basic_projection_1d):
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_1d],
            roi_threshold=4.0,
            volume_sampling={'roi_threshold': 25.0, 'n_points': 50},
        )
        cfg = sampler.volume_sampling_config
        assert cfg['roi_threshold'] == 25.0
        assert cfg['n_points'] == 50

    def test_volume_roi_threshold_defaults_to_projection(
            self, simple_2d_function, simple_bounds_2d, basic_projection_1d):
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_1d],
            roi_threshold=4.0,
            volume_sampling={'n_points': 50},
        )
        # Unset stage roi_threshold inherits the projection's.
        assert sampler.volume_sampling_config['roi_threshold'] == 4.0

    def test_volume_sampling_default_off(self, simple_2d_function,
                                         simple_bounds_2d,
                                         basic_projection_1d):
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_1d],
        )
        assert sampler.volume_sampling_config is None

    def test_invalid_volume_sampling_raises(self, simple_2d_function,
                                            simple_bounds_2d,
                                            basic_projection_1d):
        with pytest.raises(ConfigurationError, match="roi_threshold"):
            ProfileProjector(
                target_func=simple_2d_function,
                bounds=simple_bounds_2d,
                projections=[basic_projection_1d],
                roi_threshold=30.0,
                volume_sampling={'roi_threshold': -5.0},
            )
