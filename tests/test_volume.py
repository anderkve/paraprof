"""Tests for the volume-sampling building blocks (paraprof.volume)."""

import json

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import ConfigurationError
from paraprof.volume import (
    VOLUME_CONFIG_DEFAULTS,
    ProjectionEnvelope,
    assign_levels,
    default_summary_file,
    draw_envelope_seeds,
    draw_stretch,
    lnl_histogram,
    normalize_volume_config,
    umbrella_logpi,
    volume_band,
    warm_start_positions,
    write_volume_output,
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
# Config normalization and band
# --------------------------------------------------------------------------- #
class TestNormalizeVolumeConfig:

    def test_defaults_and_roi_inheritance(self):
        user = {'n_walkers': 10}
        cfg = normalize_volume_config(user, roi_threshold=4.0)
        assert cfg == dict(VOLUME_CONFIG_DEFAULTS, roi_threshold=4.0, n_walkers=10)
        assert user == {'n_walkers': 10}      # caller's dict not mutated

    def test_unknown_key(self):
        with pytest.raises(ConfigurationError, match="unknown keys.*'mode'"):
            normalize_volume_config({'mode': 'roi'}, roi_threshold=4.0)

    @pytest.mark.parametrize("config,match", [
        ({'roi_threshold': -5.0}, "roi_threshold"),
        ({'n_walkers': 1}, "n_walkers"),
        ({'n_steps': 0}, "n_steps"),
        ({'sigma_frac': np.inf}, "sigma_frac"),
        ({'partner_level_window': -1.0}, "partner_level_window"),
        ({'warm_start': 1}, "warm_start"),
        ({'output_file': ''}, "output_file"),
    ])
    def test_bad_values_rejected(self, config, match):
        with pytest.raises(ConfigurationError, match=match):
            normalize_volume_config(config, roi_threshold=4.0)

    def test_none_allowed(self):
        cfg = normalize_volume_config(
            {'eval_budget': None, 'partner_level_window': None,
             'summary_file': None}, roi_threshold=4.0)
        assert cfg['eval_budget'] is None
        assert cfg['partner_level_window'] is None

    def test_band_one_sided(self):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        assert volume_band(cfg, global_max=10.0) == (6.0, 4.0)
        wide = normalize_volume_config({'roi_threshold': 25.0}, roi_threshold=4.0)
        assert volume_band(wide, global_max=10.0) == (-15.0, 25.0)


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
            [0.0, 0.0], [0.0, 3.0],   # pass both
            [4.0, 3.0], [0.0, -3.0],  # fail x (never-activated), fail y
        ])
        np.testing.assert_array_equal(
            env.test(points, threshold_delta=4.0), [True, True, False, False])

    def test_threshold_and_widening(self):
        axis = np.linspace(-5.0, 5.0, 11)
        cells = {(i,): (0.0 if x == 0.0 else -10.0) for i, x in enumerate(axis)}
        env = ProjectionEnvelope([make_record([0], [axis], cells)],
                                 global_max=0.0, n_dims=1)
        assert env.test([[0.0]], threshold_delta=4.0)[0]
        assert not env.test([[2.0]], threshold_delta=4.0)[0]
        assert env.test([[2.0]], threshold_delta=25.0)[0]   # widened ROI

    def test_final_global_max_recomputes_membership(self):
        axis = np.linspace(-5.0, 5.0, 101)
        cells = {(i,): 0.0 for i, x in enumerate(axis) if abs(x) <= 2.0}
        env = ProjectionEnvelope([make_record([0], [axis], cells)],
                                 global_max=10.0, n_dims=1)
        assert not env.test([[0.0]], threshold_delta=4.0)[0]

    def test_covers_full_space(self):
        axis = np.linspace(-5.0, 5.0, 6)
        full = make_record([0, 1], [axis, axis], {(0, 0): 0.0})
        partial = make_record([0], [axis], {(0,): 0.0})
        assert ProjectionEnvelope([partial, full], 0.0, 2).covers_full_space
        assert not ProjectionEnvelope([partial], 0.0, 2).covers_full_space

    def test_from_projection_results_prefers_refined(self):
        coarse = make_record([0], [np.linspace(-5, 5, 6)], {(0,): 0.0})
        refined = make_record([0], [np.linspace(-5, 5, 11)],
                              {(i,): 0.0 for i in range(11)})
        env = ProjectionEnvelope.from_projection_results(
            [{'coarse_solution': coarse, 'refined_solution': refined},
             {'coarse_solution': coarse, 'refined_solution': None}], 0.0, 1)
        assert env.n_projections == 2
        assert len(env._records[0]['axes'][0]) == 11   # refined preferred
        with pytest.raises(ValueError, match="no exported grid solution"):
            ProjectionEnvelope.from_projection_results(
                [{'coarse_solution': None, 'refined_solution': None}], 0.0, 1)

    def test_cell_mapping_matches_sampler(self, simple_2d_function,
                                          simple_bounds_2d):
        sampler = ProfileProjector(
            target_func=simple_2d_function, bounds=simple_bounds_2d,
            projections=[{'dims': [0], 'grid_points': [13]}])
        rec = make_record(sampler.projection_dims, sampler.grid_axes, {})
        env = ProjectionEnvelope([rec], global_max=0.0, n_dims=2)
        rng = np.random.default_rng(7)
        points = rng.uniform(-5.0, 5.0, size=(200, 2))
        np.testing.assert_array_equal(
            env.cell_indices(0, points)[0],
            [sampler._get_grid_indices_from_point(p)[0] for p in points])


# --------------------------------------------------------------------------- #
# Ensemble primitives
# --------------------------------------------------------------------------- #
class TestEnsemblePrimitives:

    def test_assign_levels_spans_band(self):
        levels = assign_levels(10, global_max=0.0, roi_threshold=4.0)
        assert len(levels) == 10
        assert levels.max() < 0.0 and levels.min() > -4.0      # interior centres
        assert np.all(np.diff(levels) < 0)                     # descending
        np.testing.assert_allclose(np.diff(levels), -0.4)      # even spacing

    def test_umbrella_logpi(self):
        assert umbrella_logpi(-2.0, -2.0, 0.5) == 0.0          # at the level
        assert umbrella_logpi(-2.5, -2.0, 0.5) == pytest.approx(-0.5)
        assert umbrella_logpi(np.inf, -2.0, 0.5) == -np.inf
        assert umbrella_logpi(-np.inf, -2.0, 0.5) == -np.inf
        # Vectorised, with a non-finite entry.
        out = umbrella_logpi(np.array([-2.0, np.nan]), -2.0, 0.5)
        np.testing.assert_array_equal(out, [0.0, -np.inf])

    def test_draw_stretch_range(self):
        rng = np.random.default_rng(0)
        z = draw_stretch(rng, 5000, a=2.0)
        assert z.min() >= 0.5 - 1e-9 and z.max() <= 2.0 + 1e-9
        # Density ∝ 1/sqrt(z) decreases with z, but the upper interval [1, a]
        # is wider than [1/a, 1], so a bit more mass lands above 1.
        assert np.count_nonzero(z > 1.0) > np.count_nonzero(z < 1.0)

    def test_draw_envelope_seeds_inside_envelope(self):
        env = band_envelope_1d(0, -1.0, 1.0, n_dims=2)
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        seeds = draw_envelope_seeds(env, bounds, 100, threshold_delta=4.0, seed=1)
        assert len(seeds) == 100
        assert env.test(seeds, threshold_delta=4.0).all()
        assert np.all(np.abs(seeds[:, 0]) <= 1.0 + 0.05)       # constrained dim
        assert seeds[:, 1].min() < -3.0 and seeds[:, 1].max() > 3.0  # free dim

    def test_warm_start_near_level_with_fallback(self):
        levels = np.array([0.0, -2.0, -4.0])
        scan_params = np.array([[1.0, 1.0], [2.0, 2.0]])
        scan_logls = np.array([0.0, -2.0])                     # near levels 0, -2
        fallback = np.array([[9.0, 9.0], [8.0, 8.0]])
        rng = np.random.default_rng(0)
        pos = warm_start_positions(levels, 0.5, scan_params, scan_logls,
                                   fallback, rng)
        np.testing.assert_allclose(pos[0], [1.0, 1.0])         # level 0
        np.testing.assert_allclose(pos[1], [2.0, 2.0])         # level -2
        np.testing.assert_allclose(pos[2], [9.0, 9.0])         # no scan -> fallback

    def test_lnl_histogram(self):
        h = lnl_histogram(np.array([-3.5, -0.5, -0.4]), -4.0, 0.0, n_bins=4)
        assert h['bin_edges'] == [-4.0, -3.0, -2.0, -1.0, 0.0]
        assert h['counts'] == [1, 0, 0, 2]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _result(samples):
    return {
        'samples': samples, 'band_lo_final': -4.0, 'global_max': 0.0,
        'stats': {'n_walkers': 3, 'n_in_band': len(samples),
                  'lnl_histogram': lnl_histogram(
                      samples[:, -1] if len(samples) else np.empty(0),
                      -4.0, 0.0)},
    }


class TestWriteOutput:

    def test_round_trip(self, tmp_path):
        from paraprof import read_samples
        samples = np.array([[1.0, 2.0, -1.0], [3.0, 4.0, -3.0]])
        cfg = normalize_volume_config(
            {'output_file': str(tmp_path / "vol.csv")}, roi_threshold=4.0)
        out_path, summary_path = write_volume_output(_result(samples), cfg)

        np.testing.assert_allclose(read_samples(out_path), samples)
        assert summary_path == default_summary_file(cfg['output_file'])
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['n_samples'] == 2
        assert summary['band_lo_final'] == -4.0

    def test_empty_writes_summary_only(self, tmp_path):
        cfg = normalize_volume_config(
            {'output_file': str(tmp_path / "vol.csv")}, roi_threshold=4.0)
        out_path, summary_path = write_volume_output(
            _result(np.empty((0, 3))), cfg)
        assert out_path is None
        assert not (tmp_path / "vol.csv").exists()
        with open(summary_path) as f:
            assert json.load(f)['n_samples'] == 0


# --------------------------------------------------------------------------- #
# ProfileProjector integration
# --------------------------------------------------------------------------- #
def test_volume_sampling_config_stored_and_default_off(
        simple_2d_function, simple_bounds_2d, basic_projection_1d):
    sampler = ProfileProjector(
        target_func=simple_2d_function, bounds=simple_bounds_2d,
        projections=[basic_projection_1d], roi_threshold=4.0,
        volume_sampling={'roi_threshold': 25.0, 'n_walkers': 50})
    assert sampler.volume_sampling_config['roi_threshold'] == 25.0
    assert sampler.volume_sampling_config['n_walkers'] == 50

    off = ProfileProjector(
        target_func=simple_2d_function, bounds=simple_bounds_2d,
        projections=[basic_projection_1d])
    assert off.volume_sampling_config is None
