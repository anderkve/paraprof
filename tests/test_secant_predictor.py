"""Tests for the first-order (secant) continuation warm start (idea A1).

Covers ``ProfileProjector._secant_predicted_params`` and the way
``ActivationJob`` consumes a continuation prediction.
"""
import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.jobs.activation_job import ActivationJob


def _func_4d(p):
    return -float(np.sum(np.asarray(p) ** 2))


@pytest.fixture
def sampler_4d():
    """4D sampler with a 2D projection on dims (0, 1); profiled dims (2, 3)."""
    bounds = np.array([[-5.0, 5.0]] * 4)
    proj = {'dims': [0, 1], 'grid_points': [5, 5], 'patch_coarse_grid': False}
    s = ProfileProjector(
        target_func=_func_4d,
        bounds=bounds,
        projections=[proj],
        pop_per_grid_point=4,
    )
    assert s.profiled_dims == [2, 3]
    assert s.n_prof_dims == 2
    return s


def _add_cell(sampler, grid_idx, profiled_params, fitness):
    """Insert a minimal converged population state at a grid cell."""
    profiled_params = np.asarray(profiled_params, dtype=float)
    sampler.population[grid_idx] = {
        'profiled_params': np.array([profiled_params]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': 'converged',
    }


class TestSecantPredictor:
    def test_default_enabled(self, sampler_4d):
        assert sampler_4d.secant_predictor is True

    def test_toggle_off(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        proj = {'dims': [0, 1], 'grid_points': [5, 5]}
        s = ProfileProjector(
            target_func=_func_4d, bounds=bounds, projections=[proj],
            advanced_config={'continuation': {'secant_predictor': False}},
        )
        assert s.secant_predictor is False

    def test_extrapolation(self, sampler_4d):
        """phi_target ~= 2*phi_source - phi_grandparent, along a colinear chain."""
        # grandparent (0, 0) -> source (1, 1) -> target (2, 2)
        _add_cell(sampler_4d, (0, 0), [0.0, 0.0], fitness=1.0)
        source_params = np.array([1.0, 1.0])
        pred = sampler_4d._secant_predicted_params(
            target_idx=(2, 2), source_idx=(1, 1), source_params=source_params,
        )
        np.testing.assert_allclose(pred, [2.0, 2.0])

    def test_extrapolation_clipped_to_bounds(self, sampler_4d):
        """A prediction past the profiled bounds is clipped, not returned raw."""
        _add_cell(sampler_4d, (0, 0), [0.0, 0.0], fitness=1.0)
        # source near the +5 bound; raw secant would land at ~8.0
        pred = sampler_4d._secant_predicted_params(
            target_idx=(2, 2), source_idx=(1, 1),
            source_params=np.array([4.0, 4.0]),
        )
        np.testing.assert_allclose(pred, [5.0, 5.0])

    def test_none_when_grandparent_missing(self, sampler_4d):
        # No cell at the grandparent index (0, 0).
        pred = sampler_4d._secant_predicted_params(
            target_idx=(2, 2), source_idx=(1, 1),
            source_params=np.array([1.0, 1.0]),
        )
        assert pred is None

    def test_none_when_grandparent_out_of_bounds(self, sampler_4d):
        # target (0,0), source (1,1) -> grandparent (2,2) is fine; flip it so
        # the grandparent falls off the low edge: target (2,2), source (1,1)
        # was valid, so use target (1,1), source (0,0) -> grandparent (-1,-1).
        pred = sampler_4d._secant_predicted_params(
            target_idx=(1, 1), source_idx=(0, 0),
            source_params=np.array([1.0, 1.0]),
        )
        assert pred is None

    def test_none_on_degenerate_step(self, sampler_4d):
        _add_cell(sampler_4d, (1, 1), [0.0, 0.0], fitness=1.0)
        pred = sampler_4d._secant_predicted_params(
            target_idx=(1, 1), source_idx=(1, 1),
            source_params=np.array([1.0, 1.0]),
        )
        assert pred is None

    def test_none_without_source_params(self, sampler_4d):
        assert sampler_4d._secant_predicted_params((2, 2), (1, 1), None) is None


class TestActivationJobSeeding:
    def test_prediction_and_warm_start_both_seeded(self, sampler_4d):
        """With a prediction, both the first-order and zeroth-order seeds appear
        un-perturbed among the activation population's profiled params."""
        warm = np.array([1.0, 1.0])
        pred = np.array([2.0, 2.0])
        job = ActivationJob(
            job_id=0, sampler=sampler_4d, grid_idx=(2, 2),
            warm_start_params=warm, predicted_params=pred,
        )
        seeds = job.all_profiled_params
        assert any(np.allclose(s, pred) for s in seeds), "prediction not seeded"
        assert any(np.allclose(s, warm) for s in seeds), "warm start not seeded"
        # Population size is preserved exactly.
        assert len(seeds) == sampler_4d.pop_per_grid_point

    def test_no_prediction_matches_legacy(self, sampler_4d):
        """Without a prediction the first neighbour seed is warm_start_params."""
        warm = np.array([1.0, 1.0])
        job = ActivationJob(
            job_id=0, sampler=sampler_4d, grid_idx=(2, 2),
            warm_start_params=warm, predicted_params=None,
        )
        np.testing.assert_allclose(job.all_profiled_params[0], warm)
        assert len(job.all_profiled_params) == sampler_4d.pop_per_grid_point
