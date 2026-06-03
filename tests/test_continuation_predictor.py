"""Tests for the first-order continuation warm start (idea A1).

Covers ``ProfileProjector._jacobian_predicted_params`` (a direction-free local
linear fit of the profiled-optimum field) and the way ``ActivationJob``
consumes a continuation prediction.
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
    proj = {'dims': [0, 1], 'grid_points': [8, 8], 'patch_coarse_grid': False}
    s = ProfileProjector(
        target_func=_func_4d,
        bounds=bounds,
        projections=[proj],
        pop_per_grid_point=4,
    )
    assert s.profiled_dims == [2, 3]
    assert s.n_prof_dims == 2
    return s


def _add_cell(sampler, grid_idx, profiled_params, fitness=1.0):
    """Insert a minimal converged population state at a grid cell."""
    profiled_params = np.asarray(profiled_params, dtype=float)
    sampler.population[grid_idx] = {
        'profiled_params': np.array([profiled_params]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': 'converged',
    }


class TestContinuationPredictor:
    def test_default_enabled(self, sampler_4d):
        assert sampler_4d.continuation_enabled is True

    def test_toggle_off(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        proj = {'dims': [0, 1], 'grid_points': [5, 5]}
        s = ProfileProjector(
            target_func=_func_4d, bounds=bounds, projections=[proj],
            advanced_config={'continuation': {'enabled': False}},
        )
        assert s.continuation_enabled is False

    def test_single_neighbour_reduces_to_secant(self, sampler_4d):
        """With exactly one usable neighbour (the colinear "grandparent"), the
        Jacobian fit reproduces the old secant step 2*phi_source - phi_grand."""
        # grandparent (2, 2) is the only populated neighbour of source (3, 3);
        # predict target (4, 4).
        _add_cell(sampler_4d, (2, 2), [0.0, 0.0])
        source_params = np.array([1.0, 1.0])
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(4, 4), source_idx=(3, 3), source_params=source_params,
        )
        np.testing.assert_allclose(pred, [2.0, 2.0])

    def test_plane_fit_recovers_linear_field(self, sampler_4d):
        """When phi*(theta) is exactly linear over the source neighbourhood, the
        least-squares fit recovers it and the prediction is exact -- regardless
        of activation direction."""
        J_true = np.array([[0.5, -0.2],
                           [0.1, 0.3]])           # (n_prof, n_proj)
        source_idx = (4, 4)
        source_params = np.array([1.0, 1.0])
        # Populate all 8 neighbours of the source with the linear field.
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                offset = np.array([di, dj], dtype=float)
                _add_cell(sampler_4d, (4 + di, 4 + dj),
                          source_params + J_true @ offset)
        # Predict the (1, 1) neighbour (not populated).
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(5, 5), source_idx=source_idx, source_params=source_params,
        )
        expected = source_params + J_true @ np.array([1.0, 1.0])
        np.testing.assert_allclose(pred, expected, atol=1e-9)

    def test_direction_independent(self, sampler_4d):
        """Same source neighbourhood -> same prediction for a given target,
        no matter which neighbour is named as the activation source."""
        J_true = np.array([[0.4, 0.1], [-0.3, 0.2]])
        source_idx = (4, 4)
        source_params = np.array([0.5, -0.5])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                offset = np.array([di, dj], dtype=float)
                _add_cell(sampler_4d, (4 + di, 4 + dj),
                          source_params + J_true @ offset)
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(5, 4), source_idx=source_idx, source_params=source_params,
        )
        expected = source_params + J_true @ np.array([1.0, 0.0])
        np.testing.assert_allclose(pred, expected, atol=1e-9)

    def test_prediction_clipped_to_bounds(self, sampler_4d):
        """A prediction past the profiled bounds is clipped, not returned raw."""
        _add_cell(sampler_4d, (2, 2), [0.0, 0.0])
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(4, 4), source_idx=(3, 3),
            source_params=np.array([4.0, 4.0]),   # raw secant would land at ~8
        )
        np.testing.assert_allclose(pred, [5.0, 5.0])

    def test_none_without_usable_neighbour(self, sampler_4d):
        # Source has no in-population neighbours.
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(4, 4), source_idx=(3, 3),
            source_params=np.array([1.0, 1.0]),
        )
        assert pred is None

    def test_none_on_degenerate_step(self, sampler_4d):
        _add_cell(sampler_4d, (2, 2), [0.0, 0.0])
        pred = sampler_4d._jacobian_predicted_params(
            target_idx=(3, 3), source_idx=(3, 3),
            source_params=np.array([1.0, 1.0]),
        )
        assert pred is None

    def test_none_without_source_params(self, sampler_4d):
        assert sampler_4d._jacobian_predicted_params((4, 4), (3, 3), None) is None


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
