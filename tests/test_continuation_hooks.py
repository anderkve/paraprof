"""
Tests for the continuation-style hooks added to ParaProf:

1. Secant predictor warm-start (linear/secant extrapolation of ψ* from
   two already-converged neighbors along a grid axis), and
2. Online basin-switch detection (immediate post-convergence patching
   test against the best neighbor instead of deferring to the post-hoc
   patching waves).

These tests focus on the *mechanism* rather than full MPI scans, which
are covered by the integration tests. They exercise:

* the sampler-side configuration plumbing,
* :py:meth:`ProfileProjector._build_secant_predictor_candidates`,
* the multi-candidate predictor logic in :class:`LBFGSBJob`,
* the online basin-switch trigger in :meth:`LBFGSBJob.on_finish`.
"""
import collections
import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.jobs.lbfgsb_job import LBFGSBJob
from paraprof.jobs.patching_test_job import PatchingTestJob


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sampler(advanced_config=None, projections=None):
    """Build a 4D sampler with a smooth target and a tiny 2D projection.

    The target makes ψ* = (0, 0) regardless of the projection coordinates,
    so the secant predictor's extrapolations are exact — which lets us
    assert the predictor candidates *equal* the true optimum.
    """
    def target(p):
        return -float(np.sum(p ** 2))

    bounds = np.array([[-5.0, 5.0]] * 4)
    if projections is None:
        projections = [{'dims': [0, 1], 'grid_points': [5, 5],
                        'patch_coarse_grid': False}]
    sampler = ProfileProjector(
        target_func=target,
        bounds=bounds,
        projections=projections,
        pop_per_grid_point=2,
        advanced_config=advanced_config,
    )
    return sampler


def _seed_converged_cell(sampler, grid_idx, profiled_params, fitness,
                        status='optimized'):
    """Insert a fake converged population entry at ``grid_idx``.

    Mirrors the structure produced by ActivationJob / LBFGSBJob.on_finish so
    that downstream code (predictor builder, basin-switch trigger) sees a
    realistic state.
    """
    profiled_params = np.asarray(profiled_params, dtype=float)
    pop_size = max(sampler.pop_per_grid_point, 1)
    fitnesses = np.full(pop_size, -np.inf)
    profiled = np.zeros((pop_size, sampler.n_prof_dims))
    profiled[0] = profiled_params
    fitnesses[0] = fitness
    sampler.population[grid_idx] = {
        'profiled_params': profiled,
        'fitnesses': fitnesses,
        'best_fitness': fitness,
        'status': status,
        'improvement_history': collections.deque(
            maxlen=sampler.convergence_window
        ),
        'last_update_gen': 0,
        # A non-None optimizer_state is required for the neighbor-test
        # branch in LBFGSBJob.start() to seed (s, y) from this cell.
        'optimizer_state': {'s': [], 'y': []},
    }


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------

class TestContinuationConfig:

    def test_defaults(self):
        """Defaults: secant predictor ON (consistent quality win),
        basin switch OFF (mixed: helps on multimodal, can hurt on smooth
        narrow-valley targets; opt-in)."""
        sampler = _make_sampler()
        assert sampler.secant_predictor_warm_start is True
        assert sampler.online_basin_switch is False

    def test_advanced_config_disables_hooks(self):
        sampler = _make_sampler(advanced_config={
            'continuation': {
                'secant_predictor_warm_start': False,
                'online_basin_switch': False,
            }
        })
        assert sampler.secant_predictor_warm_start is False
        assert sampler.online_basin_switch is False

    def test_advanced_config_partial_override(self):
        """Disabling only one hook leaves the other at its default."""
        sampler = _make_sampler(advanced_config={
            'continuation': {'secant_predictor_warm_start': False}
        })
        assert sampler.secant_predictor_warm_start is False
        # secant flipped off, basin retains its (off) default.
        assert sampler.online_basin_switch is False

        # And the inverse: turning on basin keeps secant at its (on) default.
        sampler2 = _make_sampler(advanced_config={
            'continuation': {'online_basin_switch': True}
        })
        assert sampler2.secant_predictor_warm_start is True
        assert sampler2.online_basin_switch is True

    def test_counters_reset_between_projections(self):
        projections = [
            {'dims': [0, 1], 'grid_points': [5, 5], 'patch_coarse_grid': False},
            {'dims': [0, 2], 'grid_points': [5, 5], 'patch_coarse_grid': False},
        ]
        sampler = _make_sampler(projections=projections)
        sampler._secant_predictor_candidates_tested = 7
        sampler._online_basin_switch_tests = 11
        sampler._reset_for_new_projection(projections[1])
        assert sampler._secant_predictor_candidates_tested == 0
        assert sampler._online_basin_switch_tests == 0


# ---------------------------------------------------------------------------
# Predictor candidate builder
# ---------------------------------------------------------------------------

class TestSecantPredictorCandidates:

    def test_no_converged_neighbors_returns_empty(self):
        sampler = _make_sampler()
        cands = sampler._build_secant_predictor_candidates(
            (2, 2), base_params=np.zeros(sampler.n_prof_dims)
        )
        assert cands == []

    def test_backward_secant_extrapolation(self):
        """ψ(G-1) = a, ψ(G-2) = b ⇒ candidate = 2a - b."""
        sampler = _make_sampler()
        nprof = sampler.n_prof_dims
        # We test the cell at axis-0 index 2; neighbors at index 1 and 0.
        _seed_converged_cell(sampler, (0, 2),
                             profiled_params=np.full(nprof, 4.0),
                             fitness=-1.0)
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 2.0),
                             fitness=-0.5)
        cands = sampler._build_secant_predictor_candidates(
            (2, 2), base_params=np.full(nprof, 99.0),
        )
        # Expect at least one candidate; the backward secant along axis 0
        # gives 2*2 - 4 = 0.
        assert len(cands) >= 1
        labels = [c[0] for c in cands]
        assert 'secant-back-axis0' in labels
        back = dict(cands)['secant-back-axis0']
        np.testing.assert_allclose(back, np.full(nprof, 0.0))

    def test_centered_interpolation(self):
        """ψ(G-1) = a, ψ(G+1) = b ⇒ candidate = (a+b)/2."""
        sampler = _make_sampler()
        nprof = sampler.n_prof_dims
        # Cell at (2, 2); neighbors at (1, 2) and (3, 2) along axis 0.
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, -1.0),
                             fitness=-0.5)
        _seed_converged_cell(sampler, (3, 2),
                             profiled_params=np.full(nprof, +1.0),
                             fitness=-0.4)
        cands = sampler._build_secant_predictor_candidates(
            (2, 2), base_params=np.full(nprof, 99.0),
        )
        labels = [c[0] for c in cands]
        assert 'interp-axis0' in labels
        interp = dict(cands)['interp-axis0']
        np.testing.assert_allclose(interp, np.zeros(nprof))

    def test_candidates_are_clipped_to_bounds(self):
        sampler = _make_sampler()
        nprof = sampler.n_prof_dims
        # Build neighbors that would extrapolate to +20 (way out of bounds).
        _seed_converged_cell(sampler, (0, 2),
                             profiled_params=np.full(nprof, -10.0),
                             fitness=-1.0)
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.5)
        cands = sampler._build_secant_predictor_candidates(
            (2, 2), base_params=np.full(nprof, 99.0),
        )
        # Every candidate must be inside the bounds.
        bounds = sampler.bounds[sampler.profiled_dims]
        for label, params in cands:
            assert np.all(params >= bounds[:, 0] - 1e-9)
            assert np.all(params <= bounds[:, 1] + 1e-9)

    def test_dedup_against_base(self):
        """A predictor identical to base_params is dropped as redundant."""
        sampler = _make_sampler()
        nprof = sampler.n_prof_dims
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.5)
        _seed_converged_cell(sampler, (3, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.4)
        # Centered interpolation yields 0; same as base_params -> deduped.
        cands = sampler._build_secant_predictor_candidates(
            (2, 2), base_params=np.zeros(nprof),
        )
        labels = [c[0] for c in cands]
        assert 'interp-axis0' not in labels


# ---------------------------------------------------------------------------
# LBFGSBJob: multi-candidate predictor test
# ---------------------------------------------------------------------------

class TestLBFGSBPredictorIntegration:

    def _make_lbfgsb_job(self, sampler, grid_idx, start_params, start_fitness):
        """Build a non-running LBFGSBJob anchored at ``grid_idx``.

        Returns a job that has *not* yet had ``start()`` called.
        """
        start_full = sampler._construct_params(grid_idx, start_params)
        return LBFGSBJob(
            job_id=0,
            job_type='LBFGSB',
            sampler=sampler,
            opt_dims=tuple(sampler.profiled_dims),
            start_params=start_params,
            grid_idx=grid_idx,
            start_params_full=start_full,
            seed_history=None,
            start_fitness=start_fitness,
        )

    def test_start_emits_one_task_when_predictor_disabled(self):
        sampler = _make_sampler(advanced_config={
            'continuation': {'secant_predictor_warm_start': False}
        })
        nprof = sampler.n_prof_dims
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.1)
        _seed_converged_cell(sampler, (0, 2),
                             profiled_params=np.full(nprof, 0.5),
                             fitness=-0.4)
        # Cell at (2, 2): own params far from optimum.
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 4.0),
            'fitnesses': np.array([-100.0]),
            'best_fitness': -100.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        job = self._make_lbfgsb_job(
            sampler, (2, 2),
            start_params=np.full(nprof, 4.0),
            start_fitness=-100.0,
        )
        tasks = job.start()
        # Predictor disabled: only the legacy best-neighbor candidate.
        assert len(tasks) == 1
        assert tasks[0]['context']['sub_type'] == 'LBFGS_NEIGHBOR_TEST'

    def test_start_emits_multiple_tasks_when_predictor_enabled(self):
        sampler = _make_sampler()  # defaults: predictor enabled
        nprof = sampler.n_prof_dims
        # Three neighbors along axis 0 give backward secant + interp.
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.1)
        _seed_converged_cell(sampler, (0, 2),
                             profiled_params=np.full(nprof, 0.5),
                             fitness=-0.4)
        _seed_converged_cell(sampler, (3, 2),
                             profiled_params=np.full(nprof, -0.5),
                             fitness=-0.3)
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 4.0),
            'fitnesses': np.array([-100.0]),
            'best_fitness': -100.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        job = self._make_lbfgsb_job(
            sampler, (2, 2),
            start_params=np.full(nprof, 4.0),
            start_fitness=-100.0,
        )
        tasks = job.start()
        # Best-neighbor + at least one predictor candidate.
        assert len(tasks) >= 2
        cand_indices = {t['context']['candidate_idx'] for t in tasks}
        assert cand_indices == set(range(len(tasks)))

    def test_predictor_candidate_wins_when_better(self):
        """If a predictor candidate evaluates higher than the best-neighbor
        candidate, it must become the L-BFGS-B starting point."""
        sampler = _make_sampler()
        nprof = sampler.n_prof_dims
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 0.0),
                             fitness=-0.1)
        _seed_converged_cell(sampler, (0, 2),
                             profiled_params=np.full(nprof, 0.5),
                             fitness=-0.4)
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 4.0),
            'fitnesses': np.array([-100.0]),
            'best_fitness': -100.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        job = self._make_lbfgsb_job(
            sampler, (2, 2),
            start_params=np.full(nprof, 4.0),
            start_fitness=-100.0,
        )
        tasks = job.start()
        # Fake worker results: predictor candidate (idx 1) wins.
        for t in tasks:
            cand_idx = t['context']['candidate_idx']
            fake_fitness = -0.5 if cand_idx == 0 else -0.05
            job.process_result({
                'context': t['context'],
                'target_val': fake_fitness,
                'params': t['params'],
            })
        # After all candidates processed, current_params should equal the
        # winner (the secant predictor at +nprof[0]). The best winner's
        # fitness is -0.05, which beats start_fitness=-100, so the winner
        # is used (not fallback).
        assert job.current_fitness == pytest.approx(-0.05)
        # And the win is recorded in the diagnostic counter.
        assert sampler._secant_predictor_candidates_tested == 1
        assert sampler._secant_predictor_candidates_won == 1


# ---------------------------------------------------------------------------
# Online basin-switch trigger
# ---------------------------------------------------------------------------

class TestOnlineBasinSwitch:

    # Basin switch is OFF by default; tests in this class enable it
    # explicitly except where verifying the disabled path.
    BASIN_ON = {'continuation': {'online_basin_switch': True}}

    def _run_on_finish(self, sampler, finished_grid_idx,
                      final_params, final_fitness, job_type='LBFGSB',
                      next_job_id=1000):
        """Construct an LBFGSBJob, mark it converged, run on_finish()."""
        nprof = sampler.n_prof_dims
        start_params = np.zeros(nprof)
        start_full = sampler._construct_params(finished_grid_idx, start_params)
        job = LBFGSBJob(
            job_id=999,
            job_type=job_type,
            sampler=sampler,
            opt_dims=tuple(sampler.profiled_dims),
            start_params=start_params,
            grid_idx=finished_grid_idx,
            start_params_full=start_full,
            seed_history=None,
            start_fitness=-100.0,
        )
        job.success = True
        job._is_finished = True
        job.current_params = np.asarray(final_params, dtype=float)
        job.current_fitness = float(final_fitness)
        return job.on_finish(next_job_id)

    def test_no_spawn_when_no_better_neighbor(self):
        sampler = _make_sampler(advanced_config=self.BASIN_ON)
        nprof = sampler.n_prof_dims
        sampler.population[(2, 2)] = {
            'profiled_params': np.zeros((1, nprof)),
            'fitnesses': np.array([-0.1]),
            'best_fitness': -0.1,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        # Neighbors all have *worse* fitness.
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.full(nprof, 1.0),
                             fitness=-1.0)
        _seed_converged_cell(sampler, (3, 2),
                             profiled_params=np.full(nprof, 1.0),
                             fitness=-2.0)
        spawn = self._run_on_finish(sampler, (2, 2),
                                    final_params=np.zeros(nprof),
                                    final_fitness=-0.1)
        assert spawn is None

    def test_spawn_when_neighbor_dominates(self):
        sampler = _make_sampler(advanced_config=self.BASIN_ON)
        nprof = sampler.n_prof_dims
        # Cell finishes with mediocre fitness.
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 2.0),
            'fitnesses': np.array([-5.0]),
            'best_fitness': -5.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        # Best neighbor at (1, 2) has much better fitness with different ψ*.
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.zeros(nprof),
                             fitness=-0.1)
        spawn = self._run_on_finish(sampler, (2, 2),
                                    final_params=np.full(nprof, 2.0),
                                    final_fitness=-5.0)
        assert spawn is not None
        job, _ = spawn
        assert isinstance(job, PatchingTestJob)
        # The test job targets the just-finished cell with the neighbor's ψ*.
        assert job.grid_idx == (2, 2)
        np.testing.assert_allclose(job.test_profiled_params, np.zeros(nprof))
        # The sentinel wave_number marks it as online (not part of any wave).
        assert job.wave_number == -1
        # Sampler-level counter incremented.
        assert sampler._online_basin_switch_tests == 1

    def test_no_spawn_when_feature_disabled(self):
        sampler = _make_sampler(advanced_config={
            'continuation': {'online_basin_switch': False}
        })
        nprof = sampler.n_prof_dims
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 2.0),
            'fitnesses': np.array([-5.0]),
            'best_fitness': -5.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.zeros(nprof),
                             fitness=-0.1)
        spawn = self._run_on_finish(sampler, (2, 2),
                                    final_params=np.full(nprof, 2.0),
                                    final_fitness=-5.0)
        assert spawn is None

    def test_no_spawn_from_patching_lbfgsb(self):
        """PATCHING_LBFGSB jobs must NOT trigger another online check
        (prevents recursion through the patching infrastructure)."""
        sampler = _make_sampler(advanced_config=self.BASIN_ON)
        nprof = sampler.n_prof_dims
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 2.0),
            'fitnesses': np.array([-5.0]),
            'best_fitness': -5.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.zeros(nprof),
                             fitness=-0.1)
        spawn = self._run_on_finish(sampler, (2, 2),
                                    final_params=np.full(nprof, 2.0),
                                    final_fitness=-5.0,
                                    job_type='PATCHING_LBFGSB')
        assert spawn is None

    def test_basin_switch_counter_on_improvement(self):
        """When the spawned PatchingTestJob finds improvement, the
        sampler-level improvement counter must increment."""
        sampler = _make_sampler(advanced_config=self.BASIN_ON)
        nprof = sampler.n_prof_dims
        sampler.population[(2, 2)] = {
            'profiled_params': np.full((1, nprof), 2.0),
            'fitnesses': np.array([-5.0]),
            'best_fitness': -5.0,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': None,
        }
        _seed_converged_cell(sampler, (1, 2),
                             profiled_params=np.zeros(nprof),
                             fitness=-0.1)
        spawn = self._run_on_finish(sampler, (2, 2),
                                    final_params=np.full(nprof, 2.0),
                                    final_fitness=-5.0)
        assert spawn is not None
        patch_job, _ = spawn
        # Drive the patch job through start + process_result with a winning
        # test fitness. The target at (2, 2) with profiled = (0,..,0) is
        # better than the current best, so will_update will be True.
        tasks = patch_job.start()
        assert len(tasks) == 1
        # Simulate the worker reporting a strong improvement.
        patch_job.process_result({
            'context': tasks[0]['context'],
            'target_val': -0.01,
            'params': tasks[0]['params'],
        })
        assert patch_job.will_update is True
        # on_finish should bump the diagnostic counter.
        patch_job.on_finish(2000)
        assert sampler._online_basin_switch_improvements == 1
