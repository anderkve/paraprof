"""
Unit tests for the suspect-cell recheck detector and seed-gathering machinery.

These tests bypass the MPI master/worker loop and exercise the sampler-level
helpers directly. The integration with the master state machine is covered
indirectly by the end-to-end Himmelblau test in test_integration.py.
"""
import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.jobs.suspect_recheck_job import SuspectRecheckJob


def _make_sampler(grid_n=10, n_prof_dims=2):
    """Build a ProfileProjector with a 1-D projection over a smooth 3-D quadratic."""
    n_dims = 1 + n_prof_dims

    def target(p):
        return -float(np.sum(p ** 2))

    bounds = np.array([[-5.0, 5.0]] * n_dims)
    projection = {'dims': [0], 'grid_points': [grid_n]}
    sampler = ProfileProjector(
        target_func=target,
        bounds=bounds,
        projections=[projection],
        pop_per_grid_point=2,
    )
    return sampler


def _set_cell(sampler, idx, profiled_params, fitness):
    """Manually plant a converged population entry at grid index idx."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    profiled_params = np.asarray(profiled_params, dtype=float)
    sampler.population[idx] = {
        'profiled_params': np.array([profiled_params]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': 'optimized',
        'improvement_history': [],
        'last_update_gen': 0,
        'optimizer_state': None,
    }
    if fitness > sampler.global_max_target_val:
        sampler.global_max_target_val = float(fitness)


class TestSuspectDetection:
    def test_smooth_surface_yields_no_suspects(self):
        """A perfectly smooth profiled-params surface should flag nobody."""
        sampler = _make_sampler(grid_n=10)
        # Plant cells with smoothly-varying profiled params and matching logL.
        for i in range(sampler.grid_shape[0]):
            p = np.array([0.1 * i, -0.1 * i])
            _set_cell(sampler, (i,), p, fitness=-float(np.sum(p ** 2)))

        suspects = sampler._find_suspect_cells(
            wave_number=0, updated_points_last_wave=None
        )
        # Defensive: cells inside ROI only; the smooth case must not flag anyone
        # with both signals at default thresholds.
        assert suspects == []

    def test_param_discontinuity_strip_is_flagged(self):
        """A strip of cells with discontinuous profiled params and lower logL
        should be flagged (at least at the boundary of the strip)."""
        sampler = _make_sampler(grid_n=12)

        # Background: smooth, params near (0.1*i, -0.1*i), fitness near 0.
        for i in range(sampler.grid_shape[0]):
            p = np.array([0.1 * i, -0.1 * i])
            _set_cell(sampler, (i,), p, fitness=-float(np.sum(p ** 2)))

        # Plant a 3-cell strip in the middle with profiled params far away
        # and substantially lower fitness.
        strip = [(5,), (6,), (7,)]
        for idx in strip:
            _set_cell(sampler, idx, np.array([3.5, 3.5]), fitness=-25.0)

        # Make sure global_max stays at one of the smooth cells.
        # Use a generous ROI so the strip stays inside it for detection.
        sampler.roi_threshold = 100.0

        suspects = sampler._find_suspect_cells(
            wave_number=0, updated_points_last_wave=None
        )
        # At least one strip cell must be flagged — the strip's boundary cells
        # have the largest profiled-param distance to the neighbour median.
        # Adjacent smooth cells (e.g. (4,) and (8,)) may also be flagged because
        # their neighbour median is pulled by the strip — that's fine, the
        # recheck just finds no improvement there.
        assert any(s in strip for s in suspects), suspects

    def test_wave1_propagates_from_updated_points(self):
        """Wave >=1 returns in-population neighbours of last wave's winners."""
        sampler = _make_sampler(grid_n=10)
        for i in range(sampler.grid_shape[0]):
            _set_cell(sampler, (i,), np.zeros(2), fitness=0.0)

        suspects = sampler._find_suspect_cells(
            wave_number=1, updated_points_last_wave=[(5,)]
        )
        # Cells 4 and 6 should be picked up as neighbours of cell 5.
        assert set(suspects) == {(4,), (6,)}


class TestSeedGathering:
    def test_seed_dedup_and_safety_baseline(self):
        """Seeds include own params, drop duplicates, and pull from non-suspects."""
        sampler = _make_sampler(grid_n=10)
        # Smooth background with own params at (1.0, 1.0)
        for i in range(sampler.grid_shape[0]):
            _set_cell(sampler, (i,), np.array([1.0, 1.0]), fitness=-1.0)
        # Make cell 5 the "suspect" with very different params
        _set_cell(sampler, (5,), np.array([3.0, -2.0]), fitness=-15.0)

        seeds = sampler._gather_suspect_seeds((5,), suspect_set={(5,)})

        # Own params are present (first seed) plus at least one non-suspect
        # neighbour's params.
        assert len(seeds) >= 2
        # First seed is the cell's own current params.
        np.testing.assert_allclose(seeds[0], [3.0, -2.0])
        # At least one seed should match the smooth background.
        bg_present = any(np.allclose(s, [1.0, 1.0]) for s in seeds)
        assert bg_present

class TestSuspectJobLifecycle:
    def test_job_spawns_lbfgsb_when_seed_beats_threshold(self):
        sampler = _make_sampler(grid_n=10)
        _set_cell(sampler, (5,), np.array([3.0, -2.0]), fitness=-15.0)

        # Two seeds: a marginally better one (within polish_threshold) and a
        # clearly-better one. The job should pick the clear winner.
        seeds = [
            np.array([3.0, -1.99]),
            np.array([0.0, 0.0]),
        ]
        job = SuspectRecheckJob(
            job_id=0, sampler=sampler, grid_idx=(5,),
            candidate_seeds=seeds, wave_number=0,
        )
        tasks = job.start()
        assert len(tasks) == 2

        # Feed worker results manually.
        job.process_result({
            'context': tasks[0]['context'],
            'target_val': -14.5,
            'params': tasks[0]['params'],
        })
        assert not job.is_finished()
        job.process_result({
            'context': tasks[1]['context'],
            'target_val': -2.0,
            'params': tasks[1]['params'],
        })
        assert job.is_finished()
        assert job.will_update

        spawn = job.on_finish(next_job_id=1)
        assert spawn is not None
        new_job, next_id = spawn
        assert new_job.type == 'SUSPECT_RECHECK_LBFGSB'
        assert next_id == 2
        # The L-BFGS-B job should start from the clearly-better seed.
        np.testing.assert_allclose(new_job.start_params_partial, [0.0, 0.0])

    def test_job_skips_lbfgsb_when_no_seed_beats_threshold(self):
        sampler = _make_sampler(grid_n=10)
        _set_cell(sampler, (5,), np.array([0.0, 0.0]), fitness=-1.0)

        seeds = [np.array([0.05, 0.05]), np.array([-0.05, -0.05])]
        job = SuspectRecheckJob(
            job_id=0, sampler=sampler, grid_idx=(5,),
            candidate_seeds=seeds, wave_number=0,
        )
        tasks = job.start()
        for t in tasks:
            job.process_result({
                'context': t['context'],
                'target_val': -1.0,  # no improvement
                'params': t['params'],
            })
        assert job.is_finished()
        assert not job.will_update
        assert job.on_finish(next_job_id=1) is None


class TestConfigPlumbing:
    def test_advanced_config_overrides(self):
        def target(p):
            return -float(np.sum(p ** 2))

        sampler = ProfileProjector(
            target_func=target,
            bounds=np.array([[-1.0, 1.0]] * 3),
            projections=[{'dims': [0], 'grid_points': [5]}],
            advanced_config={
                'suspect_recheck': {
                    'enabled': False,
                    'max_waves': 7,
                    'param_k': 5.0,
                },
            },
        )
        assert sampler.suspect_recheck_enabled is False
        assert sampler.max_suspect_waves == 7
        assert sampler.suspect_param_k == 5.0
        # Untouched keys retain defaults
        assert sampler.suspect_max_fraction == pytest.approx(0.25)
