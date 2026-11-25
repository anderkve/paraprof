"""Tests for the choice of neighbours in patching.

The unit tests plant grid points by hand and call the sampler helpers
directly. The last test runs patching on virtual workers (no MPI).
"""
import numpy as np
import pytest

import octant_lab
from paraprof import ProfileProjector
from paraprof.jobs.patching_test_job import PatchingTestJob


def _make_sampler(grid_points, **kwargs):
    """Projector over a quadratic with two profiled parameters."""
    n_proj = len(grid_points)
    return ProfileProjector(
        target_func=lambda p: -float(np.sum(p ** 2)),
        bounds=np.array([[-5.0, 5.0]] * (n_proj + 2)),
        projections=[{'dims': list(range(n_proj)), 'grid_points': list(grid_points)}],
        roi_threshold=100.0,
        **kwargs,
    )


def _set_cell(sampler, idx, profiled_params, fitness):
    """Plant a finished grid point."""
    sampler.population[idx] = {
        'profiled_params': np.array([np.asarray(profiled_params, dtype=float)]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': 'optimized',
        'improvement_history': [],
        'last_update_gen': 0,
        'optimizer_state': None,
    }
    sampler.global_max_target_val = max(sampler.global_max_target_val, float(fitness))


def _plant_two_optima_1d(sampler, border):
    """Values falling with the index; grid points below ``border`` hold one
    optimum, the others a distant one. Both vary slowly along the grid."""
    for i in range(sampler.grid_shape[0]):
        vec = [0.01 * i, -0.01 * i] if i < border else [3.0 + 0.01 * i, 2.0]
        _set_cell(sampler, (i,), vec, fitness=-float(i))


class TestPatchingSources:
    def test_smooth_grid_tests_only_the_higher_neighbour(self):
        sampler = _make_sampler([12])
        _plant_two_optima_1d(sampler, border=100)
        thresholds = sampler._jump_thresholds()

        assert sampler._patching_sources((5,), thresholds) == [(4,)]
        # The highest grid point has no higher neighbour.
        assert sampler._patching_sources((0,), thresholds) == []

    def test_neighbour_across_a_jump_is_tested_even_if_lower(self):
        sampler = _make_sampler([12])
        _plant_two_optima_1d(sampler, border=7)
        thresholds = sampler._jump_thresholds()

        # (6,) holds the first optimum; (7,) is lower and holds the other.
        assert sampler._patching_sources((6,), thresholds) == [(5,), (7,)]
        # (6,) is both the highest neighbour of (7,) and across the jump.
        assert sampler._patching_sources((7,), thresholds) == [(6,)]

    def test_one_neighbour_per_optimum(self):
        sampler = _make_sampler([6, 6])
        for i in range(7):
            for j in range(7):
                vec = [0.01 * i, 0.01 * j] if i < 4 else [3.0 + 0.01 * i, 2.0 + 0.01 * j]
                _set_cell(sampler, (i, j), vec, fitness=-float(i) - 0.1 * j)
        thresholds = sampler._jump_thresholds()

        sources = sampler._patching_sources((3, 3), thresholds)
        # The highest neighbour, and of the three across the jump the one
        # along an axis.
        assert sources == [(2, 2), (4, 3)]

    def test_sources_limit_the_neighbours(self):
        sampler = _make_sampler([12])
        _plant_two_optima_1d(sampler, border=7)
        thresholds = sampler._jump_thresholds()

        assert sampler._patching_sources((6,), thresholds, sources={(7,)}) == [(7,)]
        assert sampler._patching_sources((6,), thresholds, sources={(5,)}) == [(5,)]
        assert sampler._patching_sources((6,), thresholds, sources={(9,)}) == []

    def test_periodic_parameter_at_its_bounds_is_no_jump(self):
        sampler = _make_sampler([12], periodic=[1])
        for i in range(sampler.grid_shape[0]):
            # The periodic parameter crosses its bound between (5,) and (6,).
            first = -5.0 + 0.01 * (i - 5.5) if i >= 6 else 5.0 + 0.01 * (i - 5.5)
            _set_cell(sampler, (i,), [first, 0.0], fitness=-float(i))
        thresholds = sampler._jump_thresholds()

        assert sampler._patching_sources((5,), thresholds) == [(4,)]


class TestPatchingWaveJobs:
    def test_wave_zero_tests_across_the_border_both_ways(self):
        sampler = _make_sampler([12])
        _plant_two_optima_1d(sampler, border=7)

        jobs, next_id = sampler.create_patching_wave_jobs(0, None, 0)
        tested = {(job.grid_idx, tuple(job.test_profiled_params)) for job in jobs}

        assert next_id == len(jobs)
        assert ((6,), tuple(sampler._best_profiled_params((7,)))) in tested
        assert ((7,), tuple(sampler._best_profiled_params((6,)))) in tested

    def test_later_wave_tests_only_the_updated_grid_points(self):
        sampler = _make_sampler([12])
        _plant_two_optima_1d(sampler, border=7)

        jobs, _ = sampler.create_patching_wave_jobs(1, [(7,)], 0)
        tested = {(job.grid_idx, tuple(job.test_profiled_params)) for job in jobs}

        vec = tuple(sampler._best_profiled_params((7,)))
        # (6,) is across the jump from (7,). (8,) is lower and holds the
        # same optimum, so it gets a test too; (7,) itself gets none.
        assert tested == {((6,), vec), ((8,), vec)}

    def test_wave_cap_defaults_to_the_grid_size(self):
        sampler = _make_sampler([12, 30])
        assert sampler.patching_wave_cap() == 31

        sampler = _make_sampler([12, 30], max_patching_waves=4)
        assert sampler.patching_wave_cap() == 4


class TestPatchingTestJob:
    @pytest.mark.parametrize("gain, taken", [(1e-12, False), (1e-6, True)])
    def test_gain_must_exceed_the_convergence_tolerance(self, gain, taken):
        sampler = _make_sampler([12])
        _set_cell(sampler, (3,), [0.0, 0.0], fitness=-1.0)
        job = PatchingTestJob(0, sampler, (3,), np.array([0.1, 0.1]), wave_number=0)

        job.process_result({'target_val': -1.0 + gain})

        assert job.will_update is taken


def test_patching_moves_a_displaced_border_back():
    """octant_6d, projection on s13: the upper octant is planted three grid
    points too far on the downhill side of the maximum. Patching has to take
    the lower octant from lower neighbours."""
    dims, npts = [4], [60]
    ref = octant_lab.reference(dims, npts)
    last = ref['oct'].shape[0] - 1

    def displaced(idx):
        return int(ref['oct'][max(0, min(last, idx[0] - 3))])

    rec = octant_lab.run(dims, npts, assign=displaced, ref=ref,
                         n_initial_optimizations=20)

    planted, _ = octant_lab.compare(rec['snaps']['before patching'], ref, rec['roi'])
    patched, _ = octant_lab.compare(rec['snaps']['before suspect recheck'], ref, rec['roi'])
    assert planted['wrong octant'] == 3
    assert patched['wrong octant'] == 0
    assert patched['ok'] == planted['ok'] + 3
