"""
Unit tests for the allow_skip_DE fast-convergence helper
(`de.allow_skip_DE`).

These bypass the MPI master/worker loop and exercise the sampler-level
predicate and config plumbing directly. End-to-end behaviour and the
target-call savings are covered by the A/B benchmark in
``examples/run_allow_skip_de_benchmark*.py``.
"""
import numpy as np

from paraprof import ProfileProjector


def _make_sampler(n_prof_dims=2, allow_skip_DE=False, grid_n=10):
    """ProfileProjector with a 1-D projection over a smooth quadratic."""
    n_dims = 1 + n_prof_dims

    def target(p):
        return -float(np.sum(p ** 2))

    bounds = np.array([[-5.0, 5.0]] * n_dims)
    projection = {'dims': [0], 'grid_points': [grid_n]}
    return ProfileProjector(
        target_func=target,
        bounds=bounds,
        projections=[projection],
        pop_per_grid_point=2,
        advanced_config={'de': {'allow_skip_DE': allow_skip_DE}},
    )


def _set_cell(sampler, idx, profiled_params, fitness, warm_start_best=True,
              status='optimized'):
    if not isinstance(idx, tuple):
        idx = (idx,)
    profiled_params = np.asarray(profiled_params, dtype=float)
    sampler.population[idx] = {
        'profiled_params': np.array([profiled_params]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': status,
        'improvement_history': [],
        'last_update_gen': 0,
        'optimizer_state': None,
        'warm_start_best': warm_start_best,
    }
    if fitness > sampler.global_max_target_val:
        sampler.global_max_target_val = float(fitness)


class TestConfigPlumbing:
    def test_default_is_off(self):
        sampler = _make_sampler()
        assert sampler.de_allow_skip_DE is False
        assert sampler.de_cells_skipped == 0

    def test_opt_in_flag(self):
        sampler = _make_sampler(allow_skip_DE=True)
        assert sampler.de_allow_skip_DE is True


class TestDeSkippablePredicate:
    def test_agreeing_neighbours_certify(self):
        """A cell flanked by neighbours that agree on the profiled argmax,
        with its own warm-start the best seed, is skippable."""
        sampler = _make_sampler(allow_skip_DE=True)
        # Neighbours of cell (5,) at (4,) and (6,) share the same argmax.
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (6,), [0.42, -0.41], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_de_skippable((5,)) is True

    def test_scattered_neighbours_do_not_certify(self):
        """Neighbours that disagree on the argmax (mode crossing / multimodal
        inner problem) must keep full DE."""
        sampler = _make_sampler(allow_skip_DE=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        # (6,) sits on a far-away mode: large argmax spread.
        _set_cell(sampler, (6,), [-3.50, 3.50], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_de_skippable((5,)) is False

    def test_cold_seed_won_does_not_certify(self):
        """If a cold random/pool seed beat the neighbour warm-start at
        activation (warm_start_best False), DE's global search is needed."""
        sampler = _make_sampler(allow_skip_DE=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (6,), [0.42, -0.41], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=False, status='active')
        assert sampler._is_de_skippable((5,)) is False

    def test_too_few_neighbours_do_not_certify(self):
        """A single neighbour is not enough agreement evidence."""
        sampler = _make_sampler(allow_skip_DE=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_de_skippable((5,)) is False


class TestGenerationGate:
    def test_skippable_cell_gets_reduced_window(self):
        """create_de_generation_jobs tags a fresh skippable cell with the
        reduced convergence window and counts it; a non-skippable cell is
        left on the default window."""
        from paraprof.sampler import SKIP_DE_WINDOW

        sampler = _make_sampler(allow_skip_DE=True)
        sampler.current_generation = 1
        # Build a settled, agreeing neighbourhood plus two fresh active cells:
        # (5,) skippable, (8,) not (scattered neighbours).
        for i in (3, 4, 6, 7):
            _set_cell(sampler, (i,), [0.4, -0.4], fitness=-0.32)
        _set_cell(sampler, (9,), [-3.5, 3.5], fitness=-0.34)  # outlier for (8,)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        _set_cell(sampler, (8,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        sampler.activated_grid_indices = set(sampler.population.keys())

        sampler.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

        assert sampler.population[(5,)].get('conv_window') == SKIP_DE_WINDOW
        assert sampler.population[(8,)].get('conv_window') is None
        assert sampler.de_cells_skipped == 1

    def test_off_by_default_no_tagging(self):
        """With the feature off, no cell is tagged and the counter stays 0."""
        sampler = _make_sampler(allow_skip_DE=False)
        sampler.current_generation = 1
        for i in (3, 4, 6, 7):
            _set_cell(sampler, (i,), [0.4, -0.4], fitness=-0.32)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        sampler.activated_grid_indices = set(sampler.population.keys())

        sampler.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

        assert sampler.population[(5,)].get('conv_window') is None
        assert sampler.de_cells_skipped == 0
