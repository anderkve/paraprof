"""Unit tests for the de.allow_early_DE_exit predicate and generation-gate.

These bypass the MPI loop and exercise the sampler-level logic directly;
end-to-end savings are covered by examples/run_allow_early_de_exit_benchmark*.py.
"""
import numpy as np

from paraprof import ProfileProjector
from paraprof.sampler import SKIP_DE_WINDOW


def _make_sampler(enabled=False):
    """ProfileProjector with a 1-D projection over a smooth 3-D quadratic."""
    def target(p):
        return -float(np.sum(p ** 2))

    return ProfileProjector(
        target_func=target,
        bounds=np.array([[-5.0, 5.0]] * 3),
        projections=[{'dims': [0], 'grid_points': [10]}],
        pop_per_grid_point=2,
        advanced_config={'de': {'allow_early_DE_exit': enabled}},
    )


def _set_cell(sampler, idx, profiled_params, fitness, warm_start_best=True,
              status='optimized'):
    idx = idx if isinstance(idx, tuple) else (idx,)
    sampler.population[idx] = {
        'profiled_params': np.array([np.asarray(profiled_params, dtype=float)]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': status,
        'improvement_history': [],
        'last_update_gen': 0,
        'optimizer_state': None,
        'warm_start_best': warm_start_best,
    }
    sampler.global_max_target_val = max(sampler.global_max_target_val, float(fitness))


def test_skippable_when_neighbours_agree():
    """Agreeing neighbours + warm-start-best => the cell is skip-eligible."""
    s = _make_sampler(enabled=True)
    _set_cell(s, (4,), [0.40, -0.40], fitness=-0.32)
    _set_cell(s, (6,), [0.42, -0.41], fitness=-0.34)
    _set_cell(s, (5,), [0.41, -0.40], fitness=-0.33, status='active')
    assert s._is_de_skippable((5,)) is True


def test_not_skippable_guards():
    """Scattered neighbours (argmax disagreement) or a cold seed that beat the
    warm-start both keep full DE."""
    s = _make_sampler(enabled=True)
    _set_cell(s, (4,), [0.40, -0.40], fitness=-0.32)
    _set_cell(s, (6,), [-3.50, 3.50], fitness=-0.34)  # far-away mode
    _set_cell(s, (5,), [0.41, -0.40], fitness=-0.33, status='active')
    assert s._is_de_skippable((5,)) is False          # disagreement

    s.population[(6,)]['profiled_params'][0] = [0.42, -0.41]  # now agree
    s.population[(5,)]['warm_start_best'] = False             # but cold seed won
    assert s._is_de_skippable((5,)) is False          # multimodality guard


def test_gate_tags_skippable_and_counts():
    """create_de_generation_jobs gives a skip-eligible fresh cell the reduced
    window and counts it, leaving a non-eligible cell on the full window."""
    s = _make_sampler(enabled=True)
    s.current_generation = 1
    for i in (3, 4, 6, 7):
        _set_cell(s, (i,), [0.4, -0.4], fitness=-0.32)
    _set_cell(s, (9,), [-3.5, 3.5], fitness=-0.34)            # outlier for (8,)
    _set_cell(s, (5,), [0.41, -0.40], fitness=-0.33, status='active')
    _set_cell(s, (8,), [0.41, -0.40], fitness=-0.33, status='active')
    s.activated_grid_indices = set(s.population.keys())

    s.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

    assert s.population[(5,)].get('conv_window') == SKIP_DE_WINDOW
    assert s.population[(8,)].get('conv_window') is None
    assert s.de_cells_skipped == 1


def test_off_by_default_leaves_full_window():
    s = _make_sampler(enabled=False)
    assert s.de_allow_early_DE_exit is False
    s.current_generation = 1
    for i in (3, 4, 6, 7):
        _set_cell(s, (i,), [0.4, -0.4], fitness=-0.32)
    _set_cell(s, (5,), [0.41, -0.40], fitness=-0.33, status='active')
    s.activated_grid_indices = set(s.population.keys())

    s.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

    assert s.population[(5,)].get('conv_window') is None
    assert s.de_cells_skipped == 0
