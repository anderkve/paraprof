"""
Unit tests for the smooth-certification fast-convergence helper
(`de.smooth_certify`).

These bypass the MPI master/worker loop and exercise the sampler-level
predicate and config plumbing directly. End-to-end behaviour and the
target-call savings are covered by the A/B benchmark in
``examples/run_smooth_certify_benchmark*.py``.
"""
import numpy as np

from paraprof import ProfileProjector


def _make_sampler(n_prof_dims=2, smooth_certify=False, grid_n=10):
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
        advanced_config={'de': {'smooth_certify': smooth_certify}},
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
        assert sampler.de_smooth_certify is False
        assert sampler.de_cells_smooth_certified == 0

    def test_opt_in_flag(self):
        sampler = _make_sampler(smooth_certify=True)
        assert sampler.de_smooth_certify is True


class TestSmoothCertifiablePredicate:
    def test_agreeing_neighbours_certify(self):
        """A cell flanked by neighbours that agree on the profiled argmax,
        with its own warm-start the best seed, is certifiable."""
        sampler = _make_sampler(smooth_certify=True)
        # Neighbours of cell (5,) at (4,) and (6,) share the same argmax.
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (6,), [0.42, -0.41], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_smooth_certifiable((5,)) is True

    def test_scattered_neighbours_do_not_certify(self):
        """Neighbours that disagree on the argmax (mode crossing / multimodal
        inner problem) must keep full DE."""
        sampler = _make_sampler(smooth_certify=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        # (6,) sits on a far-away mode: large argmax spread.
        _set_cell(sampler, (6,), [-3.50, 3.50], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_smooth_certifiable((5,)) is False

    def test_cold_seed_won_does_not_certify(self):
        """If a cold random/pool seed beat the neighbour warm-start at
        activation (warm_start_best False), DE's global search is needed."""
        sampler = _make_sampler(smooth_certify=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (6,), [0.42, -0.41], fitness=-0.34)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=False, status='active')
        assert sampler._is_smooth_certifiable((5,)) is False

    def test_too_few_neighbours_do_not_certify(self):
        """A single neighbour is not enough agreement evidence."""
        sampler = _make_sampler(smooth_certify=True)
        _set_cell(sampler, (4,), [0.40, -0.40], fitness=-0.32)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        assert sampler._is_smooth_certifiable((5,)) is False


class TestGenerationGate:
    def test_certifiable_cell_gets_reduced_window(self):
        """create_de_generation_jobs tags a fresh certifiable cell with the
        reduced convergence window and counts it; a non-certifiable cell is
        left on the default window."""
        from paraprof.sampler import SMOOTH_CERTIFY_WINDOW

        sampler = _make_sampler(smooth_certify=True)
        sampler.current_generation = 1
        # Build a settled, agreeing neighbourhood plus two fresh active cells:
        # (5,) certifiable, (8,) not (scattered neighbours).
        for i in (3, 4, 6, 7):
            _set_cell(sampler, (i,), [0.4, -0.4], fitness=-0.32)
        _set_cell(sampler, (9,), [-3.5, 3.5], fitness=-0.34)  # outlier for (8,)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        _set_cell(sampler, (8,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        sampler.activated_grid_indices = set(sampler.population.keys())

        sampler.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

        assert sampler.population[(5,)].get('conv_window') == SMOOTH_CERTIFY_WINDOW
        assert sampler.population[(8,)].get('conv_window') is None
        assert sampler.de_cells_smooth_certified == 1

    def test_off_by_default_no_tagging(self):
        """With the feature off, no cell is tagged and the counter stays 0."""
        sampler = _make_sampler(smooth_certify=False)
        sampler.current_generation = 1
        for i in (3, 4, 6, 7):
            _set_cell(sampler, (i,), [0.4, -0.4], fitness=-0.32)
        _set_cell(sampler, (5,), [0.41, -0.40], fitness=-0.33,
                  warm_start_best=True, status='active')
        sampler.activated_grid_indices = set(sampler.population.keys())

        sampler.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)

        assert sampler.population[(5,)].get('conv_window') is None
        assert sampler.de_cells_smooth_certified == 0


class TestPoolCertifyTrigger:
    """Second smooth-certify trigger: cross-projection pool agreement."""

    def _pool_push(self, sampler, cell_node_phi, fitness):
        """Push a full-D point landing in cell (5,) at projection node 5."""
        node0 = sampler.grid_axes[0][5]
        full = np.array([node0, cell_node_phi[0], cell_node_phi[1]])
        sampler._update_global_pool(full, fitness, grid_idx=None)
        sampler._pool_cell_best = None  # invalidate cache

    def test_pool_agreement_certifies_without_neighbours(self):
        """A fresh cell with no settled neighbours (neighbour trigger can't
        fire) is certified when an ROI-competitive pool optimum at this cell
        agrees with the cell's current best argmax."""
        sampler = _make_sampler(smooth_certify=True)
        sampler.current_generation = 1
        # Enough activated cells for DE to run, but none neighbouring (5,).
        for i in (0, 1, 2):
            _set_cell(sampler, (i,), [0.0, 0.0], fitness=-0.1)
        _set_cell(sampler, (5,), [0.20, -0.20], fitness=-0.08,
                  warm_start_best=True, status='active')
        sampler.activated_grid_indices = set(sampler.population.keys())
        # Cross-projection optimum essentially at this cell, agreeing phi.
        self._pool_push(sampler, [0.205, -0.205], fitness=-0.05)

        assert sampler._is_smooth_certifiable((5,)) is False  # no neighbours
        assert sampler._is_pool_certifiable((5,)) is True

        sampler.create_de_generation_jobs(next_job_id=0, max_num_to_evolve=None)
        assert sampler.population[(5,)].get('conv_window') is not None
        assert sampler.de_cells_certified_pool_only == 1

    def test_pool_disagreement_does_not_certify(self):
        """A pool optimum at this cell with a far-off argmax (a different inner
        mode) does not certify -- DE must run."""
        sampler = _make_sampler(smooth_certify=True)
        _set_cell(sampler, (5,), [0.20, -0.20], fitness=-0.08,
                  warm_start_best=True, status='active')
        self._pool_push(sampler, [3.5, 3.5], fitness=-0.05)  # different mode
        assert sampler._is_pool_certifiable((5,)) is False

    def test_below_roi_pool_point_does_not_certify(self):
        sampler = _make_sampler(smooth_certify=True)
        _set_cell(sampler, (5,), [0.20, -0.20], fitness=-0.08,
                  warm_start_best=True, status='active')
        sampler.global_max_target_val = 100.0  # pool point far below ROI cutoff
        self._pool_push(sampler, [0.205, -0.205], fitness=-0.05)
        assert sampler._is_pool_certifiable((5,)) is False

    def test_trigger_can_be_disabled(self):
        sampler = _make_sampler(smooth_certify=True)
        sampler.de_pool_certify_trigger = False
        _set_cell(sampler, (5,), [0.20, -0.20], fitness=-0.08,
                  warm_start_best=True, status='active')
        self._pool_push(sampler, [0.205, -0.205], fitness=-0.05)
        assert sampler._is_pool_certifiable((5,)) is False
