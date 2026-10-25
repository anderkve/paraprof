"""One evaluation at the point a derived-mode polish projected back into its cell."""
import numpy as np
from .base import Job


class DerivedSnapJob(Job):
    """Evaluate one full parameter vector for a cell of a projection over
    derived quantities. The sampler's in-cell record picks the value up through
    ``_register_target_call``; here the cell's population best is updated too."""
    def __init__(self, job_id, sampler, grid_idx, full_params):
        super().__init__(job_id, 'DERIVED_SNAP', sampler)
        self.grid_idx = grid_idx
        self.full_params = np.asarray(full_params, dtype=float)

    def start(self):
        return [{'params': self.full_params,
                 'context': {'type': self.type, 'job_id': self.id}}]

    def process_result(self, result):
        fit = self.sampler.fitness_for_cell(result, self.grid_idx)
        state = self.sampler.population.get(self.grid_idx)
        if state is not None and np.isfinite(fit) and fit > state['best_fitness']:
            k = int(np.argmax(state['fitnesses']))
            state['fitnesses'][k] = fit
            state['profiled_params'][k] = np.asarray(result['params'], dtype=float)[self.sampler.profiled_dims]
            state['best_fitness'] = fit
            self.sampler.profile_likelihood_grid[self.grid_idx] = fit
        self.success = True
        self._is_finished = True
        return []
