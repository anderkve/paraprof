"""One evaluation at a point a derived-mode polish asks for after it has finished: the end
point projected back into its cell, or the converged point moved onto the cell's edge."""
import numpy as np
from .base import Job


class DerivedSnapJob(Job):
    """Evaluate one full parameter vector for a cell of a projection over derived
    quantities. The sampler's in-cell record picks the value up through
    ``_register_target_call``; here the cell's population best is updated too."""
    def __init__(self, job_id, sampler, grid_idx, full_params):
        super().__init__(job_id, 'DERIVED_SNAP', sampler)
        self.grid_idx = grid_idx
        self.full_params = np.asarray(full_params, dtype=float)
        self.landed_inside = False
        self.result_value = -np.inf

    def start(self):
        return [{'params': self.full_params,
                 'context': {'type': self.type, 'job_id': self.id}}]

    def process_result(self, result):
        sampler = self.sampler
        fit = sampler.fitness_for_cell(result, self.grid_idx)
        self.result_value = result['target_val']
        d = result.get('derived')
        d = sampler._as_derived_array(d) if d is not None else None
        if d is not None:
            q = d[sampler.derived_idx]
            self.landed_inside = bool(np.all(np.isfinite(q)) and sampler._derived_cell_distance(q, self.grid_idx) == 0.0)
        state = sampler.population.get(self.grid_idx)
        if state is not None and np.isfinite(fit) and fit > state['best_fitness']:
            k = int(np.argmax(state['fitnesses']))
            state['fitnesses'][k] = fit
            state['profiled_params'][k] = np.asarray(result['params'], dtype=float)[sampler.profiled_dims]
            state['best_fitness'] = fit
            sampler.profile_likelihood_grid[self.grid_idx] = fit
        self.success = True
        self._is_finished = True
        return []

    def on_finish(self, next_job_id):
        return None
