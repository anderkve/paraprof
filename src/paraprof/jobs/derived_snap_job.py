"""One evaluation at the point a derived-mode polish projected back into its cell, and the
stiff second-stage polish that follows it."""
import numpy as np
from .base import Job


class DerivedSnapJob(Job):
    """Evaluate one full parameter vector for a cell of a projection over derived
    quantities. The sampler's in-cell record picks the value up through
    ``_register_target_call``; here the cell's population best is updated too.
    After a first-stage polish, a second polish with a stiffer anchoring penalty
    is started from the best point inside the cell, so that an optimum on the
    cell boundary is finished rather than left where the soft penalty put it."""
    def __init__(self, job_id, sampler, grid_idx, full_params, stage=1):
        super().__init__(job_id, 'DERIVED_SNAP', sampler)
        self.grid_idx = grid_idx
        self.full_params = np.asarray(full_params, dtype=float)
        self.stage = stage
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
        sampler = self.sampler
        factor = sampler.derived_second_stage_factor
        if self.stage != 1 or not factor or factor <= 0 or self.grid_idx not in sampler.population:
            return None
        # start the stiff polish from the best point known inside the cell
        if self.landed_inside:
            x0, v0 = self.full_params, self.result_value
        else:
            rec = sampler.derived_cell_best.get(self.grid_idx)
            if rec is None:
                return None
            x0, v0 = rec[1], rec[0]
        from .lbfgsb_job import LBFGSBJob
        scale = sampler.derived_penalty_scale * factor
        state = sampler.population[self.grid_idx]
        state['status'] = 'LBFGSB_queued'
        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='LBFGSB',
            sampler=sampler,
            opt_dims=tuple(sampler.profiled_dims),
            start_params=np.asarray(x0, dtype=float)[sampler.profiled_dims],
            grid_idx=self.grid_idx,
            start_params_full=np.asarray(x0, dtype=float),
            seed_history=None,
            start_fitness=float(v0),          # inside the cell: no penalty at any scale
        )
        job.penalty_scale = scale
        job.derived_stage = 2
        job.lbfgsb_max_iter = int(sampler.derived_second_stage_max_iter)
        return job, next_job_id + 1
