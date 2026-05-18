"""Suspect-cell recheck job: tests multiple candidate seeds at one grid cell."""
import numpy as np
from .base import Job


class SuspectRecheckJob(Job):
    def __init__(self, job_id, sampler, grid_idx, candidate_seeds, wave_number):
        super().__init__(job_id, 'SUSPECT_RECHECK', sampler)
        self.grid_idx = grid_idx
        self.candidate_seeds = candidate_seeds
        self.wave_number = wave_number

        self.current_best_fitness = sampler.population[grid_idx]['best_fitness']
        self.best_seed = None
        self.best_seed_fitness = -np.inf
        self.pending = len(candidate_seeds)
        self.will_update = False

    def start(self):
        if not self.candidate_seeds:
            self.success = True
            self._is_finished = True
            return []

        tasks = []
        for i, seed in enumerate(self.candidate_seeds):
            full_params = self.sampler._construct_params(self.grid_idx, seed)
            context = {
                'type': self.type,
                'job_id': self.id,
                'grid_idx': self.grid_idx,
                'wave_number': self.wave_number,
                'seed_idx': i,
            }
            tasks.append({'params': full_params, 'context': context})
        return tasks

    def process_result(self, result):
        f = result['target_val']
        if np.isfinite(f) and f > self.best_seed_fitness:
            self.best_seed_fitness = f
            self.best_seed = self.candidate_seeds[result['context']['seed_idx']]
        self.pending -= 1

        if self.pending == 0:
            self.success = True
            self._is_finished = True
            improvement = self.best_seed_fitness - self.current_best_fitness
            if (self.best_seed is not None
                    and improvement > self.sampler.suspect_polish_threshold):
                self.will_update = True

        return []

    def on_finish(self, next_job_id):
        if not self.success or not self.will_update:
            return None

        from .lbfgsb_job import LBFGSBJob

        start_params_full = self.sampler._construct_params(
            self.grid_idx, self.best_seed
        )
        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='SUSPECT_RECHECK_LBFGSB',
            sampler=self.sampler,
            opt_dims=tuple(self.sampler.profiled_dims),
            start_params=self.best_seed,
            grid_idx=self.grid_idx,
            start_params_full=start_params_full,
            start_fitness=self.best_seed_fitness,
        )
        return (job, next_job_id + 1)
