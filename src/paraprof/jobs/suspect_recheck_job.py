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
        self.seed_fitness = np.full(len(candidate_seeds), -np.inf)
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
        f = self.sampler.fitness_for_cell(result, self.grid_idx)   # the target value; penalised in derived mode
        self.seed_fitness[result['context']['seed_idx']] = f
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

    def _seeds_to_polish(self):
        """The seeds the recheck polishes: the best one when it beats the cell's
        value by more than polish_threshold and, in derived mode, up to
        recheck_max_polishes further seeds that lie in another basin (farther
        than 5% of the profiled extent from the cell's point and from each
        other) within recheck_tolerance of the cell's value. A cell of a
        derived-plane projection can hold several basins of the inner
        problem, and a seed from a better one, moved into the cell to first
        order, can evaluate below the incumbent by the projection error alone;
        only a polish can tell."""
        sampler = self.sampler
        chosen = []
        if self.will_update:
            chosen.append((self.best_seed, self.best_seed_fitness))
        if not sampler.derived_mode or sampler.derived_recheck_max_polishes <= 0:
            return chosen
        extent = sampler._profiled_extent()
        own = sampler._best_profiled_params(self.grid_idx)
        taken = [own] + [c[0] for c in chosen]
        order = np.argsort(-self.seed_fitness)
        for i in order:
            f = self.seed_fitness[i]
            if not np.isfinite(f) or f < self.current_best_fitness - sampler.derived_recheck_tolerance:
                break
            seed = np.asarray(self.candidate_seeds[i], dtype=float)
            if all(np.linalg.norm((seed - t) / extent) > 0.05 for t in taken):
                chosen.append((seed, f))
                taken.append(seed)
                if len(chosen) >= 1 + sampler.derived_recheck_max_polishes:
                    break
        return chosen

    def on_finish(self, next_job_id):
        if not self.success:
            return None
        seeds = self._seeds_to_polish()
        if not seeds:
            return None

        from .lbfgsb_job import LBFGSBJob

        jobs = []
        for seed, f in seeds:
            start_params_full = self.sampler._construct_params(self.grid_idx, seed)
            jobs.append(LBFGSBJob(
                job_id=next_job_id,
                job_type='SUSPECT_RECHECK_LBFGSB',
                sampler=self.sampler,
                opt_dims=tuple(self.sampler.profiled_dims),
                start_params=seed,
                grid_idx=self.grid_idx,
                start_params_full=start_params_full,
                start_fitness=float(f),
            ))
            next_job_id += 1
        return (jobs if len(jobs) > 1 else jobs[0], next_job_id)
