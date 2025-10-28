"""
Grid point activation job for initializing populations.
"""
import numpy as np
import collections
from scipy.stats.qmc import LatinHypercube as LHS
from .base import Job


class ActivationJob(Job):
    """
    A job to evaluate the initial population for a single grid point.
    Can be warm-started with parameters from a neighbor.
    """
    def __init__(self, job_id, sampler, grid_idx, warm_start_params=None):
        super().__init__(job_id, 'ACTIVATE_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.warm_start_params = warm_start_params

        self.pop_size = self.sampler.pop_per_grid_point
        self.n_cont_dims = self.sampler.n_cont_dims
        cont_bounds = self.sampler.bounds[self.sampler.continuous_dims]

        # Generate all continuous parameter sets at once
        lhs_sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = lhs_sampler.random(n=self.pop_size)
        scaled_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])

        if self.warm_start_params is not None:
            # Replace the closest LHS point with the warm-start params
            distances = np.linalg.norm(scaled_samples - self.warm_start_params, axis=1)
            closest_idx = np.argmin(distances)
            scaled_samples[closest_idx] = self.warm_start_params

        self.all_continuous_params = scaled_samples

        self.all_full_params = [
            self.sampler._construct_params(self.grid_idx, cont_params)
            for cont_params in self.all_continuous_params
        ]

        # State tracking
        self.fitnesses = np.full(self.pop_size, -np.inf)
        self.evals_remaining = self.pop_size

    def start(self):
        """Return tasks for all individuals in the population."""
        tasks = []
        for i, full_params in enumerate(self.all_full_params):
            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i
            }
            tasks.append({'params': full_params, 'context': context})
        return tasks

    def process_result(self, result):
        """Store the fitness for one individual."""
        point_idx = result['context']['point_idx']
        self.fitnesses[point_idx] = result['target_val']
        self.evals_remaining -= 1

        if self.evals_remaining == 0:
            self.success = True
            self._is_finished = True

        return [] # No new tasks are generated from a result

    def on_finish(self, next_job_id):
        """Add this grid point to the main sampler population."""
        if self.grid_idx in self.sampler.pending_activation_indices:
             self.sampler.pending_activation_indices.remove(self.grid_idx)

        if not self.success or self.grid_idx in self.sampler.population:
            return None

        best_fitness = np.max(self.fitnesses)
        self.sampler.profile_likelihood_grid[self.grid_idx] = best_fitness

        self.sampler.population[self.grid_idx] = {
            'continuous_params': self.all_continuous_params,
            'fitnesses': self.fitnesses,
            'best_fitness': best_fitness,
            'status': 'active',
            'improvement_history': collections.deque(maxlen=self.sampler.convergence_window),
            'last_update_gen': 0,
            'optimizer_state': None
        }
        self.sampler.active_grid_indices.add(self.grid_idx)

        return None
