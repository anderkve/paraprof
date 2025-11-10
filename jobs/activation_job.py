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

        # Check if we're in direct evaluation mode
        if self.sampler.direct_eval_mode:
            # Direct evaluation mode: just evaluate at grid point center
            self.pop_size = 1
            self.n_cont_dims = 0

            # Create empty continuous params (shape: 1 x 0)
            self.all_continuous_params = np.empty((1, 0))

            # Construct full params at grid point (all dims are projection dims)
            grid_coords = self.sampler._get_grid_coords_from_indices(self.grid_idx)
            self.all_full_params = [grid_coords]

            # State tracking
            self.fitnesses = np.full(1, -np.inf)
            self.evals_remaining = 1

        else:
            # Normal mode: population-based initialization
            self.pop_size = self.sampler.pop_per_grid_point
            self.n_cont_dims = self.sampler.n_cont_dims
            cont_bounds = self.sampler.bounds[self.sampler.continuous_dims]

            # --- Mixed initialization strategy ---
            # Calculate how many samples from each source
            mix_ratios = self.sampler.activation_mix_ratios
            n_from_neighbors = int(self.pop_size * mix_ratios['neighbors'])
            n_from_global = int(self.pop_size * mix_ratios['global'])
            n_from_random = self.pop_size - n_from_neighbors - n_from_global

            samples_list = []

            # 1. Neighbor samples (warm start)
            if self.warm_start_params is not None and n_from_neighbors > 0:
                # Add the warm start params
                samples_list.append(self.warm_start_params)
                # Add perturbations around it for the remaining neighbor samples
                for _ in range(n_from_neighbors - 1):
                    perturbation = np.random.normal(0, 0.1, size=self.n_cont_dims)
                    perturbed = self.warm_start_params + perturbation * (cont_bounds[:, 1] - cont_bounds[:, 0])
                    perturbed = self.sampler._ensure_bounds(perturbed, self.sampler.continuous_dims)
                    samples_list.append(perturbed)
            else:
                # If no warm start, redistribute to random
                n_from_random += n_from_neighbors

            # 2. Global pool samples
            global_samples = self.sampler._sample_from_global_pool(n_from_global)
            if global_samples is not None:
                samples_list.extend(global_samples)
            else:
                # If pool is empty, redistribute to random
                n_from_random += n_from_global

            # 3. Random LHS samples
            if n_from_random > 0:
                lhs_sampler = LHS(d=self.n_cont_dims, seed=np.random.randint(1e6, 1e12))
                unit_samples = lhs_sampler.random(n=n_from_random)
                random_samples = cont_bounds[:, 0] + unit_samples * (cont_bounds[:, 1] - cont_bounds[:, 0])
                samples_list.extend(random_samples)

            # Combine all samples
            self.all_continuous_params = np.array(samples_list)

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

        if self.sampler.direct_eval_mode:
            # Direct evaluation mode: simpler state (no population, just the value)
            self.sampler.population[self.grid_idx] = {
                'fitness': best_fitness,
                'status': 'evaluated',
                'full_params': self.all_full_params[0]
            }
        else:
            # Normal mode: full population-based state
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
