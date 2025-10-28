"""
Differential Evolution job for grid point evolution.
"""
import numpy as np
from scipy.stats import cauchy, norm
from .base import Job


class DEGridPointJob(Job):
    """
    A job to run one generation of DE for one grid point.
    """
    def __init__(self, job_id, sampler, grid_idx, parent_pool,
                 pbest_archive, successful_F_list, successful_CR_list):

        super().__init__(job_id, 'DE_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.grid_state = self.sampler.population[self.grid_idx]

        # Shared resources from master
        self.parent_pool = parent_pool
        self.pbest_archive = pbest_archive

        # Shared lists to append successful mutations to
        self.successful_F_list = successful_F_list
        self.successful_CR_list = successful_CR_list

        self.pop_size = self.sampler.pop_per_grid_point
        self.evals_remaining = self.pop_size

        # Store trial info to process results
        self.trial_info = {} # {point_idx: (trial_params, F_i, CR_i)}

    def start(self):
        """Generate all trial points and return their evaluation tasks."""
        tasks = []
        grid_state = self.grid_state

        for i in range(self.pop_size):
            mem_loc = np.random.randint(0, self.sampler.memory_size)
            mu_CR, mu_F = self.sampler.memory_CR[mem_loc], self.sampler.memory_F[mem_loc]

            CR_i = np.clip(norm.rvs(loc=mu_CR, scale=0.1), 0, 1)
            F_i = cauchy.rvs(loc=mu_F, scale=0.1)
            while F_i <= 0:
                F_i = cauchy.rvs(loc=mu_F, scale=0.1)
            F_i = min(F_i, 1.0)

            x_i_params = grid_state['continuous_params'][i]

            use_neighbor_mutation = False
            best_neighbor_params = None
            if np.random.rand() < self.sampler.neighbor_pull_probability:
                best_neighbor_fitness = -np.inf
                for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
                    if neighbor_idx in self.sampler.population:
                        neighbor_state = self.sampler.population[neighbor_idx]
                        if neighbor_state['best_fitness'] > best_neighbor_fitness:
                            best_neighbor_fitness = neighbor_state['best_fitness']
                            neighbor_best_idx = np.argmax(neighbor_state['fitnesses'])
                            best_neighbor_params = neighbor_state['continuous_params'][neighbor_best_idx]

                if best_neighbor_params is not None and best_neighbor_fitness > grid_state['best_fitness']:
                    use_neighbor_mutation = True

            if len(self.parent_pool) < 3:
                continue # Not enough parents to mutate

            mutant = None
            if use_neighbor_mutation:
                r2_p, r3_p = np.random.choice(self.parent_pool, 2, replace=False)
                r2, r3 = r2_p['continuous_params'], r3_p['continuous_params']
                mutant = x_i_params + F_i * (best_neighbor_params - x_i_params) + F_i * (r2 - r3)

            elif self.sampler.mutation_strategy == 'current-to-rand/1':
                p1_p, p2_p, p3_p = np.random.choice(self.parent_pool, 3, replace=False)
                p1, p2, p3 = p1_p['continuous_params'], p2_p['continuous_params'], p3_p['continuous_params']
                mutant = x_i_params + F_i * (p1 - x_i_params) + F_i * (p2 - p3)

            elif self.sampler.mutation_strategy == 'rand/1':
                r1_p, r2_p, r3_p = np.random.choice(self.parent_pool, 3, replace=False)
                r1, r2, r3 = r1_p['continuous_params'], r2_p['continuous_params'], r3_p['continuous_params']
                mutant = r1 + F_i * (r2 - r3)

            elif self.sampler.mutation_strategy == 'current-to-pbest/1':
                archive = self.pbest_archive if self.pbest_archive else self.parent_pool
                x_pbest_p = np.random.choice(archive)
                x_pbest = x_pbest_p['continuous_params']

                potential_diff = [p for p in self.parent_pool if not np.array_equal(p['continuous_params'], x_pbest)]
                if len(potential_diff) < 2:
                    potential_diff = self.parent_pool # Fallback
                    if len(potential_diff) < 2:
                        continue

                r2_p, r3_p = np.random.choice(potential_diff, 2, replace=False)
                r2, r3 = r2_p['continuous_params'], r3_p['continuous_params']
                mutant = x_i_params + F_i * (x_pbest - x_i_params) + F_i * (r2 - r3)

            if mutant is None:
                self.evals_remaining -= 1 # This individual won't be evaluated
                continue

            mutant = self.sampler._ensure_bounds(mutant, self.sampler.continuous_dims)

            cross_points = np.random.rand(self.sampler.n_cont_dims) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.sampler.n_cont_dims)] = True
            trial_params = np.where(cross_points, mutant, x_i_params)

            # Store info needed when result comes back
            self.trial_info[i] = (trial_params, F_i, CR_i)

            full_trial_params = self.sampler._construct_params(self.grid_idx, trial_params)
            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i
            }
            tasks.append({'params': full_trial_params, 'context': context})

        if not tasks and self.evals_remaining == 0:
            self.success = True
            self._is_finished = True

        return tasks

    def process_result(self, result):
        """Compare trial fitness with target and store successful F/CR."""
        point_idx = result['context']['point_idx']
        trial_fitness = result['target_val']

        if point_idx not in self.trial_info:
            print(f"Warning: Received result for DE point_idx {point_idx} with no trial info. Ignoring.")
            self.evals_remaining -= 1
            if self.evals_remaining <= 0:
                self.success = True
                self._is_finished = True
            return []

        trial_params, F_i, CR_i = self.trial_info[point_idx]
        grid_state = self.grid_state

        if trial_fitness > grid_state['fitnesses'][point_idx]:
            # Success! Update the individual
            grid_state['continuous_params'][point_idx] = trial_params
            grid_state['fitnesses'][point_idx] = trial_fitness
            # Append to the shared lists
            self.successful_F_list.append(F_i)
            self.successful_CR_list.append(CR_i)

        self.evals_remaining -= 1
        if self.evals_remaining <= 0:
            self.success = True
            self._is_finished = True

        return [] # No new tasks

    def on_finish(self, next_job_id):
        """
        Update the best_fitness and history for this grid point.
        If converged, spawn a new REFINEMENT job.
        """
        if not self.success:
            return None

        grid_state = self.grid_state
        old_best_fitness = grid_state['best_fitness']
        new_best_fitness = np.max(grid_state['fitnesses'])
        improvement = new_best_fitness - old_best_fitness
        grid_state['improvement_history'].append(improvement)

        if new_best_fitness > old_best_fitness:
            grid_state['best_fitness'] = new_best_fitness
            grid_state['last_update_gen'] = self.sampler.current_generation
            self.sampler.profile_likelihood_grid[self.grid_idx] = new_best_fitness

            if new_best_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = new_best_fitness

        # Check for convergence
        if grid_state['status'] == 'active' and \
           len(grid_state['improvement_history']) == self.sampler.convergence_window:

            avg_improvement = np.mean(grid_state['improvement_history'])
            if avg_improvement < self.sampler.convergence_threshold:
                print(f"--- DE Converged for {self.grid_idx}. Spawning refinement job. ---")
                # This job factory will set status to 'refining_queued'
                # and return (new_job, next_job_id + 1)
                return self.sampler.create_refinement_job_for_point(self.grid_idx, next_job_id)

        return None
