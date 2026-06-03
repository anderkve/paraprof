"""Differential Evolution job for grid point evolution."""
import numpy as np
from scipy.stats import cauchy, norm
from ..logger import get_logger
from .base import Job

logger = get_logger()


DE_CR_NORMAL_SCALE = 0.1
DE_F_CAUCHY_SCALE = 0.1
DE_F_MAX_VALUE = 1.0
DE_MIN_PARENT_POOL_SIZE = 3


class DEGridPointJob(Job):
    """Run one DE generation for a single grid point."""
    def __init__(self, job_id, sampler, grid_idx, parent_pool,
                 pbest_archive, successful_F_list, successful_CR_list):

        super().__init__(job_id, 'DE_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.grid_state = self.sampler.population[self.grid_idx]

        # Shared with the master, mutated in place
        self.parent_pool = parent_pool
        self.pbest_archive = pbest_archive
        self.successful_F_list = successful_F_list
        self.successful_CR_list = successful_CR_list

        self.pop_size = self.sampler.pop_per_grid_point
        self.evals_remaining = self.pop_size

        self.trial_info = {} # {point_idx: (trial_params, F_i, CR_i)}


    def start(self):
        """Generate all trial points and return their evaluation tasks."""
        if self.sampler.direct_eval_mode or self.sampler.n_prof_dims == 0:
            self.success = True
            self._is_finished = True
            return []

        tasks = []
        grid_state = self.grid_state

        for i in range(self.pop_size):
            mem_loc = np.random.randint(0, self.sampler.memory_size)
            mu_CR, mu_F = self.sampler.memory_CR[mem_loc], self.sampler.memory_F[mem_loc]

            CR_i = np.clip(norm.rvs(loc=mu_CR, scale=DE_CR_NORMAL_SCALE), 0, 1)
            F_i = cauchy.rvs(loc=mu_F, scale=DE_F_CAUCHY_SCALE)
            while F_i <= 0:
                F_i = cauchy.rvs(loc=mu_F, scale=DE_F_CAUCHY_SCALE)
            F_i = min(F_i, DE_F_MAX_VALUE)

            x_i_params = grid_state['profiled_params'][i]

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
                            best_neighbor_params = neighbor_state['profiled_params'][neighbor_best_idx]

                if best_neighbor_params is not None and best_neighbor_fitness > grid_state['best_fitness']:
                    use_neighbor_mutation = True

            if len(self.parent_pool) < DE_MIN_PARENT_POOL_SIZE:
                # Not enough parents to mutate; count this slot as resolved so
                # the job can finish even if every iteration hits this branch.
                self.evals_remaining -= 1
                continue

            mutant = None
            if use_neighbor_mutation:
                r2_p, r3_p = np.random.choice(self.parent_pool, 2, replace=False)
                r2, r3 = r2_p['profiled_params'], r3_p['profiled_params']
                mutant = x_i_params + F_i * (best_neighbor_params - x_i_params) + F_i * (r2 - r3)

            elif self.sampler.mutation_strategy == 'current-to-rand/1':
                p1_p, p2_p, p3_p = np.random.choice(self.parent_pool, 3, replace=False)
                p1, p2, p3 = p1_p['profiled_params'], p2_p['profiled_params'], p3_p['profiled_params']
                mutant = x_i_params + F_i * (p1 - x_i_params) + F_i * (p2 - p3)

            elif self.sampler.mutation_strategy == 'rand/1':
                r1_p, r2_p, r3_p = np.random.choice(self.parent_pool, 3, replace=False)
                r1, r2, r3 = r1_p['profiled_params'], r2_p['profiled_params'], r3_p['profiled_params']
                mutant = r1 + F_i * (r2 - r3)

            elif self.sampler.mutation_strategy == 'current-to-pbest/1':
                archive = self.pbest_archive if self.pbest_archive else self.parent_pool
                x_pbest_p = np.random.choice(archive)
                x_pbest = x_pbest_p['profiled_params']

                potential_diff = self.parent_pool
                if len(potential_diff) < 2:
                    self.evals_remaining -= 1
                    continue

                max_attempts = 10
                for _ in range(max_attempts):
                    r2_p, r3_p = np.random.choice(potential_diff, 2, replace=False)
                    if not (np.array_equal(r2_p['profiled_params'], x_pbest) or
                            np.array_equal(r3_p['profiled_params'], x_pbest)):
                        break

                r2, r3 = r2_p['profiled_params'], r3_p['profiled_params']
                mutant = x_i_params + F_i * (x_pbest - x_i_params) + F_i * (r2 - r3)

            if mutant is None:
                self.evals_remaining -= 1
                continue

            cross_points = np.random.rand(self.sampler.n_prof_dims) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.sampler.n_prof_dims)] = True
            trial_params = np.where(cross_points, mutant, x_i_params)

            trial_params = self.sampler._ensure_bounds(trial_params, self.sampler.profiled_dims)

            full_trial_params = self.sampler._construct_params(self.grid_idx, trial_params)

            self.trial_info[i] = (trial_params, F_i, CR_i)

            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i,
            }
            task = {
                'params': full_trial_params,
                'context': context,
            }
            tasks.append(task)

        if not tasks and self.evals_remaining == 0:
            self.success = True
            self._is_finished = True

        return tasks

    def process_result(self, result):
        """Compare trial fitness with target and store successful F/CR."""
        point_idx = result['context']['point_idx']
        trial_fitness = result['target_val']

        if point_idx not in self.trial_info:
            logger.warning(f"Warning: Received result for DE point_idx {point_idx} with no trial info. Ignoring.")
            self.evals_remaining -= 1
            if self.evals_remaining <= 0:
                self.success = True
                self._is_finished = True
            return []

        trial_params, F_i, CR_i = self.trial_info[point_idx]
        grid_state = self.grid_state

        if trial_fitness > grid_state['fitnesses'][point_idx]:
            grid_state['profiled_params'][point_idx] = trial_params
            grid_state['fitnesses'][point_idx] = trial_fitness
            self.successful_F_list.append(F_i)
            self.successful_CR_list.append(CR_i)

        self.evals_remaining -= 1
        if self.evals_remaining <= 0:
            self.success = True
            self._is_finished = True

        return []

    def on_finish(self, next_job_id):
        """Update best_fitness and history; spawn an LBFGSB polish job if converged."""
        if not self.success:
            return None

        if self.sampler.direct_eval_mode or self.sampler.n_prof_dims == 0:
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

            best_idx = np.argmax(grid_state['fitnesses'])
            best_profiled_params = grid_state['profiled_params'][best_idx]
            full_params = self.sampler._construct_params(self.grid_idx, best_profiled_params)
            self.sampler._update_global_pool(full_params, new_best_fitness, self.grid_idx)

        if grid_state['status'] == 'active' and \
           len(grid_state['improvement_history']) == self.sampler.convergence_window:

            avg_improvement = np.mean(grid_state['improvement_history'])
            if avg_improvement < self.sampler.convergence_threshold:
                if self.sampler.lbfgsb_polish:
                    return self.sampler.create_LBFGSB_job_for_point(self.grid_idx, next_job_id)
                else:
                    grid_state['status'] = 'optimized'
                    return None

        return None
