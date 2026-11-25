"""Differential Evolution job for grid point evolution."""
import numpy as np
from scipy.stats import cauchy, norm
from ..logger import get_logger
from .base import Job

logger = get_logger()


# Each trial vector draws its own crossover rate CR (normal, clipped to
# [0, 1]) and scale factor F (Cauchy, redrawn while <= 0, capped at
# DE_F_MAX_VALUE) around these centres.
DE_CR_CENTRE = 0.5
DE_CR_NORMAL_SCALE = 0.1
DE_F_CENTRE = 0.7
DE_F_CAUCHY_SCALE = 0.1
DE_F_MAX_VALUE = 1.0
DE_MIN_PARENT_POOL_SIZE = 3


class DEGridPointJob(Job):
    """Run one DE generation (current-to-pbest/1) for a single grid point."""
    def __init__(self, job_id, sampler, grid_idx, parent_pool, pbest_archive):

        super().__init__(job_id, 'DE_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.grid_state = self.sampler.population[self.grid_idx]

        # Shared resources from master
        self.parent_pool = parent_pool
        self.pbest_archive = pbest_archive

        self.pop_size = self.sampler.pop_per_grid_point
        self.evals_remaining = self.pop_size

        # Trial vectors by population slot, to process results
        self.trial_params = {}


    def start(self):
        """Generate all trial points and return their evaluation tasks."""
        if self.sampler.direct_eval_mode or self.sampler.n_prof_dims == 0:
            self.success = True
            self._is_finished = True
            return []

        tasks = []
        grid_state = self.grid_state
        sampler = self.sampler

        def diff(a, b):
            # a - b; the short way round for a periodic parameter.
            return sampler._param_delta(a, b, sampler.profiled_dims)

        for i in range(self.pop_size):
            CR_i = np.clip(norm.rvs(loc=DE_CR_CENTRE, scale=DE_CR_NORMAL_SCALE), 0, 1)
            F_i = cauchy.rvs(loc=DE_F_CENTRE, scale=DE_F_CAUCHY_SCALE)
            while F_i <= 0:
                F_i = cauchy.rvs(loc=DE_F_CENTRE, scale=DE_F_CAUCHY_SCALE)
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

            if use_neighbor_mutation:
                # Pull towards the best neighbour instead of a p-best parent.
                r2_p, r3_p = np.random.choice(self.parent_pool, 2, replace=False)
                r2, r3 = r2_p['profiled_params'], r3_p['profiled_params']
                mutant = x_i_params + F_i * diff(best_neighbor_params, x_i_params) + F_i * diff(r2, r3)
            else:
                archive = self.pbest_archive if self.pbest_archive else self.parent_pool
                x_pbest_p = np.random.choice(archive)
                x_pbest = x_pbest_p['profiled_params']

                max_attempts = 10
                for _ in range(max_attempts):
                    r2_p, r3_p = np.random.choice(self.parent_pool, 2, replace=False)
                    if not (np.array_equal(r2_p['profiled_params'], x_pbest) or
                            np.array_equal(r3_p['profiled_params'], x_pbest)):
                        break

                r2, r3 = r2_p['profiled_params'], r3_p['profiled_params']
                mutant = x_i_params + F_i * diff(x_pbest, x_i_params) + F_i * diff(r2, r3)

            cross_points = np.random.rand(self.sampler.n_prof_dims) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.sampler.n_prof_dims)] = True
            trial_params = np.where(cross_points, mutant, x_i_params)

            trial_params = self.sampler._ensure_bounds(trial_params, self.sampler.profiled_dims)

            full_trial_params = self.sampler._construct_params(self.grid_idx, trial_params)

            self.trial_params[i] = trial_params

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
        """Keep the trial vector if it beats the member it was made from."""
        point_idx = result['context']['point_idx']
        trial_fitness = result['target_val']

        if point_idx not in self.trial_params:
            logger.warning(f"Warning: Received result for DE point_idx {point_idx} with no trial vector. Ignoring.")
            self.evals_remaining -= 1
            if self.evals_remaining <= 0:
                self.success = True
                self._is_finished = True
            return []

        grid_state = self.grid_state

        if trial_fitness > grid_state['fitnesses'][point_idx]:
            grid_state['profiled_params'][point_idx] = self.trial_params[point_idx]
            grid_state['fitnesses'][point_idx] = trial_fitness

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

            if new_best_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = new_best_fitness

            best_idx = np.argmax(grid_state['fitnesses'])
            best_profiled_params = grid_state['profiled_params'][best_idx]
            full_params = self.sampler._construct_params(self.grid_idx, best_profiled_params)
            self.sampler._update_global_pool(full_params, new_best_fitness, self.grid_idx)

        # Effective convergence window: allow_early_DE_exit cells use a reduced
        # one (set in create_de_generation_jobs). `>=` + slice matches the old
        # `==`/full-deque for the default window (capped at convergence_window).
        eff_window = grid_state.get('conv_window') or self.sampler.convergence_window
        if grid_state['status'] == 'active' and \
           len(grid_state['improvement_history']) >= eff_window:

            avg_improvement = np.mean(list(grid_state['improvement_history'])[-eff_window:])
            if avg_improvement < self.sampler.convergence_threshold:
                if self.sampler.lbfgsb_polish:
                    return self.sampler.create_LBFGSB_job_for_point(self.grid_idx, next_job_id)
                else:
                    grid_state['status'] = 'done'
                    return None

        return None
