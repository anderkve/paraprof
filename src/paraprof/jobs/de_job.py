"""
Differential Evolution job for grid point evolution.
"""
import numpy as np
from scipy.stats import cauchy, norm
from ..logger import get_logger
from .base import Job

logger = get_logger()


# DE mutation and crossover constants
DE_CR_NORMAL_SCALE = 0.1
"""Standard deviation for sampling CR from normal distribution around memory value"""

DE_F_CAUCHY_SCALE = 0.1
"""Scale parameter for sampling F from Cauchy distribution around memory value"""

DE_F_MAX_VALUE = 1.0
"""Maximum allowed value for mutation factor F"""

DE_MIN_PARENT_POOL_SIZE = 3
"""Minimum number of parents required in pool to perform DE mutation"""

# Try to import emulator utilities
try:
    from ..emulator_utils import prepare_emulator_cache_for_worker
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False
    logger.debug("Emulator utilities not available")


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

        # Track pre-screening statistics
        self.trials_generated = 0
        self.trials_screened_out = 0


    def start(self):
        """Generate all trial points and return their evaluation tasks."""
        # Direct evaluation mode: no continuous dimensions, so no evolution needed
        if self.sampler.direct_eval_mode or self.sampler.n_cont_dims == 0:
            # Grid point already evaluated by ActivationJob, mark as converged
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

            if len(self.parent_pool) < DE_MIN_PARENT_POOL_SIZE:
                continue  # Not enough parents to mutate

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

                potential_diff = self.parent_pool
                if len(potential_diff) < 2:
                    continue

                max_attempts = 10
                for _ in range(max_attempts):
                    r2_p, r3_p = np.random.choice(potential_diff, 2, replace=False)
                    if not (np.array_equal(r2_p['continuous_params'], x_pbest) or
                            np.array_equal(r3_p['continuous_params'], x_pbest)):
                        break
                # Use r2_p, r3_p even if they match (negligible probability with large pools)

                r2, r3 = r2_p['continuous_params'], r3_p['continuous_params']
                mutant = x_i_params + F_i * (x_pbest - x_i_params) + F_i * (r2 - r3)

            if mutant is None:
                self.evals_remaining -= 1 # This individual won't be evaluated
                continue

            # Crossover first, then apply bounds once (avoids intermediate array)
            cross_points = np.random.rand(self.sampler.n_cont_dims) < CR_i
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.sampler.n_cont_dims)] = True
            trial_params = np.where(cross_points, mutant, x_i_params)

            # Apply bounds after crossover
            trial_params = self.sampler._ensure_bounds(trial_params, self.sampler.continuous_dims)

            # Track trial generation
            self.trials_generated += 1

            # Construct full params
            full_trial_params = self.sampler._construct_params(self.grid_idx, trial_params)
            target_fitness = grid_state['fitnesses'][i]

            # Store info needed when result comes back
            self.trial_info[i] = (trial_params, F_i, CR_i)

            # === PREPARE EMULATOR DATA FOR WORKER-SIDE PRE-SCREENING ===
            emulator_cache = None
            if EMULATOR_AVAILABLE and getattr(self.sampler, 'use_de_prescreening', False):
                emulator_cache = prepare_emulator_cache_for_worker(
                    sampler=self.sampler,
                    center_params=full_trial_params,
                    min_points=self.sampler.emulator_min_neighbors,
                    max_points=getattr(self.sampler, 'emulator_max_neighbors', None),
                    grid_idx=self.grid_idx  # Pass grid index for local cache gathering
                )
            # === END EMULATOR PREPARATION ===

            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i,
                'target_fitness': target_fitness  # Worker needs this for UCB comparison
            }
            task = {
                'params': full_trial_params,
                'context': context,
                'emulator_cache': emulator_cache  # None if disabled or insufficient data
            }
            tasks.append(task)

        if not tasks and self.evals_remaining == 0:
            self.success = True
            self._is_finished = True

        # Update global statistics (trials_generated tracked here, screened_out tracked in process_result)
        self.sampler.de_trials_generated += self.trials_generated

        return tasks

    def process_result(self, result):
        """Compare trial fitness with target and store successful F/CR."""
        point_idx = result['context']['point_idx']

        # Check if trial was screened out by worker-side emulator
        was_screened = result.get('emulator_screened', False)
        if was_screened:
            # Worker rejected this trial - count it and skip
            self.trials_screened_out += 1
            self.sampler.de_trials_screened_out += 1

            self.evals_remaining -= 1
            if self.evals_remaining <= 0:
                self.success = True
                self._is_finished = True
                # Log pre-screening effectiveness for this job
                if self.trials_generated > 0 and self.sampler.use_de_prescreening:
                    screen_rate = 100 * self.trials_screened_out / self.trials_generated
                    logger.info(
                        f"DE job {self.id} (grid {self.grid_idx}): "
                        f"Screened out {self.trials_screened_out}/{self.trials_generated} "
                        f"trials ({screen_rate:.1f}%)"
                    )
            return []

        # Normal evaluation result
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
            # Log pre-screening effectiveness for this job
            if self.trials_generated > 0 and self.sampler.use_de_prescreening:
                screen_rate = 100 * self.trials_screened_out / self.trials_generated
                logger.info(
                    f"DE job {self.id} (grid {self.grid_idx}): "
                    f"Screened out {self.trials_screened_out}/{self.trials_generated} "
                    f"trials ({screen_rate:.1f}%)"
                )

        return [] # No new tasks

    def on_finish(self, next_job_id):
        """
        Update the best_fitness and history for this grid point.
        If converged, spawn a new LBFGSB job.
        """
        if not self.success:
            return None

        # Direct evaluation mode: already marked as converged by ActivationJob
        if self.sampler.direct_eval_mode or self.sampler.n_cont_dims == 0:
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

            # Update global solution pool with improved solution
            best_idx = np.argmax(grid_state['fitnesses'])
            best_continuous_params = grid_state['continuous_params'][best_idx]
            # Construct full parameter vector for the pool
            full_params = self.sampler._construct_params(self.grid_idx, best_continuous_params)
            self.sampler._update_global_pool(full_params, new_best_fitness, self.grid_idx)

        # Check for convergence
        if grid_state['status'] == 'active' and \
           len(grid_state['improvement_history']) == self.sampler.convergence_window:

            avg_improvement = np.mean(grid_state['improvement_history'])
            if avg_improvement < self.sampler.convergence_threshold:
                if self.sampler.lbfgsb_refinement:
                    logger.info(f"--- DE Converged for {self.grid_idx}. Spawning L-BFGS-B refinement job. ---")
                    return self.sampler.create_LBFGSB_job_for_point(self.grid_idx, next_job_id)
                else:
                    # Mark as optimized without L-BFGS-B refinement
                    grid_state['status'] = 'optimized'
                    logger.info(f"--- DE Converged for {self.grid_idx}. Marked as optimized (L-BFGS-B refinement disabled). ---")
                    return None

        return None
