import numpy as np
import itertools
import collections
import time
from scipy.optimize import minimize
from scipy.stats import cauchy, norm
from scipy.stats.qmc import LatinHypercube as LHS
import os
import sys # Added sys for exit

# --- MPI Setup ---
# The code is designed to be run with MPI.
# e.g., mpiexec -n <number_of_cores> python test_103_MPI_complete.py
try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Task Definitions (for communication between master and workers) ---
TASK_TERMINATE = -1
TASK_LIKELIHOOD_EVAL = 1


# --- Job Base Class ---

class Job:
    """
    Abstract base class for a job.
    A job is a self-contained unit of work that can be broken down into
    one or more tasks (likelihood evaluations).
    """
    def __init__(self, job_id, job_type, sampler):
        self.id = job_id
        self.type = job_type
        self.sampler = sampler  # Reference to the main sampler state object
        self._is_finished = False
        self.success = False

    def start(self):
        """
        Returns the initial list of tasks to be queued.
        Each task is a dict: {'params': full_params, 'context': context}
        """
        raise NotImplementedError

    def process_result(self, result):
        """
        Processes a worker result associated with this job.
        Returns a list of new tasks to be queued (can be empty).
        """
        raise NotImplementedError

    def is_finished(self):
        """Returns True if the job is complete."""
        return self._is_finished

    def on_finish(self, next_job_id):
        """
        Called by the master when the job is finished.
        Use this to update the main sampler's state.
        Can optionally return (new_job, next_job_id) to spawn a child job.
        """
        pass  # Optional
        return None


# --- Concrete Job Implementations ---

class LBFGSBJob(Job):
    """
    A self-contained job for running an asynchronous L-BFGS-B optimization.
    This single class handles all the logic for initial fitness evaluation,
    gradient calculation, line searching, and history updates.
    
    For REFINEMENT jobs, it includes logic to test a neighbor's parameters
    and seed the Hessian (s_hist, y_hist) from a converged neighbor.
    """
    def __init__(self, job_id, job_type, sampler, opt_dims, start_params, 
                 grid_idx, start_params_full, seed_history=None, start_fitness=-np.inf):
        
        super().__init__(job_id, job_type, sampler)
        
        # L-BFGS-B parameters from sampler
        self.refinement_ftol = sampler.refinement_ftol
        self.refinement_max_iter = sampler.refinement_max_iter
        self.refinement_gradient_method = sampler.refinement_gradient_method

        # Job-specific state
        self.grid_idx = grid_idx
        self.opt_dims = opt_dims # Dimensions to optimize (relative to full params)
        self.n_opt_dims = len(opt_dims)
        
        # start_params are the *partial* parameters corresponding to opt_dims
        self.start_params_partial = start_params
        self.start_params_full = start_params_full # Full params for *initial* eval
        self.start_fitness = start_fitness # The fitness of start_params_partial
        self.improvement = 0.0 # For patching

        # L-BFGS-B internal state
        if self.type == 'REFINEMENT':
            self.status = 'NEEDS_NEIGHBOR_TEST'
            self.fallback_params = self.start_params_partial # Own best params
        else:
            self.status = 'NEEDS_INITIAL_F'
            
        self.current_params = self.start_params_partial
        self.current_fitness = -np.inf # Likelihood value (maximization)
        self.current_objective = np.inf # Objective function value (minimization)
        
        self.gradient_components = {}
        self.pending_grad_evals = 0
        self.current_gradient = None
        
        self.s_hist = collections.deque(maxlen=10)
        self.y_hist = collections.deque(maxlen=10)
        if seed_history:
            self.s_hist.extend(seed_history['s'])
            self.y_hist.extend(seed_history['y'])
            
        self.iteration = 0
        
        self.search_direction = None
        self.line_search_alpha = 1.0
        self.pending_s_k = None
        self.pending_g_old = None

        # For neighbor test
        self.neighbor_params_to_test = None

    def _get_full_params(self, partial_params):
        """Constructs full parameters from partial optimization parameters."""
        if self.grid_idx is None:
            # Global optimization: partial_params are already full_params
            return partial_params
        else:
            # Grid-anchored optimization
            return self.sampler._construct_params(self.grid_idx, partial_params)

    def _get_partial_params_from_full(self, full_params):
        """Extracts optimization parameters from a full parameter vector."""
        return full_params[list(self.opt_dims)]

    def _construct_full_params_for_task(self, partial_params_to_eval):
        """
        Creates the full parameter vector for a task, handling
        global (grid_idx=None) vs grid-anchored optimization.
        """
        if self.grid_idx is None:
            # Global optimization: partial_params_to_eval is the full vector
            # We must ensure it's bounded.
            return self.sampler._ensure_bounds(partial_params_to_eval, self.opt_dims)
        else:
            # Grid-anchored: partial_params_to_eval is *only* the continuous dims
            # We must ensure *they* are bounded.
            bounded_partial = self.sampler._ensure_bounds(
                partial_params_to_eval, 
                self.sampler.continuous_dims
            )
            return self.sampler._construct_params(self.grid_idx, bounded_partial)


    def start(self):
        """Returns the first task(s) for the job."""
        
        if self.status == 'NEEDS_NEIGHBOR_TEST':
            # --- Logic from serial _refine_single_point ---
            best_neighbor_state = None
            best_neighbor_fitness = -np.inf
            
            for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
                if neighbor_idx in self.sampler.population:
                    neighbor_state = self.sampler.population[neighbor_idx]
                    # Check if neighbor is 'refined' (post-LBFGSB) or 'converged' (post-DE)
                    # and has an optimizer state to seed from.
                    if (neighbor_state['status'] in ['refined', 'converged']) and \
                       (neighbor_state.get('optimizer_state') is not None):
                        
                        if neighbor_state['best_fitness'] > best_neighbor_fitness:
                            best_neighbor_fitness = neighbor_state['best_fitness']
                            best_neighbor_state = neighbor_state
            
            if best_neighbor_state:
                # 1. Seed the history
                self.s_hist.clear()
                self.y_hist.clear()
                self.s_hist.extend(best_neighbor_state['optimizer_state']['s'])
                self.y_hist.extend(best_neighbor_state['optimizer_state']['y'])
                
                # 2. Get neighbor's best params to test
                neighbor_best_idx = np.argmax(best_neighbor_state['fitnesses'])
                self.neighbor_params_to_test = best_neighbor_state['continuous_params'][neighbor_best_idx]
                
                # 3. Create a task to test these params at *our* grid point
                full_params_test = self.sampler._construct_params(self.grid_idx, self.neighbor_params_to_test)
                
                context = {
                    'type': self.type, 
                    'job_id': self.id, 
                    'sub_type': 'LBFGS_NEIGHBOR_TEST'
                }
                return [{'params': full_params_test, 'context': context}]

            else:
                # No valid neighbor found, skip test and just evaluate our own params
                self.status = 'NEEDS_INITIAL_F'
                # Fall through to the 'NEEDS_INITIAL_F' logic

        if self.status == 'NEEDS_INITIAL_F':
            # This is the fallback: evaluate the starting params
            # (either for global opt, or for refinement with no neighbors)
            context = {
                'type': self.type, 
                'job_id': self.id, 
                'sub_type': 'LBFGS_INITIAL_F'
            }
            # self.start_params_full was constructed by the factory
            return [{'params': self.start_params_full, 'context': context}]

        # Should not be reached
        return []

    def process_result(self, result):
        """Main dispatcher for L-BFGS-B state machine."""
        context = result['context']
        sub_type = context.get('sub_type', 'NONE')
        new_tasks = []

        if self.status == 'NEEDS_NEIGHBOR_TEST' and sub_type == 'LBFGS_NEIGHBOR_TEST':
            neighbor_fitness = result['target_val']
            
            if neighbor_fitness > self.start_fitness:
                # Neighbor's params are a better starting point
                self.current_params = self.neighbor_params_to_test
                self.current_fitness = neighbor_fitness
                self.current_objective = -neighbor_fitness
            else:
                # Our original params are better
                self.current_params = self.fallback_params
                self.current_fitness = self.start_fitness
                self.current_objective = -self.start_fitness
                
            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks()

        elif self.status == 'NEEDS_INITIAL_F' and sub_type == 'LBFGS_INITIAL_F':
            # Came here from global opt or refinement with no neighbors
            self.current_fitness = result['target_val']
            self.current_objective = -self.current_fitness
            self.current_params = self._get_partial_params_from_full(result['params'])
            
            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks()

        elif self.status == 'NEEDS_GRADIENT' and sub_type == 'LBFGS_GRADIENT':
            # This method will collect gradient components and,
            # when complete, calculate the search direction and return line search tasks.
            new_tasks = self._process_gradient_result(result)
            
        elif self.status == 'NEEDS_LINE_SEARCH' and sub_type == 'LBFGS_LINE_SEARCH':
            # This method will check Armijo, and either
            # 1. Accept step, calc new gradient (returns gradient tasks)
            # 2. Reject step, reduce alpha (returns new line search task)
            # 3. Fail (returns no tasks, sets status to FINISHED)
            new_tasks = self._process_line_search_result(result)
        
        return new_tasks

    def _calculate_gradient_tasks(self, eps=1e-8):
        """Generates tasks needed to numerically calculate the gradient."""
        tasks = []
        x = self.current_params
        self.gradient_components = {} # Clear old components
            
        if self.refinement_gradient_method == "central":
            self.pending_grad_evals = 2 * self.n_opt_dims
            for i in range(self.n_opt_dims):
                # Positive step
                x_plus = x.copy()
                x_plus[i] += eps
                full_params_plus = self._construct_full_params_for_task(x_plus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': 1}
                tasks.append({'params': full_params_plus, 'context': context})
                
                # Negative step
                x_minus = x.copy()
                x_minus[i] -= eps
                full_params_minus = self._construct_full_params_for_task(x_minus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': -1}
                tasks.append({'params': full_params_minus, 'context': context})
                
        elif self.refinement_gradient_method == "forward":
            self.pending_grad_evals = self.n_opt_dims
            for i in range(self.n_opt_dims):
                x_plus = x.copy()
                x_plus[i] += eps
                full_params_plus = self._construct_full_params_for_task(x_plus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': 1}
                tasks.append({'params': full_params_plus, 'context': context})
        else:
            raise Exception(f"Gradient method {self.refinement_gradient_method} not implemented.")
                
        return tasks

    def _process_gradient_result(self, result):
        """Processes a returned likelihood evaluation for a gradient calculation."""
        context = result['context']
        dim, sign = context['dim'], context['sign']
        
        self.gradient_components[(dim, sign)] = -result['target_val'] # Store objective
        self.pending_grad_evals -= 1

        if self.pending_grad_evals < 0:
            raise Exception("LBFGSBJob: pending_grad_evals < 0. This should not happen.")
        
        # Check if all components for the gradient have been computed
        if self.pending_grad_evals == 0:
            grad = np.zeros(self.n_opt_dims)
            f = self.current_objective
            
            if self.refinement_gradient_method == "central":
                for i in range(self.n_opt_dims):
                    f_plus = self.gradient_components[(i, 1)]
                    f_minus = self.gradient_components[(i, -1)]
                    grad[i] = (f_plus - f_minus) / (2 * 1e-8)
            elif self.refinement_gradient_method == "forward":
                 for i in range(self.n_opt_dims):
                    f_plus = self.gradient_components[(i, 1)]
                    grad[i] = (f_plus - f) / 1e-8
            
            self.current_gradient = grad
            
            # --- History update (if pending from line search) ---
            if self.pending_s_k is not None:
                s_k = self.pending_s_k
                g_old = self.pending_g_old
                g_new = self.current_gradient
                y_k = g_new - g_old
                if np.dot(y_k, s_k) > 1e-10:
                    self.s_hist.append(s_k)
                    self.y_hist.append(y_k)
                # Clear pending
                self.pending_s_k = None
                self.pending_g_old = None
            # --- End History update ---
            
            # --- L-BFGS two-loop recursion to find search direction ---
            q = grad
            a = []
            s_hist, y_hist = self.s_hist, self.y_hist
            
            for s, y in zip(reversed(s_hist), reversed(y_hist)):
                rho = 1.0 / np.dot(y, s)
                alpha = rho * np.dot(s, q)
                q = q - alpha * y
                a.append(alpha)
            
            if s_hist:
                gamma = np.dot(s_hist[-1], y_hist[-1]) / np.dot(y_hist[-1], y_hist[-1])
                z = gamma * q
            else:
                z = q

            for (s, y), alpha in zip(zip(s_hist, y_hist), reversed(a)):
                rho = 1.0 / np.dot(y, s)
                beta = rho * np.dot(y, z)
                z = z + s * (alpha - beta)
                
            self.search_direction = -z
            self.status = 'NEEDS_LINE_SEARCH'
            self.line_search_alpha = 1.0 # Reset for new line search
            
            # Return a new task for the first step of the line search
            return [self._calculate_line_search_task()]
        
        return [] # Not ready yet, no new task

    def _calculate_line_search_task(self):
        """Generates the next task for a backtracking line search."""
        alpha = self.line_search_alpha
        x = self.current_params
        d = self.search_direction
        x_new = x + alpha * d
        
        # We must construct the *full* params for the task
        full_params_new = self._construct_full_params_for_task(x_new)
        
        context = {
            'type': self.type, 
            'job_id': self.id, 
            'sub_type': 'LBFGS_LINE_SEARCH', 
            'alpha': alpha
        }
        return {'params': full_params_new, 'context': context}

    def _process_line_search_result(self, result):
        """Processes a line search result and determines the next step."""
        f_new = -result['target_val'] # Objective value
        alpha = result['context']['alpha']
        
        x_old = self.current_params
        f_old = self.current_objective
        g_old = self.current_gradient
        d = self.search_direction
        c1 = 1e-4

        # Re-calculate x_new based on alpha
        x_new = x_old + alpha * d
        
        opt_indices = self.opt_dims
        if self.grid_idx is not None:
            opt_indices = self.sampler.continuous_dims
            
        x_new_bounded = self.sampler._ensure_bounds(x_new, opt_indices)

        # Armijo condition check
        if f_new <= f_old + c1 * alpha * np.dot(g_old, x_new_bounded - x_old):
            # Step accepted, move to the next L-BFGS iteration
            self.iteration += 1
            
            # Check for convergence
            if self.iteration >= self.refinement_max_iter or np.abs(f_old - f_new) < self.refinement_ftol:
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                self.current_params = x_new_bounded # Save final params
                self.current_fitness = -f_new       # Save final fitness
                return [] # Job is done

            # --- Not converged, prepare for next iteration ---
            
            # Store s_k and g_old so we can update history *after* new gradient is computed
            self.pending_s_k = x_new_bounded - x_old
            self.pending_g_old = g_old

            # Update state for next iteration
            self.current_params = x_new_bounded
            self.current_fitness = -f_new
            self.current_objective = f_new
            
            # Generate tasks to calculate the new gradient
            self.status = 'NEEDS_GRADIENT'
            return self._calculate_gradient_tasks()

        else:
            # Step not accepted, reduce alpha and try again
            self.line_search_alpha *= 0.5
            if self.line_search_alpha < 1e-10: # Failsafe
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = False # Line search failed
                return [] # Job is done
            
            return [self._calculate_line_search_task()]

    def on_finish(self, next_job_id):
        """Finalize a job, updating the sampler state."""
        
        if self.success:
            # For patching, record the improvement
            self.improvement = self.current_fitness - self.start_fitness
        
        if not self.success:
            # For refinement, if it fails, set status back to 'converged'
            # so it can be picked up by patching later.
            if self.type == 'REFINEMENT' and self.grid_idx in self.sampler.population:
                self.sampler.population[self.grid_idx]['status'] = 'converged'
            return None # Don't record failed jobs

        if self.type == 'INITIAL_OPTIMIZATION':
            final_params = self._construct_full_params_for_task(self.current_params)
            final_target_val = self.current_fitness
            
            self.sampler.initial_maxima.append({'point': final_params, 'target_val': final_target_val})
            if final_target_val > self.sampler.global_max_target_val: 
                self.sampler.global_max_target_val = final_target_val

        elif self.type == 'REFINEMENT':
            grid_idx = self.grid_idx
            if grid_idx in self.sampler.population:
                state = self.sampler.population[grid_idx]
                state['optimizer_state'] = {'s': list(self.s_hist), 'y': list(self.y_hist)}
                state['status'] = 'refined' # Mark as fully complete
                
                # Update the best individual with the refined result
                if self.current_fitness > state['best_fitness']:
                     state['best_fitness'] = self.current_fitness
                     best_idx = np.argmax(state['fitnesses'])
                     state['continuous_params'][best_idx] = self.current_params
                     state['fitnesses'][best_idx] = self.current_fitness
                     self.sampler.profile_likelihood_grid[self.grid_idx] = self.current_fitness
                     
                     if self.current_fitness > self.sampler.global_max_target_val:
                         self.sampler.global_max_target_val = self.current_fitness
        
        return None # This job doesn't spawn children


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


# --- Main Sampler Class (Refactored to hold state) ---
class GridAnchoredDESampler:
    def __init__(self,
                 target_func,
                 bounds,
                 projections,
                 pop_per_grid_point=1,
                 mutation_strategy='current-to-rand/1',
                 pbest_fraction=0.1,
                 n_initial_optimizations=20,
                 roi_threshold=3.0,
                 convergence_threshold=1e-5,
                 convergence_window=25,
                 neighbor_pull_probability=0.3,
                 refinement_ftol=1e-7,
                 refinement_max_iter=50,
                 refinement_gradient_method="central",
                 patching_fraction=0.1,
                 patching_conv_threshold=0.01,
                 max_patching_iterations=100,
                 memory_size=100,
                 samples_output_file=None):
        """
        Initializes the Grid-Anchored DE Sampler.
        This class now primarily holds state and configuration.
        The execution logic is in the Job classes and master_main.
        """
        self.target_func = target_func
        self.bounds = np.array(bounds)
        self.dims = len(self.bounds)
        self.projections = projections

        # --- Algorithm parameters ---
        self.pop_per_grid_point = pop_per_grid_point
        allowed_strategies = ['current-to-rand/1', 'rand/1', 'current-to-pbest/1']
        if mutation_strategy not in allowed_strategies:
            raise ValueError(f"mutation_strategy must be one of {allowed_strategies}")
        self.mutation_strategy = mutation_strategy
        self.pbest_fraction = pbest_fraction

        self.n_initial_optimizations = n_initial_optimizations
        self.roi_threshold = roi_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.neighbor_pull_probability = neighbor_pull_probability
        self.refinement_ftol = refinement_ftol
        self.refinement_max_iter = refinement_max_iter
        self.refinement_gradient_method = refinement_gradient_method
        self.patching_fraction = patching_fraction
        self.patching_conv_threshold = patching_conv_threshold
        self.max_patching_iterations = max_patching_iterations
        self.memory_size = memory_size
        
        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        if self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000

        # --- Persistent State (across projections) ---
        self.target_calls = 0
        self.global_max_target_val = -np.inf
        
        # --- Per-Projection State (will be reset) ---
        self.projection_dims = None
        self.grid_points_per_dim = None
        self.initial_maxima = []
        self.population = {} # {grid_idx: state_dict}
        self.active_grid_indices = set()
        self.pending_activation_indices = set() # For dynamic activation
        self.current_generation = 0 # DE generation
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

        # --- Per-Projection State (reset) ---
        self._reset_for_new_projection(self.projections[0])


    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        print("\n" + "="*80)
        print(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        print("="*80 + "\n")

        self.projection_dims = sorted(projection_config['dims'])
        # _Anders: Add +1 to number of grid points, to get nicer coordinates
        grid_points = list(projection_config['grid_points']) # Copy
        for i in range(len(grid_points)):
            grid_points[i] += 1
        self.grid_points_per_dim = grid_points

        if len(self.projection_dims) != len(self.grid_points_per_dim):
            raise ValueError("Length of projection_dims must match length of grid_points_per_dim.")
        if any(d >= self.dims for d in self.projection_dims):
            raise ValueError("projection_dims contains an index out of bounds.")

        self.continuous_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

        self.grid_shape = tuple(self.grid_points_per_dim)

        self.grid_axes = [np.linspace(self.bounds[d, 0], self.bounds[d, 1], n) for d, n in zip(self.projection_dims, self.grid_points_per_dim)]
        self.profile_likelihood_grid = {} # Use a dict for sparse grid

        # Reset state variables
        self.initial_maxima = []
        self.population = {}
        self.active_grid_indices = set()
        self.pending_activation_indices = set()
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0


    def _flush_samples_buffer(self):
        """Writes the content of the samples buffer to the output file."""
        if not self.samples_output_file or not self.samples_buffer:
            return
        
        try:
            with open(self.samples_output_file, 'a') as f:
                for params, target_val in self.samples_buffer:
                    param_str = ", ".join([f"{p:.6e}" for p in params])
                    f.write(f"{param_str}, {target_val:.6e}\n")
            
            self.samples_buffer = []
        except IOError as e:
            print(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val):
        """Registers a completed target call (only on master)."""
        self.target_calls += 1
        if hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, target_val))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()
        # Updating global max is now handled by the jobs
        # to ensure it happens at the right time (e.g., after refinement).


    def _get_grid_indices_from_point(self, point, grid_axes=None):
        """Converts a point's projection coordinates to the closest grid indices."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        
        grid_coords = point[self.projection_dims]
        indices = []
        for i, coord in enumerate(grid_coords):
            axis = grid_axes[i]
            index = np.argmin(np.abs(axis - coord))
            indices.append(index)
        return tuple(indices)


    def _get_grid_coords_from_indices(self, grid_idx, grid_axes=None):
        """Converts grid indices to projection parameter values."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])


    def _construct_params(self, grid_idx, continuous_params, grid_axes=None):
        """Constructs a full parameter vector from grid and continuous parts."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes)
        full_params[self.continuous_dims] = continuous_params
        return full_params


    def _ensure_bounds(self, vec, dims_to_check):
        """Ensures a vector's components are within the defined bounds."""
        # Ensure dims_to_check is a list or array of indices
        dims_to_check = np.array(dims_to_check, dtype=int)
        if vec.shape != self.bounds[dims_to_check, 0].shape:
             # This happens in global optimization, vec is (N_dims,)
             # but dims_to_check might be smaller.
             # We only want to clip the dimensions specified.
             clipped_vec = vec.copy()
             for i, dim_idx in enumerate(dims_to_check):
                 clipped_vec[i] = np.clip(vec[i], self.bounds[dim_idx, 0], self.bounds[dim_idx, 1])
             return clipped_vec
        else:
             return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])


    def _get_valid_neighbors(self, grid_idx, include_center=False):
        """Generator to yield valid neighbor indices for a given grid point."""
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue
            
            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))
            
            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                yield neighbor_idx


    # --- Job Factory Methods (Master Only) ---

    def create_initial_optimization_jobs(self, next_job_id):
        """Generates L-BFGS-B jobs for finding initial maxima."""
        print(f"--- Generating {self.n_initial_optimizations} initial optimization jobs ---")
        jobs = []
        sampler = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = sampler.random(n=self.n_initial_optimizations)
        start_points = self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])
        
        for i, start_point in enumerate(start_points):
            job = LBFGSBJob(
                job_id=next_job_id,
                job_type='INITIAL_OPTIMIZATION',
                sampler=self,
                opt_dims=tuple(range(self.dims)), # Global optimization
                start_params=start_point,          # Full vector
                grid_idx=None,                     # No grid anchor
                start_params_full=start_point,     # Full vector
                seed_history=None
            )
            jobs.append(job)
            next_job_id += 1
            
        return jobs, next_job_id


    def create_activation_jobs(self, next_job_id):
        """Generates ActivationJobs for grid points near found maxima."""
        if not self.initial_maxima:
            print("Warning: No initial maxima found. Cannot create activation jobs.")
            return [], next_job_id

        jobs = []
        activation_job_created_for_grid_points = set()
        
        for maximum in self.initial_maxima:
            point = maximum['point']
            grid_idx = self._get_grid_indices_from_point(point)

            if grid_idx in activation_job_created_for_grid_points:
                continue

            for neighbor_idx in self._get_valid_neighbors(grid_idx, include_center=True):
                if (neighbor_idx in activation_job_created_for_grid_points) or \
                   (neighbor_idx in self.population):
                    continue

                job = ActivationJob(
                    job_id=next_job_id,
                    sampler=self,
                    grid_idx=neighbor_idx,
                    warm_start_params=point[self.continuous_dims] # Warm start
                )
                jobs.append(job)
                activation_job_created_for_grid_points.add(neighbor_idx)
                self.pending_activation_indices.add(neighbor_idx)
                next_job_id += 1
        
        print(f"--- Generating {len(jobs)} activation jobs ---")
        return jobs, next_job_id
        

    def create_de_generation_jobs(self, next_job_id, max_num_to_evolve):
        """Generates all DEGridPointJobs for one generation."""
        
        successful_F = []
        successful_CR = []
        
        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']
        
        if not unconverged_indices:
            print("All active points have converged. Ending DE phase.")
            return [], next_job_id, successful_F, successful_CR

        # --- Prioritize which grid points to evolve ---
        priority_scores = []
        for idx in unconverged_indices:
            state = self.population[idx]
            fitness_score = max(0, state['best_fitness'] - (self.global_max_target_val - 2 * self.roi_threshold))
            improvement_rate = np.mean(state['improvement_history']) if state['improvement_history'] else 0
            improvement_score = improvement_rate * 10
            pri_score = fitness_score + improvement_score + (1./len(unconverged_indices))
            priority_scores.append(pri_score)
        
        priority_scores = np.array(priority_scores)
        probabilities = priority_scores / np.sum(priority_scores)

        if max_num_to_evolve is not None:
            num_to_evolve = min(len(unconverged_indices), max_num_to_evolve)
        else:
            num_to_evolve = len(unconverged_indices)
        
        if num_to_evolve == 0:
             return [], next_job_id, successful_F, successful_CR
             
        indices_to_process_map = np.random.choice(
            np.arange(len(unconverged_indices)),
            size=num_to_evolve, 
            replace=False, 
            p=probabilities
        )
        indices_to_process = [unconverged_indices[i] for i in indices_to_process_map]

        active_pop_list = list(self.active_grid_indices)
        if len(active_pop_list) < 4:
            print("Not enough active points (<4) to perform DE. Waiting.")
            return [], next_job_id, successful_F, successful_CR

        # --- Create parent and p-best pools ---
        parent_pool = []
        for idx in active_pop_list:
            state = self.population[idx]
            best_idx = np.argmax(state['fitnesses'])
            parent_pool.append({
                'continuous_params': state['continuous_params'][best_idx],
                'fitness': state['fitnesses'][best_idx]
            })

        pbest_archive = []
        if self.mutation_strategy == 'current-to-pbest/1':
            parent_pool.sort(key=lambda p: p['fitness'], reverse=True)
            pbest_size = max(1, int(len(parent_pool) * self.pbest_fraction))
            pbest_archive = parent_pool[:pbest_size]

        # --- Create a job for each selected grid point ---
        jobs = []
        for grid_idx in indices_to_process:
            job = DEGridPointJob(
                job_id=next_job_id,
                sampler=self,
                grid_idx=grid_idx,
                parent_pool=parent_pool,
                pbest_archive=pbest_archive,
                successful_F_list=successful_F, # Pass shared list
                successful_CR_list=successful_CR # Pass shared list
            )
            jobs.append(job)
            next_job_id += 1
            
        return jobs, next_job_id, successful_F, successful_CR

    def update_de_memory(self, successful_F, successful_CR):
        """Updates the F and CR memory after a DE generation."""
        if successful_F:
            weights = np.ones(len(successful_F)) # Simple mean
            muF = np.sum(weights * np.array(successful_F)**2) / np.sum(weights * np.array(successful_F))
            muCR = np.sum(weights * np.array(successful_CR)) / np.sum(weights)
            
            self.memory_F[self.memory_idx] = muF
            self.memory_CR[self.memory_idx] = muCR
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def create_refinement_job_for_point(self, grid_idx, next_job_id):
        """
        Creates a new L-BFGS-B refinement job for a single converged grid point.
        """
        state = self.population.get(grid_idx)
        
        # Safety check: only refine active/converged/refined points
        if not state or state['status'] == 'refining_queued':
            return None

        # Mark as claimed
        state['status'] = 'refining_queued'
        
        # Find the best individual to start from
        best_ind_idx = np.argmax(state['fitnesses'])
        start_params_partial = state['continuous_params'][best_ind_idx]
        start_fitness = state['fitnesses'][best_ind_idx]
        
        # Construct the full parameter vector for the initial task
        start_params_full = self._construct_params(grid_idx, start_params_partial)

        # Get the seed history if it exists
        seed_history = state.get('optimizer_state')
        
        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='REFINEMENT',
            sampler=self,
            opt_dims=tuple(self.continuous_dims), # Optimize continuous dims
            start_params=start_params_partial,     # Partial vector
            grid_idx=grid_idx,                     # Grid anchor
            start_params_full=start_params_full,   # Full vector for first eval
            seed_history=seed_history,
            start_fitness=start_fitness            # Pass current best fitness
        )
        
        return (job, next_job_id + 1)

    def create_dynamic_activation_jobs(self, next_job_id):
        """
        Creates ActivationJobs for neighbors of high-likelihood grid points
        that have not yet been activated.
        """
        new_jobs = []
        roi_cutoff = self.global_max_target_val - self.roi_threshold
        
        # Iterate over a copy as population might be modified
        active_points_in_roi = [
            idx for idx, state in self.population.items() 
            if state['best_fitness'] > roi_cutoff
        ]

        for grid_idx in active_points_in_roi:
            for neighbor_idx in self._get_valid_neighbors(grid_idx):
                if neighbor_idx not in self.population and neighbor_idx not in self.pending_activation_indices:
                    
                    # Find best warm_start_params from neighbor's neighbors (incl. self)
                    best_warm_start_params = None
                    best_warm_start_fitness = -np.inf
                    
                    for potential_source_idx in self._get_valid_neighbors(neighbor_idx, include_center=True):
                        if potential_source_idx in self.population:
                            source_state = self.population[potential_source_idx]
                            if source_state['best_fitness'] > best_warm_start_fitness:
                                best_warm_start_fitness = source_state['best_fitness']
                                source_best_idx = np.argmax(source_state['fitnesses'])
                                best_warm_start_params = source_state['continuous_params'][source_best_idx]

                    job = ActivationJob(
                        job_id=next_job_id, 
                        sampler=self, 
                        grid_idx=neighbor_idx, 
                        warm_start_params=best_warm_start_params
                    )
                    new_jobs.append(job)
                    self.pending_activation_indices.add(neighbor_idx)
                    next_job_id += 1
                    
        return new_jobs, next_job_id

    def create_patching_refinement_jobs(self, next_job_id):
        """
        Identifies candidates for patching and creates refinement jobs for them.
        """
        # 1. Identify all candidates
        roi_cutoff = self.global_max_target_val - self.roi_threshold
        candidate_indices = []
        for grid_idx, state in self.population.items():
            if state['best_fitness'] >= roi_cutoff:
                candidate_indices.append(grid_idx)
            else:
                # Also check points bordering the ROI (from test_38.py)
                neighbor_count = 0
                roi_neighbor_count = 0
                for neighbor_idx in self._get_valid_neighbors(grid_idx):
                    neighbor_count += 1
                    if self.profile_likelihood_grid.get(neighbor_idx, -np.inf) >= roi_cutoff:
                        roi_neighbor_count += 1
                if neighbor_count > 0 and (roi_neighbor_count / neighbor_count) > 0.5:
                    candidate_indices.append(grid_idx)

        if not candidate_indices:
            return [], next_job_id

        # 2. Calculate priority scores for each candidate
        priority_scores = []
        for grid_idx in candidate_indices:
            current_logL = self.profile_likelihood_grid.get(grid_idx, -np.inf)
            if current_logL == -np.inf:
                continue

            scalar_gradient_sum = 0.0
            neighbor_logL_sum = 0.0

            for dim_idx in range(self.n_proj_dims):
                for direction in [-1, 1]:
                    offset = np.zeros(self.n_proj_dims, dtype=int)
                    offset[dim_idx] = direction
                    neighbor_idx = tuple(np.array(grid_idx) + offset)

                    if not all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                        continue

                    neighbor_logL = self.profile_likelihood_grid.get(neighbor_idx, -np.inf)

                    if neighbor_logL > -np.inf and neighbor_logL >= roi_cutoff:
                        gradient = neighbor_logL - current_logL
                        if gradient > 0:
                            scalar_gradient_sum += gradient
                            neighbor_logL_sum += neighbor_logL

            if scalar_gradient_sum > 0:
                likelihood_weight = max(neighbor_logL_sum - roi_cutoff, 0) + 1.0
                priority_score = scalar_gradient_sum * likelihood_weight
                priority_scores.append((priority_score, grid_idx))

        if not priority_scores:
            return [], next_job_id

        # 3. Sort points and select the top fraction
        priority_scores.sort(key=lambda x: x[0], reverse=True)
        num_to_patch = max(1, int(len(priority_scores) * self.patching_fraction))
        points_to_patch = [idx for _, idx in priority_scores[:num_to_patch]]

        # 4. Create refinement jobs for these points
        new_jobs = []
        for grid_idx in points_to_patch:
            state = self.population.get(grid_idx)
            # Only patch if it exists and isn't already being refined
            if state and state['status'] != 'refining_queued':
                spawn_result = self.create_refinement_job_for_point(grid_idx, next_job_id)
                if spawn_result:
                    job, next_job_id = spawn_result
                    new_jobs.append(job)
        
        return new_jobs, next_job_id

# --- End of class GridAnchoredDESampler ---







# --- Test Functions and Plotting ---

def rosenbrock_4D(params):
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)

def himmelblau_4d(params):
    x1, x2, x3, x4 = params
    term1 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    term2 = (x3**2 + x4 - 11)**2 + (x3 + x4**2 - 7)**2
    scale = 0.05
    return -1 * scale * (term1 + term2)

def get_test_function(name):
    """Factory function to get a test likelihood, its bounds, and true peaks."""
    if name == "rosenbrock_4D":
        bounds = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]]
        peaks = [np.array([1.0, 1.0, 1.0, 1.0])]
        return rosenbrock_4D, bounds, peaks
    
    elif name == "himmelblau_4d":
        bounds = [[-6, 6], [-6, 6], [-6, 6], [-6, 6]]
        peaks = [
            np.array([3.0, 2.0, 3.0, 2.0]),
            np.array([-2.805118, 3.131312, -2.805118, 3.131312]),
            np.array([-3.779310, -3.283186, -3.779310, -3.283186]),
            np.array([3.584428, -1.848126, 3.584428, -1.848126])
        ]
        return himmelblau_4d, bounds, peaks
    else:
        raise ValueError(f"Unknown test function: {name}")


def plot_profiles(sampler, fig, axes):
    """Generates and displays the 2D profile likelihood plot."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
        return

    ax = axes[0]
    ax.clear()

    if sampler.n_proj_dims != 2:
        ax.text(0.5, 0.5, 'Plotting only supported for 2D projections.', 
                horizontalalignment='center', verticalalignment='center')
        fig.canvas.draw()
        plt.pause(0.01)
        return

    dim1, dim2 = sampler.projection_dims
    
    # --- Create 2D grid from sparse dict ---
    profile_2d = np.full(sampler.grid_shape, -np.inf)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_2d[grid_idx] = fitness
    # ---
    
    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    plot_baseline = sampler.global_max_target_val
    vmin = plot_baseline - 3.0
    vmax = plot_baseline
    
    masked_profile = np.ma.masked_where(profile_2d == -np.inf, profile_2d)
    
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')

    im = ax.imshow(masked_profile.T, extent=extent, aspect='auto', origin='lower', 
                   cmap=cmap, vmin=vmin, vmax=vmax)
    
    active_points = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
             coords = sampler._get_grid_coords_from_indices(grid_idx)
             active_points.append(coords)
    
    if active_points:
        active_points = np.array(active_points)
        ax.scatter(active_points[:, 0], active_points[:, 1], c='cyan', s=3, 
                   edgecolor='black', lw=0.5, label='Active DE Points')

    if sampler.initial_maxima:
        peaks = np.array([m['point'] for m in sampler.initial_maxima])
        ax.plot(peaks[:, dim1], peaks[:, dim2], 'r*', markersize=10, 
                label='Found Maxima', markeredgecolor='k')

    ax.set_title(f'Profile Likelihood (Gen: {sampler.current_generation}, Dims: {sampler.projection_dims})')
    ax.set_xlabel(f'Parameter {dim1}')
    ax.set_ylabel(f'Parameter {dim2}')
    if ax.get_legend() is None: # Avoid duplicate legends
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    cax = axes[1]
    cax.clear()
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log Likelihood')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)




# --- MPI Worker and Master Main Functions ---

def worker_main(comm):
    """Main loop for a worker process."""
    # rank = comm.Get_rank()
    # First, receive the target function from the master.
    target_func = comm.bcast(None, root=0)
    print(f"Worker {myrank}: Received target function. Ready for tasks.")

    while True:
        # print(f"rank {myrank}: DEBUG: worker_main: START new iteration.", flush=True)

        # Wait for a task from the master
        task = comm.recv(source=0, tag=MPI.ANY_TAG)
        
        if task == TASK_TERMINATE:
            print(f"Worker {myrank}: Received terminate signal. Exiting.")
            break
            
        # Execute the task (a single target evaluation)
        params = task['params']
        target_val = target_func(params)
        
        # Send the result back to the master
        context = task['context']
        context['worker_rank'] = myrank
        result = {'target_val': target_val, 'params': params, 'context': context}
        comm.send(result, dest=0)
        # print(f"rank {myrank}: DEBUG: worker_main: END iteration.", flush=True)


def master_main(comm, sampler, num_generations, max_num_to_evolve, 
                plot_callback, plot_interval, skip_init_opt_on_warm_start=True,
                fig=None, axes=None): # Add fig/axes for plotting
    """
    Main control loop for the master process.
    Acts as a state machine, dispatching jobs and processing results.
    """
    n_workers = comm.Get_size() - 1
    if n_workers <= 0:
        print("Error: This script requires at least 2 MPI processes (1 master, 1+ workers).")
        return

    print(f"rank {myrank}: DEBUG: master_main: STARTING with {n_workers} workers.")
    
    # Broadcast the target function to all workers
    comm.bcast(sampler.target_func, root=0)

    # --- Master state ---
    free_workers = list(range(1, n_workers + 1))
    
    # Define the workflow stages
    stages = ['INITIAL_OPTIMIZATION', 'ACTIVATION', 'DE_LOOP', 'PATCHING']
    
    current_stage = stages.pop(0) if stages else None

    active_jobs = {} # {job_id: Job object}
    
    # --- MODIFICATION: Create priority task queues ---
    high_prio_tasks = collections.deque()
    low_prio_tasks = collections.deque()
    # task_queue = collections.deque() # Old queue
    # --- END MODIFICATION ---
    
    next_job_id = 0
    tasks_sent = 0
    tasks_completed = 0
    
    # DE stage state
    de_generation = 0 # Counter for DE stage
    de_successful_F = [] # Shared list for F/CR
    de_successful_CR = []

    # Patching stage state
    patching_iteration = 0
    last_patching_improvement = np.inf
    current_patching_batch_ids = set()
    current_patching_batch_improvements = []

    last_plot_time = time.time()
    de_gen_start_time = time.time() # Add timer for DE generations
    
    # --- Main Event Loop ---
    # --- MODIFICATION: Update loop condition to check both queues ---
    while current_stage or active_jobs or high_prio_tasks or low_prio_tasks or (tasks_sent > tasks_completed):
    # --- END MODIFICATION ---

        # --- 1. Generate new jobs if a stage is starting or continuing ---
        # This block only runs when no jobs are active and no tasks are queued.
        # --- MODIFICATION: Update check to include both queues ---
        if not active_jobs and not high_prio_tasks and not low_prio_tasks and (tasks_sent == tasks_completed):
        # --- END MODIFICATION ---
            
            if not current_stage:
                break # All stages and jobs are complete

            print(f"--- Master: Entering stage: {current_stage} ---")
            new_jobs = []
            
            if current_stage == 'INITIAL_OPTIMIZATION':
                if skip_init_opt_on_warm_start and sampler.initial_maxima:
                     print("Skipping initial optimization on warm start.")
                else:
                    new_jobs, next_job_id = sampler.create_initial_optimization_jobs(next_job_id)
                
                if not new_jobs:
                    current_stage = stages.pop(0) if stages else None
                    continue
            
            elif current_stage == 'ACTIVATION':
                new_jobs, next_job_id = sampler.create_activation_jobs(next_job_id)
                if not new_jobs:
                    print("No activation jobs created (no initial maxima?). Moving to next stage.")
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'DE_LOOP':
                # This stage loops
                if de_generation >= num_generations:
                    print("--- Master: DE generations complete. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue
                
                # Update F/CR memory from *previous* generation
                if de_generation > 0:
                    sampler.update_de_memory(de_successful_F, de_successful_CR)

                de_generation += 1
                sampler.current_generation = de_generation # Update sampler state
                print(f"--- Master: Starting DE Generation {de_generation} ---")
                
                new_de_jobs, next_job_id, de_successful_F, de_successful_CR = sampler.create_de_generation_jobs(
                    next_job_id, max_num_to_evolve
                )
                
                # --- Add dynamic activation jobs ---
                new_act_jobs, next_job_id = sampler.create_dynamic_activation_jobs(next_job_id)
                new_jobs = new_de_jobs + new_act_jobs
                
                # --- Print Generation Summary ---
                elapsed = time.time() - de_gen_start_time
                de_gen_start_time = time.time() # Reset timer
                
                total_grid_points = len(sampler.population)
                active_count = len([s for s in sampler.population.values() if s['status'] == 'active'])
                converged_count = total_grid_points - active_count
                newly_activated_count = len(new_act_jobs)

                print(f"Gen {sampler.current_generation:4d} | Calls: {sampler.target_calls/1e3:6.1f}k | "
                      f"Grid Pts (act/conv/tot): {active_count:4d}/{converged_count:4d}/{total_grid_points:4d} | "
                      f"Global Max logL: {sampler.global_max_target_val:.4e} | "
                      f"New Activations: {newly_activated_count:3d} | "
                      f"Elapsed: {elapsed:.1f}s")
                # --- End Print ---

                if not new_jobs and not de_successful_F: # DE converged and no new activations
                    print("--- Master: DE converged and no new activations. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue
            
            elif current_stage == 'PATCHING':
                if patching_iteration >= sampler.max_patching_iterations or \
                   last_patching_improvement < sampler.patching_conv_threshold:
                    print("--- Master: Patching converged or max iterations reached. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue
                
                print(f"--- Master: Starting Patching Iteration {patching_iteration + 1} ---")
                new_jobs, next_job_id = sampler.create_patching_refinement_jobs(next_job_id)

                if not new_jobs:
                    print("--- Master: No patching candidates found. Ending patching. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue
                
                # Reset batch state
                current_patching_batch_ids = {j.id for j in new_jobs}
                current_patching_batch_improvements = []
                patching_iteration += 1
                current_stage = 'WAITING_FOR_PATCHING' # Go to wait state
            
            elif current_stage == 'WAITING_FOR_PATCHING':
                # This stage just waits for jobs to complete.
                # If we are here with no active jobs, it means the batch finished.
                pass

            
            # --- Add new jobs to active pool and queue initial tasks ---
            for job in new_jobs:
                active_jobs[job.id] = job
                initial_tasks = job.start()
                
                # --- MODIFICATION: Add to correct priority queue ---
                if job.type in ['INITIAL_OPTIMIZATION', 'REFINEMENT']:
                    high_prio_tasks.extend(initial_tasks)
                else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                    low_prio_tasks.extend(initial_tasks)
                # --- END MODIFICATION ---
            
            # If we just finished a non-looping stage, advance to the next
            if current_stage not in ['DE_LOOP', 'WAITING_FOR_PATCHING']:
                 current_stage = stages.pop(0) if stages else None

        # --- 3. (MODIFIED) Check for and process ALL available results ---
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            # A message is waiting, so now we do a blocking (but instant) receive
            result = comm.recv(source=MPI.ANY_SOURCE)
            worker_rank = result['context']['worker_rank']
            free_workers.append(worker_rank)
            tasks_completed += 1
            
            # Register the call centrally
            sampler._register_target_call(result['params'], result['target_val'])
            
            job_id = result['context'].get('job_id', -1)
            if job_id not in active_jobs:
                print(f"Warning: Received result for unknown/finished job {job_id}. Ignoring.")
                continue
                
            job = active_jobs[job_id]
            new_tasks = job.process_result(result)
            
            # --- MODIFICATION: Add to correct priority queue ---
            if job.type in ['INITIAL_OPTIMIZATION', 'REFINEMENT']:
                high_prio_tasks.extend(new_tasks)
            else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                low_prio_tasks.extend(new_tasks)
            # --- END MODIFICATION ---
            
            if job.is_finished():
                job_id_finished = job.id
                was_patching_job = job_id_finished in current_patching_batch_ids
                
                if was_patching_job:
                    if job.success and hasattr(job, 'improvement') and job.improvement > 0:
                        current_patching_batch_improvements.append(job.improvement)
                    current_patching_batch_ids.remove(job_id_finished)

                print(f"--- Master: Job {job.id} ({job.type}) finished. Success: {job.success} ---")
                # This updates the sampler state and can spawn a new job
                spawn_result = job.on_finish(next_job_id) 
                del active_jobs[job_id_finished]
                
                if spawn_result:
                    new_job, next_job_id = spawn_result
                    active_jobs[new_job.id] = new_job
                    initial_tasks = new_job.start()
                    
                    # --- MODIFICATION: Add to correct priority queue ---
                    if new_job.type in ['INITIAL_OPTIMIZATION', 'REFINEMENT']:
                        high_prio_tasks.extend(initial_tasks)
                    else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                        low_prio_tasks.extend(initial_tasks)
                    # --- END MODIFICATION ---
                    
                    print(f"--- Master: Spawned new job {new_job.id} ({new_job.type}) for grid {new_job.grid_idx} ---")

                # Check if the patching batch is now complete
                if current_stage == 'WAITING_FOR_PATCHING' and not current_patching_batch_ids:
                    last_patching_improvement = sum(current_patching_batch_improvements)
                    print(f"--- Master: Patching Iteration {patching_iteration} complete. Total improvement: {last_patching_improvement} ---")
                    current_stage = 'PATCHING' # Loop back to start next iteration
        
        # --- End of Iprobe receive loop ---


        # --- 2. (Now Step 3) Dispatch tasks to free workers ---
        # --- MODIFICATION: Prioritize high_prio_tasks ---
        while free_workers and (high_prio_tasks or low_prio_tasks):
            worker_rank = free_workers.pop(0)
            
            if high_prio_tasks:
                task = high_prio_tasks.popleft()
            else:
                task = low_prio_tasks.popleft()
            
            comm.send(task, dest=worker_rank)
            tasks_sent += 1
        # --- END MODIFICATION ---

        # --- 3. Wait for and process a result ---
        # Only check for results if we are expecting any
        # --- (MODIFICATION: This block is now removed and replaced by Iprobe loop) ---
        # if tasks_sent > tasks_completed:
        #    ... (old blocking recv logic removed) ...
        # --- (END MODIFICATION) ---
                
        # --- 4. Plotting (optional) & Polite Sleep ---
        current_time = time.time()
        if plot_callback and (current_time - last_plot_time > plot_interval):
             # Only plot during DE phase for this example
            if sampler.current_generation > 0:
                print(f"Plotting... (Gen {sampler.current_generation})")
                plot_callback(sampler, fig, axes)
                last_plot_time = current_time
        
        # If no tasks are ready to send and no workers are free,
        # but we are still waiting for results, sleep for a tiny bit
        # to prevent a 100% CPU busy-wait.
        if not free_workers and not high_prio_tasks and not low_prio_tasks and (tasks_sent > tasks_completed):
            time.sleep(0.001) # 1ms sleep


    # --- End of Main Event Loop ---
    
    print(f"rank {myrank}: DEBUG: master_main: Workflow finished.")
    
    # Final flush of sample buffer
    sampler._flush_samples_buffer()

    # --- Print Final Summary ---
    print("\n" + "="*80)
    print("--- Master: Workflow Complete ---")
    print(f"  Total Target Function Calls: {sampler.target_calls}")
    print(f"  Final Global Max logL: {sampler.global_max_target_val:.6e}")
    print(f"  Total Grid Points Explored: {len(sampler.population)}")
    print("="*80 + "\n")

    # --- 5. Terminate workers ---
    print(f"rank {myrank}: DEBUG: master_main: send TASK_TERMINATE to workers.", flush=True)
    for rank in range(1, n_workers + 1):
        if rank in free_workers:
            comm.send(TASK_TERMINATE, dest=rank)
        else:
            # This logic is for a non-blocking setup.
            # In our blocking setup, all workers should be free.
            # But as a failsafe:
            print(f"Waiting for worker {rank} to finish last task before terminating...")
            result = comm.recv(source=rank)
            tasks_completed += 1
            # We don't process this last result, just receive it.
            comm.send(TASK_TERMINATE, dest=rank)
            
    print(f"rank {myrank}: DEBUG: master_main: All workers terminated.")



if __name__ == '__main__':
    
    # --- Configuration (shared by all processes, but only master uses most of it) ---
    TEST_FUNCTION = "himmelblau_4d"
    OUTPUT_FILE = f"samples_rank_{myrank}.csv"
    
    PROJECTIONS_TO_RUN = [
        {'dims': [0, 1], 'grid_points': [100, 100], 'patching': True, 'refining': True},
        # {'dims': [0, 2], 'grid_points': [100, 100], 'patching': True, 'refining': True},
    ]
    
    log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

    if myrank == 0:
        # --- Master process ---
        
        # Setup plotting
        try:
            import matplotlib.pyplot as plt
            plt.ion() # Interactive mode on
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [10, 1]})
        except ImportError:
            fig, axes = None, None
            print("Matplotlib not found. Plotting will be disabled.")


        sampler = GridAnchoredDESampler(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=PROJECTIONS_TO_RUN,
            pop_per_grid_point=1, # Increased for better DE
            mutation_strategy='current-to-pbest/1',
            pbest_fraction=0.1,
            n_initial_optimizations=30, # Increased
            roi_threshold=3.2,
            convergence_threshold=1e-3, # Tighter -> Looser (match serial)
            convergence_window=2,      # Longer window -> Shorter (match serial)
            neighbor_pull_probability=0.5,
            refinement_ftol=1e-9, 
            refinement_max_iter=20,
            refinement_gradient_method="forward", # "central",
            patching_fraction=0.05,
            patching_conv_threshold=0.01,
            max_patching_iterations=10, # Limit patching
            memory_size=len(PROJECTIONS_TO_RUN[0]['grid_points']) * 25,
            samples_output_file=OUTPUT_FILE,
        )

        def plot_func_wrapper(s, fig, axes):
            plot_profiles(s, fig, axes)

        master_main(
            comm=comm,
            sampler=sampler,
            num_generations=100000, # Set a finite number of generations
            max_num_to_evolve=None, # Limit evals per gen -> Evolve all
            plot_callback=plot_func_wrapper,
            plot_interval=100, # Plot every 2 seconds
            skip_init_opt_on_warm_start=False,
            fig=fig,
            axes=axes
        )
        
        if fig:
            print("Master: Final plot. Press Enter to exit.")
            plot_func_wrapper(sampler, fig, axes)
            plt.ioff()
            plt.show()

    else:
        # --- Worker process ---
        worker_main(comm)

print(f"rank {myrank}: Done.")


