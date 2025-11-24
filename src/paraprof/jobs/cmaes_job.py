"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) job for grid point optimization.

This implementation uses neighbor-informed initialization to leverage the grid structure.
"""
import numpy as np
from ..logger import get_logger
from .base import Job

logger = get_logger()

# Try to import emulator utilities
try:
    from ..emulator_utils import prepare_emulator_cache_for_worker
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False
    logger.debug("Emulator utilities not available")


class CMAESGridPointJob(Job):
    """
    A job to run one generation of CMA-ES for one grid point.

    Uses neighbor-informed initialization to warm-start the optimization
    from solutions found at neighboring grid points.
    """
    def __init__(self, job_id, sampler, grid_idx, initial_mean=None,
                 initial_sigma=None, initial_C=None):

        super().__init__(job_id, 'CMAES_GRID_POINT', sampler)
        self.grid_idx = grid_idx
        self.grid_state = self.sampler.population[self.grid_idx]

        self.n_dims = self.sampler.n_cont_dims
        self.lambda_ = sampler.cmaes_lambda  # Population size
        self.mu = sampler.cmaes_mu  # Number of parents

        # CMA-ES state variables
        self._initialize_cmaes_state(initial_mean, initial_sigma, initial_C)

        # Track which offspring have been evaluated
        self.offspring_params = []  # List of parameter vectors
        self.offspring_fitness = []  # List of fitness values
        self.evals_remaining = self.lambda_

        # Track pre-screening statistics
        self.trials_generated = 0
        self.trials_screened_out = 0

        # Generation tracking
        self.generation = 0
        self.max_generations = sampler.cmaes_max_generations

    def _initialize_cmaes_state(self, initial_mean, initial_sigma, initial_C):
        """
        Initialize CMA-ES state variables with neighbor-informed values.
        """
        # === MEAN INITIALIZATION (Opportunity 3) ===
        if initial_mean is not None:
            self.m = initial_mean.copy()
        else:
            self.m = self._get_neighbor_informed_mean()

        # === STEP SIZE INITIALIZATION (Opportunity 2) ===
        if initial_sigma is not None:
            self.sigma = initial_sigma
        else:
            self.sigma = self._estimate_initial_sigma()

        # === COVARIANCE INITIALIZATION WITH NEIGHBOR INHERITANCE ===
        if initial_C is not None:
            # Explicit initialization provided
            self.C = initial_C.copy()
        else:
            # Try to inherit from best neighbor
            neighbor_C = self._get_neighbor_covariance()
            if neighbor_C is not None:
                # Mix neighbor covariance with identity for regularization
                # This prevents premature convergence to neighbor's narrow distribution
                mixing_factor = 1.0  # 70% neighbor, 30% identity
                self.C = mixing_factor * neighbor_C + (1 - mixing_factor) * np.eye(self.n_dims)
                logger.debug(f"CMA-ES at {self.grid_idx}: Initialized C with {mixing_factor:.0%} neighbor + {1-mixing_factor:.0%} identity")
            else:
                # No suitable neighbor found, use identity matrix
                self.C = np.eye(self.n_dims)

        # CMA-ES algorithm parameters (standard settings)
        self.pc = np.zeros(self.n_dims)  # Evolution path for C
        self.ps = np.zeros(self.n_dims)  # Evolution path for sigma

        # Weights for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights**2)

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.n_dims) / (self.n_dims + 4 + 2 * self.mueff / self.n_dims)
        self.cs = (self.mueff + 2) / (self.n_dims + self.mueff + 5)
        self.c1 = 2 / ((self.n_dims + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n_dims + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n_dims + 1)) - 1) + self.cs

        # Expectation of ||N(0,I)||
        self.chiN = self.n_dims**0.5 * (1 - 1/(4*self.n_dims) + 1/(21*self.n_dims**2))

        # Eigendecomposition of C (will be updated)
        self.eigeneval = 0
        self.B = np.eye(self.n_dims)
        self.D = np.ones(self.n_dims)
        self.invsqrtC = np.eye(self.n_dims)

    def _get_neighbor_informed_mean(self):
        """
        Initialize mean from best neighbor solution (Opportunity 3).

        Returns
        -------
        np.ndarray
            Initial mean vector for CMA-ES
        """
        # First try to get best neighbor solution
        best_neighbor_params = None
        best_neighbor_fitness = -np.inf

        for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
            if neighbor_idx in self.sampler.population:
                neighbor_state = self.sampler.population[neighbor_idx]
                if neighbor_state['best_fitness'] > best_neighbor_fitness:
                    best_neighbor_fitness = neighbor_state['best_fitness']
                    best_idx = np.argmax(neighbor_state['fitnesses'])
                    best_neighbor_params = neighbor_state['continuous_params'][best_idx]

        if best_neighbor_params is not None:
            logger.debug(f"CMA-ES at {self.grid_idx}: Initializing mean from neighbor (fitness {best_neighbor_fitness:.4e})")
            return best_neighbor_params.copy()

        # Fallback: try global solution pool
        global_samples = self.sampler._sample_from_global_pool(1)
        if global_samples is not None and len(global_samples) > 0:
            logger.debug(f"CMA-ES at {self.grid_idx}: Initializing mean from global pool")
            return global_samples[0]

        # Last resort: center of bounds for continuous dimensions
        logger.debug(f"CMA-ES at {self.grid_idx}: Initializing mean from bounds center")
        bounds = self.sampler.bounds[self.sampler.continuous_dims]
        return 0.5 * (bounds[:, 0] + bounds[:, 1])

    def _get_neighbor_covariance(self):
        """
        Inherit covariance matrix from best neighbor with CMA-ES state.

        Returns
        -------
        np.ndarray or None
            Covariance matrix from best neighbor, or None if no suitable neighbor found
        """
        best_C = None
        best_fitness = -np.inf
        best_neighbor_idx = None

        # Search for best neighbor with stored CMA-ES state
        for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
            if neighbor_idx in self.sampler.population:
                neighbor_state = self.sampler.population[neighbor_idx]

                # Check if this neighbor has CMA-ES covariance stored
                if 'cmaes_C' not in neighbor_state:
                    continue

                # Check if this neighbor is better than previous candidates
                if neighbor_state['best_fitness'] > best_fitness:
                    C_candidate = neighbor_state['cmaes_C']

                    # Check condition number to ensure well-conditioned matrix
                    try:
                        eigvals = np.linalg.eigvalsh(C_candidate)
                        if np.min(eigvals) <= 0:
                            logger.debug(f"CMA-ES at {self.grid_idx}: Neighbor {neighbor_idx} has non-positive eigenvalues, skipping")
                            continue

                        cond = eigvals.max() / eigvals.min()
                        if cond < 1e10:  # Not too ill-conditioned
                            best_C = C_candidate
                            best_fitness = neighbor_state['best_fitness']
                            best_neighbor_idx = neighbor_idx
                        else:
                            logger.debug(f"CMA-ES at {self.grid_idx}: Neighbor {neighbor_idx} has poor condition number {cond:.2e}, skipping")
                    except np.linalg.LinAlgError:
                        logger.debug(f"CMA-ES at {self.grid_idx}: Failed to compute eigenvalues for neighbor {neighbor_idx}, skipping")
                        continue

        if best_C is not None:
            logger.info(f"CMA-ES at {self.grid_idx}: Inheriting covariance from neighbor {best_neighbor_idx} (fitness {best_fitness:.4e})")
        else:
            logger.debug(f"CMA-ES at {self.grid_idx}: No suitable neighbor covariance found, using identity")

        return best_C

    def _estimate_initial_sigma(self):
        """
        Estimate initial step size based on neighbor fitness variance (Opportunity 2).

        Returns
        -------
        float
            Initial step size
        """
        # Collect fitness values from neighbors
        neighbor_fitnesses = []
        for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
            if neighbor_idx in self.sampler.population:
                neighbor_state = self.sampler.population[neighbor_idx]
                neighbor_fitnesses.append(neighbor_state['best_fitness'])

        # Estimate typical parameter scale
        bounds = self.sampler.bounds[self.sampler.continuous_dims]
        typical_scale = np.mean(bounds[:, 1] - bounds[:, 0])

        if len(neighbor_fitnesses) >= 2:
            # If fitness varies significantly, use larger sigma (steep region)
            # If fitness is relatively flat, use smaller sigma
            fitness_std = np.std(neighbor_fitnesses)
            fitness_range = np.max(neighbor_fitnesses) - np.min(neighbor_fitnesses)

            if fitness_range > self.sampler.roi_threshold / 2:
                # Steep region - explore more
                sigma = 0.3 * typical_scale
                logger.debug(f"CMA-ES at {self.grid_idx}: Steep region detected, sigma={sigma:.4e}")
            else:
                # Flat region - exploit more
                sigma = 0.1 * typical_scale
                logger.debug(f"CMA-ES at {self.grid_idx}: Flat region detected, sigma={sigma:.4e}")
        else:
            # Default: moderate exploration
            sigma = 0.2 * typical_scale
            logger.debug(f"CMA-ES at {self.grid_idx}: Using default sigma={sigma:.4e}")

        return sigma

    def _sample_offspring(self):
        """
        Sample lambda offspring from the CMA-ES distribution N(m, σ²C).

        Uses neighbor solutions to seed some of the initial population.

        Returns
        -------
        np.ndarray
            Array of shape (lambda, n_dims) with offspring parameters
        """
        offspring = []

        # For the first generation, mix in neighbor solutions (Opportunity 3)
        if self.generation == 0:
            # Collect neighbor solutions
            neighbor_solutions = []
            for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
                if neighbor_idx in self.sampler.population:
                    neighbor_state = self.sampler.population[neighbor_idx]
                    best_idx = np.argmax(neighbor_state['fitnesses'])
                    neighbor_solutions.append(neighbor_state['continuous_params'][best_idx])

            # Use up to 30% of population from neighbors
            n_from_neighbors = min(len(neighbor_solutions), self.lambda_ // 3)
            if n_from_neighbors > 0:
                logger.debug(f"CMA-ES at {self.grid_idx}: Seeding {n_from_neighbors}/{self.lambda_} offspring from neighbors")
                # Take the most diverse neighbors
                np.random.shuffle(neighbor_solutions)
                offspring.extend(neighbor_solutions[:n_from_neighbors])

        # Generate remaining offspring from CMA distribution
        n_remaining = self.lambda_ - len(offspring)

        # Update eigendecomposition if needed (every ~1/(c1+cmu)/n_dims generations)
        if self.generation - self.eigeneval > 1 / (self.c1 + self.cmu) / self.n_dims / 10:
            self.eigeneval = self.generation
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # Enforce symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
            self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T

        for _ in range(n_remaining):
            # Sample from N(0, C) = B * D * N(0, I)
            z = np.random.randn(self.n_dims)
            y = self.B @ (self.D * z)
            x = self.m + self.sigma * y

            # Ensure bounds
            x = self.sampler._ensure_bounds(x, self.sampler.continuous_dims)
            offspring.append(x)

        return np.array(offspring)

    def start(self):
        """Generate offspring and return their evaluation tasks."""
        # Direct evaluation mode: no continuous dimensions, so no evolution needed
        if self.sampler.direct_eval_mode or self.sampler.n_cont_dims == 0:
            # Grid point already evaluated by ActivationJob, mark as converged
            self.success = True
            self._is_finished = True
            return []

        # Sample offspring
        offspring = self._sample_offspring()
        self.offspring_params = offspring
        self.offspring_fitness = [None] * self.lambda_

        # Create evaluation tasks
        tasks = []
        for i, params in enumerate(offspring):
            self.trials_generated += 1

            # Construct full params
            full_params = self.sampler._construct_params(self.grid_idx, params)

            # Current best fitness for emulator comparison
            target_fitness = self.grid_state['best_fitness']

            # === PREPARE EMULATOR DATA FOR WORKER-SIDE PRE-SCREENING ===
            emulator_cache = None
            if EMULATOR_AVAILABLE and getattr(self.sampler, 'use_de_prescreening', False):
                emulator_cache = prepare_emulator_cache_for_worker(
                    sampler=self.sampler,
                    center_params=full_params,
                    min_points=self.sampler.emulator_min_neighbors,
                    max_points=getattr(self.sampler, 'emulator_max_neighbors', None),
                    grid_idx=self.grid_idx
                )
            # === END EMULATOR PREPARATION ===

            context = {
                'type': self.type,
                'job_id': self.id,
                'offspring_idx': i,
                'target_fitness': target_fitness
            }
            task = {
                'params': full_params,
                'context': context,
                'emulator_cache': emulator_cache
            }
            tasks.append(task)

        # Update global statistics
        if hasattr(self.sampler, 'de_trials_generated'):
            self.sampler.de_trials_generated += self.trials_generated

        return tasks

    def process_result(self, result):
        """Process an offspring evaluation result."""
        offspring_idx = result['context']['offspring_idx']

        # Check if trial was screened out by worker-side emulator
        was_screened = result.get('emulator_screened', False)
        if was_screened:
            # Worker rejected this trial - mark as very poor fitness
            self.trials_screened_out += 1
            if hasattr(self.sampler, 'de_trials_screened_out'):
                self.sampler.de_trials_screened_out += 1

            self.offspring_fitness[offspring_idx] = -np.inf
        else:
            # Normal evaluation result
            self.offspring_fitness[offspring_idx] = result['target_val']

        self.evals_remaining -= 1

        # When all offspring evaluated, update CMA-ES state
        if self.evals_remaining <= 0:
            self._update_cmaes_state()

            # Check convergence or max generations
            if self._check_convergence() or self.generation >= self.max_generations:
                self.success = True
                self._is_finished = True

                # Log pre-screening effectiveness
                if self.trials_generated > 0 and getattr(self.sampler, 'use_de_prescreening', False):
                    screen_rate = 100 * self.trials_screened_out / self.trials_generated
                    logger.info(
                        f"CMA-ES job {self.id} (grid {self.grid_idx}): "
                        f"Screened out {self.trials_screened_out}/{self.trials_generated} "
                        f"trials ({screen_rate:.1f}%)"
                    )
            else:
                # Next generation
                self.generation += 1
                self.trials_generated = 0
                self.trials_screened_out = 0

                # Sample new offspring
                offspring = self._sample_offspring()
                self.offspring_params = offspring
                self.offspring_fitness = [None] * self.lambda_
                self.evals_remaining = self.lambda_

                # Create new tasks
                tasks = []
                for i, params in enumerate(offspring):
                    self.trials_generated += 1
                    full_params = self.sampler._construct_params(self.grid_idx, params)
                    target_fitness = self.grid_state['best_fitness']

                    # Emulator cache
                    emulator_cache = None
                    if EMULATOR_AVAILABLE and getattr(self.sampler, 'use_de_prescreening', False):
                        emulator_cache = prepare_emulator_cache_for_worker(
                            sampler=self.sampler,
                            center_params=full_params,
                            min_points=self.sampler.emulator_min_neighbors,
                            max_points=getattr(self.sampler, 'emulator_max_neighbors', None),
                            grid_idx=self.grid_idx
                        )

                    context = {
                        'type': self.type,
                        'job_id': self.id,
                        'offspring_idx': i,
                        'target_fitness': target_fitness
                    }
                    task = {
                        'params': full_params,
                        'context': context,
                        'emulator_cache': emulator_cache
                    }
                    tasks.append(task)

                # Update global statistics
                if hasattr(self.sampler, 'de_trials_generated'):
                    self.sampler.de_trials_generated += self.trials_generated

                return tasks

        return []

    def _update_cmaes_state(self):
        """Update CMA-ES state variables after evaluating all offspring."""
        # Sort offspring by fitness
        fitness_array = np.array(self.offspring_fitness)
        sorted_indices = np.argsort(fitness_array)[::-1]  # Descending order

        # Select mu best offspring
        best_indices = sorted_indices[:self.mu]
        best_offspring = self.offspring_params[best_indices]

        # Update mean (weighted recombination)
        m_old = self.m.copy()
        self.m = np.sum(self.weights[:, np.newaxis] * best_offspring, axis=0)

        # Update evolution paths
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  self.invsqrtC @ (self.m - m_old) / self.sigma

        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs)**(2 * (self.generation + 1))) /
                self.chiN < 1.4 + 2 / (self.n_dims + 1))

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * \
                  (self.m - m_old) / self.sigma

        # Update covariance matrix
        artmp = (best_offspring - m_old) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) +
                           (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * artmp.T @ np.diag(self.weights) @ artmp

        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) *
                            (np.linalg.norm(self.ps) / self.chiN - 1))

        # Update grid state with best solution
        best_fitness = fitness_array[best_indices[0]]
        best_params = best_offspring[0]

        # Update population state (store best individual)
        best_idx_in_pop = np.argmax(self.grid_state['fitnesses'])
        if best_fitness > self.grid_state['fitnesses'][best_idx_in_pop]:
            self.grid_state['continuous_params'][best_idx_in_pop] = best_params
            self.grid_state['fitnesses'][best_idx_in_pop] = best_fitness

            # Update improvement history
            improvement = best_fitness - self.grid_state['best_fitness']
            self.grid_state['improvement_history'].append(improvement)

            if best_fitness > self.grid_state['best_fitness']:
                self.grid_state['best_fitness'] = best_fitness
                self.grid_state['last_update_gen'] = self.sampler.current_generation
                self.sampler.profile_likelihood_grid[self.grid_idx] = best_fitness

                if best_fitness > self.sampler.global_max_target_val:
                    self.sampler.global_max_target_val = best_fitness

                # Update global solution pool
                full_params = self.sampler._construct_params(self.grid_idx, best_params)
                self.sampler._update_global_pool(full_params, best_fitness, self.grid_idx)

    def _check_convergence(self):
        """
        Check if CMA-ES has converged.

        Returns
        -------
        bool
            True if converged
        """
        # Check improvement history (similar to DE convergence)
        if len(self.grid_state['improvement_history']) >= self.sampler.convergence_window:
            recent_improvements = list(self.grid_state['improvement_history'])[-self.sampler.convergence_window:]
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement < self.sampler.convergence_threshold:
                return True

        # Check step size (CMA-ES specific)
        if self.sigma < 1e-10:
            logger.debug(f"CMA-ES at {self.grid_idx}: Converged due to small sigma ({self.sigma:.2e})")
            return True

        # Check condition number of C
        if self.generation > 0 and self.generation % 10 == 0:
            eigvals = np.linalg.eigvalsh(self.C)
            condition_number = np.max(eigvals) / np.min(eigvals)
            if condition_number > 1e14:
                logger.debug(f"CMA-ES at {self.grid_idx}: Converged due to ill-conditioned C (cond={condition_number:.2e})")
                return True

        return False

    def on_finish(self, next_job_id):
        """
        Update grid state after CMA-ES completes.
        Store CMA-ES state for neighbor inheritance.
        Optionally spawn L-BFGS-B refinement job.
        """
        if not self.success:
            return None

        # Direct evaluation mode: already marked as converged
        if self.sampler.direct_eval_mode or self.sampler.n_cont_dims == 0:
            return None

        # Store CMA-ES state in grid state for future neighbor inheritance
        self.grid_state['cmaes_C'] = self.C.copy()
        self.grid_state['cmaes_sigma'] = self.sigma
        self.grid_state['cmaes_B'] = self.B.copy()
        self.grid_state['cmaes_D'] = self.D.copy()
        logger.debug(f"CMA-ES at {self.grid_idx}: Stored converged covariance matrix for neighbor inheritance")

        # Check if we should spawn L-BFGS-B refinement
        if self._check_convergence():
            if self.sampler.lbfgsb_refinement:
                logger.info(f"--- CMA-ES Converged for {self.grid_idx}. Spawning L-BFGS-B refinement job. ---")
                # Mark status and spawn refinement job
                return self.sampler.create_LBFGSB_job_for_point(self.grid_idx, next_job_id)
            else:
                # Mark as optimized without L-BFGS-B refinement
                self.grid_state['status'] = 'optimized'
                logger.info(f"--- CMA-ES Converged for {self.grid_idx}. Marked as optimized (L-BFGS-B refinement disabled). ---")
                return None
        else:
            # Max generations reached without convergence
            logger.info(f"--- CMA-ES at {self.grid_idx} reached max generations ({self.max_generations}). ---")
            if self.sampler.lbfgsb_refinement:
                return self.sampler.create_LBFGSB_job_for_point(self.grid_idx, next_job_id)
            else:
                self.grid_state['status'] = 'optimized'
                return None
