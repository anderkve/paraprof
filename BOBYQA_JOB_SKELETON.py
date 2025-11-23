"""
BOBYQA optimization job - Phase 1 Implementation Skeleton

This is a detailed skeleton showing how BOBYQAJob would be implemented
following ParaProf's job-based architecture.

Key Design Decisions:
1. Use batch evaluation mode (simpler than full function interception)
2. Build quadratic model incrementally with parallel evaluations
3. Follow LBFGSBJob patterns for consistency
4. Support neighbor warm-starting (Hessian seeding for Phase 2)
"""

import numpy as np
import collections
from scipy.linalg import solve
from .base import Job


class BOBYQAJob(Job):
    """
    BOBYQA (Bound Optimization BY Quadratic Approximation) job.

    This job implements derivative-free optimization using a quadratic
    interpolation model within a trust region framework. The algorithm
    is parallelized by breaking model evaluations into asynchronous tasks.

    Algorithm Overview:
    1. Build initial interpolation model (2n+1 points)
    2. Solve trust region subproblem to get candidate step
    3. Evaluate candidate and update trust radius
    4. Optionally improve model geometry
    5. Repeat until convergence

    Parallelization Strategy:
    - Initial model points: Evaluated in parallel
    - Trust region step: Single evaluation (sequential)
    - Model improvement: 1-3 points in parallel (opportunistic)
    """

    def __init__(self, job_id, job_type, sampler, opt_dims, start_params,
                 grid_idx, start_params_full, seed_history=None,
                 start_fitness=-np.inf, initial_trust_radius=0.1,
                 max_iterations=50):
        """
        Initialize BOBYQA optimization job.

        Parameters
        ----------
        job_id : int
            Unique job identifier
        job_type : str
            Job type ('POST_ACTIVATION_BOBYQA', 'BOBYQA_LOOP', etc.)
        sampler : GridAnchoredDESampler
            Reference to main sampler
        opt_dims : tuple of int
            Dimensions to optimize (indices into full parameter vector)
        start_params : np.ndarray
            Starting parameters (partial, just optimization dimensions)
        grid_idx : tuple or None
            Grid index for grid-anchored optimization (None for global)
        start_params_full : np.ndarray
            Full parameter vector for initial evaluation
        seed_history : dict, optional
            Neighbor's optimizer state for warm-starting (Phase 2 feature)
            Expected keys: 'hessian', 'gradient', 'trust_radius'
        start_fitness : float
            Known fitness at starting parameters (default: -inf, will evaluate)
        initial_trust_radius : float
            Starting trust region radius (default: 0.1)
        max_iterations : int
            Maximum BOBYQA iterations (default: 50)
        """
        super().__init__(job_id, job_type, sampler)

        # Problem dimensions
        self.grid_idx = grid_idx
        self.opt_dims = opt_dims
        self.n_opt_dims = len(opt_dims)

        # Starting point
        self.start_params_partial = start_params
        self.start_params_full = start_params_full
        self.start_fitness = start_fitness

        # BOBYQA algorithm parameters
        self.initial_trust_radius = initial_trust_radius
        self.max_iterations = max_iterations
        self.convergence_tolerance = sampler.LBFGSB_ftol  # Reuse L-BFGS-B tol

        # Current optimization state
        self.current_params = start_params.copy()
        self.current_fitness = start_fitness
        self.current_objective = -start_fitness  # Minimization objective

        # Trust region state
        self.trust_radius = initial_trust_radius
        self.min_trust_radius = 1e-6
        self.max_trust_radius = 1.0

        # Quadratic model: f(x) ≈ c + g^T(x-x0) + 0.5*(x-x0)^T*H*(x-x0)
        self.model_center = start_params.copy()
        self.model_constant = 0.0
        self.model_gradient = None  # Will be n_opt_dims vector
        self.model_hessian = None   # Will be n_opt_dims x n_opt_dims matrix

        # Interpolation set (for building model)
        self.interpolation_points = []  # List of (params, fitness) tuples
        self.n_model_points = 2 * self.n_opt_dims + 1  # Standard BOBYQA

        # Iteration tracking
        self.iteration = 0
        self.best_fitness_seen = start_fitness
        self.no_improvement_count = 0

        # State machine status
        if self.type == 'BOBYQA':
            # Full BOBYQA with neighbor testing
            self.status = 'NEEDS_NEIGHBOR_TEST'
            self.neighbor_params_to_test = None
        else:
            # Direct BOBYQA (post-activation, loop, etc.)
            if start_fitness == -np.inf:
                self.status = 'NEEDS_INITIAL_F'
            else:
                self.status = 'NEEDS_INITIAL_MODEL'

        # Seed from neighbor (Phase 2 feature)
        if seed_history:
            self._apply_warm_start(seed_history)

        # Temporary state for multi-point evaluations
        self.pending_evaluations = {}  # {eval_id: {'params': ..., 'purpose': ...}}
        self.next_eval_id = 0

    # ========================================================================
    # Parameter Conversion Methods (same as LBFGSBJob)
    # ========================================================================

    def _get_full_params(self, partial_params):
        """Constructs full parameters from partial optimization parameters."""
        if self.grid_idx is None:
            return partial_params
        else:
            return self.sampler._construct_params(self.grid_idx, partial_params)

    def _get_partial_params_from_full(self, full_params):
        """Extracts optimization parameters from a full parameter vector."""
        return full_params[list(self.opt_dims)]

    def _construct_full_params_for_task(self, partial_params_to_eval):
        """
        Creates the full parameter vector for a task, ensuring bounds.
        """
        if self.grid_idx is None:
            return self.sampler._ensure_bounds(partial_params_to_eval, self.opt_dims)
        else:
            bounded_partial = self.sampler._ensure_bounds(
                partial_params_to_eval,
                self.sampler.continuous_dims
            )
            return self.sampler._construct_params(self.grid_idx, bounded_partial)

    # ========================================================================
    # Warm Starting (Phase 2 Feature - Stub for now)
    # ========================================================================

    def _apply_warm_start(self, seed_history):
        """
        Apply warm-starting from a neighbor's optimizer state.

        Phase 2 feature: Use neighbor's Hessian approximation to seed
        the quadratic model, potentially reducing model-building evaluations.

        Parameters
        ----------
        seed_history : dict
            Contains 'hessian', 'gradient', 'trust_radius' from neighbor
        """
        # Phase 2 implementation:
        # - Use neighbor's Hessian as initial model_hessian
        # - Adjust trust_radius based on neighbor's experience
        # - Potentially skip some interpolation point evaluations

        # For Phase 1: Just log that we received warm-start data
        self.sampler.logger.debug(
            f"BOBYQA job {self.id}: Received warm-start data "
            f"(Phase 2 feature, not yet implemented)"
        )
        pass

    # ========================================================================
    # Trust Region Subproblem Solver
    # ========================================================================

    def _solve_trust_region_subproblem(self):
        """
        Solve the trust region subproblem to find next candidate point.

        Minimize: m(s) = g^T*s + 0.5*s^T*H*s
        Subject to: ||s|| <= trust_radius
                    l <= x0 + s <= u

        Returns
        -------
        s : np.ndarray
            Trust region step
        predicted_reduction : float
            Predicted decrease in objective (for ratio calculation)
        """
        g = self.model_gradient
        H = self.model_hessian
        delta = self.trust_radius
        x0 = self.model_center

        # Get parameter bounds
        if self.grid_idx is None:
            opt_indices = self.opt_dims
        else:
            opt_indices = self.sampler.continuous_dims

        # Bounds on step: l - x0 <= s <= u - x0
        lower_bounds = self.sampler.bounds[opt_indices, 0] - x0
        upper_bounds = self.sampler.bounds[opt_indices, 1] - x0

        # Simple Cauchy point strategy (guaranteed descent)
        # More sophisticated: Could use dogleg, CG-Steihaug, or exact solver

        # Step 1: Compute Cauchy point (steepest descent to TR boundary or bounds)
        if np.linalg.norm(g) < 1e-10:
            # Gradient is zero, no clear direction
            return np.zeros(self.n_opt_dims), 0.0

        # Steepest descent direction (for minimization)
        d = -g / np.linalg.norm(g)

        # Step to trust region boundary
        alpha_tr = delta

        # Step to parameter bounds
        alpha_bounds = np.inf
        for i in range(self.n_opt_dims):
            if d[i] > 0:
                alpha_bounds = min(alpha_bounds, upper_bounds[i] / d[i])
            elif d[i] < 0:
                alpha_bounds = min(alpha_bounds, -lower_bounds[i] / (-d[i]))

        # Take minimum feasible step
        alpha = min(alpha_tr, alpha_bounds)

        # Cauchy step
        s_cauchy = alpha * d

        # Predicted reduction (for accept/reject decision)
        predicted_reduction = -(np.dot(g, s_cauchy) + 0.5 * np.dot(s_cauchy, H @ s_cauchy))

        # TODO (Phase 2): Implement more sophisticated solver
        # - Dogleg method
        # - Conjugate gradient (Steihaug-Toint)
        # - Exact trust region solver (hard case handling)

        return s_cauchy, predicted_reduction

    # ========================================================================
    # Interpolation Set Management
    # ========================================================================

    def _generate_initial_interpolation_points(self):
        """
        Generate points for building initial quadratic model.

        Standard BOBYQA: Use 2*n + 1 points
        - 1 center point (current_params)
        - n coordinate directions (+step)
        - n coordinate directions (-step)

        Returns
        -------
        points : list of np.ndarray
            List of parameter vectors to evaluate
        """
        points = []
        x0 = self.current_params

        # Scale step size based on trust radius
        step_size = self.trust_radius / 2.0

        # Get bounds for each dimension
        if self.grid_idx is None:
            opt_indices = self.opt_dims
        else:
            opt_indices = self.sampler.continuous_dims

        # Coordinate perturbations
        for i in range(self.n_opt_dims):
            # Positive perturbation
            x_plus = x0.copy()
            param_range = (self.sampler.bounds[opt_indices[i], 1] -
                          self.sampler.bounds[opt_indices[i], 0])
            step = min(step_size, param_range / 4.0)  # Don't step too far
            x_plus[i] = x0[i] + step

            # Ensure within bounds
            x_plus[i] = np.clip(x_plus[i],
                               self.sampler.bounds[opt_indices[i], 0],
                               self.sampler.bounds[opt_indices[i], 1])
            points.append(x_plus)

            # Negative perturbation
            x_minus = x0.copy()
            x_minus[i] = x0[i] - step
            x_minus[i] = np.clip(x_minus[i],
                                self.sampler.bounds[opt_indices[i], 0],
                                self.sampler.bounds[opt_indices[i], 1])
            points.append(x_minus)

        return points

    def _build_quadratic_model(self):
        """
        Fit a quadratic model to the interpolation set.

        Model: f(x) ≈ c + g^T(x-x0) + 0.5*(x-x0)^T*H*(x-x0)

        Uses least-squares fitting if we have more points than coefficients,
        otherwise uses exact interpolation.
        """
        if len(self.interpolation_points) < self.n_opt_dims + 1:
            # Not enough points yet
            return

        x0 = self.model_center
        f0 = self.current_objective

        # Collect differences from center
        n_points = len(self.interpolation_points)
        X = np.zeros((n_points, self.n_opt_dims))
        F = np.zeros(n_points)

        for i, (params, fitness) in enumerate(self.interpolation_points):
            X[i, :] = params - x0
            F[i] = -fitness - f0  # Objective difference

        # Fit linear model first: f ≈ f0 + g^T*s
        # Use least squares if overdetermined
        if n_points >= self.n_opt_dims:
            g, _, _, _ = np.linalg.lstsq(X, F, rcond=None)
            self.model_gradient = g
        else:
            # Underdetermined - use minimum norm solution
            self.model_gradient = np.zeros(self.n_opt_dims)

        # Fit Hessian: Use finite difference approximation from coordinate pairs
        H = np.zeros((self.n_opt_dims, self.n_opt_dims))

        # Simple diagonal Hessian from coordinate pairs
        # For each dimension, we have points at x0 ± step
        for i in range(self.n_opt_dims):
            if 2*i+1 < len(self.interpolation_points):
                # Points are ordered: (+dir_0, -dir_0, +dir_1, -dir_1, ...)
                params_plus, f_plus = self.interpolation_points[2*i]
                params_minus, f_minus = self.interpolation_points[2*i+1]

                step = params_plus[i] - x0[i]
                if abs(step) > 1e-10:
                    # Second derivative: (f+ + f- - 2*f0) / step^2
                    H[i, i] = ((-f_plus) + (-f_minus) - 2*f0) / (step**2)

        self.model_hessian = H

        # Ensure positive definite for trust region (add small regularization)
        min_eig = np.min(np.linalg.eigvalsh(H))
        if min_eig < 1e-8:
            H += (1e-6 - min_eig) * np.eye(self.n_opt_dims)
            self.model_hessian = H

    # ========================================================================
    # Accept/Reject Trust Region Step
    # ========================================================================

    def _update_trust_region(self, s, f_new, predicted_reduction):
        """
        Accept or reject trust region step and update radius.

        Parameters
        ----------
        s : np.ndarray
            Step taken
        f_new : float
            Objective value at new point
        predicted_reduction : float
            Predicted reduction from quadratic model

        Returns
        -------
        accepted : bool
            Whether step was accepted
        """
        f_old = self.current_objective
        actual_reduction = f_old - f_new

        # Compute ratio of actual to predicted reduction
        if abs(predicted_reduction) < 1e-10:
            ratio = 0.0
        else:
            ratio = actual_reduction / predicted_reduction

        # Trust region update rules (standard BOBYQA)
        if ratio < 0.1:
            # Poor agreement, shrink trust region
            self.trust_radius *= 0.5
            accepted = False
        elif ratio > 0.75 and np.linalg.norm(s) > 0.9 * self.trust_radius:
            # Excellent agreement and step hit boundary, expand
            self.trust_radius = min(2.0 * self.trust_radius, self.max_trust_radius)
            accepted = True
        elif ratio > 0.1:
            # Acceptable agreement, accept step
            accepted = True
        else:
            accepted = False

        # Ensure trust radius doesn't get too small
        self.trust_radius = max(self.trust_radius, self.min_trust_radius)

        return accepted

    # ========================================================================
    # Job Interface Methods
    # ========================================================================

    def start(self):
        """
        Returns the first task(s) for the job.

        Depending on status, this generates:
        - Neighbor test evaluation
        - Initial fitness evaluation
        - Initial interpolation set evaluations
        """
        # Safety check
        if self.n_opt_dims == 0:
            self.success = False
            self._is_finished = True
            return []

        # --- Neighbor Testing (like LBFGSBJob) ---
        if self.status == 'NEEDS_NEIGHBOR_TEST':
            # Find best neighbor with optimizer state
            best_neighbor_state = None
            best_neighbor_fitness = -np.inf

            for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
                if neighbor_idx in self.sampler.population:
                    neighbor_state = self.sampler.population[neighbor_idx]
                    if (neighbor_state['status'] in ['optimized', 'converged', 'bobyqa_optimized']) and \
                       (neighbor_state.get('optimizer_state') is not None):

                        if neighbor_state['best_fitness'] > best_neighbor_fitness:
                            best_neighbor_fitness = neighbor_state['best_fitness']
                            best_neighbor_state = neighbor_state

            if best_neighbor_state:
                # Test neighbor's parameters at our grid point
                neighbor_best_idx = np.argmax(best_neighbor_state['fitnesses'])
                self.neighbor_params_to_test = best_neighbor_state['continuous_params'][neighbor_best_idx]

                full_params_test = self.sampler._construct_params(
                    self.grid_idx, self.neighbor_params_to_test
                )

                context = {
                    'type': self.type,
                    'job_id': self.id,
                    'sub_type': 'BOBYQA_NEIGHBOR_TEST'
                }
                return [{'params': full_params_test, 'context': context}]
            else:
                # No neighbor found, proceed to initial evaluation
                self.status = 'NEEDS_INITIAL_F'

        # --- Initial Fitness Evaluation ---
        if self.status == 'NEEDS_INITIAL_F':
            context = {
                'type': self.type,
                'job_id': self.id,
                'sub_type': 'BOBYQA_INITIAL_F'
            }
            return [{'params': self.start_params_full, 'context': context}]

        # --- Initial Interpolation Set ---
        if self.status == 'NEEDS_INITIAL_MODEL':
            # Generate all interpolation points
            interp_points = self._generate_initial_interpolation_points()

            tasks = []
            for params in interp_points:
                eval_id = self.next_eval_id
                self.next_eval_id += 1

                full_params = self._construct_full_params_for_task(params)

                context = {
                    'type': self.type,
                    'job_id': self.id,
                    'sub_type': 'BOBYQA_MODEL_POINT',
                    'eval_id': eval_id
                }

                self.pending_evaluations[eval_id] = {
                    'params': params,
                    'purpose': 'initial_model'
                }

                tasks.append({'params': full_params, 'context': context})

            self.status = 'BUILDING_INITIAL_MODEL'
            return tasks

        return []

    def process_result(self, result):
        """
        Process a worker result and return new tasks.

        State machine dispatcher that handles results based on sub_type.
        """
        context = result['context']
        sub_type = context.get('sub_type', 'NONE')
        new_tasks = []

        # --- Neighbor Test Result ---
        if self.status == 'NEEDS_NEIGHBOR_TEST' and sub_type == 'BOBYQA_NEIGHBOR_TEST':
            neighbor_fitness = result['target_val']

            if neighbor_fitness > self.start_fitness:
                # Neighbor's params are better starting point
                self.current_params = self.neighbor_params_to_test
                self.current_fitness = neighbor_fitness
                self.current_objective = -neighbor_fitness
                self.model_center = self.neighbor_params_to_test.copy()
            else:
                # Our params are better
                self.current_fitness = self.start_fitness
                self.current_objective = -self.start_fitness
                self.model_center = self.start_params_partial.copy()

            self.best_fitness_seen = self.current_fitness
            self.status = 'NEEDS_INITIAL_MODEL'
            new_tasks = self.start()  # Trigger model building

        # --- Initial Fitness Result ---
        elif self.status == 'NEEDS_INITIAL_F' and sub_type == 'BOBYQA_INITIAL_F':
            self.current_fitness = result['target_val']
            self.current_objective = -self.current_fitness
            self.best_fitness_seen = self.current_fitness
            self.model_center = self.current_params.copy()

            self.status = 'NEEDS_INITIAL_MODEL'
            new_tasks = self.start()  # Trigger model building

        # --- Model Point Evaluation ---
        elif self.status == 'BUILDING_INITIAL_MODEL' and sub_type == 'BOBYQA_MODEL_POINT':
            eval_id = context['eval_id']
            fitness = result['target_val']

            if eval_id in self.pending_evaluations:
                params = self.pending_evaluations[eval_id]['params']
                self.interpolation_points.append((params, fitness))
                del self.pending_evaluations[eval_id]

            # Check if all model points evaluated
            if len(self.pending_evaluations) == 0:
                # Build the quadratic model
                self._build_quadratic_model()

                # Move to trust region iteration
                self.status = 'NEEDS_TRUST_STEP'
                new_tasks = self._generate_trust_region_step()

        # --- Trust Region Step Evaluation ---
        elif self.status == 'NEEDS_TRUST_STEP' and sub_type == 'BOBYQA_TRUST_STEP':
            f_new = -result['target_val']  # Objective (minimization)
            s = context['step']
            predicted_reduction = context['predicted_reduction']

            # Decide whether to accept step
            accepted = self._update_trust_region(s, f_new, predicted_reduction)

            if accepted:
                # Update current point
                self.current_params = self.model_center + s
                self.current_fitness = result['target_val']
                self.current_objective = f_new
                self.model_center = self.current_params.copy()

                # Add to interpolation set
                self.interpolation_points.append((self.current_params, self.current_fitness))

                # Keep interpolation set bounded
                if len(self.interpolation_points) > self.n_model_points:
                    # Remove oldest point (simple strategy)
                    self.interpolation_points.pop(0)

                # Rebuild model
                self._build_quadratic_model()

                # Track improvement
                if self.current_fitness > self.best_fitness_seen:
                    self.best_fitness_seen = self.current_fitness
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.iteration += 1

            # Check convergence
            if self._check_convergence():
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                return []

            # Generate next trust region step
            self.status = 'NEEDS_TRUST_STEP'
            new_tasks = self._generate_trust_region_step()

        return new_tasks

    def _generate_trust_region_step(self):
        """Generate task for next trust region step."""
        # Solve trust region subproblem
        s, predicted_reduction = self._solve_trust_region_subproblem()

        # Create evaluation task
        x_new = self.model_center + s
        full_params_new = self._construct_full_params_for_task(x_new)

        context = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'BOBYQA_TRUST_STEP',
            'step': s,
            'predicted_reduction': predicted_reduction
        }

        return [{'params': full_params_new, 'context': context}]

    def _check_convergence(self):
        """Check if BOBYQA has converged."""
        # Trust radius too small
        if self.trust_radius < self.min_trust_radius:
            self.sampler.logger.info(f"BOBYQA job {self.id}: Converged (trust radius)")
            return True

        # Max iterations reached
        if self.iteration >= self.max_iterations:
            self.sampler.logger.info(f"BOBYQA job {self.id}: Max iterations reached")
            return True

        # No improvement for many iterations
        if self.no_improvement_count > 10:
            self.sampler.logger.info(f"BOBYQA job {self.id}: Converged (no improvement)")
            return True

        # Gradient very small
        if self.model_gradient is not None and np.linalg.norm(self.model_gradient) < 1e-8:
            self.sampler.logger.info(f"BOBYQA job {self.id}: Converged (gradient)")
            return True

        return False

    def on_finish(self, next_job_id):
        """
        Finalize BOBYQA job and update sampler state.

        Similar to LBFGSBJob.on_finish(), updates population state
        and global solution pool.
        """
        if not self.success:
            # Mark as converged so patching can pick it up
            if self.type in ['BOBYQA', 'BOBYQA_LOOP'] and self.grid_idx in self.sampler.population:
                self.sampler.population[self.grid_idx]['status'] = 'converged'
            return None

        # Save optimizer state for warm-starting neighbors (Phase 2)
        optimizer_state = {
            'hessian': self.model_hessian.copy() if self.model_hessian is not None else None,
            'gradient': self.model_gradient.copy() if self.model_gradient is not None else None,
            'trust_radius': self.trust_radius
        }

        # Update based on job type
        if self.type == 'INITIAL_OPTIMIZATION':
            # Global optimization result
            final_params = self._construct_full_params_for_task(self.current_params)
            self.sampler.initial_maxima.append({
                'point': final_params,
                'target_val': self.current_fitness
            })
            if self.current_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = self.current_fitness
            self.sampler._update_global_pool(final_params, self.current_fitness, grid_idx=None)

        elif self.type in ['BOBYQA', 'BOBYQA_LOOP', 'POST_ACTIVATION_BOBYQA']:
            # Grid-anchored optimization result
            grid_idx = self.grid_idx
            if grid_idx in self.sampler.population:
                state = self.sampler.population[grid_idx]
                state['optimizer_state'] = optimizer_state

                # Mark status
                if self.type == 'BOBYQA_LOOP':
                    state['status'] = 'converged'  # Enable dynamic activation
                else:
                    state['status'] = 'bobyqa_optimized'  # Final state

                # Update best individual
                if self.current_fitness > state['best_fitness']:
                    state['best_fitness'] = self.current_fitness
                    best_idx = np.argmax(state['fitnesses'])
                    state['continuous_params'][best_idx] = self.current_params
                    state['fitnesses'][best_idx] = self.current_fitness
                    self.sampler.profile_likelihood_grid[grid_idx] = self.current_fitness

                    if self.current_fitness > self.sampler.global_max_target_val:
                        self.sampler.global_max_target_val = self.current_fitness

                    # Update global solution pool
                    full_params = self.sampler._construct_params(grid_idx, self.current_params)
                    self.sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        elif self.type == 'REFINEMENT_BOBYQA':
            # Refinement optimization (similar to REFINEMENT_LBFGSB)
            grid_idx = self.grid_idx
            self.sampler.population[grid_idx] = {
                'continuous_params': np.array([self.current_params]),
                'fitnesses': np.array([self.current_fitness]),
                'best_fitness': self.current_fitness,
                'status': 'bobyqa_optimized',
                'improvement_history': [],
                'last_update_gen': 0,
                'optimizer_state': optimizer_state
            }

            self.sampler.profile_likelihood_grid[grid_idx] = self.current_fitness

            if self.current_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = self.current_fitness

            full_params = self.sampler._construct_params(grid_idx, self.current_params)
            self.sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        return None


# ============================================================================
# Helper Exception for Function Interception (Alternative Approach)
# ============================================================================

class PendingEvaluationException(Exception):
    """
    Raised when BOBYQA algorithm requests a function evaluation.

    This exception allows us to intercept synchronous function calls
    and convert them into asynchronous tasks.

    Alternative to batch evaluation approach shown above.
    """
    def __init__(self, task_id):
        self.task_id = task_id
        super().__init__(f"Pending evaluation for task {task_id}")


# ============================================================================
# Usage Example (Integration with Sampler)
# ============================================================================

"""
# In sampler.py:

def create_post_activation_bobyqa_jobs(self, next_job_id):
    '''Create BOBYQA jobs for all activated grid points.'''
    from .jobs.bobyqa_job import BOBYQAJob

    jobs = []
    for grid_idx, state in self.population.items():
        if state['status'] == 'active':
            # Skip if no continuous dimensions
            if self.direct_eval_mode or self.n_cont_dims == 0:
                state['status'] = 'bobyqa_optimized'
                continue

            # Mark as claimed
            state['status'] = 'BOBYQA_queued'

            # Find best individual to start from
            best_ind_idx = np.argmax(state['fitnesses'])
            start_params_partial = state['continuous_params'][best_ind_idx]
            start_fitness = state['fitnesses'][best_ind_idx]
            start_params_full = self._construct_params(grid_idx, start_params_partial)

            # Create BOBYQA job
            job = BOBYQAJob(
                job_id=next_job_id,
                job_type='POST_ACTIVATION_BOBYQA',
                sampler=self,
                opt_dims=tuple(self.continuous_dims),
                start_params=start_params_partial,
                grid_idx=grid_idx,
                start_params_full=start_params_full,
                start_fitness=start_fitness,
                initial_trust_radius=self.bobyqa_initial_trust_radius,
                max_iterations=self.bobyqa_max_iterations
            )
            jobs.append(job)
            next_job_id += 1

    self.logger.info(f"Created {len(jobs)} post-activation BOBYQA jobs")
    return jobs, next_job_id


# In master.py workflow:

elif stage == 'POST_ACTIVATION_BOBYQA':
    if sampler.optimization_method == 'bobyqa':
        jobs, next_job_id = sampler.create_post_activation_bobyqa_jobs(next_job_id)
        # ... (same job queue logic as other stages)
"""
