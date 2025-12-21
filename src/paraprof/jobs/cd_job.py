"""
Ultra-fast Coordinate Descent job for refinement warm-start optimization.

This job implements a minimal coordinate descent algorithm designed for
fine-tuning interpolated starting points during grid refinement. It uses
a simple 3-point parabolic line search for speed.
"""
import numpy as np
from .base import Job


class CoordinateDescentJob(Job):
    """
    Fast coordinate descent optimizer for warm-start refinement.

    This job performs a few cycles of coordinate descent with minimal
    line searches (3-4 evaluations per coordinate) to quickly improve
    starting points that are already close to the optimum.

    Designed specifically for grid refinement where interpolated starting
    points are typically very close to the true optimum.
    """

    def __init__(self, job_id, job_type, sampler, opt_dims, start_params,
                 grid_idx, start_params_full, start_fitness=-np.inf,
                 max_cycles=3, step_fraction=0.01):
        """
        Initialize the coordinate descent job.

        Parameters
        ----------
        job_id : int
            Unique job identifier
        job_type : str
            Job type identifier
        sampler : ProfileProjector
            Reference to main sampler
        opt_dims : tuple
            Dimensions to optimize (relative to full params)
        start_params : np.ndarray
            Starting parameters (partial, just the optimization dimensions)
        grid_idx : tuple
            Grid index for grid-anchored optimization
        start_params_full : np.ndarray
            Full parameter vector for initial evaluation
        start_fitness : float
            Known fitness of starting parameters (default: -inf, will evaluate)
        max_cycles : int
            Maximum number of complete coordinate cycles (default: 3)
        step_fraction : float
            Initial step size as fraction of parameter range (default: 0.01)
        """
        super().__init__(job_id, job_type, sampler)

        self.grid_idx = grid_idx
        self.opt_dims = opt_dims
        self.n_opt_dims = len(opt_dims)

        self.start_params_partial = start_params
        self.start_params_full = start_params_full
        self.start_fitness = start_fitness

        # CD algorithm parameters
        self.max_cycles = max_cycles
        self.step_fraction = step_fraction

        # Current state
        self.status = 'NEEDS_INITIAL_F' if start_fitness == -np.inf else 'READY_FOR_CYCLE'
        self.current_params = start_params.copy()
        self.current_fitness = start_fitness

        # Cycle/coordinate tracking
        self.current_cycle = 0
        self.current_dim_idx = 0
        self.coordinate_order = None  # Will be randomized each cycle

        # Line search state for current coordinate
        self.line_search_state = None

        # Convergence tracking
        self.fitness_at_cycle_start = start_fitness
        self.improvement_tolerance = sampler.lbfgsb_ftol  # Reuse L-BFGS-B tolerance

    def _get_full_params(self, partial_params):
        """Constructs full parameters from partial optimization parameters."""
        if self.grid_idx is None:
            return partial_params
        else:
            return self.sampler._construct_params(self.grid_idx, partial_params)

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

    def _get_step_size(self, dim_idx):
        """
        Calculate adaptive step size for a given dimension.

        Uses fraction of the parameter's allowed range for robustness.
        """
        opt_indices = self.opt_dims if self.grid_idx is None else self.sampler.continuous_dims
        param_idx = opt_indices[dim_idx]

        param_range = self.sampler.bounds[param_idx, 1] - self.sampler.bounds[param_idx, 0]
        return self.step_fraction * param_range

    def _start_new_cycle(self):
        """Initialize a new coordinate descent cycle."""
        self.current_cycle += 1
        self.current_dim_idx = 0
        self.fitness_at_cycle_start = self.current_fitness

        # Randomize coordinate order for this cycle
        self.coordinate_order = np.random.permutation(self.n_opt_dims)

    def _start_coordinate_line_search(self):
        """
        Start a line search along the current coordinate.

        Uses a simple 3-point parabolic fit strategy:
        1. Evaluate at current point (already known)
        2. Evaluate at +step
        3. Evaluate at -step
        4. Fit parabola and step to predicted minimum
        """
        dim_idx = self.coordinate_order[self.current_dim_idx]
        step = self._get_step_size(dim_idx)

        # Initialize line search state
        self.line_search_state = {
            'dim_idx': dim_idx,
            'step': step,
            'x0': self.current_params[dim_idx],
            'f0': self.current_fitness,
            'evaluations': {},
            'pending': ['plus', 'minus']
        }

        # Generate tasks for +step and -step
        tasks = []

        # Positive step
        params_plus = self.current_params.copy()
        params_plus[dim_idx] = self.current_params[dim_idx] + step
        full_params_plus = self._construct_full_params_for_task(params_plus)
        context_plus = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'CD_LINE_SEARCH',
            'direction': 'plus'
        }
        tasks.append({'params': full_params_plus, 'context': context_plus})

        # Negative step
        params_minus = self.current_params.copy()
        params_minus[dim_idx] = self.current_params[dim_idx] - step
        full_params_minus = self._construct_full_params_for_task(params_minus)
        context_minus = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'CD_LINE_SEARCH',
            'direction': 'minus'
        }
        tasks.append({'params': full_params_minus, 'context': context_minus})

        return tasks

    def _finish_coordinate_line_search(self):
        """
        Complete the line search using parabolic interpolation.

        Returns
        -------
        new_tasks : list
            New tasks if refinement needed, empty list otherwise
        """
        state = self.line_search_state
        dim_idx = state['dim_idx']
        step = state['step']
        x0 = state['x0']
        f0 = state['f0']
        f_plus = state['evaluations']['plus']
        f_minus = state['evaluations']['minus']

        # Simple strategy: pick the best of the three points
        best_offset = 0.0
        best_fitness = f0

        if f_plus > best_fitness:
            best_offset = step
            best_fitness = f_plus
        if f_minus > best_fitness:
            best_offset = -step
            best_fitness = f_minus

        # Parabolic refinement if we found improvement
        if best_offset != 0.0:
            # Fit parabola through 3 points: (-step, f_minus), (0, f0), (+step, f_plus)
            # f(x) = a*x^2 + b*x + c
            # Minimum at x = -b/(2*a)

            # Using equally spaced points, coefficients simplify:
            a = (f_plus + f_minus - 2*f0) / (2 * step**2)
            b = (f_plus - f_minus) / (2 * step)

            # Only use parabolic minimum if curvature is negative (concave down for maximization)
            if a < 0:
                x_min = -b / (2 * a)
                # Clamp to reasonable range (within 2*step)
                x_min = np.clip(x_min, -2*step, 2*step)

                # Evaluate at parabolic minimum if it's different from our best point
                if abs(x_min - best_offset) > 0.1 * step:
                    params_refined = self.current_params.copy()
                    params_refined[dim_idx] = x0 + x_min
                    full_params_refined = self._construct_full_params_for_task(params_refined)

                    context_refined = {
                        'type': self.type,
                        'job_id': self.id,
                        'sub_type': 'CD_PARABOLIC_REFINEMENT',
                        'x_min': x_min
                    }

                    # Mark that we're waiting for parabolic refinement
                    state['pending'] = ['parabolic']
                    state['x_min'] = x_min

                    return [{'params': full_params_refined, 'context': context_refined}]

        # No parabolic refinement needed, update parameters
        self.current_params[dim_idx] = x0 + best_offset
        self.current_fitness = best_fitness
        self.line_search_state = None

        return []

    def start(self):
        """Returns the first task(s) for the job."""
        if self.n_opt_dims == 0:
            self.success = False
            self._is_finished = True
            return []

        if self.status == 'NEEDS_INITIAL_F':
            # Evaluate starting fitness
            context = {
                'type': self.type,
                'job_id': self.id,
                'sub_type': 'CD_INITIAL_F'
            }
            return [{'params': self.start_params_full, 'context': context}]

        elif self.status == 'READY_FOR_CYCLE':
            # Start first cycle
            self._start_new_cycle()
            self.status = 'IN_LINE_SEARCH'
            return self._start_coordinate_line_search()

        return []

    def process_result(self, result):
        """Process a worker result and return new tasks."""
        context = result['context']
        sub_type = context.get('sub_type', 'NONE')
        new_tasks = []

        if sub_type == 'CD_INITIAL_F':
            # Initial fitness evaluated
            self.current_fitness = result['target_val']
            self.fitness_at_cycle_start = self.current_fitness
            self.status = 'READY_FOR_CYCLE'

            # Start first cycle
            self._start_new_cycle()
            self.status = 'IN_LINE_SEARCH'
            new_tasks = self._start_coordinate_line_search()

        elif sub_type == 'CD_LINE_SEARCH':
            # Store line search evaluation
            direction = context['direction']
            self.line_search_state['evaluations'][direction] = result['target_val']
            self.line_search_state['pending'].remove(direction)

            # Check if all line search evaluations complete
            if len(self.line_search_state['pending']) == 0:
                new_tasks = self._finish_coordinate_line_search()

                # If no parabolic refinement needed, move to next coordinate
                if len(new_tasks) == 0:
                    new_tasks = self._advance_to_next_coordinate()

        elif sub_type == 'CD_PARABOLIC_REFINEMENT':
            # Parabolic refinement result
            f_parabolic = result['target_val']
            state = self.line_search_state

            dim_idx = state['dim_idx']
            x0 = state['x0']
            x_min = state['x_min']

            # Compare with best simple step
            f_plus = state['evaluations']['plus']
            f_minus = state['evaluations']['minus']
            f0 = state['f0']

            best_simple = max(f0, f_plus, f_minus)

            if f_parabolic > best_simple:
                self.current_params[dim_idx] = x0 + x_min
                self.current_fitness = f_parabolic
            else:
                if f_plus > best_simple:
                    self.current_params[dim_idx] = x0 + state['step']
                    self.current_fitness = f_plus
                elif f_minus > best_simple:
                    self.current_params[dim_idx] = x0 - state['step']
                    self.current_fitness = f_minus

            self.line_search_state = None
            new_tasks = self._advance_to_next_coordinate()

        return new_tasks

    def _advance_to_next_coordinate(self):
        """Move to next coordinate or next cycle."""
        self.current_dim_idx += 1

        if self.current_dim_idx >= self.n_opt_dims:
            # Completed a full cycle
            improvement = self.current_fitness - self.fitness_at_cycle_start

            # Check convergence
            if improvement < self.improvement_tolerance or self.current_cycle >= self.max_cycles:
                # Converged or max cycles reached
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                return []
            else:
                # Start next cycle
                self._start_new_cycle()
                return self._start_coordinate_line_search()
        else:
            # Start line search for next coordinate
            return self._start_coordinate_line_search()

    def on_finish(self, next_job_id):
        """
        Finalize the CD job and optionally chain to L-BFGS-B.

        For refinement jobs, we create the population entry directly.
        For other job types, behavior can be customized.
        """
        if not self.success:
            return None

        if self.type == 'REFINEMENT_CD':
            # Create population entry for refined grid point
            grid_idx = self.grid_idx

            self.sampler.population[grid_idx] = {
                'continuous_params': np.array([self.current_params]),
                'fitnesses': np.array([self.current_fitness]),
                'best_fitness': self.current_fitness,
                'status': 'cd_optimized',  # Mark as CD-optimized
                'improvement_history': [],
                'last_update_gen': 0,
                'optimizer_state': None  # No L-BFGS-B history
            }

            # Update profile likelihood grid
            self.sampler.profile_likelihood_grid[grid_idx] = self.current_fitness

            # Update global maximum if needed
            if self.current_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = self.current_fitness

            # Update global solution pool
            full_params = self.sampler._construct_params(grid_idx, self.current_params)
            self.sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        return None
