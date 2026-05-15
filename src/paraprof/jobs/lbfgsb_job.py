"""
L-BFGS-B optimization job for asynchronous gradient-based optimization.
"""
import numpy as np
import collections
from .base import Job


class LBFGSBJob(Job):
    """
    A self-contained job for running an asynchronous L-BFGS-B optimization.
    This single class handles all the logic for initial fitness evaluation,
    gradient calculation, line searching, and history updates.

    For LBFGSB jobs, it includes logic to test a neighbor's parameters
    and seed the Hessian (s_hist, y_hist) from a converged neighbor.
    """
    def __init__(self, job_id, job_type, sampler, opt_dims, start_params,
                 grid_idx, start_params_full, seed_history=None, start_fitness=-np.inf):

        super().__init__(job_id, job_type, sampler)

        # L-BFGS-B parameters from sampler
        self.lbfgsb_ftol = sampler.lbfgsb_ftol
        self.lbfgsb_max_iter = sampler.lbfgsb_max_iter
        self.lbfgsb_gradient_method = sampler.lbfgsb_gradient_method
        # When True, piggyback compute_gradient=True on initial-f, neighbor-test,
        # and first-attempt line-search tasks so the worker also returns
        # grad_func(params) and we can skip FD for the dims the user provided.
        self.has_user_grad = sampler.grad_func is not None

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
        if self.type == 'LBFGSB':
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
        # Sign-flipped user gradient (∇objective = -∇target_func) at
        # ``current_params``, sliced to opt_dims. NaN entries signal "use
        # finite differences for this dim". Set by ``_seed_user_gradient``
        # when a value+grad result for the current x is received.
        self._user_grad_opt = None

        self.s_hist = collections.deque(maxlen=10)
        self.y_hist = collections.deque(maxlen=10)
        if seed_history:
            self.s_hist.extend(seed_history['s'])
            self.y_hist.extend(seed_history['y'])

        self.iteration = 0

        # Pre-allocate array for epsilon calculations (performance optimization)
        self._eps_array = np.empty(self.n_opt_dims)

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
            # Grid-anchored: partial_params_to_eval is *only* the profiled dims
            # We must ensure *they* are bounded.
            bounded_partial = self.sampler._ensure_bounds(
                partial_params_to_eval,
                self.sampler.profiled_dims
            )
            return self.sampler._construct_params(self.grid_idx, bounded_partial)


    def start(self):
        """Returns the first task(s) for the job."""
        # Safety check: if there are no dimensions to optimize, finish immediately.
        # The master loop has a defensive cleanup for this case; log a warning so
        # any future caller that hits this path is visible rather than silent.
        if self.n_opt_dims == 0:
            self.sampler.logger.warning(
                f"LBFGSBJob {self.id} ({self.type}) created with zero opt_dims; "
                "finishing immediately. This usually indicates the caller should "
                "be using a direct evaluation path instead."
            )
            self.success = False
            self._is_finished = True
            return []

        if self.status == 'NEEDS_NEIGHBOR_TEST':
            best_neighbor_state = None
            best_neighbor_fitness = -np.inf

            for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
                if neighbor_idx in self.sampler.population:
                    neighbor_state = self.sampler.population[neighbor_idx]
                    # Check if neighbor is 'optimized' (post-LBFGSB) or 'converged' (post-DE)
                    # and has an optimizer state to seed from.
                    if (neighbor_state['status'] in ['optimized', 'converged']) and \
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
                self.neighbor_params_to_test = best_neighbor_state['profiled_params'][neighbor_best_idx]

                # 3. Create a task to test these params at *our* grid point
                full_params_test = self.sampler._construct_params(self.grid_idx, self.neighbor_params_to_test)

                context = {
                    'type': self.type,
                    'job_id': self.id,
                    'sub_type': 'LBFGS_NEIGHBOR_TEST',
                    'compute_gradient': self.has_user_grad,
                }
                return [{'params': full_params_test, 'context': context}]

            else:
                # No valid neighbor found, skip test and just evaluate our own params
                self.status = 'NEEDS_INITIAL_F'
                # Fall through to the 'NEEDS_INITIAL_F' logic

        if self.status == 'NEEDS_INITIAL_F':
            # This is the fallback: evaluate the starting params
            # (either for global opt, or for optimization with no neighbors)
            context = {
                'type': self.type,
                'job_id': self.id,
                'sub_type': 'LBFGS_INITIAL_F',
                'compute_gradient': self.has_user_grad,
            }
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
                user_grad_for_current = result.get('user_gradient')
            else:
                # Our original params are better — the gradient that came
                # back with the neighbor test was at the neighbor's params,
                # not ours, so discard it.
                self.current_params = self.fallback_params
                self.current_fitness = self.start_fitness
                self.current_objective = -self.start_fitness
                user_grad_for_current = None

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks(
                user_grad_full=user_grad_for_current
            )

        elif self.status == 'NEEDS_INITIAL_F' and sub_type == 'LBFGS_INITIAL_F':
            # Came here from global opt or optimization with no neighbors
            self.current_fitness = result['target_val']
            self.current_objective = -self.current_fitness
            self.current_params = self._get_partial_params_from_full(result['params'])

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks(
                user_grad_full=result.get('user_gradient')
            )

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

    def _seed_user_grad_opt(self, user_grad_full):
        """Slice a full-dim user gradient (``∇target_func``) to opt_dims and
        flip to the objective frame (``∇objective = -∇target``). Missing
        entries (NaN) stay NaN — those dims still need FD.

        Returns a length-``n_opt_dims`` array. If ``user_grad_full`` is
        None or unusable, returns an all-NaN array (FD for all dims).
        """
        out = np.full(self.n_opt_dims, np.nan)
        if user_grad_full is None:
            return out
        try:
            arr = np.asarray(user_grad_full, dtype=float)
        except (TypeError, ValueError):
            return out
        if arr.size < (max(self.opt_dims) + 1 if self.n_opt_dims else 0):
            return out
        sliced = arr[list(self.opt_dims)]
        # Negate to convert ∇target → ∇objective. NaN stays NaN.
        with np.errstate(invalid='ignore'):
            out = np.where(np.isfinite(sliced), -sliced, np.nan)
        return out

    def _calculate_gradient_tasks(self, base_eps=1e-8, user_grad_full=None):
        """Generates tasks needed to compute the gradient at ``current_params``.

        When ``user_grad_full`` is provided (a full-dim user gradient already
        evaluated at this x), FD tasks are issued only for the dimensions
        the user did not supply. Each skipped FD task is counted in
        ``sampler.target_calls_saved_by_user_gradient``. If the user covered
        every opt dim, no FD tasks are issued and the gradient is finalized
        immediately (the job transitions directly to the line-search task).
        """
        tasks = []
        x = self.current_params
        self.gradient_components = {}

        # Adaptive step size per dimension: scale with parameter magnitude
        # This ensures numerical stability across parameters with different scales
        # Use pre-allocated array for performance
        np.maximum(np.abs(x) * base_eps, 1e-10, out=self._eps_array)
        eps = self._eps_array

        # Store eps array for later use in gradient reconstruction
        self.current_eps = eps

        self._user_grad_opt = self._seed_user_grad_opt(user_grad_full)

        # Per-dim FD cost in the active method, used to count savings.
        per_dim_fd_cost = 2 if self.lbfgsb_gradient_method == "central" else 1
        if self.lbfgsb_gradient_method not in ("central", "forward"):
            raise Exception(f"Gradient method {self.lbfgsb_gradient_method} not implemented.")

        self.pending_grad_evals = 0
        for i in range(self.n_opt_dims):
            if np.isfinite(self._user_grad_opt[i]):
                # User supplied this component; skip FD entirely.
                self.sampler.target_calls_saved_by_user_gradient += per_dim_fd_cost
                continue

            # FD path for this dim.
            x_plus = x.copy()
            x_plus[i] += eps[i]
            full_params_plus = self._construct_full_params_for_task(x_plus)
            tasks.append({
                'params': full_params_plus,
                'context': {'type': self.type, 'job_id': self.id,
                            'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': 1},
            })
            self.pending_grad_evals += 1

            if self.lbfgsb_gradient_method == "central":
                x_minus = x.copy()
                x_minus[i] -= eps[i]
                full_params_minus = self._construct_full_params_for_task(x_minus)
                tasks.append({
                    'params': full_params_minus,
                    'context': {'type': self.type, 'job_id': self.id,
                                'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': -1},
                })
                self.pending_grad_evals += 1

        # If every dim is covered by the user gradient, there is nothing to
        # wait for — finalize the gradient now and emit the first line-search
        # task.
        if self.pending_grad_evals == 0:
            return self._finalize_gradient_and_line_search()

        return tasks

    def _process_gradient_result(self, result):
        """Processes a returned likelihood evaluation for a gradient calculation."""
        context = result['context']
        dim, sign = context['dim'], context['sign']

        self.gradient_components[(dim, sign)] = -result['target_val'] # Store objective
        self.pending_grad_evals -= 1

        # Check if all components for the gradient have been computed
        if self.pending_grad_evals == 0:
            return self._finalize_gradient_and_line_search()

        return [] # Not ready yet, no new task

    def _finalize_gradient_and_line_search(self):
        """Assemble the full gradient (mixing user-supplied components with FD
        components), update the L-BFGS history, compute the search direction,
        and return the first line-search task. Shared by the all-FD,
        partial-user-grad, and full-user-grad paths.
        """
        grad = np.zeros(self.n_opt_dims)
        f = self.current_objective

        for i in range(self.n_opt_dims):
            if self._user_grad_opt is not None and np.isfinite(self._user_grad_opt[i]):
                # Already in the objective frame (negated when seeded).
                grad[i] = self._user_grad_opt[i]
                continue
            if self.lbfgsb_gradient_method == "central":
                f_plus = self.gradient_components[(i, 1)]
                f_minus = self.gradient_components[(i, -1)]
                grad[i] = (f_plus - f_minus) / (2 * self.current_eps[i])
            else:  # "forward"
                f_plus = self.gradient_components[(i, 1)]
                grad[i] = (f_plus - f) / self.current_eps[i]

        self.current_gradient = grad

        # --- History update (if pending from line search) ---
        if self.pending_s_k is not None:
            s_k = self.pending_s_k
            g_old = self.pending_g_old
            g_new = self.current_gradient
            y_k = g_new - g_old
            # Wolfe/relative curvature condition: require y·s to be a
            # meaningful fraction of ||y||*||s||, not just > tiny absolute.
            ys = float(np.dot(y_k, s_k))
            yn = float(np.linalg.norm(y_k))
            sn = float(np.linalg.norm(s_k))
            if ys > 1e-10 * max(yn * sn, 1e-30):
                self.s_hist.append(s_k)
                self.y_hist.append(y_k)
            # Clear pending
            self.pending_s_k = None
            self.pending_g_old = None
        # --- End History update ---

        # --- L-BFGS two-loop recursion to find search direction ---
        # Pairs that fail the curvature condition are skipped defensively
        # so that a degenerate dot product can never produce inf/nan rho.
        q = grad
        a = []
        usable_pairs = []
        for s, y in zip(self.s_hist, self.y_hist):
            ys = float(np.dot(y, s))
            if ys > 1e-30:
                usable_pairs.append((s, y, ys))

        for s, y, ys in reversed(usable_pairs):
            rho = 1.0 / ys
            alpha = rho * np.dot(s, q)
            q = q - alpha * y
            a.append(alpha)

        if usable_pairs:
            s_last, y_last, _ = usable_pairs[-1]
            yy = float(np.dot(y_last, y_last))
            if yy > 1e-30:
                gamma = float(np.dot(s_last, y_last)) / yy
            else:
                gamma = 1.0
            z = gamma * q
        else:
            z = q

        for (s, y, ys), alpha in zip(usable_pairs, reversed(a)):
            rho = 1.0 / ys
            beta = rho * np.dot(y, z)
            z = z + s * (alpha - beta)

        self.search_direction = -z
        self.status = 'NEEDS_LINE_SEARCH'
        self.line_search_alpha = 1.0 # Reset for new line search

        # Return a new task for the first step of the line search
        return [self._calculate_line_search_task()]

    def _calculate_line_search_task(self):
        """Generates the next task for a backtracking line search."""
        alpha = self.line_search_alpha
        x = self.current_params
        d = self.search_direction
        x_new = x + alpha * d

        # We must construct the *full* params for the task
        full_params_new = self._construct_full_params_for_task(x_new)

        # Piggyback the user gradient request on every line-search attempt.
        # The whole point of grad_func is to skip an FD round once a step
        # is accepted, so we want the gradient at WHATEVER alpha the line
        # search ends up accepting. Asking on every attempt wastes some
        # grad_func calls on rejected steps, but that cost is bounded
        # (~ log2(1/alpha) per iteration) and small relative to the FD
        # round it saves.
        context = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'LBFGS_LINE_SEARCH',
            'alpha': alpha,
            'compute_gradient': self.has_user_grad,
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
            opt_indices = self.sampler.profiled_dims

        x_new_bounded = self.sampler._ensure_bounds(x_new, opt_indices)

        # Armijo condition check
        if f_new <= f_old + c1 * alpha * np.dot(g_old, x_new_bounded - x_old):
            # Step accepted, move to the next L-BFGS iteration
            self.iteration += 1

            # Check for convergence
            if self.iteration >= self.lbfgsb_max_iter or np.abs(f_old - f_new) < self.lbfgsb_ftol:
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

            # Generate tasks to calculate the new gradient. The worker may
            # already have computed grad_func at x_new (piggybacked on the
            # first-attempt line-search task); pass it through so FD only
            # fires for the dims the user did not supply.
            self.status = 'NEEDS_GRADIENT'
            return self._calculate_gradient_tasks(
                user_grad_full=result.get('user_gradient')
            )

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
            # For optimization, if it fails, set status back to 'converged'
            # so it can be picked up by patching later.
            if self.type in ['LBFGSB', 'PATCHING_LBFGSB'] and self.grid_idx in self.sampler.population:
                self.sampler.population[self.grid_idx]['status'] = 'converged'
            return None # Don't record failed jobs

        if self.type == 'INITIAL_OPTIMIZATION':
            final_params = self._construct_full_params_for_task(self.current_params)
            final_target_val = self.current_fitness

            self.sampler.initial_maxima.append({'point': final_params, 'target_val': final_target_val})
            if final_target_val > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = final_target_val

            # Update global solution pool with discovered maximum
            self.sampler._update_global_pool(final_params, final_target_val, grid_idx=None)

        elif self.type in ['LBFGSB', 'PATCHING_LBFGSB', 'LBFGSB_LOOP', 'POST_ACTIVATION_LBFGSB']:
            grid_idx = self.grid_idx
            if grid_idx in self.sampler.population:
                state = self.sampler.population[grid_idx]
                state['optimizer_state'] = {'s': list(self.s_hist), 'y': list(self.y_hist)}

                # For LBFGSB_LOOP, mark as 'converged' to enable dynamic activation
                # For others, mark as 'optimized' (final state)
                if self.type == 'LBFGSB_LOOP':
                    state['status'] = 'converged'
                else:
                    state['status'] = 'optimized'

                # Update the best individual with the optimized result
                if self.current_fitness > state['best_fitness']:
                     state['best_fitness'] = self.current_fitness
                     best_idx = np.argmax(state['fitnesses'])
                     state['profiled_params'][best_idx] = self.current_params
                     state['fitnesses'][best_idx] = self.current_fitness
                     self.sampler.profile_likelihood_grid[self.grid_idx] = self.current_fitness

                     if self.current_fitness > self.sampler.global_max_target_val:
                         self.sampler.global_max_target_val = self.current_fitness

                     # Update global solution pool with optimized solution
                     # Construct full parameter vector for the pool
                     full_params = self.sampler._construct_params(self.grid_idx, self.current_params)
                     self.sampler._update_global_pool(full_params, self.current_fitness, self.grid_idx)

        elif self.type == 'REFINEMENT_LBFGSB':
            # For refinement, we create a new population entry directly (no prior DE)
            grid_idx = self.grid_idx

            # Create a minimal population state for this grid point
            self.sampler.population[grid_idx] = {
                'profiled_params': np.array([self.current_params]),
                'fitnesses': np.array([self.current_fitness]),
                'best_fitness': self.current_fitness,
                'status': 'optimized',
                'improvement_history': [],
                'last_update_gen': 0,
                'optimizer_state': {'s': list(self.s_hist), 'y': list(self.y_hist)}
            }

            # Update profile likelihood grid
            self.sampler.profile_likelihood_grid[grid_idx] = self.current_fitness

            # Update global maximum if needed
            if self.current_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = self.current_fitness

            # Update global solution pool
            full_params = self.sampler._construct_params(grid_idx, self.current_params)
            self.sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        return None # This job doesn't spawn children
