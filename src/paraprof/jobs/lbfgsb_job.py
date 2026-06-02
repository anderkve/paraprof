"""L-BFGS-B optimization job for asynchronous gradient-based optimization."""
import numpy as np
import collections
from .base import Job


class LBFGSBJob(Job):
    """Self-contained async L-BFGS-B run (initial eval, gradient, line search, history).

    For grid-anchored LBFGSB jobs the run starts with a neighbor-test
    step: if a converged neighbor has better fitness at our cell, we
    adopt its params and seed the (s, y) history from its optimizer state.
    """
    def __init__(self, job_id, job_type, sampler, opt_dims, start_params,
                 grid_idx, start_params_full, seed_history=None, start_fitness=-np.inf):

        super().__init__(job_id, job_type, sampler)

        self.lbfgsb_ftol = sampler.lbfgsb_ftol
        self.lbfgsb_max_iter = sampler.lbfgsb_max_iter
        self.lbfgsb_gradient_method = sampler.lbfgsb_gradient_method
        # When True we piggyback compute_gradient=True on value-evaluating tasks.
        self.has_user_grad = sampler.grad_func is not None

        self.grid_idx = grid_idx
        self.opt_dims = opt_dims
        self.n_opt_dims = len(opt_dims)

        # start_params are the partial parameters corresponding to opt_dims;
        # start_params_full is the full vector for the initial eval.
        self.start_params_partial = start_params
        self.start_params_full = start_params_full
        self.start_fitness = start_fitness
        self.improvement = 0.0

        if self.type == 'LBFGSB':
            self.status = 'NEEDS_NEIGHBOR_TEST'
            self.fallback_params = self.start_params_partial
        else:
            self.status = 'NEEDS_INITIAL_F'

        self.current_params = self.start_params_partial
        self.current_fitness = -np.inf   # likelihood (maximize)
        self.current_objective = np.inf  # objective = -likelihood (minimize)

        self.gradient_components = {}
        self.pending_grad_evals = 0
        self.current_gradient = None
        # User gradient sliced to opt_dims, in the objective frame; NaN = "use FD".
        self._user_grad_opt = None

        self.s_hist = collections.deque(maxlen=10)
        self.y_hist = collections.deque(maxlen=10)
        if seed_history:
            self.s_hist.extend(seed_history['s'])
            self.y_hist.extend(seed_history['y'])

        self.iteration = 0
        self._eps_array = np.empty(self.n_opt_dims)

        self.search_direction = None
        self.line_search_alpha = 1.0
        self.pending_s_k = None
        self.pending_g_old = None

        self.neighbor_params_to_test = None

    def _get_full_params(self, partial_params):
        """Build full params from the opt-dim slice (passthrough for global opt)."""
        if self.grid_idx is None:
            return partial_params
        return self.sampler._construct_params(self.grid_idx, partial_params)

    def _get_partial_params_from_full(self, full_params):
        """Slice a full parameter vector down to opt_dims."""
        return full_params[list(self.opt_dims)]

    def _construct_full_params_for_task(self, partial_params_to_eval):
        """Full bounded parameter vector for a worker task (global or grid-anchored)."""
        if self.grid_idx is None:
            return self.sampler._ensure_bounds(partial_params_to_eval, self.opt_dims)
        bounded_partial = self.sampler._ensure_bounds(
            partial_params_to_eval, self.sampler.profiled_dims,
        )
        return self.sampler._construct_params(self.grid_idx, bounded_partial)


    def start(self):
        """Return the first task(s) for this job."""
        # The master loop also defensively cleans up zero-opt-dim jobs, but
        # warn here so any future caller hitting this path is visible.
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
                self.current_params = self.neighbor_params_to_test
                self.current_fitness = neighbor_fitness
                self.current_objective = -neighbor_fitness
                user_grad_for_current = result.get('user_gradient')
            else:
                # Fall back to our own params; discard the gradient since it
                # was computed at the neighbor's params, not ours.
                self.current_params = self.fallback_params
                self.current_fitness = self.start_fitness
                self.current_objective = -self.start_fitness
                user_grad_for_current = None

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks(user_grad_full=user_grad_for_current)

        elif self.status == 'NEEDS_INITIAL_F' and sub_type == 'LBFGS_INITIAL_F':
            self.current_fitness = result['target_val']
            self.current_objective = -self.current_fitness
            self.current_params = self._get_partial_params_from_full(result['params'])

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks(user_grad_full=result.get('user_gradient'))

        elif self.status == 'NEEDS_GRADIENT' and sub_type == 'LBFGS_GRADIENT':
            new_tasks = self._process_gradient_result(result)

        elif self.status == 'NEEDS_LINE_SEARCH' and sub_type == 'LBFGS_LINE_SEARCH':
            new_tasks = self._process_line_search_result(result)

        return new_tasks

    def _seed_user_grad_opt(self, user_grad_full):
        """Slice the user gradient (∇target_func) to opt_dims and flip the
        sign for the objective. NaN entries stay NaN (need FD)."""
        if user_grad_full is None:
            return np.full(self.n_opt_dims, np.nan)
        sliced = np.asarray(user_grad_full)[list(self.opt_dims)]
        return np.where(np.isfinite(sliced), -sliced, np.nan)

    def _calculate_gradient_tasks(self, base_eps=1e-8, user_grad_full=None):
        """Generates tasks for the gradient at ``current_params``. Issues FD
        tasks only for dims the user did not supply; finalizes immediately
        if the user covered every dim."""
        if self.lbfgsb_gradient_method not in ("central", "forward"):
            raise Exception(f"Gradient method {self.lbfgsb_gradient_method} not implemented.")

        tasks = []
        x = self.current_params
        self.gradient_components = {}

        # Per-dim step scaled with magnitude for numerical stability.
        np.maximum(np.abs(x) * base_eps, 1e-10, out=self._eps_array)
        eps = self._eps_array
        self.current_eps = eps

        self._user_grad_opt = self._seed_user_grad_opt(user_grad_full)
        per_dim_fd_cost = 2 if self.lbfgsb_gradient_method == "central" else 1
        signs = (1, -1) if self.lbfgsb_gradient_method == "central" else (1,)

        self.pending_grad_evals = 0
        for i in range(self.n_opt_dims):
            if np.isfinite(self._user_grad_opt[i]):
                self.sampler.target_calls_saved_by_user_gradient += per_dim_fd_cost
                continue
            for sign in signs:
                x_step = x.copy()
                x_step[i] += sign * eps[i]
                tasks.append({
                    'params': self._construct_full_params_for_task(x_step),
                    'context': {'type': self.type, 'job_id': self.id,
                                'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': sign},
                })
                self.pending_grad_evals += 1

        # Nothing to wait for if the user covered every dim.
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
        """Assemble grad (user-supplied + FD), update history, return the
        first line-search task."""
        grad = np.zeros(self.n_opt_dims)
        f = self.current_objective

        for i in range(self.n_opt_dims):
            if self._user_grad_opt is not None and np.isfinite(self._user_grad_opt[i]):
                grad[i] = self._user_grad_opt[i]
            elif self.lbfgsb_gradient_method == "central":
                f_plus = self.gradient_components[(i, 1)]
                f_minus = self.gradient_components[(i, -1)]
                grad[i] = (f_plus - f_minus) / (2 * self.current_eps[i])
            else:  # "forward"
                grad[i] = (self.gradient_components[(i, 1)] - f) / self.current_eps[i]

        self.current_gradient = grad

        if self.pending_s_k is not None:
            s_k = self.pending_s_k
            g_old = self.pending_g_old
            g_new = self.current_gradient
            y_k = g_new - g_old
            # Wolfe/relative curvature: require y·s to be a meaningful fraction
            # of ||y||*||s||, not just > a tiny absolute threshold.
            ys = float(np.dot(y_k, s_k))
            yn = float(np.linalg.norm(y_k))
            sn = float(np.linalg.norm(s_k))
            if ys > 1e-10 * max(yn * sn, 1e-30):
                self.s_hist.append(s_k)
                self.y_hist.append(y_k)
            self.pending_s_k = None
            self.pending_g_old = None

        # L-BFGS two-loop recursion. Pairs failing the curvature condition are
        # skipped so a degenerate dot product can't produce inf/nan rho.
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
        self.line_search_alpha = 1.0

        return [self._calculate_line_search_task()]

    def _calculate_line_search_task(self):
        """Next task for the backtracking line search."""
        alpha = self.line_search_alpha
        x_new = self.current_params + alpha * self.search_direction
        full_params_new = self._construct_full_params_for_task(x_new)

        # Piggyback grad_func on every attempt; rejected steps waste a
        # grad call but accepted ones save an FD round.
        context = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'LBFGS_LINE_SEARCH',
            'alpha': alpha,
            'compute_gradient': self.has_user_grad,
        }
        return {'params': full_params_new, 'context': context}

    def _process_line_search_result(self, result):
        """Handle a line-search result; accept, shrink alpha, or fail."""
        f_new = -result['target_val']
        alpha = result['context']['alpha']

        x_old = self.current_params
        f_old = self.current_objective
        g_old = self.current_gradient
        d = self.search_direction
        c1 = 1e-4

        x_new = x_old + alpha * d

        opt_indices = self.sampler.profiled_dims if self.grid_idx is not None else self.opt_dims

        x_new_bounded = self.sampler._ensure_bounds(x_new, opt_indices)

        # Armijo condition
        if f_new <= f_old + c1 * alpha * np.dot(g_old, x_new_bounded - x_old):
            self.iteration += 1

            if self.iteration >= self.lbfgsb_max_iter or np.abs(f_old - f_new) < self.lbfgsb_ftol:
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                self.current_params = x_new_bounded
                self.current_fitness = -f_new
                return []

            # Stash s_k and g_old so history is updated only after the next
            # gradient eval lands.
            self.pending_s_k = x_new_bounded - x_old
            self.pending_g_old = g_old

            self.current_params = x_new_bounded
            self.current_fitness = -f_new
            self.current_objective = f_new

            self.status = 'NEEDS_GRADIENT'
            return self._calculate_gradient_tasks(user_grad_full=result.get('user_gradient'))

        # Step rejected: shrink alpha and retry; bail out if it underflows.
        self.line_search_alpha *= 0.5
        if self.line_search_alpha < 1e-10:
            self.status = 'FINISHED'
            self._is_finished = True
            self.success = False
            return []

        return [self._calculate_line_search_task()]

    def on_finish(self, next_job_id):
        """Finalize the job and write the result back to the sampler state."""

        if self.success:
            self.improvement = self.current_fitness - self.start_fitness
        else:
            # Failed cell-anchored optimization: keep the cell as 'converged'
            # so patching can pick it up later.
            if self.type in ['LBFGSB', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK_LBFGSB'] and self.grid_idx in self.sampler.population:
                self.sampler.population[self.grid_idx]['status'] = 'converged'
            return None

        if self.type == 'INITIAL_OPTIMIZATION':
            final_params = self._construct_full_params_for_task(self.current_params)
            final_target_val = self.current_fitness

            self.sampler.initial_maxima.append({'point': final_params, 'target_val': final_target_val})
            if final_target_val > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = final_target_val

            self.sampler._update_global_pool(final_params, final_target_val, grid_idx=None)

            # Online basin detection: cluster this endpoint into the distinct-optima
            # registry that drives the rolling multistart's stopping rule.
            if self.sampler.basin_detection_enabled:
                self.sampler.register_initial_optimum(final_params, final_target_val)

        elif self.type in ['LBFGSB', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK_LBFGSB', 'LBFGSB_LOOP', 'POST_ACTIVATION_LBFGSB']:
            grid_idx = self.grid_idx
            if grid_idx in self.sampler.population:
                state = self.sampler.population[grid_idx]
                state['optimizer_state'] = {'s': list(self.s_hist), 'y': list(self.y_hist)}
                # LBFGSB_LOOP cells stay 'converged' so dynamic activation can
                # re-promote them; everything else terminates at 'optimized'.
                state['status'] = 'converged' if self.type == 'LBFGSB_LOOP' else 'optimized'

                if self.current_fitness > state['best_fitness']:
                    state['best_fitness'] = self.current_fitness
                    best_idx = np.argmax(state['fitnesses'])
                    state['profiled_params'][best_idx] = self.current_params
                    state['fitnesses'][best_idx] = self.current_fitness
                    self.sampler.profile_likelihood_grid[self.grid_idx] = self.current_fitness

                    if self.current_fitness > self.sampler.global_max_target_val:
                        self.sampler.global_max_target_val = self.current_fitness

                    full_params = self.sampler._construct_params(self.grid_idx, self.current_params)
                    self.sampler._update_global_pool(full_params, self.current_fitness, self.grid_idx)

        elif self.type == 'REFINEMENT_LBFGSB':
            # Refinement cells skip DE and get a minimal population state here.
            grid_idx = self.grid_idx
            self.sampler.population[grid_idx] = {
                'profiled_params': np.array([self.current_params]),
                'fitnesses': np.array([self.current_fitness]),
                'best_fitness': self.current_fitness,
                'status': 'optimized',
                'improvement_history': [],
                'last_update_gen': 0,
                'optimizer_state': {'s': list(self.s_hist), 'y': list(self.y_hist)},
            }
            self.sampler.profile_likelihood_grid[grid_idx] = self.current_fitness
            if self.current_fitness > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = self.current_fitness

            full_params = self.sampler._construct_params(grid_idx, self.current_params)
            self.sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        return None
