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

        # For neighbor test (legacy, kept for single-candidate fallback).
        self.neighbor_params_to_test = None

        # Multi-candidate predictor test state. When the
        # secant_predictor_warm_start hook is enabled we evaluate the best
        # neighbor's ψ* (zeroth-order predictor) plus a set of linear / secant
        # extrapolation candidates along each grid axis (first-order
        # predictor) in parallel. The state below tracks which candidates are
        # outstanding and their fitnesses.
        self._candidate_params = []          # list[np.ndarray]
        self._candidate_labels = []          # list[str]
        self._candidate_fitnesses = {}       # idx -> float
        self._pending_candidate_tests = 0

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
                # 1. Seed the history from the best neighbor (this is the
                #    L-BFGS curvature inheritance; it stays at the
                #    best-fitness neighbor even when the predictor adds extra
                #    candidate starting points below).
                self.s_hist.clear()
                self.y_hist.clear()
                self.s_hist.extend(best_neighbor_state['optimizer_state']['s'])
                self.y_hist.extend(best_neighbor_state['optimizer_state']['y'])

                # 2. Build the candidate list. Candidate 0 is always the
                #    best neighbor's profiled params (zeroth-order predictor,
                #    the legacy behavior). When secant_predictor_warm_start
                #    is enabled we additionally test linear/secant
                #    extrapolations along each grid axis from already-
                #    converged neighbors.
                neighbor_best_idx = int(np.argmax(best_neighbor_state['fitnesses']))
                neighbor_best_params = best_neighbor_state['profiled_params'][neighbor_best_idx]

                self._candidate_params = [np.array(neighbor_best_params, copy=True)]
                self._candidate_labels = ['best-neighbor']

                if getattr(self.sampler, 'secant_predictor_warm_start', False):
                    secant_candidates = self.sampler._build_secant_predictor_candidates(
                        self.grid_idx, neighbor_best_params,
                    )
                    for label, params in secant_candidates:
                        self._candidate_params.append(np.array(params, copy=True))
                        self._candidate_labels.append(label)

                # Legacy alias used by tests / debugging tools.
                self.neighbor_params_to_test = self._candidate_params[0]

                # 3. Issue test tasks for every candidate in parallel.
                self._pending_candidate_tests = len(self._candidate_params)
                self._candidate_fitnesses = {}
                tasks = []
                for i, params in enumerate(self._candidate_params):
                    full_params_test = self.sampler._construct_params(
                        self.grid_idx, params
                    )
                    context = {
                        'type': self.type,
                        'job_id': self.id,
                        'sub_type': 'LBFGS_NEIGHBOR_TEST',
                        'candidate_idx': i,
                    }
                    tasks.append({'params': full_params_test, 'context': context})
                return tasks

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
                'sub_type': 'LBFGS_INITIAL_F'
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
            # Multi-candidate predictor test: collect fitnesses across all
            # candidates, only proceed when the last one comes in.
            cand_idx = context.get('candidate_idx', 0)
            self._candidate_fitnesses[cand_idx] = result['target_val']
            self._pending_candidate_tests -= 1

            if self._pending_candidate_tests > 0:
                return []

            # All candidates evaluated: pick the best one.
            best_cand_idx = max(
                self._candidate_fitnesses,
                key=lambda i: self._candidate_fitnesses[i],
            )
            best_cand_fitness = self._candidate_fitnesses[best_cand_idx]
            best_cand_params = self._candidate_params[best_cand_idx]

            # Diagnostics: per-cell + sampler-level counts of how often a
            # non-legacy predictor candidate (i.e. a secant/interp candidate)
            # wins the comparison. The legacy "best-neighbor" candidate is
            # idx 0; anything else is a continuation predictor.
            if len(self._candidate_params) > 1:
                self.sampler._secant_predictor_candidates_tested += 1
                if best_cand_idx != 0:
                    self.sampler._secant_predictor_candidates_won += 1

            if best_cand_fitness > self.start_fitness:
                # A predictor candidate beats our own start: use it.
                self.current_params = best_cand_params
                self.current_fitness = best_cand_fitness
                self.current_objective = -best_cand_fitness
            else:
                # Our original params are still best.
                self.current_params = self.fallback_params
                self.current_fitness = self.start_fitness
                self.current_objective = -self.start_fitness

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks()

        elif self.status == 'NEEDS_INITIAL_F' and sub_type == 'LBFGS_INITIAL_F':
            # Came here from global opt or optimization with no neighbors
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

    def _calculate_gradient_tasks(self, base_eps=1e-8):
        """Generates tasks needed to numerically calculate the gradient."""
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

        if self.lbfgsb_gradient_method == "central":
            self.pending_grad_evals = 2 * self.n_opt_dims
            for i in range(self.n_opt_dims):
                # Positive step
                x_plus = x.copy()
                x_plus[i] += eps[i]
                full_params_plus = self._construct_full_params_for_task(x_plus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': 1}
                tasks.append({'params': full_params_plus, 'context': context})

                # Negative step
                x_minus = x.copy()
                x_minus[i] -= eps[i]
                full_params_minus = self._construct_full_params_for_task(x_minus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': -1}
                tasks.append({'params': full_params_minus, 'context': context})

        elif self.lbfgsb_gradient_method == "forward":
            self.pending_grad_evals = self.n_opt_dims
            for i in range(self.n_opt_dims):
                x_plus = x.copy()
                x_plus[i] += eps[i]
                full_params_plus = self._construct_full_params_for_task(x_plus)
                context = {'type': self.type, 'job_id': self.id, 'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'sign': 1}
                tasks.append({'params': full_params_plus, 'context': context})
        else:
            raise Exception(f"Gradient method {self.lbfgsb_gradient_method} not implemented.")

        return tasks

    def _process_gradient_result(self, result):
        """Processes a returned likelihood evaluation for a gradient calculation."""
        context = result['context']
        dim, sign = context['dim'], context['sign']

        self.gradient_components[(dim, sign)] = -result['target_val'] # Store objective
        self.pending_grad_evals -= 1

        # Check if all components for the gradient have been computed
        if self.pending_grad_evals == 0:
            grad = np.zeros(self.n_opt_dims)
            f = self.current_objective

            if self.lbfgsb_gradient_method == "central":
                for i in range(self.n_opt_dims):
                    f_plus = self.gradient_components[(i, 1)]
                    f_minus = self.gradient_components[(i, -1)]
                    grad[i] = (f_plus - f_minus) / (2 * self.current_eps[i])
            elif self.lbfgsb_gradient_method == "forward":
                 for i in range(self.n_opt_dims):
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

    def _maybe_spawn_basin_switch_check(self, next_job_id):
        """
        Online basin-switch detection.

        After this cell's L-BFGS-B optimization completes, scan converged
        neighbors for one whose best_fitness is strictly higher than this
        cell's just-updated best_fitness. If found, spawn a
        :class:`PatchingTestJob` that evaluates the neighbor's profiled
        params at this cell. If the test confirms improvement, the patching
        job itself spawns a ``PATCHING_LBFGSB`` polish from there. This
        catches the basin-handoff case during the main scan instead of
        deferring it to the post-hoc patching waves.

        Returns
        -------
        (job, next_job_id) tuple or None
            ``None`` if no promising neighbor was found (or the feature is
            inapplicable at this cell).
        """
        # Late import to avoid a circular dependency with patching_test_job,
        # which itself imports LBFGSBJob.
        from .patching_test_job import PatchingTestJob

        if self.grid_idx is None or self.grid_idx not in self.sampler.population:
            return None

        state = self.sampler.population[self.grid_idx]
        my_best = state['best_fitness']
        my_best_idx = int(np.argmax(state['fitnesses']))
        my_params = state['profiled_params'][my_best_idx]

        best_check_neighbor_idx = None
        best_check_neighbor_fitness = my_best
        best_check_params = None

        for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
            if neighbor_idx not in self.sampler.population:
                continue
            nstate = self.sampler.population[neighbor_idx]
            if nstate['status'] not in ('optimized', 'converged'):
                continue
            n_fit = nstate['best_fitness']
            if n_fit <= best_check_neighbor_fitness:
                continue
            n_best_idx = int(np.argmax(nstate['fitnesses']))
            n_params = nstate['profiled_params'][n_best_idx]
            # Skip neighbors whose ψ* is identical to ours — that test would
            # carry no information.
            if np.allclose(n_params, my_params, atol=1e-12):
                continue
            best_check_neighbor_fitness = n_fit
            best_check_neighbor_idx = neighbor_idx
            best_check_params = n_params

        if best_check_neighbor_idx is None:
            return None

        # Sampler-level diagnostic counter; the actual improvement check
        # happens in PatchingTestJob.on_finish, so we track tests issued
        # here and let the test job update the "improvement" counter when
        # it spawns the polish.
        self.sampler._online_basin_switch_tests += 1

        job = PatchingTestJob(
            job_id=next_job_id,
            sampler=self.sampler,
            grid_idx=self.grid_idx,
            test_profiled_params=np.array(best_check_params, copy=True),
            wave_number=-1,  # sentinel for "online" test (not part of any wave)
        )
        # Tag the job so PatchingTestJob.on_finish can bump the
        # online-basin-switch improvement counter when it spawns a polish.
        job.online_basin_switch_check = True

        return (job, next_job_id + 1)


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

                # Online basin-switch detection. As soon as this cell finishes
                # its L-BFGS-B optimization, check whether any already-
                # converged neighbor has a profiled-param solution that would
                # yield a strictly better fitness at this cell, and if so
                # spawn an immediate PatchingTestJob to verify it (and a
                # polish if confirmed). The continuation interpretation is:
                # the local "track" we followed may not be the dominant
                # branch, and the neighbor lives in a different basin that
                # now dominates ours.
                #
                # Guarded against:
                #   - PATCHING_LBFGSB jobs (avoid recursion through patching)
                #   - sampler-level toggle (online_basin_switch=False)
                #   - direct-eval mode (no profiled params)
                if (self.type != 'PATCHING_LBFGSB'
                        and getattr(self.sampler, 'online_basin_switch', False)
                        and not self.sampler.direct_eval_mode):
                    spawn = self._maybe_spawn_basin_switch_check(next_job_id)
                    if spawn is not None:
                        return spawn

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
