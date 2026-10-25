"""L-BFGS-B optimization job for asynchronous gradient-based optimization."""
import numpy as np
import collections
from .base import Job

# Finite-difference step as a fraction of the per-dimension scale.
FD_REL = 1e-7
# Curvature-probe step as a fraction of the box width.
CURVATURE_PROBE_FRAC = 0.01
# Smallest scale a curvature estimate may set, as a fraction of the box width.
SCALE_MIN_FRAC = 1e-6
# Largest coordinate move of a history-free step, as a fraction of the box width.
FIRST_STEP_CAP = 0.25


class LBFGSBJob(Job):
    """Self-contained async L-BFGS-B run (initial eval, gradient, line search, history).

    For grid-anchored jobs (types ``LBFGSB`` and ``LBFGSB_LOOP``) the run
    starts with a neighbor-test step: if a converged neighbor has better
    fitness at our cell, we adopt its params. The post-DE polish (``LBFGSB``)
    also seeds its (s, y) history from that neighbor's optimizer state; the
    lbfgsb-mode loop does not (inherited curvature was found to cost more
    than it saves in higher-dimensional inner problems).

    A rejected line-search step whose Armijo-predicted decrease at the full
    step is below ``ftol`` ends the run as converged: the direction cannot
    buy a measurable improvement, which is the case for a start that is
    already stationary. Without this exit such a run halved alpha down to
    1e-10. The test is on the full step on purpose: on a stiff problem the
    first accepted step can be tiny and the halving has to be allowed to
    reach it.

    Bounds are handled with an active set: a variable sitting at a bound
    whose objective gradient points outward is held for that iteration (its
    gradient and direction components are zeroed), finite-difference steps
    are taken inward at a bound, and a line search that underflows keeps the
    best point seen instead of discarding the run.

    The two-loop recursion runs in scaled coordinates u = x / scale, with a
    per-dimension scale fixed for the job's lifetime. The scale comes from a
    one-round curvature probe after the first gradient: one evaluation per
    dimension at x0 + h_i e_i (h_i = 1% of the box width) gives a diagonal
    second derivative h_ii, and scale_i = 1 / sqrt(h_ii), clipped to
    [1e-6, 1] times the box width, used only when every probed direction is
    convex; otherwise, and for a start that is already converged (no probe
    issued), the box width is the scale in every dimension. With an empty
    history the first step is then a diagonal Newton step; a history-free
    step is in any case capped to move no coordinate by more than a quarter
    of its box. The stored (s, y) pairs stay in raw
    units, so neighbour seeding and ``optimizer_state`` are unaffected. The
    finite-difference step is ``FD_REL * scale`` per dimension.
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

        # Grid-anchored jobs (the post-DE polish and the lbfgsb-mode loop) start
        # with a neighbour test and inherit the neighbour's quasi-Newton history.
        if self.type in ('LBFGSB', 'LBFGSB_LOOP'):
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
        # Derived quantities at the current point and at the finite-difference
        # points of the last gradient (derived mode: the snap step after the
        # polish needs the Jacobian of q); None where unknown.
        self._current_derived = self.sampler._cached_derived_full(start_params_full) if start_params_full is not None else None
        self._grad_derived = {}
        self._grad_base_derived = None
        # User gradient sliced to opt_dims, in the objective frame; NaN = "use FD".
        self._user_grad_opt = None

        self.s_hist = collections.deque(maxlen=10)
        self.y_hist = collections.deque(maxlen=10)
        if seed_history:
            self.s_hist.extend(seed_history['s'])
            self.y_hist.extend(seed_history['y'])

        self.iteration = 0
        # True if the run hit the function tolerance (a real stationary point),
        # not just lbfgsb_max_iter.
        self.converged = False
        self._eps_array = np.empty(self.n_opt_dims)

        # Per-dimension scale of the quasi-Newton metric: box width until the
        # curvature probe replaces it (see class docstring).
        if self.n_opt_dims > 0:
            lo, hi = self._opt_bounds()
            width = np.asarray(hi - lo, dtype=float)
            width = np.where(np.isfinite(width) & (width > 0), width, 1.0)
        else:
            width = np.zeros(0)
        self.width = width
        self.scale = width.copy()
        self.scale_from_curvature = np.zeros(self.n_opt_dims, dtype=bool)
        self._probe_done = False
        self.curvature_components = {}
        self.pending_curvature_evals = 0

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

    def _value(self, result):
        """The fitness this job optimises for an evaluation result: the target
        value, or in derived mode the cell-anchored penalised value."""
        return self.sampler.fitness_for_cell(result, self.grid_idx)

    def _derived_snap_point(self):
        """Derived mode: the end point of the polish projected back into its
        cell with the finite-difference Jacobian of q from the last gradient;
        None if the point is inside the cell or the Jacobian is unavailable."""
        sampler = self.sampler
        x_full = self._construct_full_params_for_task(self.current_params)
        d = self._current_derived
        if d is None:
            d = sampler._cached_derived_full(x_full)
        if d is None or self._grad_base_derived is None:
            return None
        idx = sampler.derived_idx
        q = np.asarray(d, dtype=float)[idx]
        lo, hi = [], []
        for ax, k in zip(sampler.grid_axes, self.grid_idx):
            w = (ax[-1] - ax[0]) / max(len(ax) - 1, 1)
            lo.append(ax[k] - 0.48 * w)
            hi.append(ax[k] + 0.48 * w)
        lo, hi = np.array(lo), np.array(hi)
        if np.all(q >= lo) and np.all(q <= hi):
            return None
        n = len(self.current_params)
        J = np.zeros((len(idx), n))
        base = np.asarray(self._grad_base_derived, dtype=float)[idx]
        for i in range(n):
            ent = self._grad_derived.get((i, 0))
            if ent is None or ent[0] is None or ent[1] == 0:
                return None
            J[:, i] = (np.asarray(ent[0], dtype=float)[idx] - base) / ent[1]
        if not np.all(np.isfinite(J)):
            return None
        t = np.clip(q, lo, hi)
        JJt = J @ J.T
        try:
            delta = J.T @ np.linalg.solve(JJt, t - q)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        x_part = sampler._ensure_bounds(np.asarray(self.current_params, dtype=float) + delta, list(self.opt_dims))
        return self._construct_full_params_for_task(x_part)

    def _get_partial_params_from_full(self, full_params):
        """Slice a full parameter vector down to opt_dims."""
        return full_params[list(self.opt_dims)]

    def _opt_bounds(self):
        """(lo, hi) arrays of the box over the optimised dims."""
        if self.grid_idx is None:
            b = self.sampler.bounds[list(self.opt_dims)]
        else:
            b = self.sampler._profiled_bounds
        return b[:, 0], b[:, 1]

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
                # 1. Seed the history (post-DE polish only; see class docstring)
                if self.type == 'LBFGSB':
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
            # The start value is usually already known (population member,
            # patching / recheck seed): go straight to the gradient instead
            # of re-evaluating it. Re-evaluate when it is unknown (global
            # starts, refinement) or when the user gradient must be fetched
            # alongside the value.
            if np.isfinite(self.start_fitness) and not self.has_user_grad:
                self.current_params = np.asarray(self.start_params_partial, dtype=float)
                self.current_fitness = float(self.start_fitness)
                self.current_objective = -self.current_fitness
                self.status = 'NEEDS_GRADIENT'
                return self._calculate_gradient_tasks()
            # Fallback: evaluate the starting params
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
            neighbor_fitness = self._value(result)

            if neighbor_fitness > self.start_fitness:
                self.current_params = self.neighbor_params_to_test
                self.current_fitness = neighbor_fitness
                self.current_objective = -neighbor_fitness
                self._current_derived = result.get('derived')
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
            self.current_fitness = self._value(result)
            self.current_objective = -self.current_fitness
            self.current_params = self._get_partial_params_from_full(result['params'])
            self._current_derived = result.get('derived')

            self.status = 'NEEDS_GRADIENT'
            new_tasks = self._calculate_gradient_tasks(user_grad_full=result.get('user_gradient'))

        elif self.status == 'NEEDS_GRADIENT' and sub_type == 'LBFGS_GRADIENT':
            new_tasks = self._process_gradient_result(result)

        elif self.status == 'NEEDS_CURVATURE' and sub_type == 'LBFGS_CURVATURE':
            new_tasks = self._process_curvature_result(result)

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

    def _calculate_gradient_tasks(self, user_grad_full=None):
        """Generates tasks for the gradient at ``current_params``. Issues FD
        tasks only for dims the user did not supply; finalizes immediately
        if the user covered every dim."""
        if self.lbfgsb_gradient_method not in ("central", "forward"):
            raise Exception(f"Gradient method {self.lbfgsb_gradient_method} not implemented.")

        tasks = []
        x = self.current_params
        self.gradient_components = {}
        self._grad_derived = {}
        self._grad_base_derived = self._current_derived

        # Per-dim step tied to the scale of the metric, so the difference is
        # taken over the same fraction of the local length scale in every
        # direction.
        np.maximum(self.scale * FD_REL, 1e-10, out=self._eps_array)
        eps = self._eps_array
        self.current_eps = eps

        self._user_grad_opt = self._seed_user_grad_opt(user_grad_full)
        per_dim_fd_cost = 2 if self.lbfgsb_gradient_method == "central" else 1
        signs = (1, -1) if self.lbfgsb_gradient_method == "central" else (1,)
        lo, hi = self._opt_bounds()

        self.pending_grad_evals = 0
        for i in range(self.n_opt_dims):
            if np.isfinite(self._user_grad_opt[i]):
                self.sampler.target_calls_saved_by_user_gradient += per_dim_fd_cost
                continue
            # Steps that leave the box are dropped (the task would be clipped
            # back onto x and the difference would read zero). A forward step
            # at a bound is taken inward instead; a central difference at a
            # bound becomes the one-sided inward difference.
            steps = [sign * eps[i] for sign in signs if lo[i] <= x[i] + sign * eps[i] <= hi[i]]
            if not steps:
                steps = [-eps[i] if x[i] - eps[i] >= lo[i] else eps[i]]
            for k, step in enumerate(steps):
                x_step = x.copy()
                x_step[i] += step
                tasks.append({
                    'params': self._construct_full_params_for_task(x_step),
                    'context': {'type': self.type, 'job_id': self.id,
                                'sub_type': 'LBFGS_GRADIENT', 'dim': i, 'k': k, 'step': float(step)},
                })
                self.pending_grad_evals += 1

        # Nothing to wait for if the user covered every dim.
        if self.pending_grad_evals == 0:
            return self._finalize_gradient()

        return tasks

    def _process_gradient_result(self, result):
        """Processes a returned likelihood evaluation for a gradient calculation."""
        context = result['context']
        dim, k = context['dim'], context['k']

        # (objective value, FD step actually taken)
        self.gradient_components[(dim, k)] = (-self._value(result), context['step'])
        self._grad_derived[(dim, k)] = (result.get('derived'), context['step'])
        self.pending_grad_evals -= 1

        # Check if all components for the gradient have been computed
        if self.pending_grad_evals == 0:
            return self._finalize_gradient()

        return [] # Not ready yet, no new task

    def _finalize_gradient(self):
        """Assemble grad (user-supplied + FD) and update the history; then
        either issue the curvature probe (first gradient of a run that will
        iterate) or go on to the search direction."""
        grad = np.zeros(self.n_opt_dims)
        f = self.current_objective

        for i in range(self.n_opt_dims):
            if self._user_grad_opt is not None and np.isfinite(self._user_grad_opt[i]):
                grad[i] = self._user_grad_opt[i]
            elif (i, 1) in self.gradient_components:   # two-sided
                f_a, step_a = self.gradient_components[(i, 0)]
                f_b, step_b = self.gradient_components[(i, 1)]
                grad[i] = (f_a - f_b) / (step_a - step_b)
            else:                                       # one-sided (forward, or central at a bound)
                f_a, step_a = self.gradient_components[(i, 0)]
                grad[i] = (f_a - f) / step_a

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

        if not self._probe_done:
            self._probe_done = True
            # A start that is already converged (the predicted decrease of a
            # box-scaled gradient step is below ftol) gets no probe: the run
            # ends on the flat-step exit after one line-search evaluation.
            predicted = 1e-4 * float(np.sum((grad * self.width) ** 2))
            if predicted >= self.lbfgsb_ftol:
                return self._calculate_curvature_tasks()

        return self._compute_direction_and_line_search()

    def _calculate_curvature_tasks(self):
        """One evaluation per optimised dim at x0 + h_i e_i (h_i = 1% of the
        box width, taken inward at a bound) for the diagonal curvature."""
        x = self.current_params
        lo, hi = self._opt_bounds()
        self.curvature_components = {}
        self.pending_curvature_evals = 0
        tasks = []
        for i in range(self.n_opt_dims):
            h = CURVATURE_PROBE_FRAC * self.width[i]
            if x[i] + h > hi[i]:
                h = -h
            x_step = x.copy()
            x_step[i] += h
            tasks.append({
                'params': self._construct_full_params_for_task(x_step),
                'context': {'type': self.type, 'job_id': self.id,
                            'sub_type': 'LBFGS_CURVATURE', 'dim': i, 'step': float(h),
                            'compute_gradient': False},
            })
            self.pending_curvature_evals += 1
        self.status = 'NEEDS_CURVATURE'
        return tasks

    def _process_curvature_result(self, result):
        context = result['context']
        self.curvature_components[context['dim']] = (-self._value(result), context['step'])
        self.pending_curvature_evals -= 1
        if self.pending_curvature_evals == 0:
            self._set_scale_from_curvature()
            return self._compute_direction_and_line_search()
        return []

    def _set_scale_from_curvature(self):
        """scale_i = 1 / sqrt(h_ii) from the probe, clipped to [SCALE_MIN_FRAC, 1]
        times the box width, used only when every probed direction is convex
        (h_ii finite and positive). A start outside the convex region keeps
        the box width in every dimension: the local curvature there says
        little about the basin (on a quartic it changes by orders of
        magnitude along the way), and mixing probe scales with box widths
        gave metrics with a 60x anisotropy that made L-BFGS creep."""
        f0 = self.current_objective
        g = self.current_gradient
        scales = {}
        for i, (f_i, h) in self.curvature_components.items():
            h_ii = 2.0 * (f_i - f0 - g[i] * h) / (h * h)
            if not (np.isfinite(h_ii) and h_ii > 0):
                return
            sc = 1.0 / np.sqrt(h_ii)
            scales[i] = min(max(sc, SCALE_MIN_FRAC * self.width[i]), self.width[i])
        for i, sc in scales.items():
            self.scale[i] = sc
            self.scale_from_curvature[i] = True

    def _compute_direction_and_line_search(self):
        """Quasi-Newton direction (two-loop recursion in scaled coordinates,
        active-set projection at the bounds); returns the first line-search task."""
        grad = self.current_gradient
        sc = self.scale

        # Active set: a variable at a bound whose objective gradient points
        # outward is held this iteration. The direction is built from the
        # projected gradient and zeroed in the held components, so the
        # clipped step is the step the quasi-Newton model asked for. Without
        # this the free components are computed as if the held one moved
        # too, and the Armijo test can fail at every step length.
        lo, hi = self._opt_bounds()
        x = self.current_params
        held = ((x <= lo) & (grad > 0)) | ((x >= hi) & (grad < 0))

        # L-BFGS two-loop recursion in u = x / scale: the gradient enters as
        # grad * scale, the pairs as (s / scale, y * scale); s·y is invariant.
        # Pairs failing the curvature condition are skipped so a degenerate
        # dot product can't produce inf/nan rho.
        q = np.where(held, 0.0, grad) * sc
        a = []
        usable_pairs = []
        for s_raw, y_raw in zip(self.s_hist, self.y_hist):
            ys = float(np.dot(y_raw, s_raw))
            if ys > 1e-30:
                usable_pairs.append((s_raw / sc, y_raw * sc, ys))

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

        d = -(z * sc)
        d[held] = 0.0
        if not np.any(d):
            # Every component is held: a stationary point of the box-constrained problem.
            self.status = 'FINISHED'
            self._is_finished = True
            self.success = True
            self.converged = True
            return []

        if not usable_pairs:
            # No curvature pairs yet: the step length is only as good as the
            # scale. Cap it so no coordinate moves more than a fixed fraction
            # of its box; a far start then needs a few iterations instead of
            # twenty halvings.
            m = float(np.max(np.abs(d) / self.width))
            if m > FIRST_STEP_CAP:
                d *= FIRST_STEP_CAP / m

        self.search_direction = d
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
        f_new = -self._value(result)
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

            reached_ftol = np.abs(f_old - f_new) < self.lbfgsb_ftol
            if self.iteration >= self.lbfgsb_max_iter or reached_ftol:
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                # Converged only via the tolerance, not by running out of iterations.
                self.converged = bool(reached_ftol)
                self.current_params = x_new_bounded
                self.current_fitness = -f_new
                self._current_derived = result.get('derived')
                return []

            # Stash s_k and g_old so history is updated only after the next
            # gradient eval lands.
            self.pending_s_k = x_new_bounded - x_old
            self.pending_g_old = g_old

            self.current_params = x_new_bounded
            self.current_fitness = -f_new
            self.current_objective = f_new
            self._current_derived = result.get('derived')

            self.status = 'NEEDS_GRADIENT'
            return self._calculate_gradient_tasks(user_grad_full=result.get('user_gradient'))

        # Step rejected while the decrease the Armijo test asks for at the
        # full step (alpha = 1) is itself below ftol: the direction cannot
        # buy a measurable improvement at any step length (this is the
        # "start is already stationary" case, where halving alpha down to
        # 1e-10 would only burn evaluations). The test uses the full step,
        # not the current alpha: on a stiff problem a steep direction can
        # need alpha ~ 1e-6 before the Armijo test passes, and that tiny
        # accepted step is what teaches the quasi-Newton scaling the
        # curvature, so halving must be allowed to get there. A rejected
        # step with f_new == f_old but a large predicted decrease (an
        # overshoot to a mirror point) is not caught here either.
        x_full_bounded = self.sampler._ensure_bounds(x_old + d, opt_indices)
        predicted = abs(c1 * np.dot(g_old, x_full_bounded - x_old))
        if predicted < self.lbfgsb_ftol:
            self.status = 'FINISHED'
            self._is_finished = True
            self.success = True
            self.converged = True
            return []

        # Step rejected: shrink alpha and retry. If alpha underflows, no step
        # length passed the Armijo test: the run ends at the current point,
        # which is the best it has seen (every accepted step improved on the
        # previous one), and that point is written back like any other
        # endpoint. Discarding it would throw away the progress made so far;
        # the run just does not count as converged.
        self.line_search_alpha *= 0.5
        if self.line_search_alpha < 1e-10:
            self.status = 'FINISHED'
            self._is_finished = True
            self.success = True
            self.converged = False
            return []

        return [self._calculate_line_search_task()]

    def on_finish(self, next_job_id):
        """Finalize the job and write the result back to the sampler state."""

        if self.success:
            self.improvement = self.current_fitness - self.start_fitness
        else:
            # Failed cell-anchored optimization: return the cell to its
            # terminal status so patching (or, in lbfgsb mode, dynamic
            # activation) can pick it up later instead of leaving it queued.
            cell_types = ['LBFGSB', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK_LBFGSB',
                          'LBFGSB_LOOP', 'POST_ACTIVATION_LBFGSB']
            if self.type in cell_types and self.grid_idx in self.sampler.population:
                self.sampler.population[self.grid_idx]['status'] = (
                    'optimized' if self.type == 'POST_ACTIVATION_LBFGSB' else 'converged')
            return None

        if self.type == 'INITIAL_OPTIMIZATION':
            final_params = self._construct_full_params_for_task(self.current_params)
            final_target_val = self.current_fitness

            self.sampler.initial_maxima.append({'point': final_params, 'target_val': final_target_val})
            if final_target_val > self.sampler.global_max_target_val:
                self.sampler.global_max_target_val = final_target_val

            self.sampler._update_global_pool(final_params, final_target_val, grid_idx=None)

            # Cluster this endpoint into the distinct-optima registry driving
            # the stopping rule -- but only if it converged: a truncated descent
            # isn't a basin minimizer and would inflate W.
            if self.converged:
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

            if self.sampler.derived_mode and self.sampler.derived_snap:
                snap = self._derived_snap_point()
                if snap is not None:
                    from .derived_snap_job import DerivedSnapJob
                    return DerivedSnapJob(next_job_id, self.sampler, grid_idx, snap), next_job_id + 1

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
