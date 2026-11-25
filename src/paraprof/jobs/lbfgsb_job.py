"""L-BFGS-B optimization job for asynchronous gradient-based optimization."""
import collections

import numpy as np

from .base import Job

ARMIJO_C1 = 1e-4
# First trial displacement of a line search without curvature information and
# longest trial displacement, as fractions of the bound extent.
FIRST_STEP = 0.1
MAX_STEP = 0.5
# Ratio between consecutive trial steps of the backtracking ladder, and the
# cap on its rungs (see _start_line_search).
LADDER_FACTOR = 0.5
MAX_LADDER_RUNGS = 60
# Rungs beyond the one being decided that idle workers may evaluate early.
LOOKAHEAD = 8

# A finite-difference Hessian costs n(n+3)/2 evaluations; it is measured for
# runs over at most this many parameters.
HESSIAN_MAX_DIMS = 12
# A Hessian is measured again, at most HESSIAN_MAX_REFRESH times per run, when
# the line search cut the step HESSIAN_REFRESH_RUNG times, the run is further
# than HESSIAN_REFRESH_DISTANCE (fraction of the bound extent) from where the
# Hessian was measured, and the step gained more than HESSIAN_REFRESH_GAIN
# tolerances. Measuring again in place does not fix a model that failed there,
# and the n(n+3)/2 evaluations are not repaid near the end of a run.
HESSIAN_REFRESH_RUNG = 2
HESSIAN_REFRESH_DISTANCE = 0.05
HESSIAN_REFRESH_GAIN = 100.0
HESSIAN_MAX_REFRESH = 3
# Eigenvalues of the finite-difference Hessian are kept at least this fraction
# of the largest one in magnitude, and made positive.
HESSIAN_EIG_FLOOR = 1e-4

GRID_JOB_TYPES = ('LBFGSB', 'LBFGSB_LOOP', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK_LBFGSB')


class LBFGSBJob(Job):
    """Self-contained async L-BFGS-B run (initial eval, gradient, line search, history).

    Works in bound-normalised coordinates z = (x - lower) / (upper - lower):
    gradients, curvature pairs and step lengths refer to z.

    A periodic parameter (``ProfileProjector(periodic=...)``) has no bounds
    to stop at. A trial point that leaves its range is wrapped to the other
    side when it is evaluated or accepted; the step is measured before that.

    The line search tries a fixed ladder of steps alpha_0 * factor**j and
    accepts the first rung that passes the Armijo test, so the master may
    evaluate later rungs early (see `speculate`) without changing the result.

    With `hessian` the run also measures a finite-difference Hessian with the
    gradient (n(n+3)/2 evaluations) and starts the L-BFGS recursion from its
    inverse. A cell takes the Hessian of a finished neighbour instead.

    For grid-anchored LBFGSB jobs the run starts with a neighbor-test
    step: if a finished neighbor has better fitness at our cell, we
    adopt its params and seed the (s, y) history from its optimizer state.
    """
    def __init__(self, job_id, job_type, sampler, opt_dims, start_params,
                 grid_idx, start_params_full, seed_history=None, start_fitness=-np.inf):

        super().__init__(job_id, job_type, sampler)

        self.lbfgsb_ftol = sampler.lbfgsb_ftol
        self.lbfgsb_max_iter = sampler.lbfgsb_max_iter
        self.lbfgsb_gradient_method = sampler.lbfgsb_gradient_method
        if self.lbfgsb_gradient_method not in ("central", "forward"):
            raise Exception(f"Gradient method {self.lbfgsb_gradient_method} not implemented.")
        # Finite-difference step, as a fraction of each parameter's bound
        # extent. A trial step shorter than this cannot be told from its start.
        self.fd_step = sampler.lbfgsb_fd_step_value
        # An accepted step that changes f by less than a few times the noise
        # of the target (0 if unknown) is as converged as the run can get.
        self.ftol_eff = max(self.lbfgsb_ftol, 4.0 * sampler.lbfgsb_noise)
        # When True we piggyback compute_gradient=True on value-evaluating tasks.
        self.has_user_grad = sampler.grad_func is not None

        self.grid_idx = grid_idx
        self.opt_dims = opt_dims
        self.n_opt_dims = len(opt_dims)

        b = np.asarray(sampler.bounds, dtype=float)[list(opt_dims)]
        self._lo = b[:, 0]
        self._hi = b[:, 1]
        self._scale = self._hi - self._lo
        self._periodic = sampler._periodic_mask[list(opt_dims)]
        self._any_periodic = bool(self._periodic.any())

        # start_params are the partial parameters corresponding to opt_dims;
        # start_params_full is the full vector for the initial eval.
        self.start_params_partial = np.asarray(start_params, dtype=float)
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
        self.current_gradient = None     # d objective / d z

        # Curvature pairs, in normalised coordinates.
        self.s_hist = collections.deque(maxlen=10)
        self.y_hist = collections.deque(maxlen=10)
        self.use_hessian = 0 < self.n_opt_dims <= HESSIAN_MAX_DIMS and not self.has_user_grad
        self.hessian_step = sampler.lbfgsb_hessian_step_value()
        # Inverse Hessian of the objective in z, from a neighbour, from the
        # sampler's hint, or measured by the run. None: not known. The array
        # is shared between jobs and never modified in place.
        self.Hinv = None
        self._hessian_z = None           # where this run measured its Hessian
        self._hessian_refreshes = 0
        # Cap on the trial displacement of the next line search (set in _accept).
        self._trust = None
        if seed_history:
            self._seed_curvature(seed_history)
        if self.use_hessian and self.Hinv is None and self.type != 'INITIAL_OPTIMIZATION':
            hint = sampler.lbfgsb_hessian_hint
            if hint is not None and hint.shape == (self.n_opt_dims, self.n_opt_dims):
                self.Hinv = hint
        self.last_gamma = None

        self.iteration = 0
        # True if the run ended at a stationary point within its resolution
        # (function tolerance met, or no trial step down to the finite-difference
        # step improves), not by running out of iterations.
        self.converged = False

        self.pending_s_k = None
        self.pending_g_old = None

        self.neighbor_params_to_test = None

        # Gradient evaluations by id. A gradient may be requested for a point
        # whose value is still in flight (see speculate).
        self._grads = {}
        self._next_gid = 0
        self._gid = None       # gradient the run is waiting for
        self._ls = None        # state of the current line search

    def _get_partial_params_from_full(self, full_params):
        """Slice a full parameter vector down to opt_dims."""
        return np.asarray(full_params, dtype=float)[list(self.opt_dims)]

    def _into_bounds(self, x):
        """``x`` (opt-dim slice) clipped to the bounds; periodic parameters are wrapped."""
        if self._any_periodic:
            x = np.where(self._periodic, self._lo + np.mod(x - self._lo, self._scale), x)
        return np.clip(x, self._lo, self._hi)

    def _construct_full_params_for_task(self, partial_params_to_eval):
        """Full bounded parameter vector for a worker task (global or grid-anchored)."""
        bounded = self._into_bounds(partial_params_to_eval)
        if self.grid_idx is None:
            return bounded
        return self.sampler._construct_params(self.grid_idx, bounded)

    def _ctx(self, sub_type, **extra):
        return {'type': self.type, 'job_id': self.id, 'sub_type': sub_type, **extra}

    def _seed_curvature(self, state):
        """Take the curvature of another run's ``optimizer_state``."""
        hinv = state.get('Hinv')
        if self.use_hessian and hinv is not None:
            # The Hessian carries the curvature; the pairs were built on the
            # other run's start and are not reused with it.
            self.Hinv = hinv
        else:
            self.s_hist.extend(state['s'])
            self.y_hist.extend(state['y'])

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
                    # A finished neighbour with an optimizer state to seed from.
                    if (neighbor_state['status'] == 'done') and \
                       (neighbor_state.get('optimizer_state') is not None):

                        if neighbor_state['best_fitness'] > best_neighbor_fitness:
                            best_neighbor_fitness = neighbor_state['best_fitness']
                            best_neighbor_state = neighbor_state

            if best_neighbor_state:
                # 1. Seed the history
                self.s_hist.clear()
                self.y_hist.clear()
                self._seed_curvature(best_neighbor_state['optimizer_state'])

                # 2. Get neighbor's best params to test
                neighbor_best_idx = np.argmax(best_neighbor_state['fitnesses'])
                self.neighbor_params_to_test = best_neighbor_state['profiled_params'][neighbor_best_idx]

                # 3. Create a task to test these params at *our* grid point
                full_params_test = self.sampler._construct_params(self.grid_idx, self.neighbor_params_to_test)

                return [{'params': full_params_test,
                         'context': self._ctx('LBFGS_NEIGHBOR_TEST',
                                              compute_gradient=self.has_user_grad)}]

            # No valid neighbor found, skip test and just evaluate our own params
            self.status = 'NEEDS_INITIAL_F'

        if self.status == 'NEEDS_INITIAL_F':
            # A cell job knows the value at its start point. Evaluate it again
            # only to collect a user gradient there.
            if np.isfinite(self.start_fitness) and not self.has_user_grad:
                self._set_current(self.start_params_partial, self.start_fitness)
                return self._begin_gradient(self._new_gid())
            return [{'params': self.start_params_full,
                     'context': self._ctx('LBFGS_INITIAL_F',
                                          compute_gradient=self.has_user_grad)}]

        # Should not be reached
        return []

    def _set_current(self, params_partial, fitness):
        self.current_params = self._into_bounds(params_partial)
        self.current_fitness = fitness
        self.current_objective = -fitness

    def process_result(self, result):
        """Main dispatcher for L-BFGS-B state machine."""
        context = result['context']
        sub_type = context.get('sub_type', 'NONE')

        if sub_type == 'LBFGS_NEIGHBOR_TEST':
            neighbor_fitness = result['target_val']
            if neighbor_fitness > self.start_fitness:
                self._set_current(self.neighbor_params_to_test, neighbor_fitness)
                user_grad = result.get('user_gradient')
            else:
                # Fall back to our own params; discard the gradient since it
                # was computed at the neighbor's params, not ours.
                self._set_current(self.fallback_params, self.start_fitness)
                user_grad = None
            if not np.isfinite(self.current_fitness):
                return self._finish(success=False)
            return self._begin_gradient(self._new_gid(), user_grad_full=user_grad)

        if sub_type == 'LBFGS_INITIAL_F':
            self._set_current(self._get_partial_params_from_full(result['params']),
                              result['target_val'])
            if not np.isfinite(self.current_fitness):
                # Nothing to climb from an invalid point.
                return self._finish(success=False)
            return self._begin_gradient(self._new_gid(),
                                        user_grad_full=result.get('user_gradient'))

        if sub_type in ('LBFGS_GRADIENT', 'LBFGS_HESSIAN'):
            return self._process_gradient_result(result)

        if sub_type == 'LBFGS_LINE_SEARCH':
            return self._process_line_search_result(result)

        return []

    def _new_gid(self):
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _user_grad_opt(self, user_grad_full):
        """Slice the user gradient (of target_func) to opt_dims and flip the
        sign for the objective. NaN entries stay NaN (need FD)."""
        if user_grad_full is None:
            return np.full(self.n_opt_dims, np.nan)
        sliced = np.asarray(user_grad_full, dtype=float)[list(self.opt_dims)]
        return np.where(np.isfinite(sliced), -sliced, np.nan)

    def _make_gradient(self, gid, x, user_grad_full=None):
        """Plan the probes of the gradient at x (opt-dim slice).

        A probe steps into the box: a forward step that would leave it is
        replaced by a backward one, so a parameter on its upper bound still
        gets a derivative and no probe coincides with x. A periodic parameter
        has room on both sides.
        """
        h = self.fd_step * self._scale
        ug = self._user_grad_opt(user_grad_full)
        probes = {}
        per_dim_cost = 2 if self.lbfgsb_gradient_method == "central" else 1
        for i in range(self.n_opt_dims):
            if np.isfinite(ug[i]):
                self.sampler.target_calls_saved_by_user_gradient += per_dim_cost
                continue
            up_ok = self._periodic[i] or x[i] + h[i] <= self._hi[i]
            dn_ok = self._periodic[i] or x[i] - h[i] >= self._lo[i]
            if self.lbfgsb_gradient_method == "central" and up_ok and dn_ok:
                signs = (1, -1)
            elif up_ok or not dn_ok:
                signs = (1,)
            else:
                signs = (-1,)
            for sign in signs:
                probes[(i, sign)] = None
        self._grads[gid] = {'x': x, 'h': h, 'user': ug, 'probes': probes,
                            'sent': set(), 'f': None, 'hess': None}

    def _add_hessian_probes(self, gid):
        """Plan the second-difference probes of the Hessian at the point of gradient gid.

        Per dimension two points, x + s h e_i and x - s h e_i (or x + 2 s h e_i
        where x - s h e_i would leave the box), and per pair one point,
        x + s_i h e_i + s_j h e_j. The sign s_i points into the box.
        """
        g = self._grads[gid]
        x = g['x']
        n = self.n_opt_dims
        h = self.hessian_step * self._scale
        sign = np.where(self._periodic | (x + h <= self._hi), 1.0, -1.0)
        offsets = {}
        for i in range(n):
            e = np.zeros(n)
            e[i] = sign[i] * h[i]
            offsets[('d', i, 1)] = e
            far = x[i] - sign[i] * h[i]
            central = self._periodic[i] or self._lo[i] <= far <= self._hi[i]
            offsets[('d', i, -1 if central else 2)] = (-e if central else 2 * e)
        for i in range(n):
            for j in range(i + 1, n):
                e = np.zeros(n)
                e[i] = sign[i] * h[i]
                e[j] = sign[j] * h[j]
                offsets[('m', i, j)] = e
        g['hess'] = {'sign': sign, 'offsets': offsets, 'vals': dict.fromkeys(offsets)}

    def _hessian_task(self, gid, key):
        g = self._grads[gid]
        return {'params': self._construct_full_params_for_task(g['x'] + g['hess']['offsets'][key]),
                'context': self._ctx('LBFGS_HESSIAN', hkey=key, gid=gid)}

    def _assemble_hessian(self, g):
        """Inverse of the finite-difference Hessian (normalised coordinates),
        with its eigenvalues made positive; None if a probe was invalid."""
        hs = g['hess']
        n = self.n_opt_dims
        f0 = g['f']
        v = hs['vals']
        if not all(np.isfinite(val) for val in v.values()):
            return None
        hz = self.hessian_step            # the step in normalised units
        H = np.zeros((n, n))
        for i in range(n):
            f1 = v[('d', i, 1)]
            if ('d', i, -1) in v:
                H[i, i] = (f1 - 2.0 * f0 + v[('d', i, -1)]) / hz ** 2
            else:
                H[i, i] = (v[('d', i, 2)] - 2.0 * f1 + f0) / hz ** 2
        for i in range(n):
            for j in range(i + 1, n):
                mixed = (v[('m', i, j)] - v[('d', i, 1)] - v[('d', j, 1)] + f0) / hz ** 2
                H[i, j] = H[j, i] = hs['sign'][i] * hs['sign'][j] * mixed
        w, V = np.linalg.eigh(H)
        wmax = float(np.max(np.abs(w)))
        if not np.isfinite(wmax) or wmax == 0.0:
            return None
        w = np.maximum(np.abs(w), HESSIAN_EIG_FLOOR * wmax)
        return (V / w) @ V.T

    def _probe_task(self, gid, key):
        g = self._grads[gid]
        i, sign = key
        x_step = g['x'].copy()
        x_step[i] += sign * g['h'][i]
        g['sent'].add(key)
        return {'params': self._construct_full_params_for_task(x_step),
                'context': self._ctx('LBFGS_GRADIENT', dim=i, sign=sign, gid=gid)}

    def _begin_gradient(self, gid, user_grad_full=None, measure_hessian=False):
        """Wait for the gradient at the current point; returns the probes not yet sent."""
        if gid not in self._grads:
            self._make_gradient(gid, self.current_params, user_grad_full)
        g = self._grads[gid]
        g['f'] = self.current_objective
        self._gid = gid
        self._ls = None
        self.status = 'NEEDS_GRADIENT'
        # Drop gradients requested for trial points that were not accepted.
        for other in [k for k in self._grads if k != gid]:
            del self._grads[other]
        tasks = [self._probe_task(gid, key) for key in g['probes'] if key not in g['sent']]
        if self.use_hessian and (self.Hinv is None or measure_hessian):
            self._add_hessian_probes(gid)
            tasks += [self._hessian_task(gid, key) for key in g['hess']['vals']]
        if self._gradient_complete(g):
            return self._finalize_gradient_and_line_search()
        return tasks

    @staticmethod
    def _gradient_complete(g):
        return (all(v is not None for v in g['probes'].values())
                and (g['hess'] is None or all(v is not None for v in g['hess']['vals'].values())))

    def _process_gradient_result(self, result):
        """Processes a returned likelihood evaluation for a gradient or Hessian calculation."""
        context = result['context']
        g = self._grads.get(context['gid'])
        if g is None:
            return []          # probe of a trial point that was not accepted
        if context['sub_type'] == 'LBFGS_HESSIAN':
            g['hess']['vals'][context['hkey']] = -result['target_val']
        else:
            g['probes'][(context['dim'], context['sign'])] = -result['target_val']  # objective
        if (self.status == 'NEEDS_GRADIENT' and context['gid'] == self._gid
                and self._gradient_complete(g)):
            return self._finalize_gradient_and_line_search()
        return []

    def _assemble_gradient(self, g):
        """Gradient of the objective with respect to z."""
        grad = np.zeros(self.n_opt_dims)
        f0 = g['f']
        for i in range(self.n_opt_dims):
            if np.isfinite(g['user'][i]):
                grad[i] = g['user'][i]
                continue
            fp = g['probes'].get((i, 1))
            fm = g['probes'].get((i, -1))
            h = g['h'][i]
            if fp is not None and fm is not None:
                grad[i] = (fp - fm) / (2 * h)
            elif fp is not None:
                grad[i] = (fp - f0) / h
            else:
                grad[i] = (f0 - fm) / h
        # A probe on an invalid point (objective +inf) says nothing about the slope.
        grad = np.where(np.isfinite(grad), grad, 0.0)
        return grad * self._scale

    def _finalize_gradient_and_line_search(self):
        """Assemble grad (user-supplied + FD), update history, start the line search."""
        store = self._grads[self._gid]
        g_new = self._assemble_gradient(store)
        self.current_gradient = g_new

        if store['hess'] is not None:
            hinv = self._assemble_hessian(store)
            if hinv is not None:
                # A measured Hessian replaces what the run had, pairs included.
                self.Hinv = hinv
                self._hessian_z = (store['x'] - self._lo) / self._scale
                self.s_hist.clear()
                self.y_hist.clear()
                self.pending_s_k = None
                self.pending_g_old = None

        if self.pending_s_k is not None:
            s_k = self.pending_s_k
            y_k = g_new - self.pending_g_old
            # Wolfe/relative curvature: require y·s to be a meaningful fraction
            # of ||y||*||s||, not just > a tiny absolute threshold.
            ys = float(np.dot(y_k, s_k))
            yn = float(np.linalg.norm(y_k))
            sn = float(np.linalg.norm(s_k))
            if ys > 1e-10 * max(yn * sn, 1e-30):
                self.s_hist.append(s_k)
                self.y_hist.append(y_k)
                self.last_gamma = ys / max(yn * yn, 1e-300)
            self.pending_s_k = None
            self.pending_g_old = None

        return self._start_line_search()

    def _two_loop(self, grad):
        """L-BFGS direction -H grad, or None without usable curvature pairs.

        Pairs failing the curvature condition are skipped so a degenerate dot
        product can't produce inf/nan rho.
        """
        usable_pairs = []
        for s, y in zip(self.s_hist, self.y_hist):
            ys = float(np.dot(y, s))
            if ys > 1e-30:
                usable_pairs.append((s, y, ys))
        if not usable_pairs and self.Hinv is None:
            return None

        q = grad
        a = []
        for s, y, ys in reversed(usable_pairs):
            alpha = np.dot(s, q) / ys
            q = q - alpha * y
            a.append(alpha)

        if self.Hinv is not None:
            z = self.Hinv @ q
        else:
            s_last, y_last, _ = usable_pairs[-1]
            yy = float(np.dot(y_last, y_last))
            gamma = float(np.dot(s_last, y_last)) / yy if yy > 1e-30 else 1.0
            z = gamma * q

        for (s, y, ys), alpha in zip(usable_pairs, reversed(a)):
            beta = np.dot(y, z) / ys
            z = z + s * (alpha - beta)
        return -z

    def _bounded_newton(self, g, z):
        """Newton step in the subspace of the parameters that are free to move.

        A parameter on a bound whose gradient pushes it outward is held, and
        the step of the others is taken with the Hessian restricted to them,
        (H_FF)^-1 = Hinv_FF - Hinv_FA (Hinv_AA)^-1 Hinv_AF. Projecting the
        unconstrained step instead would leave the free parameters with a step
        computed as if the held ones could move. Returns None when no
        parameter is held or no Hessian is known.
        """
        if self.Hinv is None:
            return None
        held = (((z <= 0.0) & (g > 0.0)) | ((z >= 1.0) & (g < 0.0))) & ~self._periodic
        if not held.any():
            return None
        free = ~held
        d = np.zeros(self.n_opt_dims)
        if not free.any():
            return d
        Hi = self.Hinv
        reduced = Hi[np.ix_(free, free)] - Hi[np.ix_(free, held)] @ np.linalg.solve(
            Hi[np.ix_(held, held)], Hi[np.ix_(held, free)])
        d[free] = -reduced @ g[free]
        return d

    def _project(self, d, z):
        """Zero the components that push a parameter sitting on a bound outward."""
        d = d.copy()
        d[(z <= 0.0) & (d < 0.0) & ~self._periodic] = 0.0
        d[(z >= 1.0) & (d > 0.0) & ~self._periodic] = 0.0
        return d

    def _start_line_search(self):
        """Choose the direction and the ladder of trial steps; return the first rung."""
        g = self.current_gradient
        x0 = self.current_params
        z0 = (x0 - self._lo) / self._scale

        d = self._bounded_newton(g, z0)
        if d is None:
            d = self._two_loop(g)
        scaled = d is not None
        if scaled:
            d = self._project(d, z0)
            if float(np.dot(g, d)) >= 0.0:
                scaled = False            # not a descent direction: fall back
        if not scaled:
            # No usable curvature: steepest descent, with the length scale of
            # earlier runs of this kind if the sampler has one.
            gamma = self.sampler.lbfgsb_scale_hint(self.type)
            d = self._project(-(gamma if gamma else 1.0) * g, z0)
            scaled = bool(gamma)

        gd = float(np.dot(g, d))
        dmax = float(np.max(np.abs(d)))
        if dmax == 0.0 or gd >= 0.0:
            # Every component of the gradient is held by a bound: stationary.
            return self._finish(converged=True)
        if self.Hinv is not None and scaled and -0.5 * gd < self.ftol_eff:
            # -g.d/2 is the decrease the quadratic model expects from a Newton
            # step. Below the function tolerance there is nothing left to gain.
            return self._finish(converged=True)

        alpha0 = 1.0 if scaled else FIRST_STEP / dmax
        alpha0 = min(alpha0, MAX_STEP / dmax)
        # A line search that had to cut its step says how far the model can be
        # trusted here; the next one starts from that length, not from the top.
        if self._trust is not None:
            alpha0 = min(alpha0, self._trust / dmax)
        # Beyond alpha_cap every parameter that moves sits on its bound and
        # the clipped trial point no longer changes.
        with np.errstate(divide='ignore', invalid='ignore'):
            room = np.where(d > 0, (1.0 - z0) / d, np.where(d < 0, -z0 / d, 0.0))
        if self._any_periodic:
            # A periodic parameter that moves never reaches a bound.
            room = np.where(self._periodic & (d != 0), np.inf, room)
        alpha_cap = float(np.max(room))
        if alpha_cap > 0.0:
            alpha0 = min(alpha0, alpha_cap)

        # The ladder ends when the trial displacement falls below what a
        # finite difference resolves.
        alphas = []
        alpha = alpha0
        while len(alphas) < MAX_LADDER_RUNGS:
            alphas.append(alpha)
            alpha *= LADDER_FACTOR
            if alpha * dmax < self.fd_step:
                break

        self._ls = {
            'x0': x0, 'z0': z0, 'f0': self.current_objective,
            'g': g, 'd': d, 'alphas': alphas, 'vals': {}, 'ugrad': {}, 'sent': set(),
            'decided': 0, 'scaled': scaled, 'probe_gid': None,
        }
        self.status = 'NEEDS_LINE_SEARCH'
        return [self._rung_task(0)]

    def _rung_point(self, j):
        """Trial point of rung j. A periodic parameter is not stopped at its
        bound and may lie outside its range here (see _into_bounds)."""
        ls = self._ls
        z_free = ls['z0'] + ls['alphas'][j] * ls['d']
        z = np.clip(z_free, 0.0, 1.0)
        x = np.clip(self._lo + z * self._scale, self._lo, self._hi)
        if self._any_periodic:
            x = np.where(self._periodic, self._lo + z_free * self._scale, x)
        return x

    def _rung_task(self, j):
        """Task for rung j of the current ladder.

        Piggyback grad_func on every attempt; rejected steps waste a
        grad call but accepted ones save an FD round.
        """
        ls = self._ls
        ls['sent'].add(j)
        return {'params': self._construct_full_params_for_task(self._rung_point(j)),
                'context': self._ctx('LBFGS_LINE_SEARCH', rung=j, lid=self.iteration,
                                     compute_gradient=self.has_user_grad)}

    def _process_line_search_result(self, result):
        """Record a rung; accept the first rung that passes, in ladder order."""
        ctx = result['context']
        ls = self._ls
        if ls is None or ctx['lid'] != self.iteration:
            return []          # rung of an earlier line search
        j = ctx['rung']
        ls['vals'][j] = -result['target_val']
        ls['ugrad'][j] = result.get('user_gradient')
        return self._advance_line_search()

    def _advance_line_search(self):
        ls = self._ls
        while ls['decided'] in ls['vals']:
            j = ls['decided']
            f_new = ls['vals'][j]
            x_new = self._rung_point(j)
            dz = (x_new - ls['x0']) / self._scale
            # Armijo condition, on the displacement actually made after clipping.
            if np.any(dz != 0.0) and f_new <= ls['f0'] + ARMIJO_C1 * float(np.dot(ls['g'], dz)):
                return self._accept(j, x_new, f_new, dz)
            ls['decided'] += 1
            if ls['probe_gid'] is not None:
                # The probes sent ahead were for rung 0.
                del self._grads[ls['probe_gid']]
                ls['probe_gid'] = None

        j = ls['decided']
        if j >= len(ls['alphas']):
            # No rung improves on the current point: stationary within the resolution.
            return self._finish(converged=True)
        if j not in ls['sent']:
            return [self._rung_task(j)]
        return []

    def _accept(self, j, x_new, f_new, dz):
        ls = self._ls
        f_old = ls['f0']
        g_old = ls['g']
        self.iteration += 1
        # dz is the step as it was taken; from here on the point is in range.
        x_new = self._into_bounds(x_new)
        self.current_params = x_new
        self.current_fitness = -f_new
        self.current_objective = f_new

        reached_ftol = np.abs(f_old - f_new) < self.ftol_eff
        if self.iteration >= self.lbfgsb_max_iter or reached_ftol:
            # Converged only via the tolerance, not by running out of iterations.
            return self._finish(converged=bool(reached_ftol))

        # Stash s_k and g_old so history is updated only after the next
        # gradient eval lands.
        self.pending_s_k = dz
        self.pending_g_old = g_old

        self._trust = max(2.0 * float(np.max(np.abs(dz))), 4.0 * self.fd_step)

        # The Hessian in use is poor here (see HESSIAN_REFRESH_*): measure it again.
        measure_hessian = False
        if (self.use_hessian and j >= HESSIAN_REFRESH_RUNG
                and self._hessian_refreshes < HESSIAN_MAX_REFRESH
                and np.abs(f_old - f_new) > HESSIAN_REFRESH_GAIN * self.ftol_eff):
            z_new = (x_new - self._lo) / self._scale
            if self._hessian_z is not None:
                moved = z_new - self._hessian_z
                if self._any_periodic:
                    moved = np.where(self._periodic, moved - np.round(moved), moved)
            measure_hessian = (
                self._hessian_z is None
                or float(np.max(np.abs(moved))) > HESSIAN_REFRESH_DISTANCE)
            if measure_hessian:
                self._hessian_refreshes += 1

        if ls['probe_gid'] is not None:
            gid = ls['probe_gid']           # probes sent ahead for this point
        else:
            gid = self._new_gid()
            self._make_gradient(gid, x_new, ls['ugrad'][j])
        return self._begin_gradient(gid, measure_hessian=measure_hessian)

    @property
    def wants_speculation(self):
        return self._ls is not None

    def speculate(self):
        """One task for a worker that would otherwise idle, or None.

        Either a later rung of the ladder, whose value is then there if the
        rungs before it are rejected, or a gradient probe at rung 0, so the
        next iteration can start at once if rung 0 is accepted. Probes come
        first for a step from curvature information, rungs otherwise.
        Neither changes which rung is accepted.
        """
        if not self.wants_speculation:
            return None
        ls = self._ls
        order = ('probe', 'rung') if ls['scaled'] else ('rung', 'probe')
        for kind in order:
            if kind == 'rung':
                # A step from curvature information rarely needs more than one
                # or two cuts; the whole ladder is only worth sending ahead
                # for a step of unknown length.
                ahead = min(LOOKAHEAD, 2) if ls['scaled'] else LOOKAHEAD
                stop = min(len(ls['alphas']), ls['decided'] + 1 + ahead)
                for j in range(ls['decided'], stop):
                    if j not in ls['sent']:
                        return self._rung_task(j)
            elif ls['decided'] == 0 and not self.has_user_grad:
                if ls['probe_gid'] is None:
                    ls['probe_gid'] = self._new_gid()
                    self._make_gradient(ls['probe_gid'], self._into_bounds(self._rung_point(0)))
                g = self._grads[ls['probe_gid']]
                for key in g['probes']:
                    if key not in g['sent']:
                        return self._probe_task(ls['probe_gid'], key)
        return None

    def _finish(self, converged=False, success=True):
        self.status = 'FINISHED'
        self._is_finished = True
        self.success = success
        self.converged = converged
        self._ls = None
        self._grads = {}
        return []

    def on_finish(self, next_job_id):
        """Finalize the job and write the result back to the sampler state.

        The best accepted iterate is written back however the run ended.
        """
        sampler = self.sampler
        usable = np.isfinite(self.current_fitness)
        if usable:
            self.improvement = self.current_fitness - self.start_fitness
        if self.last_gamma is not None:
            sampler.note_lbfgsb_scale(self.type, self.last_gamma)

        if self.type == 'INITIAL_OPTIMIZATION':
            if not usable:
                return None
            final_params = self._construct_full_params_for_task(self.current_params)
            final_target_val = self.current_fitness

            sampler.initial_maxima.append({'point': final_params, 'target_val': final_target_val})
            if final_target_val > sampler.global_max_target_val:
                sampler.global_max_target_val = final_target_val

            sampler._update_global_pool(final_params, final_target_val, grid_idx=None)

            # Cluster this endpoint into the distinct-optima registry driving
            # the stopping rule -- but only if it converged: a truncated descent
            # isn't a basin minimizer and would inflate W.
            if self.converged:
                sampler.register_initial_optimum(final_params, final_target_val)

        elif self.type in GRID_JOB_TYPES:
            grid_idx = self.grid_idx
            if grid_idx in sampler.population:
                state = sampler.population[grid_idx]
                state['optimizer_state'] = {'s': list(self.s_hist), 'y': list(self.y_hist),
                                            'Hinv': self.Hinv}
                if self._hessian_z is not None:
                    sampler.lbfgsb_hessian_hint = self.Hinv
                # A run that could not start (invalid start point) ends here
                # too: the cell keeps its value, and patching may improve it.
                state['status'] = 'done'

                if usable and self.current_fitness > state['best_fitness']:
                    state['best_fitness'] = self.current_fitness
                    best_idx = np.argmax(state['fitnesses'])
                    state['profiled_params'][best_idx] = self.current_params
                    state['fitnesses'][best_idx] = self.current_fitness

                    if self.current_fitness > sampler.global_max_target_val:
                        sampler.global_max_target_val = self.current_fitness

                    full_params = sampler._construct_params(self.grid_idx, self.current_params)
                    sampler._update_global_pool(full_params, self.current_fitness, self.grid_idx)

        elif self.type == 'REFINEMENT_LBFGSB':
            if not usable:
                return None
            # Refinement cells skip DE and get a minimal population state here.
            grid_idx = self.grid_idx
            sampler.population[grid_idx] = {
                'profiled_params': np.array([self.current_params]),
                'fitnesses': np.array([self.current_fitness]),
                'best_fitness': self.current_fitness,
                'status': 'done',
                'improvement_history': [],
                'optimizer_state': {'s': list(self.s_hist), 'y': list(self.y_hist)},
            }
            if self.current_fitness > sampler.global_max_target_val:
                sampler.global_max_target_val = self.current_fitness

            full_params = sampler._construct_params(grid_idx, self.current_params)
            sampler._update_global_pool(full_params, self.current_fitness, grid_idx)

        return None
