"""Jobs for the volume-sampling stage (tiers 2 and 3 of the funnel).

``VolumeProbeJob`` evaluates the target once at each anchor point (tier 2).
``VolumeSearchJob`` runs an anchored L-BFGS-B search (tier 3) for anchors
whose probe failed: it minimizes the bounds-scaled distance to the anchor
plus a hinged band-violation penalty, terminating at the first in-band point
within the coverage radius. The penalized objective steers the search only;
recorded representatives always carry their true logL.
"""
import numpy as np

from ..volume import SOURCE_PROBE
from .base import Job
from .lbfgsb_job import LBFGSBJob

# Tier-3 anchored search: κ = SEARCH_PENALTY_STRENGTH / roi_threshold², so a
# band violation of roi_threshold costs this many units of scaled distance².
# Normalized by roi_threshold² (hence scale-free), 1.0 is a robust default —
# κ shapes the search path, not its in-band outcome.
SEARCH_PENALTY_STRENGTH = 1.0

# Band violations are clamped before squaring so a failed evaluation
# (logL = -inf) yields a huge-but-finite penalty instead of overflowing.
VIOLATION_CLAMP = 1e100

# Interior walks stop refining once within this fraction of the band depth
# from the drawn depth target.
WALK_DEPTH_TOL_FRAC = 0.05

# Depth corrections allowed per tangential round before the round reverts.
TANGENT_CORRECTIONS = 2


class VolumeProbeJob(Job):
    """Evaluate the target once at each of the given anchors (tier 2).

    On finish, writes per-anchor probe records into the
    :class:`~paraprof.volume.AnchorSet` (kept unconditionally, so the
    uniform subset and volume estimate can be re-derived after global-max
    drift) and registers in-band probes as their own anchor's
    representative (distance 0).
    """

    def __init__(self, job_id, sampler, anchor_set, anchor_indices,
                 band_lo):
        super().__init__(job_id, 'VOLUME_PROBE', sampler)
        self.anchor_set = anchor_set
        self.anchor_indices = np.asarray(anchor_indices, dtype=int)
        self.band_lo = float(band_lo)
        self.target_vals = np.full(len(self.anchor_indices), -np.inf)
        self.evals_remaining = len(self.anchor_indices)

    def start(self):
        if len(self.anchor_indices) == 0:
            self.success = True
            self._is_finished = True
            return []

        tasks = []
        for i, anchor_idx in enumerate(self.anchor_indices):
            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i,
            }
            tasks.append({'params': self.anchor_set.anchors[anchor_idx].copy(),
                          'context': context})
        return tasks

    def process_result(self, result):
        idx = result['context']['point_idx']
        self.target_vals[idx] = result['target_val']
        self.evals_remaining -= 1
        if self.evals_remaining <= 0:
            self.success = True
            self._is_finished = True
        return []

    def on_finish(self, next_job_id):
        aset = self.anchor_set
        n_hits = 0
        for i, anchor_idx in enumerate(self.anchor_indices):
            logl = float(self.target_vals[i])
            aset.probed[anchor_idx] = True
            aset.probe_logls[anchor_idx] = logl
            if np.isfinite(logl) and logl >= self.band_lo:
                n_hits += 1
                aset.offer_to_anchor(anchor_idx, aset.anchors[anchor_idx].copy(),
                                     logl, 0.0, SOURCE_PROBE)
        if len(self.anchor_indices):
            self.sampler.logger.info(
                f"--- Volume probe: {n_hits}/{len(self.anchor_indices)} "
                f"anchors in band ---"
            )
        return None


class VolumeSearchJob(LBFGSBJob):
    """Anchored search (tier 3): pull an evaluation into the band near the anchor.

    Maximized fitness: ``-(dist² + κ·v²)`` where ``dist`` is the bounds-scaled
    distance to the anchor and ``v = max(0, band_lo - logL)`` is the band
    violation. In-band evaluations get a fully analytic gradient (distance term
    only); out-of-band evaluations chain-rule a user ``grad_func`` if provided,
    otherwise fall back to FD. The job ends at the first in-band point within
    the coverage radius (``hit``), or at L-BFGS-B termination classified as
    ``projected`` (some in-band point beyond the radius) or ``hole`` (never
    in-band; closest-approach point recorded).
    """

    # Distance slack for interior walks from 'projected' entry points: the
    # walk may move up to this factor times the entry distance from the
    # anchor (hit-entry walks are capped at the coverage radius instead).
    PROJECTED_WALK_SLACK = 1.5

    def __init__(self, job_id, sampler, anchor_set, anchor_index,
                 band_lo, kappa, start_params, max_iter=None,
                 interior_steps=0, band_depth=None, depth_exponent=None,
                 draw_depth_target=None):
        start_params = np.asarray(start_params, dtype=float)
        super().__init__(
            job_id, 'VOLUME_SEARCH', sampler,
            opt_dims=tuple(range(sampler.dims)),
            start_params=start_params,
            grid_idx=None,
            start_params_full=start_params,
            seed_history=None,
            start_fitness=-np.inf,
        )
        if max_iter is not None:
            self.lbfgsb_max_iter = max_iter

        self.anchor_index = int(anchor_index)
        self.anchor = anchor_set.anchors[self.anchor_index]
        self.coverage_radius = anchor_set.coverage_radius
        self._lo = anchor_set.bounds[:, 0]
        self._hi = anchor_set.bounds[:, 1]
        self._extent = anchor_set.bounds[:, 1] - anchor_set.bounds[:, 0]
        self.band_lo = float(band_lo)
        self.kappa = float(kappa)
        self.interior_steps = int(interior_steps)
        # Band depth in logL units (the stage's roi_threshold); sets the
        # interior walk's depth-target scale.
        self.band_depth = float(band_depth) if band_depth is not None \
            else float(sampler.roi_threshold)
        # Exponent γ of the depth-target draw t = band_depth·U^γ (see
        # volume.depth_law_exponent). None = the quadratic-basin volume law.
        self.depth_exponent = depth_exponent
        # Stage-level adaptive draw (VolumeStageState.draw_depth_target);
        # overrides the i.i.d. draw so censored bins get retried.
        self.draw_depth_target = draw_depth_target

        self.hit = False
        self.best_inband_point = None
        self.best_inband_logl = -np.inf
        self.best_inband_dist = np.inf
        self.best_viol_point = None
        self.best_viol_logl = -np.inf
        self.best_viol_dist = np.inf
        self.best_viol = np.inf

        # Interior-walk state (off unless interior_steps > 0). interior_point
        # is the walk's deepest accepted point; the stage state makes it the
        # anchor's representative.
        self.interior_point = None
        self.interior_logl = -np.inf
        self.interior_dist = np.inf
        self._walk_done = False
        self._walk_steps_left = 0
        self._walk_dir_scaled = None
        self._walk_dist_cap = np.inf
        self._walk_target_logl = np.inf
        self._walk_bisecting = False
        self._walk_low_point = None
        self._walk_reaimed = False
        self._walk_fail_count = 0
        self._march_point = None
        self._walk_entry_point = None
        # Tangential-randomization state. _tan_grad is a Broyden estimate of
        # the scaled-space logL gradient (the shell normal), rank-1 updated
        # from every tangent evaluation.
        self._tan_grad = None
        self._tan_h = 0.0
        self._tan_correct_left = 0
        self.tangent_moves = 0
        self._aim_cache = None
        self._final_success = None

    def _scaled_dist(self, params):
        return float(np.linalg.norm((params - self.anchor) / self._extent))

    def _violation(self, logl):
        return min(max(0.0, self.band_lo - logl), VIOLATION_CLAMP)

    def _transformed_gradient(self, params, logl, violation, raw_gradient):
        """Gradient of the maximized fitness at ``params`` (None = use FD).

        In-band the hinge is locally flat, so the gradient is the analytic
        distance term alone — no ∇logL needed. Below the band, chain-rule a
        user-supplied ∇logL if present (NaN entries fall through to FD
        per dim, as in the base class).
        """
        ddist2 = 2.0 * (params - self.anchor) / self._extent ** 2
        if violation == 0.0:
            return -ddist2
        if raw_gradient is None:
            return None
        # Below the band (v = band_lo - logL), so ∇v = -∇logL.
        grad = np.asarray(raw_gradient, dtype=float)
        return -ddist2 + 2.0 * self.kappa * violation * grad

    def process_result(self, result):
        sub_type = result['context'].get('sub_type')
        if sub_type == 'VOLUME_INTERIOR':
            return self._process_interior_result(result)
        if sub_type == 'VOLUME_TANGENT':
            return self._process_tangent_result(result)

        logl = result['target_val']
        params = np.asarray(result['params'], dtype=float)
        dist = self._scaled_dist(params)
        violation = self._violation(logl)

        # Track what was seen — also for stray results arriving after a
        # hit (e.g. leftover FD evaluations while an interior walk runs).
        if violation == 0.0:
            if dist < self.best_inband_dist:
                self.best_inband_point = params.copy()
                self.best_inband_logl = float(logl)
                self.best_inband_dist = dist
        elif (violation < self.best_viol
              or (violation == self.best_viol and dist < self.best_viol_dist)):
            self.best_viol_point = params.copy()
            self.best_viol_logl = float(logl)
            self.best_viol_dist = dist
            self.best_viol = violation

        if self.status in ('INTERIOR_WALK', 'TANGENT_WALK') \
                or self._is_finished:
            return []

        if violation == 0.0 and dist <= self.coverage_radius:
            # A covering point, not a stationary one: stop the search. In
            # interior-steps mode, first walk a few steps off the band edge.
            self.hit = True
            walk_tasks = self._start_interior_walk(
                params, float(logl), dist, dist_cap=self.coverage_radius)
            if walk_tasks is not None:
                self._final_success = True
                return walk_tasks
            self.status = 'FINISHED'
            self._is_finished = True
            self.success = True
            return []

        transformed = dict(result)
        transformed['target_val'] = -(dist * dist
                                      + self.kappa * violation * violation)
        transformed['user_gradient'] = self._transformed_gradient(
            params, logl, violation, result.get('user_gradient'))
        out = super().process_result(transformed)

        # The base machinery just terminated (ftol/max_iter/line-search
        # failure) with an in-band point seen beyond the radius: detour
        # through an interior walk from it before reporting 'projected'.
        if self._is_finished and not self._walk_done \
                and self.interior_steps > 0 \
                and self.best_inband_point is not None:
            walk_tasks = self._start_interior_walk(
                self.best_inband_point, self.best_inband_logl,
                self.best_inband_dist,
                dist_cap=max(self.coverage_radius,
                             self.best_inband_dist
                             * self.PROJECTED_WALK_SLACK))
            if walk_tasks is not None:
                self._final_success = self.success
                self._is_finished = False
                return walk_tasks
        return out

    # ------------------------------------------------------------------ #
    # Interior walk: steps from the band edge toward a
    # drawn depth target (ΔlnL below the band top; see draw_depth_target
    # and depth_law_exponent). It marches along the aim direction
    # (_walk_aim_direction), bisects to land precisely on the target, and
    # spends any leftover budget on tangential randomization. All moves
    # stay within the distance cap, preserving the coverage guarantee.
    # ------------------------------------------------------------------ #
    def _start_interior_walk(self, origin, origin_logl, origin_dist,
                             dist_cap):
        """Arm the walk; returns the first task list, or None to skip
        (interior_steps off, walk already ran, or the entry point already
        reaches the drawn depth target)."""
        if self.interior_steps <= 0 or self._walk_done:
            return None
        self._walk_done = True
        self.interior_point = np.asarray(origin, dtype=float).copy()
        self.interior_logl = float(origin_logl)
        self.interior_dist = float(origin_dist)

        # Depth target: logL >= band_top - t, with band_top the band's
        # upper logL edge (global max at stage start) and the depth t drawn
        # either adaptively at stage level (quota over the law's residual
        # need) or i.i.d. from the law t = band_depth·U^γ.
        band_top = self.band_lo + self.band_depth
        if self.draw_depth_target is not None:
            target_depth = float(self.draw_depth_target())
        else:
            exponent = self.depth_exponent
            if exponent is None:
                exponent = 2.0 / max(len(self.anchor), 1)
            u = float(np.random.random())
            target_depth = self.band_depth * u ** exponent
        self._walk_target_logl = band_top - target_depth
        if self.interior_logl >= self._walk_target_logl:
            tol = WALK_DEPTH_TOL_FRAC * self.band_depth
            if abs(self.interior_logl - self._walk_target_logl) <= tol:
                # The entry already sits at the drawn depth: spend the
                # whole walk budget on tangential randomization instead.
                self._walk_entry_point = self.interior_point.copy()
                self._walk_dist_cap = float(dist_cap)
                self._walk_step = (2.0 * self.coverage_radius
                                   / self.interior_steps)
                self._walk_steps_left = self.interior_steps
                tasks = self._enter_tangent_phase()
                if tasks is not None:
                    return tasks
            return None

        self._march_point = self.interior_point.copy()
        self._walk_entry_point = self.interior_point.copy()
        self._walk_dist_cap = float(dist_cap)
        self._walk_dir_scaled = self._walk_aim_direction(self.interior_point)
        # Step length sized so the full step budget can traverse the cap
        # ball's diameter (a rim entry may need to cross to the far side);
        # the bisection phase restores depth precision afterwards.
        self._walk_step = 2.0 * self.coverage_radius / self.interior_steps
        self._walk_steps_left = self.interior_steps
        self.status = 'INTERIOR_WALK'
        return [self._interior_task()]

    def _aim_candidates(self):
        """Known deep points from the scan (global pool + initial maxima),
        as ``(points, logls)`` arrays. Snapshotted once per job."""
        if self._aim_cache is None:
            points, logls = [], []
            for fitness, _, entry in getattr(self.sampler,
                                             'global_solution_pool', []):
                if np.isfinite(fitness):
                    points.append(np.asarray(entry['full_params'],
                                             dtype=float))
                    logls.append(float(fitness))
            for maximum in getattr(self.sampler, 'initial_maxima', []):
                val = float(maximum['target_val'])
                if np.isfinite(val):
                    points.append(np.asarray(maximum['point'], dtype=float))
                    logls.append(val)
            self._aim_cache = (
                np.asarray(points, dtype=float).reshape(-1, len(self.anchor)),
                np.asarray(logls, dtype=float),
            )
        return self._aim_cache

    def _walk_aim_direction(self, origin):
        """Scaled-space unit direction for the walk: toward the nearest
        scan-known point (global pool / initial maxima) at least as deep as
        the target. Aim points beyond the distance cap are projected onto
        the cap sphere around the anchor. Falls back to the inward
        continuation of ``origin - anchor`` when no candidate qualifies.
        """
        origin_scaled = (np.asarray(origin, dtype=float) - self._lo) \
            / self._extent
        anchor_scaled = (self.anchor - self._lo) / self._extent
        points, logls = self._aim_candidates()
        if len(points):
            deep_enough = logls >= self._walk_target_logl
            if deep_enough.any():
                cands = (points[deep_enough] - self._lo) / self._extent
                d2 = np.einsum('ij,ij->i', cands - origin_scaled,
                               cands - origin_scaled)
                aim = cands[int(np.argmin(d2))]
                rel = aim - anchor_scaled
                rel_norm = float(np.linalg.norm(rel))
                if np.isfinite(self._walk_dist_cap) \
                        and rel_norm > self._walk_dist_cap > 0:
                    aim = anchor_scaled + rel * (self._walk_dist_cap
                                                 / rel_norm)
                direction = aim - origin_scaled
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    return direction / norm
        direction = origin_scaled - anchor_scaled
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.random.standard_normal(len(direction))
            norm = float(np.linalg.norm(direction))
        return direction / norm

    def _interior_task(self):
        scaled = ((self._march_point - self._lo) / self._extent
                  + self._walk_step * self._walk_dir_scaled)
        params = np.clip(self._lo + scaled * self._extent, self._lo, self._hi)
        return {'params': params,
                'context': {'type': self.type, 'job_id': self.id,
                            'sub_type': 'VOLUME_INTERIOR'}}

    def _process_interior_result(self, result):
        logl = float(result['target_val'])
        params = np.asarray(result['params'], dtype=float)
        dist = self._scaled_dist(params)
        in_band = self._violation(logl) == 0.0
        # Tiny slack so a step landing exactly on the cap is not lost to
        # floating-point round-trip noise.
        acceptable = in_band and dist <= self._walk_dist_cap + 1e-12
        self._walk_steps_left -= 1
        tol = WALK_DEPTH_TOL_FRAC * self.band_depth

        if not self._walk_bisecting:
            # Marching phase: fixed steps along the aim ray until the
            # target is crossed. Any within-cap step advances the march —
            # including logL dips and out-of-band points (curved band
            # sheets put thin out-of-band slivers across straight chords;
            # the walk passes through them, spending budget, and only
            # in-band points are ever adopted as representatives). Only
            # leaving the distance cap stalls the walk.
            if dist <= self._walk_dist_cap + 1e-12:
                below_target = self._march_point.copy()
                self._march_point = params.copy()
                if in_band and logl > self.interior_logl:
                    self.interior_point = params.copy()
                    self.interior_logl = logl
                    self.interior_dist = dist
                if in_band and logl >= self._walk_target_logl:
                    # Crossed the target: refine toward it by bisection if
                    # the overshoot is large and budget remains, so the
                    # realized depth is not quantized to the step lattice.
                    self.interior_point = params.copy()
                    self.interior_logl = logl
                    self.interior_dist = dist
                    if self._walk_steps_left > 0 \
                            and logl - self._walk_target_logl > tol:
                        self._walk_bisecting = True
                        self._walk_low_point = below_target
                        return [self._bisect_task()]
                elif self._walk_steps_left > 0:
                    return [self._interior_task()]
            elif self._walk_steps_left > 0 \
                    and self.interior_logl < self._walk_target_logl:
                # The step left the distance cap below the target. The cap
                # boundary lies between the march position and the failed
                # step, so shrink the step (twice per aim) and retry before
                # spending the one re-aim toward the nearest known deeper
                # point. All retries spend walk budget.
                if self._walk_fail_count < 2:
                    self._walk_fail_count += 1
                    self._walk_step *= 0.5
                    return [self._interior_task()]
                if not self._walk_reaimed:
                    self._walk_reaimed = True
                    self._walk_fail_count = 0
                    self._walk_step = (2.0 * self.coverage_radius
                                       / self.interior_steps)
                    self._walk_dir_scaled = self._walk_aim_direction(
                        self._march_point)
                    return [self._interior_task()]
        else:
            # Bisection phase: tighten the bracket around the target depth.
            # Points at/above the target replace the upper end (the
            # representative); anything else replaces the lower end.
            if acceptable and logl >= self._walk_target_logl:
                if logl < self.interior_logl:
                    self.interior_point = params.copy()
                    self.interior_logl = logl
                    self.interior_dist = dist
            else:
                self._walk_low_point = params.copy()
            if self._walk_steps_left > 0 \
                    and self.interior_logl - self._walk_target_logl > tol:
                return [self._bisect_task()]
        return self._maybe_tangent_or_finish()

    def _bisect_task(self):
        params = 0.5 * (self._walk_low_point + self.interior_point)
        return {'params': params,
                'context': {'type': self.type, 'job_id': self.id,
                            'sub_type': 'VOLUME_INTERIOR'}}

    # ------------------------------------------------------------------ #
    # Tangential randomization: spend leftover walk budget moving the
    # representative ALONG its iso-likelihood shell, so walked points do
    # not under-fill the transverse directions of fixed-depth level sets.
    # Each round hops perpendicular to the Broyden normal estimate, then
    # Newton-corrects any depth drift back onto the target (midpoint
    # fallback). Cap and bounds are checked before any evaluation is
    # spent; failed rounds revert and shrink the hop.
    # ------------------------------------------------------------------ #
    def _enter_tangent_phase(self):
        """Arm the tangent phase; returns the first task list or None
        (no budget, or no usable hop direction)."""
        if self._walk_steps_left <= 0:
            return None
        cur_s = (self.interior_point - self._lo) / self._extent
        # Secant normal: the direction along which the walk crossed the
        # target depth; entry/anchor fallbacks for at-entry cases.
        for ref in (self._walk_low_point, self._walk_entry_point,
                    self.anchor):
            if ref is None:
                continue
            normal = cur_s - (np.asarray(ref) - self._lo) / self._extent
            norm = float(np.linalg.norm(normal))
            if norm > 1e-9:
                self._tan_grad = normal / norm
                break
        else:
            return None
        travel = 0.0
        if self._walk_entry_point is not None:
            travel = float(np.linalg.norm(
                cur_s - (self._walk_entry_point - self._lo) / self._extent))
        self._tan_h = max(travel, 2.0 * self._walk_step)
        self._tan_correct_left = TANGENT_CORRECTIONS
        task = self._tangent_task()
        if task is None:
            return None
        self.status = 'TANGENT_WALK'
        return [task]

    def _maybe_tangent_or_finish(self):
        """Terminal of the interior walk: enter the tangent phase when the
        representative sits at the drawn depth and budget remains."""
        tol = WALK_DEPTH_TOL_FRAC * self.band_depth
        if self.interior_point is not None \
                and self._violation(self.interior_logl) == 0.0 \
                and abs(self.interior_logl - self._walk_target_logl) <= tol:
            tasks = self._enter_tangent_phase()
            if tasks is not None:
                return tasks
        self._finish_walk()
        return []

    def _tangent_task(self):
        """One tangential hop proposal (cap and bounds pre-checked without
        spending evaluations); None when no usable direction remains."""
        cur_s = (self.interior_point - self._lo) / self._extent
        anchor_s = (self.anchor - self._lo) / self._extent
        g_norm = float(np.linalg.norm(self._tan_grad))
        if g_norm < 1e-12:
            return None
        n_hat = self._tan_grad / g_norm
        h = self._tan_h
        for _ in range(20):
            g = np.random.standard_normal(len(cur_s))
            v = g - np.dot(g, n_hat) * n_hat
            norm = float(np.linalg.norm(v))
            if norm < 1e-12:
                continue
            cand_s = cur_s + h * (v / norm)
            cand = self._lo + cand_s * self._extent
            if np.all(cand >= self._lo) and np.all(cand <= self._hi) \
                    and float(np.linalg.norm(cand_s - anchor_s)) \
                    <= self._walk_dist_cap + 1e-12:
                self._tan_h = h
                return {'params': cand,
                        'context': {'type': self.type, 'job_id': self.id,
                                    'sub_type': 'VOLUME_TANGENT'}}
            h *= 0.8
        return None

    def _process_tangent_result(self, result):
        logl = float(result['target_val'])
        params = np.asarray(result['params'], dtype=float)
        dist = self._scaled_dist(params)
        self._walk_steps_left -= 1
        tol = WALK_DEPTH_TOL_FRAC * self.band_depth

        # Secant/Broyden rank-1 update of the gradient estimate from this
        # evaluation and the on-target current point: the normal then
        # tracks shell curvature as accepted hops move the point, at zero
        # evaluation cost.
        if np.isfinite(logl):
            d = (params - self.interior_point) / self._extent
            d2 = float(np.dot(d, d))
            if d2 > 1e-18:
                resid = (logl - self.interior_logl) \
                    - float(np.dot(self._tan_grad, d))
                self._tan_grad = self._tan_grad + (resid / d2) * d

        on_target = (self._violation(logl) == 0.0
                     and dist <= self._walk_dist_cap + 1e-12
                     and abs(logl - self._walk_target_logl) <= tol)
        if on_target:
            # Accept the move; start a fresh round from the new position.
            self.interior_point = params.copy()
            self.interior_logl = logl
            self.interior_dist = dist
            self.tangent_moves += 1
            self._tan_correct_left = TANGENT_CORRECTIONS
        elif self._tan_correct_left > 0 and self._walk_steps_left > 0:
            # Curvature drift: Newton-correct along the estimated gradient
            # (preserves the tangential displacement, unlike bisecting
            # back toward the current point); fall back to the midpoint
            # when the model step is unreasonable or leaves cap/bounds.
            self._tan_correct_left -= 1
            cand = None
            g2 = float(np.dot(self._tan_grad, self._tan_grad))
            if np.isfinite(logl) and g2 > 1e-18:
                step_s = self._tan_grad \
                    * ((self._walk_target_logl - logl) / g2)
                if float(np.linalg.norm(step_s)) <= self._tan_h:
                    cand_s = (params - self._lo) / self._extent + step_s
                    cand_p = self._lo + cand_s * self._extent
                    anchor_s = (self.anchor - self._lo) / self._extent
                    if np.all(cand_p >= self._lo) \
                            and np.all(cand_p <= self._hi) \
                            and float(np.linalg.norm(cand_s - anchor_s)) \
                            <= self._walk_dist_cap + 1e-12:
                        cand = cand_p
            if cand is None:
                cand = 0.5 * (params + self.interior_point)
            return [{'params': cand,
                     'context': {'type': self.type, 'job_id': self.id,
                                 'sub_type': 'VOLUME_TANGENT'}}]
        else:
            # Round failed: revert and try a smaller hop next round.
            self._tan_h *= 0.5
            self._tan_correct_left = TANGENT_CORRECTIONS
        if self._walk_steps_left > 0 and self._tan_h > 1e-6:
            task = self._tangent_task()
            if task is not None:
                return [task]
        self._finish_walk()
        return []

    def _finish_walk(self):
        self.status = 'FINISHED'
        self._is_finished = True
        self.success = bool(self._final_success)

    def outcome(self):
        """'hit', 'projected', or 'hole' — meaningful once the job finished."""
        if self.hit:
            return 'hit'
        if self.best_inband_point is not None:
            return 'projected'
        return 'hole'

    def on_finish(self, next_job_id):
        # All bookkeeping lives in the stage state (the orchestrator calls
        # VolumeStageState.record_search_job); never touch the projection
        # grid state the base class manages.
        return None
