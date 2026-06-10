"""Jobs for the volume-sampling stage (tiers 2 and 3 of the funnel).

``VolumeProbeJob`` evaluates the target once at each anchor point (tier 2).
``VolumeSearchJob`` runs an anchored L-BFGS-B search (tier 3) for an anchor
whose probe failed: it minimizes the bounds-scaled distance to the anchor
plus a hinged band-violation penalty, terminating at the first evaluation
that lands in-band within the coverage radius. The penalized objective
steers the *search only*; reported representatives always carry their true
logL, so band membership stays re-derivable (see
docs/volume_sampling_plan.md).
"""
import numpy as np

from ..volume import SOURCE_PROBE
from .base import Job
from .lbfgsb_job import LBFGSBJob

# Band violations are clamped before squaring so a failed evaluation
# (logL = -inf) yields a huge-but-finite penalty instead of overflowing.
VIOLATION_CLAMP = 1e100


class VolumeProbeJob(Job):
    """Evaluate the target once at each of the given anchors (tier 2).

    On finish, writes per-anchor probe records into the
    :class:`~paraprof.volume.AnchorSet` (kept unconditionally, so the
    uniform subset and volume estimate can be re-derived after global-max
    drift) and registers in-band probes as their own anchor's
    representative (distance 0).
    """

    def __init__(self, job_id, sampler, anchor_set, anchor_indices,
                 band_lo, band_hi):
        super().__init__(job_id, 'VOLUME_PROBE', sampler)
        self.anchor_set = anchor_set
        self.anchor_indices = np.asarray(anchor_indices, dtype=int)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
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
            if np.isfinite(logl) and self.band_lo <= logl <= self.band_hi:
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

    Maximized fitness: ``-(dist² + κ·v²)`` with ``dist`` the bounds-scaled
    Euclidean distance to the anchor and ``v`` the band violation
    ``max(0, band_lo - logL) + max(0, logL - band_hi)``. The whole L-BFGS-B
    machinery (FD gradients, line search, ftol) is inherited and runs on
    the transformed values: ``process_result`` rewrites each raw result
    before delegating to the base class.

    Gradients: the distance term is analytic, so an evaluation *inside* the
    band gets a fully analytic gradient and spends zero FD evaluations
    (credited to ``target_calls_saved_by_user_gradient``, since it rides
    the user-gradient path). Outside the band, a user ``grad_func``
    contributes via the chain rule; otherwise plain FD runs on the
    transformed objective.

    The job ends successfully at the first evaluation that is in-band and
    within ``coverage_radius`` of the anchor (a covering point, not a
    stationary one). Otherwise it runs the base machinery to termination
    and the outcome is classified from what was seen: ``projected`` (some
    in-band point, all beyond the radius) or ``hole`` (never in-band; the
    minimum-violation evaluation is kept as the closest approach).
    """

    # Distance slack for interior walks from 'projected' entry points: the
    # walk may move up to this factor times the entry distance from the
    # anchor (hit-entry walks are capped at the coverage radius instead).
    PROJECTED_WALK_SLACK = 1.5

    def __init__(self, job_id, sampler, anchor_set, anchor_index,
                 band_lo, band_hi, kappa, start_params, max_iter=None,
                 interior_steps=0, band_depth=None):
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
        self.band_hi = float(band_hi)
        self.kappa = float(kappa)
        self.interior_steps = int(interior_steps)
        # Band depth in logL units (roi_threshold for roi mode); sets the
        # interior walk's depth-target scale.
        self.band_depth = float(band_depth) if band_depth is not None \
            else float(sampler.roi_threshold)

        self.hit = False
        self.best_inband_point = None
        self.best_inband_logl = -np.inf
        self.best_inband_dist = np.inf
        self.best_viol_point = None
        self.best_viol_logl = -np.inf
        self.best_viol_dist = np.inf
        self.best_viol = np.inf

        # Interior-walk state (experimental, off unless interior_steps > 0).
        # interior_point is the walk's deepest accepted point; the stage
        # state makes it the anchor's representative.
        self.interior_point = None
        self.interior_logl = -np.inf
        self.interior_dist = np.inf
        self._walk_done = False
        self._walk_steps_left = 0
        self._walk_dir_scaled = None
        self._walk_dist_cap = np.inf
        self._walk_target_logl = np.inf
        self._final_success = None

    def _scaled_dist(self, params):
        return float(np.linalg.norm((params - self.anchor) / self._extent))

    def _violation(self, logl):
        v = max(0.0, self.band_lo - logl) + max(0.0, logl - self.band_hi)
        return min(v, VIOLATION_CLAMP)

    def _transformed_gradient(self, params, logl, violation, raw_gradient):
        """Gradient of the maximized fitness at ``params`` (None = use FD).

        In-band the hinge is locally flat, so the gradient is the analytic
        distance term alone — no ∇logL needed. Out of band, chain-rule a
        user-supplied ∇logL if present (NaN entries fall through to FD
        per dim, as in the base class).
        """
        ddist2 = 2.0 * (params - self.anchor) / self._extent ** 2
        if violation == 0.0:
            return -ddist2
        if raw_gradient is None:
            return None
        sign = -1.0 if logl < self.band_lo else 1.0
        grad = np.asarray(raw_gradient, dtype=float)
        return -ddist2 - 2.0 * self.kappa * violation * sign * grad

    def process_result(self, result):
        if result['context'].get('sub_type') == 'VOLUME_INTERIOR':
            return self._process_interior_result(result)

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

        if self.status == 'INTERIOR_WALK' or self._is_finished:
            return []

        if violation == 0.0 and dist <= self.coverage_radius:
            # A covering point, not a stationary one: stop the search. In
            # interior-steps mode, first walk a few steps off the band edge.
            self.hit = True
            self.converged = True
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
    # Interior walk (experimental, roi mode only): a few cheap steps off
    # the band edge. Direction: the inward continuation of
    # (entry - anchor), the line the projection arrived along — no
    # gradient needed. Each walk draws a *depth target*: for a locally
    # quadratic basin, uniform-in-volume depth means ΔlnL below the
    # maximum distributed as depth·U^(2/d), which correctly concentrates
    # near the band edge in high dimensions (most of a d-ball's volume is
    # near its surface) and spreads uniformly in 2D. The walk ascends
    # until its accepted point reaches the target, the step cap, the
    # distance cap, or a non-improving step.
    # ------------------------------------------------------------------ #
    def _start_interior_walk(self, origin, origin_logl, origin_dist,
                             dist_cap):
        """Arm the walk; returns the first task list, or None to skip
        (interior_steps off, shell mode, walk already ran, or the entry
        point already reaches the drawn depth target)."""
        if self.interior_steps <= 0 or self._walk_done \
                or np.isfinite(self.band_hi):
            return None
        self._walk_done = True
        self.interior_point = np.asarray(origin, dtype=float).copy()
        self.interior_logl = float(origin_logl)
        self.interior_dist = float(origin_dist)

        # Depth target: logL >= band_top - band_depth * U^(2/d), with
        # band_top the band's upper logL edge (global max at stage start).
        band_top = self.band_lo + self.band_depth
        u = float(np.random.random())
        self._walk_target_logl = band_top - self.band_depth * u ** (
            2.0 / max(len(self.anchor), 1))
        if self.interior_logl >= self._walk_target_logl:
            return None

        direction = (self.interior_point - self.anchor) / self._extent
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.random.standard_normal(len(direction))
            norm = float(np.linalg.norm(direction))
        self._walk_dir_scaled = direction / norm
        self._walk_step = self.coverage_radius / self.interior_steps
        self._walk_steps_left = self.interior_steps
        self._walk_dist_cap = float(dist_cap)
        self.status = 'INTERIOR_WALK'
        return [self._interior_task()]

    def _interior_task(self):
        scaled = ((self.interior_point - self._lo) / self._extent
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
        deeper = logl > self.interior_logl

        # Tiny slack so a step landing exactly on the cap is not lost to
        # floating-point round-trip noise.
        if in_band and dist <= self._walk_dist_cap + 1e-12 and deeper:
            self.interior_point = params.copy()
            self.interior_logl = logl
            self.interior_dist = dist
            self._walk_steps_left -= 1
            if self._walk_steps_left > 0 \
                    and self.interior_logl < self._walk_target_logl:
                return [self._interior_task()]
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
