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

    def __init__(self, job_id, sampler, anchor_set, anchor_index,
                 band_lo, band_hi, kappa, start_params, max_iter=None):
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
        self._extent = anchor_set.bounds[:, 1] - anchor_set.bounds[:, 0]
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.kappa = float(kappa)

        self.hit = False
        self.best_inband_point = None
        self.best_inband_logl = -np.inf
        self.best_inband_dist = np.inf
        self.best_viol_point = None
        self.best_viol_logl = -np.inf
        self.best_viol_dist = np.inf
        self.best_viol = np.inf

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
        logl = result['target_val']
        params = np.asarray(result['params'], dtype=float)
        dist = self._scaled_dist(params)
        violation = self._violation(logl)

        if violation == 0.0:
            if dist < self.best_inband_dist:
                self.best_inband_point = params.copy()
                self.best_inband_logl = float(logl)
                self.best_inband_dist = dist
            if dist <= self.coverage_radius:
                # A covering point, not a stationary one: stop immediately.
                self.hit = True
                self.status = 'FINISHED'
                self._is_finished = True
                self.success = True
                self.converged = True
                return []
        elif (violation < self.best_viol
              or (violation == self.best_viol and dist < self.best_viol_dist)):
            self.best_viol_point = params.copy()
            self.best_viol_logl = float(logl)
            self.best_viol_dist = dist
            self.best_viol = violation

        transformed = dict(result)
        transformed['target_val'] = -(dist * dist
                                      + self.kappa * violation * violation)
        transformed['user_gradient'] = self._transformed_gradient(
            params, logl, violation, result.get('user_gradient'))
        return super().process_result(transformed)

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
