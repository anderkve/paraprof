"""Estimate the numerical noise of the target and a finite-difference step that suits it.

A forward difference with step h has an error of about eps_f / h + |f''| h / 2,
smallest for h ~ sqrt(eps_f / |f''|), where eps_f is the scatter of the target
between neighbouring points. The job measures both at a few base points:

1. eps_f from the difference table of the target on equally spaced points
   along a random line (the ECNoise procedure of Moré and Wild, SIAM J. Sci.
   Comput. 33 (2011) 1292);
2. |f''| from second differences along each coordinate, with a step large
   enough for the second difference to stand above the noise (Moré and Wild,
   ACM Trans. Math. Softw. 38 (2012) 19).

The evaluations of a phase are independent: the job takes two or three wall steps.
"""
import numpy as np

from .base import Job

N_LINE = 9                    # points per line of the difference table
LINE_SPACING = 1e-7           # first spacing of those points (fraction of the bound extent)
MAX_LINE_ATTEMPTS = 3
MARGIN = 0.1                  # base points keep this distance (fraction) from the bounds


def ecnoise(fval):
    """Noise level of equally spaced function values (ECNoise).

    Returns ``(noise, inform)``: inform 1 noise detected, 2 spacing too small
    (at least half of the first differences vanish), 3 spacing too large (the
    values are dominated by the trend of the function).
    """
    f = np.asarray(fval, dtype=float)
    nf = len(f)
    fmin, fmax = f.min(), f.max()
    scale = max(abs(fmin), abs(fmax))
    if scale > 0 and (fmax - fmin) / scale > 0.1:
        return 0.0, 3
    level = np.zeros(nf - 1)
    dsgn = np.zeros(nf - 1, dtype=bool)
    gamma = 1.0
    d = f.copy()
    for j in range(1, nf):
        d = d[1:] - d[:-1]
        if j == 1 and np.sum(d == 0.0) >= nf / 2:
            return 0.0, 2
        gamma *= 0.5 * (j / (2.0 * j - 1.0))
        level[j - 1] = np.sqrt(gamma * np.mean(d ** 2))
        dsgn[j - 1] = (d.min() * d.max() < 0.0)
    for k in range(nf - 3):
        emin, emax = level[k:k + 3].min(), level[k:k + 3].max()
        if emax <= 4.0 * emin and dsgn[k]:
            return float(level[k]), 1
    return 0.0, 3


class NoiseEstimateJob(Job):
    """Measure the noise of the target and set the sampler's finite-difference step."""

    def __init__(self, job_id, sampler, base_points, step_min, step_max, step_fallback):
        super().__init__(job_id, 'NOISE_ESTIMATE', sampler)
        b = np.asarray(sampler.bounds, dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.extent = self.hi - self.lo
        self.n_dims = len(self.lo)
        self.step_min, self.step_max, self.step_fallback = step_min, step_max, step_fallback
        # keep the base points away from the bounds: the probes need room on both sides
        z = (np.asarray(base_points, dtype=float) - self.lo) / self.extent
        self.base_z = np.clip(z, MARGIN, 1.0 - MARGIN)
        rng = np.random.default_rng(np.random.randint(1, 2 ** 31))
        self.directions = []
        for _ in self.base_z:
            v = rng.standard_normal(self.n_dims)
            self.directions.append(v / np.linalg.norm(v))
        self.spacing = [LINE_SPACING] * len(self.base_z)
        self.attempt = [0] * len(self.base_z)
        self.line_vals = [dict() for _ in self.base_z]
        self.noise = [None] * len(self.base_z)      # per base point; None = no estimate
        self.base_val = [None] * len(self.base_z)
        self.curv_vals = {}
        self.curv_step = None
        self.pending = 0
        self.phase = 'LINE'
        self.result = None

    # --- tasks ----------------------------------------------------------------
    def _point(self, z):
        return self.lo + np.clip(z, 0.0, 1.0) * self.extent

    def _line_tasks(self, b):
        tasks = []
        self.line_vals[b] = {}
        for i in range(N_LINE):
            z = self.base_z[b] + (i - N_LINE // 2) * self.spacing[b] * self.directions[b]
            tasks.append({'params': self._point(z),
                          'context': {'type': self.type, 'job_id': self.id, 'sub_type': 'LINE',
                                      'base': b, 'i': i}})
        self.pending += len(tasks)
        return tasks

    def start(self):
        tasks = []
        for b in range(len(self.base_z)):
            tasks += self._line_tasks(b)
        return tasks

    def process_result(self, result):
        ctx = result['context']
        self.pending -= 1
        new_tasks = []
        if ctx['sub_type'] == 'LINE':
            b = ctx['base']
            self.line_vals[b][ctx['i']] = result['target_val']
            if len(self.line_vals[b]) == N_LINE:
                new_tasks += self._close_line(b)
        else:
            self.curv_vals[(ctx['base'], ctx['dim'], ctx['sign'])] = result['target_val']

        if self.pending == 0 and not new_tasks:
            if self.phase == 'LINE':
                new_tasks = self._begin_curvature()
            if self.pending == 0:
                self._conclude()
        return new_tasks

    def _close_line(self, b):
        vals = np.array([self.line_vals[b][i] for i in range(N_LINE)])
        if not np.all(np.isfinite(vals)):
            return []                      # invalid region: no estimate from this base point
        self.base_val[b] = float(vals[N_LINE // 2])
        noise, inform = ecnoise(vals)
        floor = 2.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(vals))))
        if inform == 1:
            self.noise[b] = max(noise, floor)
            return []
        if inform == 2:
            # Identical neighbouring values: the target is exact to rounding at this spacing.
            self.noise[b] = floor
            return []
        # Trend of the function dominates: shorten the line and try again.
        self.attempt[b] += 1
        if self.attempt[b] >= MAX_LINE_ATTEMPTS:
            return []
        self.spacing[b] *= 1e-2
        return self._line_tasks(b)

    def _begin_curvature(self):
        self.phase = 'CURVATURE'
        est = [n for n in self.noise if n is not None]
        self.eps_f = float(np.median(est)) if est else None
        if not est:
            return []
        # Step of the second difference: large enough to stand above the noise.
        self.curv_step = float(np.clip(self.eps_f ** 0.25, 1e-4, 0.05))
        tasks = []
        for b in range(len(self.base_z)):
            if self.noise[b] is None:
                continue
            for i in range(self.n_dims):
                for sign in (1, -1):
                    z = self.base_z[b].copy()
                    z[i] += sign * self.curv_step
                    tasks.append({'params': self._point(z),
                                  'context': {'type': self.type, 'job_id': self.id,
                                              'sub_type': 'CURVATURE', 'base': b, 'dim': i,
                                              'sign': sign}})
        self.pending += len(tasks)
        return tasks

    # --- conclusion -------------------------------------------------------------
    def _conclude(self):
        eps_f = self.eps_f
        if eps_f is None:
            self.result = {'noise': None, 'fd_step': self.step_fallback, 'curvature': None,
                           'reason': 'no base point gave a noise estimate'}
        else:
            curv = []
            for b in range(len(self.base_z)):
                if self.noise[b] is None:
                    continue
                f0 = self.base_val[b]
                for i in range(self.n_dims):
                    fp = self.curv_vals[(b, i, 1)]
                    fm = self.curv_vals[(b, i, -1)]
                    if not (np.isfinite(fp) and np.isfinite(fm)):
                        continue
                    second = abs(fp - 2.0 * f0 + fm)
                    if second > 100.0 * eps_f:        # a difference that is above the noise
                        curv.append(second / self.curv_step ** 2)
            if curv:
                mu = float(np.median(curv))
                step = 8.0 ** 0.25 * np.sqrt(eps_f / mu)
                reason = None
            else:
                mu = None
                step = self.step_fallback
                reason = 'no second difference above the noise'
            self.result = {'noise': eps_f, 'curvature': mu,
                           'fd_step': float(np.clip(step, self.step_min, self.step_max)),
                           'reason': reason}
        self.success = True
        self._is_finished = True

    def on_finish(self, next_job_id):
        s = self.sampler
        r = self.result
        s._noise_estimated = True
        s.lbfgsb_fd_step_value = r['fd_step']
        if r['noise'] is not None:
            s.lbfgsb_noise = r['noise']
        if r['noise'] is None:
            s.logger.warning(
                f"--- Noise estimate failed ({r['reason']}); "
                f"finite-difference step set to {r['fd_step']:.1e} of the bound extent ---")
        else:
            curv = f"{r['curvature']:.2e}" if r['curvature'] is not None else "not resolved"
            s.logger.info(
                f"--- Noise of the target: {r['noise']:.2e}; curvature scale: {curv}; "
                f"finite-difference step: {r['fd_step']:.1e} of the bound extent ---")
        return None
