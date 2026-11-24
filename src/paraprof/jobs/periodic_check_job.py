"""Check that the target agrees at the two bounds of each periodic parameter."""
import numpy as np

from .base import Job

# Relative difference between the two values above which a warning is logged.
REL_TOLERANCE = 1e-6
# The same, in units of the noise of the target where that is known.
NOISE_TOLERANCE = 100.0


class PeriodicCheckJob(Job):
    """Evaluate the target at the lower and the upper bound of each periodic
    parameter, the other parameters taken from a few base points.

    The two values are equal if the bounds span one period. A difference is
    logged as a warning; the scan goes on either way.
    """

    def __init__(self, job_id, sampler, base_points):
        super().__init__(job_id, 'PERIODIC_CHECK', sampler)
        self.base_points = np.atleast_2d(np.asarray(base_points, dtype=float))
        self.values = {}          # (dim, base index, 0 or 1) -> target value
        self.pending = 0

    def start(self):
        tasks = []
        for dim in self.sampler.periodic_dims:
            for b, base in enumerate(self.base_points):
                for side in (0, 1):
                    point = base.copy()
                    point[dim] = self.sampler.bounds[dim, side]
                    tasks.append({'params': point,
                                  'context': {'type': self.type, 'job_id': self.id,
                                              'dim': dim, 'base': b, 'side': side}})
        self.pending = len(tasks)
        if not tasks:
            self.success = True
            self._is_finished = True
        return tasks

    def process_result(self, result):
        ctx = result['context']
        self.values[(ctx['dim'], ctx['base'], ctx['side'])] = result['target_val']
        self.pending -= 1
        if self.pending == 0:
            self.success = True
            self._is_finished = True
        return []

    def on_finish(self, next_job_id):
        sampler = self.sampler
        sampler._periodic_checked = True
        for dim in sampler.periodic_dims:
            worst = 0.0
            n_compared = 0
            for b in range(len(self.base_points)):
                f_lo = self.values.get((dim, b, 0), -np.inf)
                f_hi = self.values.get((dim, b, 1), -np.inf)
                if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
                    continue
                n_compared += 1
                tol = max(REL_TOLERANCE * max(1.0, abs(f_lo), abs(f_hi)),
                          NOISE_TOLERANCE * sampler.lbfgsb_noise)
                if abs(f_lo - f_hi) > tol:
                    worst = max(worst, abs(f_lo - f_hi))
            name = (sampler.parameter_names[dim] if sampler.parameter_names
                    else f"parameter {dim}")
            if worst > 0.0:
                sampler.logger.warning(
                    f"{name} is marked periodic, but the target differs by up to "
                    f"{worst:.3g} between its lower and its upper bound. The bounds "
                    f"must span exactly one period.")
            elif n_compared == 0:
                sampler.logger.info(
                    f"--- Periodic check: no valid target value at the bounds of {name}; "
                    f"not checked ---")
            else:
                sampler.logger.info(
                    f"--- Periodic check: the target agrees at the bounds of {name} ---")
        return None
