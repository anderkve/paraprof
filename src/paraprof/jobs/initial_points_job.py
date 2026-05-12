"""
Job for evaluating user-provided initial points.

Routes the evaluations through the normal master/result loop so they share
the same bookkeeping path as every other target call: ``_register_target_call``
(which feeds ``samples_output_file``) and ``_update_global_pool``.
"""
import numpy as np

from .base import Job


class InitialPointEvalJob(Job):
    """Evaluate every entry in ``sampler.initial_points`` exactly once.

    Successful evaluations land in ``sampler.initial_maxima`` (consumed by
    the activation stage) and in ``sampler.global_solution_pool``.
    """

    def __init__(self, job_id, sampler, points):
        super().__init__(job_id, 'INITIAL_POINT_EVAL', sampler)
        self.points = np.asarray(points)
        self.target_vals = np.full(len(self.points), -np.inf)
        self.evals_remaining = len(self.points)

    def start(self):
        if len(self.points) == 0:
            self.success = True
            self._is_finished = True
            return []

        tasks = []
        for i, point in enumerate(self.points):
            context = {
                'type': self.type,
                'job_id': self.id,
                'point_idx': i,
            }
            tasks.append({'params': point, 'context': context})
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
        sampler = self.sampler
        for idx, target_val in enumerate(self.target_vals):
            point = self.points[idx]
            target_val = float(target_val)
            sampler.initial_maxima.append({
                'point': point,
                'target_val': target_val,
            })
            if target_val > sampler.global_max_target_val:
                sampler.global_max_target_val = target_val
            if np.isfinite(target_val):
                sampler._update_global_pool(
                    np.asarray(point), target_val, grid_idx=None
                )

        sampler._initial_points_evaluated = True

        if len(self.points) > 0:
            best = float(np.max(self.target_vals))
            sampler.logger.info(
                f"--- Evaluated {len(self.points)} initial points. "
                f"Best: {best:.4e} ---"
            )
        return None
