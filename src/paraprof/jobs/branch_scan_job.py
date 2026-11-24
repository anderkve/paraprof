"""Start of the optimisation confined to a branch."""
import numpy as np

from .base import Job

# Values of a split parameter tried along a line, besides the mirrored one.
LINE_POINTS = 16


class BranchScanJob(Job):
    """Look for an optimum inside a branch and start the confined L-BFGS-B
    run there (see ProfileProjector.create_branch_seed_jobs).

    ``point`` is the best known point; it lies outside the branch in at
    least one split parameter. The reference start is ``point`` with each of
    those parameters mirrored into the branch at the nearest end of its
    sub-range, which is the other solution if the two are symmetric about
    the split point.

    They need not be. For each such parameter the target is evaluated along
    a line through the reference start: the parameter takes LINE_POINTS
    values spread over its sub-range, the others stay. An optimum of the
    branch shows as a local maximum along the line. The highest one, over
    all lines, becomes the start. The end of a line next to ``point`` does
    not count: a maximum there is the tail of the optimum that ``point``
    belongs to. Without a local maximum the reference start is used.
    """

    def __init__(self, job_id, sampler, branch, point):
        super().__init__(job_id, 'BRANCH_SEED', sampler)
        self.branch = branch
        point = np.asarray(point, dtype=float)
        self.reference = point.copy()
        moved = {}           # dim -> True if the point lies below the sub-range
        for d, (lo, hi) in branch['ranges'].items():
            if point[d] < lo:
                self.reference[d] = min(hi, 2.0 * lo - point[d])
                moved[d] = True
            elif point[d] > hi:
                self.reference[d] = max(lo, 2.0 * hi - point[d])
                moved[d] = False

        # Per line: the dim, its values in ascending order with the reference
        # value among them, and the index of the end next to the point.
        self.lines = []
        for d, below in moved.items():
            lo, hi = branch['ranges'][d]
            values = lo + (np.arange(LINE_POINTS) + 0.5) / LINE_POINTS * (hi - lo)
            values = np.unique(np.append(values, self.reference[d]))
            self.lines.append((d, values, 0 if below else len(values) - 1))
        self.results = [np.full(len(values), -np.inf) for _, values, _ in self.lines]
        self.evals_remaining = sum(len(values) for _, values, _ in self.lines)

    def start(self):
        if self.evals_remaining == 0:
            self.success = True
            self._is_finished = True
            return []
        tasks = []
        for i, (d, values, _) in enumerate(self.lines):
            for k, value in enumerate(values):
                params = self.reference.copy()
                params[d] = value
                tasks.append({'params': params,
                              'context': {'type': self.type, 'job_id': self.id,
                                          'sub_type': 'BRANCH_LINE', 'line': i, 'point_idx': k}})
        return tasks

    def process_result(self, result):
        context = result['context']
        self.results[context['line']][context['point_idx']] = result['target_val']
        self.evals_remaining -= 1
        if self.evals_remaining == 0:
            self.success = True
            self._is_finished = True
        return []

    def choose_start(self):
        """``(point, value)`` of the highest local maximum along the lines, or
        the reference start with its value (-inf if it was not evaluated)."""
        best_point, best_value = None, -np.inf
        reference_value = -np.inf
        for (d, values, near), f in zip(self.lines, self.results):
            at_reference = int(np.argmin(np.abs(values - self.reference[d])))
            reference_value = f[at_reference]
            for k in range(len(values)):
                if k == near or not np.isfinite(f[k]) or f[k] <= best_value:
                    continue
                rises = k == 0 or f[k] > f[k - 1]
                holds = k == len(values) - 1 or f[k] >= f[k + 1]
                if rises and holds:
                    best_value = f[k]
                    best_point = self.reference.copy()
                    best_point[d] = values[k]
        if best_point is None:
            return self.reference, reference_value
        return best_point, best_value

    def on_finish(self, next_job_id):
        """Spawn the confined L-BFGS-B run from the chosen start."""
        start, value = self.choose_start()
        job = self.sampler.branch_seed_run(next_job_id, self.branch, start, start_fitness=value)
        return job, next_job_id + 1
