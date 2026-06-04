"""Cross-projection pool-certificate job (idea 3, flavor a).

Tests, at one grid cell, a profiled-params vector taken from a *higher-fitness*
full-D point that an earlier projection already evaluated and that projects into
this cell. By the one-sided structure of profiling, adopting the result can only
raise the cell value, never lower it -- so this is a regression-proof accuracy
amplifier, not a heuristic. It reuses information already in
``global_solution_pool``; the single confirming evaluation at the exact grid
node is the only cost.
"""
from .base import Job


class PoolCertificateJob(Job):
    """Confirm a cross-projection candidate phi at one cell; polish on improvement."""
    def __init__(self, job_id, sampler, grid_idx, candidate_phi, pool_fitness):
        super().__init__(job_id, 'POOL_CERTIFICATE', sampler)
        self.grid_idx = grid_idx
        self.candidate_phi = candidate_phi
        self.pool_fitness = pool_fitness  # fitness the point had in its origin projection
        self.current_best_fitness = sampler.population[grid_idx]['best_fitness']
        self.test_fitness = None
        self.will_update = False

    def start(self):
        self.sampler.pool_cert_tests += 1
        full_params = self.sampler._construct_params(self.grid_idx, self.candidate_phi)
        context = {'type': self.type, 'job_id': self.id, 'grid_idx': self.grid_idx}
        return [{'params': full_params, 'context': context}]

    def process_result(self, result):
        self.test_fitness = result['target_val']
        self.success = True
        self._is_finished = True
        if self.test_fitness > self.current_best_fitness + self.sampler.suspect_polish_threshold:
            self.will_update = True
            self.sampler.pool_cert_raises += 1
            self.sampler.pool_cert_gain += float(self.test_fitness - self.current_best_fitness)
        return []

    def on_finish(self, next_job_id):
        if not self.success or not self.will_update:
            return None

        # Reuse the patching L-BFGS-B polish path: it adopts via max (one-sided
        # safe), updates the grid + global pool, and is already wired through
        # the master loop, so no new job-type plumbing is needed.
        from .lbfgsb_job import LBFGSBJob

        start_params_full = self.sampler._construct_params(self.grid_idx, self.candidate_phi)
        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='PATCHING_LBFGSB',
            sampler=self.sampler,
            opt_dims=tuple(self.sampler.profiled_dims),
            start_params=self.candidate_phi,
            grid_idx=self.grid_idx,
            start_params_full=start_params_full,
            seed_history=None,
            start_fitness=self.test_fitness,
        )
        job.grid_idx = self.grid_idx
        return (job, next_job_id + 1)
