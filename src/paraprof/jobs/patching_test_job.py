"""Patching test job for wave-based patching verification."""
import numpy as np
from .base import Job


class PatchingTestJob(Job):
    """Test whether a grid point improves when using a neighbor's profiled params."""
    def __init__(self, job_id, sampler, grid_idx, test_profiled_params,
                 wave_number):
        super().__init__(job_id, 'PATCHING_TEST', sampler)
        self.grid_idx = grid_idx
        self.test_profiled_params = test_profiled_params
        self.wave_number = wave_number

        # Store current best for comparison
        self.current_best_fitness = self.sampler.population[self.grid_idx]['best_fitness']

        # Result storage
        self.test_fitness = None
        self.will_update = False

    def start(self):
        """Return task to evaluate the test parameters."""
        full_params = self.sampler._construct_params(self.grid_idx, self.test_profiled_params)
        context = {
            'type': self.type,
            'job_id': self.id,
            'grid_idx': self.grid_idx,
            'wave_number': self.wave_number
        }
        return [{'params': full_params, 'context': context}]

    def process_result(self, result):
        """Store the test fitness result."""
        self.test_fitness = result['target_val']
        self.success = True
        self._is_finished = True

        # Check if we found an improvement
        if self.test_fitness > self.current_best_fitness:
            self.will_update = True

        return []  # No new tasks

    def on_finish(self, next_job_id):
        """
        If test found improvement, spawn L-BFGS-B job to optimize from test point.
        """
        if not self.success or not self.will_update:
            return None

        # Import here to avoid circular dependency
        from .lbfgsb_job import LBFGSBJob

        # Construct full parameter vector for initial evaluation
        start_params_full = self.sampler._construct_params(
            self.grid_idx, self.test_profiled_params
        )

        # Create L-BFGS-B job to optimize from the improved starting point
        # Use special job type to track it's part of patching wave
        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='PATCHING_LBFGSB',
            sampler=self.sampler,
            opt_dims=tuple(self.sampler.profiled_dims),
            start_params=self.test_profiled_params,
            grid_idx=self.grid_idx,
            start_params_full=start_params_full,
            seed_history=None,
            start_fitness=self.test_fitness
        )

        # Store wave context in the job for tracking
        job.wave_number = self.wave_number
        job.grid_idx = self.grid_idx

        return (job, next_job_id + 1)
