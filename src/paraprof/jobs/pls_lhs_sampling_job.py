"""
PLS LHS Sampling job for generating unbiased training data.

This job handles the evaluation of LHS samples around converged grid points,
which are used to create unbiased training data for PLS subspace learning.
"""
import numpy as np
from .base import Job
from ..logger import get_logger


class PLSLHSSamplingJob(Job):
    """
    A job to evaluate LHS samples for PLS training data.

    This job is spawned automatically when a PLS-LBFGSB optimization
    completes at a grid point. It evaluates points uniformly sampled
    in a hypercube around the best-fit solution.
    """

    def __init__(self, job_id, sampler, grid_idx, best_continuous_params):
        """
        Initialize PLS LHS sampling job.

        Parameters
        ----------
        job_id : int
            Unique job identifier
        sampler : ProfileProjector
            Reference to the sampler instance
        grid_idx : tuple
            Grid point index where sampling is performed
        best_continuous_params : np.ndarray
            Best continuous parameters found at this grid point
        """
        super().__init__(job_id, 'PLS_LHS_SAMPLING', sampler)

        self.grid_idx = grid_idx
        self.best_continuous_params = best_continuous_params

        # Storage for results
        self.X_samples = []
        self.y_samples = []
        self.evals_remaining = 0

        self.logger = get_logger()

    def start(self):
        """Generate LHS sampling tasks."""
        # Generate LHS samples
        tasks = self.sampler.generate_pls_lhs_samples(
            self.grid_idx,
            self.best_continuous_params
        )

        if tasks is None or len(tasks) == 0:
            # LHS sampling is disabled or no continuous dimensions
            self._is_finished = True
            self.success = True
            return []

        # Add job_id to all task contexts
        for task in tasks:
            task['context']['job_id'] = self.id

        self.evals_remaining = len(tasks)

        self.logger.debug(
            f"Job {self.id} (PLS-LHS): Starting {len(tasks)} LHS evaluations "
            f"at grid point {self.grid_idx}"
        )

        return tasks

    def process_result(self, result):
        """Process LHS evaluation result."""
        context = result['context']

        # Extract continuous params and fitness
        continuous_params = context['continuous_params']
        fitness = result['target_val']

        # Store the result
        self.X_samples.append(continuous_params)
        self.y_samples.append(fitness)

        self.evals_remaining -= 1

        # Check if all evaluations are complete
        if self.evals_remaining == 0:
            self._is_finished = True
            self.success = True

        return []

    def on_finish(self, next_job_id):
        """Cache the evaluated LHS samples."""
        if not self.success:
            self.logger.warning(
                f"Job {self.id} (PLS-LHS) failed at grid point {self.grid_idx}"
            )
            return None

        # Cache the training data
        X_samples = np.array(self.X_samples)
        y_samples = np.array(self.y_samples)

        self.sampler.cache_pls_training_data(
            self.grid_idx,
            X_samples,
            y_samples
        )

        return None
