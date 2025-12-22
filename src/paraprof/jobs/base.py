"""
Base class for asynchronous jobs.
"""


class Job:
    """
    Abstract base class for a job.
    A job is a self-contained unit of work that can be broken down into
    one or more tasks (likelihood evaluations).
    """
    def __init__(self, job_id, job_type, sampler):
        self.id = job_id
        self.type = job_type
        self.sampler = sampler  # Reference to the main sampler state object
        self._is_finished = False
        self.success = False

    def start(self):
        """
        Returns the initial list of tasks to be queued.
        Each task is a dict: {'params': full_params, 'context': context}
        """
        raise NotImplementedError

    def process_result(self, result):
        """
        Processes a worker result associated with this job.
        Returns a list of new tasks to be queued (can be empty).
        """
        raise NotImplementedError

    def is_finished(self):
        """Returns True if the job is complete."""
        return self._is_finished

    def on_finish(self, next_job_id):
        """
        Called by the master when the job is finished.
        Use this to update the main sampler's state.
        Can optionally return (new_job, next_job_id) to spawn a child job.
        """
        return None
