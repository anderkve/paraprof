"""Base class for asynchronous jobs."""


class Job:
    """A unit of work decomposable into one or more worker tasks (likelihood evals)."""

    def __init__(self, job_id, job_type, sampler):
        self.id = job_id
        self.type = job_type
        self.sampler = sampler
        self._is_finished = False
        self.success = False

    def start(self):
        """Initial task list. Each task: ``{'params': full_params, 'context': context}``."""
        raise NotImplementedError

    def process_result(self, result):
        """Handle a worker result; return any follow-up tasks (possibly empty)."""
        raise NotImplementedError

    def is_finished(self):
        return self._is_finished

    def on_finish(self, next_job_id):
        """Called on completion. May return ``(new_job, next_job_id)`` to spawn a child."""
        return None
