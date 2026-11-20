"""Run the paraprof master against virtual workers, without MPI processes.

`VirtualComm` stands in for the communicator of `master_main`. Each virtual
worker evaluates the target in this process and returns the result after a
simulated evaluation time, so the master's scheduling can be studied for any
number of workers on one core. The master code runs unchanged.

    from virtual_pool import run_virtual_scan
    out = run_virtual_scan(target, bounds, projections, n_workers=100, **projector_kwargs)
    out['wall'], out['evaluations'], out['mean_busy_workers']

Limits: the cost of the master is not on the simulated clock
(`out['master_seconds_per_result']` gives it in real time), and every worker
takes the same simulated time up to the optional jitter.
"""
import copy
import heapq
import time

import numpy as np

import paraprof
import paraprof.master as _master
from paraprof.worker import evaluate_task


class _DoneRequest:
    """Request of a send that has already completed."""

    def Test(self):
        return True


class _MPIShim:
    """The names `paraprof.master` takes from `mpi4py.MPI`."""
    ANY_SOURCE = -1

    class Request:
        @staticmethod
        def Waitall(requests):
            return None


class VirtualComm:
    """Communicator with `n_workers` simulated workers (ranks 1..n_workers).

    An evaluation takes one unit of simulated time; `jitter` is the relative
    spread of that duration (lognormal sigma).
    """

    def __init__(self, n_workers, target_func, grad_func=None, jitter=0.0, seed=0):
        self.n_workers = int(n_workers)
        self.target_func = target_func
        self.grad_func = grad_func
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.logger = paraprof.get_logger()
        self.now = 0.0
        self._heap = []          # (t_done, seq, result)
        self._seq = 0
        self._idle_probe = False
        self.log = []            # one row per evaluation: (rank, t_start, t_end, job type)

    # --- the subset of the mpi4py API that the master uses -----------------
    def Get_size(self):
        return self.n_workers + 1

    def isend(self, task, dest):
        self._idle_probe = False
        if isinstance(task, int) and task == _master.TASK_TERMINATE:
            return _DoneRequest()
        result = evaluate_task(task, self.target_func, self.grad_func, dest, self.logger)
        dur = 1.0
        if self.jitter:
            dur *= float(np.exp(self.jitter * self.rng.standard_normal()))
        self.log.append((dest, self.now, self.now + dur, task['context'].get('type', '')))
        heapq.heappush(self._heap, (self.now + dur, self._seq, result))
        self._seq += 1
        return _DoneRequest()

    def Iprobe(self, source):
        if not self._heap:
            return False
        if self._heap[0][0] <= self.now:
            return True
        # Nothing has arrived yet. The first empty probe lets the master
        # dispatch what it has queued; if it probes again without having
        # sent anything, it is waiting, and the clock moves to the next
        # completion.
        if self._idle_probe:
            self.now = self._heap[0][0]
            self._idle_probe = False
            return True
        self._idle_probe = True
        return False

    def recv(self, source):
        t_done, _, result = heapq.heappop(self._heap)
        self.now = max(self.now, t_done)
        self._idle_probe = False
        return result


def run_virtual_scan(target_func, bounds, projections, n_workers, grad_func=None,
                     jitter=0.0, seed=0, log_level='WARNING', sampler_hook=None,
                     save_plots=False, **projector_kwargs):
    """One paraprof scan on `n_workers` virtual workers.

    Returns a dict with the sampler state after the last projection and the
    simulated timeline; times are in units of one evaluation. `projections`
    is deep-copied. `sampler_hook(sampler)` is called before the scan starts,
    for tests that instrument the sampler.
    """
    paraprof.set_log_level(log_level)
    projections = copy.deepcopy(projections)
    comm = VirtualComm(n_workers, target_func, grad_func=grad_func, jitter=jitter, seed=seed)
    real_mpi = _master.MPI
    _master.MPI = _MPIShim
    t_real = time.perf_counter()
    try:
        with paraprof.ProfileProjector(target_func=target_func, bounds=bounds,
                                       projections=projections, grad_func=grad_func,
                                       **projector_kwargs) as sampler:
            if sampler_hook is not None:
                sampler_hook(sampler)
            results = paraprof.run_scan(comm=comm, sampler=sampler, projections=projections,
                                        save_plots=save_plots, broadcast_target_func=False,
                                        myrank=0)
            out = {
                'results': results,
                'population': {idx: {'status': st['status'],
                                     'best_fitness': float(st['best_fitness'])}
                               for idx, st in sampler.population.items()},
                'grid_axes': [np.array(a) for a in sampler.grid_axes],
                'projection_dims': list(sampler.projection_dims),
                'global_max': float(sampler.global_max_target_val),
                'target_calls': int(sampler.target_calls),
            }
    finally:
        _master.MPI = real_mpi
    t_real = time.perf_counter() - t_real

    rank, start, end, job_type = zip(*comm.log)
    start, end = np.array(start), np.array(end)
    wall = float(end.max() - start.min())
    mean_busy = float(np.sum(end - start)) / wall
    out.update({
        'evaluations': len(start),
        'wall': wall,
        'mean_busy_workers': mean_busy,
        'utilisation': mean_busy / n_workers,
        'timeline': (np.array(rank), start, end, list(job_type)),
        'master_seconds_per_result': t_real / len(start),
    })
    return out
