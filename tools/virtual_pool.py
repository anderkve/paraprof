"""Run the paraprof master against virtual workers, without MPI processes.

`VirtualComm` stands in for the communicator that `master_main` uses. Each
virtual worker evaluates the target in this process and hands the result back
after a simulated evaluation time, so the master's scheduling can be studied
for any number of workers on one core: how many workers are busy at a time,
how long a scan takes in units of one evaluation, how many evaluations it
spends.

The master code runs unchanged. Only the clock is simulated: it advances to
the next completion whenever the master has nothing left to dispatch.

    from virtual_pool import run_virtual_scan
    out = run_virtual_scan(target, bounds, projections, n_workers=100, **projector_kwargs)
    out['wall'], out['evaluations'], out['mean_busy_workers']

Limits: the cost of the master itself is not part of the simulated clock
(`out['master_seconds_per_result']` reports it in real time), and an
evaluation takes the same simulated time on every worker up to the optional
jitter.
"""
import heapq
import time

import numpy as np

import paraprof
import paraprof.master as _master

TASK_TERMINATE = -1


class _DoneRequest:
    """Request of a send that has already completed."""

    def wait(self, *args, **kwargs):
        return None

    Wait = wait

    def test(self, *args, **kwargs):
        return (True, None)

    def Test(self, *args, **kwargs):
        return True


class _MPIShim:
    """The names `paraprof.master` takes from `mpi4py.MPI`."""
    ANY_SOURCE = -1
    ANY_TAG = -1

    class Request:
        @staticmethod
        def Waitall(requests, statuses=None):
            return None

        @staticmethod
        def Testsome(requests, statuses=None):
            return list(range(len(requests)))


class VirtualComm:
    """Communicator with `n_workers` simulated workers (ranks 1..n_workers).

    eval_time : simulated duration of one evaluation.
    jitter    : relative spread of that duration (lognormal sigma).
    """

    def __init__(self, n_workers, target_func, grad_func=None, eval_time=1.0,
                 jitter=0.0, seed=0):
        self.n_workers = int(n_workers)
        self.target_func = target_func
        self.grad_func = grad_func
        self.eval_time = float(eval_time)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.now = 0.0
        self._heap = []          # (t_done, seq, result)
        self._seq = 0
        self._idle_probe = False
        # one row per evaluation: (rank, t_start, t_end, job type, sub type)
        self.log = []

    # --- the subset of the mpi4py API that the master uses -----------------
    def Get_size(self):
        return self.n_workers + 1

    def Get_rank(self):
        return 0

    def bcast(self, obj, root=0):
        return obj

    def isend(self, task, dest, tag=0):
        self._idle_probe = False
        if isinstance(task, int) and task == TASK_TERMINATE:
            return _DoneRequest()
        result = self._evaluate(task, dest)
        dur = self.eval_time
        if self.jitter:
            dur *= float(np.exp(self.jitter * self.rng.standard_normal()))
        ctx = task['context']
        self.log.append((dest, self.now, self.now + dur, ctx.get('type', ''),
                         ctx.get('sub_type', '')))
        heapq.heappush(self._heap, (self.now + dur, self._seq, result))
        self._seq += 1
        return _DoneRequest()

    send = isend

    def Iprobe(self, source=-1, tag=-1):
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

    def recv(self, source=-1, tag=-1, status=None):
        t_done, _, result = heapq.heappop(self._heap)
        if t_done > self.now:
            self.now = t_done
        self._idle_probe = False
        return result

    # --- what a worker does with one task (as in paraprof.worker) ----------
    def _evaluate(self, task, rank):
        params = task['params']
        context = task['context']
        error = None
        try:
            val = self.target_func(params)
        except Exception as e:  # noqa: BLE001 - mirror worker_main
            val = -np.inf
            error = f"{type(e).__name__}: {e}"
        if val is None or not np.isfinite(val):
            if not (val == -np.inf):
                error = error or f"Non-finite target value {val!r}"
                val = -np.inf
        user_gradient = None
        user_gradient_error = None
        if context.get('compute_gradient') and self.grad_func is not None and error is None:
            from paraprof.worker import _normalize_user_gradient
            try:
                user_gradient, user_gradient_error = _normalize_user_gradient(
                    self.grad_func(params), len(params))
            except Exception as e:  # noqa: BLE001
                user_gradient_error = f"{type(e).__name__}: {e}"
        context['worker_rank'] = rank
        return {'target_val': val, 'params': params, 'context': context,
                'error': error, 'user_gradient': user_gradient,
                'user_gradient_error': user_gradient_error}

    # --- bookkeeping --------------------------------------------------------
    def timeline(self):
        """Arrays (rank, start, end) and lists (job type, sub type)."""
        if not self.log:
            z = np.zeros(0)
            return z.astype(int), z, z, [], []
        rank = np.array([r[0] for r in self.log])
        start = np.array([r[1] for r in self.log])
        end = np.array([r[2] for r in self.log])
        return rank, start, end, [r[3] for r in self.log], [r[4] for r in self.log]


def busy_profile(start, end, n_workers):
    """Fraction of the wall time with k workers busy, k = 0..n_workers, and the mean."""
    if len(start) == 0:
        return np.zeros(n_workers + 1), 0.0
    t0, t1 = start.min(), end.max()
    times = np.concatenate([start, end])
    delta = np.concatenate([np.ones(len(start)), -np.ones(len(end))])
    order = np.lexsort((delta, times))      # ends before starts at equal times
    times, delta = times[order], delta[order]
    level = np.cumsum(delta)[:-1]
    width = np.diff(times)
    hist = np.zeros(n_workers + 1)
    np.add.at(hist, np.clip(np.round(level).astype(int), 0, n_workers), width)
    span = t1 - t0
    return hist / span, float(np.sum(level * width) / span)


def run_virtual_scan(target_func, bounds, projections, n_workers, grad_func=None,
                     eval_time=1.0, jitter=0.0, seed=0, log_level='WARNING',
                     **projector_kwargs):
    """One paraprof scan on `n_workers` virtual workers.

    Returns a dict with the sampler state after the last projection and the
    simulated timeline. `projections` is deep-copied.
    """
    import copy

    paraprof.set_log_level(log_level)
    projections = copy.deepcopy(projections)
    comm = VirtualComm(n_workers, target_func, grad_func=grad_func,
                       eval_time=eval_time, jitter=jitter, seed=seed)
    real_mpi = _master.MPI
    _master.MPI = _MPIShim
    t_real = time.perf_counter()
    try:
        with paraprof.ProfileProjector(target_func=target_func, bounds=bounds,
                                       projections=projections, grad_func=grad_func,
                                       **projector_kwargs) as sampler:
            results = paraprof.run_scan(comm=comm, sampler=sampler, projections=projections,
                                        save_plots=False, broadcast_target_func=False, myrank=0)
            population = {idx: {'status': st['status'], 'best_fitness': float(st['best_fitness'])}
                          for idx, st in sampler.population.items()}
            grid_axes = [np.array(a) for a in sampler.grid_axes]
            out = {
                'results': results,
                'population': population,
                'grid_axes': grid_axes,
                'projection_dims': list(sampler.projection_dims),
                'global_max': float(sampler.global_max_target_val),
                'target_calls': int(sampler.target_calls),
            }
    finally:
        _master.MPI = real_mpi
    t_real = time.perf_counter() - t_real

    rank, start, end, jtype, stype = comm.timeline()
    hist, mean_busy = busy_profile(start, end, n_workers)
    wall = float(end.max() - start.min()) if len(start) else 0.0
    out.update({
        'n_workers': n_workers,
        'evaluations': int(len(start)),
        'wall': wall,                               # simulated, in units of eval_time if that is 1
        'mean_busy_workers': mean_busy,
        'utilisation': mean_busy / n_workers if n_workers else 0.0,
        'busy_histogram': hist,
        'timeline': (rank, start, end, jtype, stype),
        'master_seconds_per_result': t_real / max(len(start), 1),
    })
    return out


def by_job_type(out):
    """Evaluations, simulated wall span and busy workers per job type."""
    rank, start, end, jtype, _ = out['timeline']
    table = {}
    jtype = np.array(jtype)
    for name in dict.fromkeys(jtype.tolist()):
        m = jtype == name
        table[name] = {'evaluations': int(m.sum()),
                       'first_start': float(start[m].min()),
                       'last_end': float(end[m].max())}
    return table
