"""Benchmark user-supplied gradients on Himmelblau-4D and Rosenbrock-4D.

Runs each test function twice — with and without grad_func — on the same
projection and reports target-function calls, FD calls saved, and the
final maximum reached.

Usage (run by the harness, not by hand):
    mpiexec -n 4 python tools/bench_user_gradients.py <function> <method> <grad>

where:
    <function> ∈ {himmelblau_4d, rosenbrock_4d}
    <method>   ∈ {de, lbfgsb}
    <grad>     ∈ {none, full}
"""
import json
import sys

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, run_all_projections, terminate_workers, worker_main,
    get_test_function, set_log_level,
)

set_log_level('WARNING')


# ---------------------------------------------------------------------------
# Analytic gradients of the maximized log-likelihoods
# ---------------------------------------------------------------------------

def himmelblau_4d_grad(params):
    """∇ of paraprof's himmelblau_4d (which is the negated, 0.05-scaled
    sum of two 2-D Himmelblau copies)."""
    x1, x2, x3, x4 = params
    s = 0.05
    g = np.empty(4)
    g[0] = -s * (4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7))
    g[1] = -s * (2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7))
    g[2] = -s * (4 * x3 * (x3**2 + x4 - 11) + 2 * (x3 + x4**2 - 7))
    g[3] = -s * (2 * (x3**2 + x4 - 11) + 4 * x4 * (x3 + x4**2 - 7))
    return g


def rosenbrock_4d_grad(params):
    """∇ of paraprof's rosenbrock_nd: -0.1 * Σ_i [100(x_{i+1}-x_i²)² + (1-x_i)²]."""
    x = np.asarray(params, dtype=float)
    s = 0.1
    n = x.size
    g = np.zeros(n)
    # Contributions from term i = 0..n-2.
    for i in range(n - 1):
        a = x[i + 1] - x[i] ** 2
        b = 1.0 - x[i]
        dgi_dxi = -400.0 * x[i] * a - 2.0 * b
        dgi_dxip1 = 200.0 * a
        g[i] += dgi_dxi
        g[i + 1] += dgi_dxip1
    return -s * g


GRAD_FUNCS = {
    'himmelblau_4d': himmelblau_4d_grad,
    'rosenbrock_4d': rosenbrock_4d_grad,
}


def main():
    func_name = sys.argv[1]            # himmelblau_4d | rosenbrock_4d
    method = sys.argv[2]               # de | lbfgsb
    grad_flag = sys.argv[3]            # none | full

    target_func, bounds, _ = get_test_function(func_name)
    grad_func = GRAD_FUNCS[func_name] if grad_flag == 'full' else None

    projections = [
        {'dims': [0, 1], 'grid_points': [20, 20], 'optimization_method': method},
    ]

    np.random.seed(20260513)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        with ProfileProjector(
            target_func=target_func,
            bounds=bounds,
            projections=projections,
            grad_func=grad_func,
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=20,
            max_patching_waves=5,
            lbfgsb_max_iter=15,
        ) as sampler:
            comm.bcast((sampler.target_func, sampler.grad_func), root=0)
            results = run_all_projections(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False, myrank=rank,
            )
            metrics = results[0]['metrics']
            grid = sampler.export_grid_solution()
            payload = {
                'function': func_name,
                'method': method,
                'grad': grad_flag,
                'calls': metrics['total_target_calls'],
                'saved': sampler.target_calls_saved_by_user_gradient,
                'grad_errors': sampler.user_gradient_errors,
                'max_logL': float(metrics['global_max']),
                'n_activated_cells': len(grid['solutions']),
            }
            print('RESULT', json.dumps(payload), flush=True)
        terminate_workers(comm, rank)
    else:
        worker_main(comm, rank)


if __name__ == '__main__':
    main()
