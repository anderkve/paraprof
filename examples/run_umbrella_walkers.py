"""
Prototype: umbrella-sampling-in-lnL walkers as an alternative to the
volume-sampling funnel.

Runs the six 2D projections of a 4D test function (to build the same
ProjectionEnvelope and global max the funnel uses), seeds N walkers at
envelope-filtered anchor points, and runs each as an independent
random-walk Metropolis chain whose target is a log-Gaussian in lnL:

    pi_l(theta)  proportional to  exp(-(lnL(theta) - l)^2 / (2 sigma^2))

Each walker's home level l is drawn uniform-in-DeltalnL across the band
[global_max - roi, global_max], so the ensemble tiles the lnL range
(lnL-filling); the spread of seeds + the random walk give space-filling.
No gradients, no normals, no coverage radius -- only lnL evaluations.

Every evaluated point is logged with its lnL (accepted or not), since each
is a valid populated ROI sample. Outputs umbrella_<func>.csv
([params..., lnL, accepted, level]) and umbrella_<func>_summary.json.

    mpiexec -n <ncores> python run_umbrella_walkers.py <func> <roi> \\
        [--grid G] [--steps S] [--sigma-frac F] [--eval-budget B] \\
        [--n-walkers N]

Defaults give the scan-matched budget: eval_budget = projection evals, and
N = eval_budget / (steps + 1).
"""
import argparse
import json
import time

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, get_test_function, run_all_projections,
    set_log_level, terminate_workers, worker_main,
)
from paraprof.volume import ProjectionEnvelope, generate_anchors
from paraprof.worker import TASK_TERMINATE

set_log_level('WARNING')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('func')
    p.add_argument('roi', type=float)
    p.add_argument('--grid', type=int, default=20)
    p.add_argument('--steps', type=int, default=30,
                   help='MCMC steps per walker (after the seed eval)')
    p.add_argument('--sigma-frac', type=float, default=0.15,
                   help='umbrella width sigma as a fraction of roi')
    p.add_argument('--eval-budget', type=int, default=None,
                   help='default: projection evals')
    p.add_argument('--n-walkers', type=int, default=None,
                   help='default: eval_budget // (steps + 1)')
    p.add_argument('--label', default='')
    p.add_argument('--no-precondition', action='store_true',
                   help='use isotropic proposals (disable the local '
                        'gradient preconditioner)')
    p.add_argument('--burn-in', type=int, default=4,
                   help='isotropic steps before the preconditioner engages')
    return p.parse_args()


args = parse_args()
func_name = args.func
roi_threshold = args.roi
roi_volume = roi_threshold + 2.0          # same band as the funnel test
suffix = f"_{args.label}" if args.label else ""
precondition = not args.no_precondition
np.random.seed(20260613)

log_likelihood, bounds, _ = get_test_function(func_name)
bounds = np.asarray(bounds, dtype=float)
n_dims = len(bounds)
lo, extent = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

PROJECTIONS = [
    {'dims': [i, j], 'grid_points': [args.grid, args.grid]}
    for i in range(n_dims) for j in range(i + 1, n_dims)
]


def umbrella_logpi(lnl, level, sigma):
    if not np.isfinite(lnl):
        return -np.inf
    return -0.5 * ((lnl - level) / sigma) ** 2


if rank == 0:
    comm.bcast((log_likelihood, None), root=0)
    t0 = time.time()

    with ProfileProjector(
        target_func=log_likelihood, bounds=bounds, projections=PROJECTIONS,
        roi_threshold=roi_threshold, pop_per_grid_point=3,
        n_initial_optimizations=60,
        samples_output_file=f"samples_{func_name}_umbrella.csv",
    ) as sampler:
        results = run_all_projections(comm=comm, sampler=sampler,
                                      projections=PROJECTIONS,
                                      save_plots=False, myrank=rank)
        n_projection_evals = sampler.target_calls
        global_max = sampler.global_max_target_val
        t_proj = time.time()

        eval_budget = (args.eval_budget if args.eval_budget is not None
                       else n_projection_evals)
        n_walkers = (args.n_walkers if args.n_walkers is not None
                     else max(eval_budget // (args.steps + 1), 1))
        sigma = args.sigma_frac * roi_volume
        band_lo = global_max - roi_volume

        # Envelope-seeded anchors == walker start points (projection info
        # for efficiency); each walker gets a home level uniform in DeltalnL.
        envelope = ProjectionEnvelope.from_projection_results(
            results, global_max, n_dims)
        anchor_set = generate_anchors(envelope, bounds, n_walkers,
                                      roi_volume, seed=1)
        starts = anchor_set.anchors
        n_walkers = len(starts)
        rng = np.random.default_rng(7)
        levels = global_max - rng.uniform(0.0, roi_volume, size=n_walkers)

        # --- Walker state ---
        theta = starts.copy()
        lnl_cur = np.full(n_walkers, np.nan)        # NaN until seed eval done
        step = np.full(n_walkers, 0.05)             # tangential proposal sd
        grad = np.zeros((n_walkers, n_dims))        # scaled-space lnL gradient
        n_done = np.zeros(n_walkers, dtype=int)
        proposal = [None] * n_walkers
        records = []                                # (params, lnl, acc, level)

        workers = [r for r in range(comm.Get_size()) if r != rank]
        free = list(workers)
        ready = list(range(n_walkers))              # need a seed eval first
        busy = {}                                   # worker_rank -> walker idx
        evals = 0

        def propose(w):
            scaled = (theta[w] - lo) / extent
            z = rng.standard_normal(n_dims)
            gn = np.linalg.norm(grad[w])
            if precondition and n_done[w] >= args.burn_in and gn > 1e-8:
                # Anisotropic step: small along the lnL gradient (sized so a
                # step changes lnL by ~sigma, keeping the walker in its
                # umbrella), large in the tangent plane (free along the
                # iso-lnL surface). Rank-1 preconditioner, updated each step.
                n = grad[w] / gn
                z_par = float(z @ n)
                z_perp = z - z_par * n
                s_par = float(np.clip(sigma / gn, 1e-3, 0.3))
                delta = step[w] * z_perp + s_par * z_par * n
            else:
                delta = step[w] * z
            return lo + np.clip(scaled + delta, 0.0, 1.0) * extent

        pending = []
        while True:
            while free and ready and evals < eval_budget:
                w = ready.pop()
                if np.isnan(lnl_cur[w]):
                    params = theta[w].copy()         # seed eval at the anchor
                else:
                    params = propose(w)
                    proposal[w] = params
                wr = free.pop()
                pending.append(comm.isend(
                    {'params': params, 'context': {'walker': w}}, dest=wr))
                busy[wr] = w
                evals += 1

            if not busy:
                break

            result = comm.recv(source=MPI.ANY_SOURCE)
            wr = result['context']['worker_rank']
            w = result['context']['walker']
            free.append(wr)
            del busy[wr]
            lnl = result['target_val']
            params = np.asarray(result['params'], dtype=float)

            if np.isnan(lnl_cur[w]):
                lnl_cur[w] = lnl
                theta[w] = params
                records.append((params, lnl, 1, levels[w]))
            else:
                # Broyden secant update of the local lnL gradient from this
                # proposal (zero extra evals; uses every proposed point).
                if precondition and np.isfinite(lnl):
                    dscaled = (params - theta[w]) / extent
                    d2 = float(dscaled @ dscaled)
                    if d2 > 1e-18:
                        grad[w] = grad[w] + (
                            (lnl - lnl_cur[w]) - grad[w] @ dscaled) / d2 * dscaled
                d = (umbrella_logpi(lnl, levels[w], sigma)
                     - umbrella_logpi(lnl_cur[w], levels[w], sigma))
                acc = np.log(rng.uniform()) < d
                records.append((params, lnl, int(acc), levels[w]))
                if acc:
                    theta[w] = params
                    lnl_cur[w] = lnl
                step[w] = float(np.clip(
                    step[w] * np.exp(0.1 * ((1.0 if acc else 0.0) - 0.3)),
                    1e-3, 0.5))
                n_done[w] += 1

            if n_done[w] < args.steps and evals < eval_budget:
                ready.append(w)
        if pending:
            MPI.Request.Waitall(pending)
        t_walk = time.time()

        # --- Write outputs ---
        data = np.array([list(p) + [lnl, acc, lev]
                         for p, lnl, acc, lev in records])
        np.savetxt(f"umbrella_{func_name}{suffix}.csv", data, delimiter=',')

        lnls = data[:, n_dims]
        in_band = np.isfinite(lnls) & (lnls >= band_lo)
        summary = {
            'function': func_name, 'label': args.label,
            'precondition': precondition, 'burn_in': args.burn_in,
            'roi_threshold': roi_threshold,
            'roi_volume': roi_volume, 'sigma': sigma, 'grid': args.grid,
            'global_max': float(global_max),
            'n_projection_evals': int(n_projection_evals),
            'eval_budget': int(eval_budget), 'n_walkers': int(n_walkers),
            'steps_per_walker': args.steps,
            'n_umbrella_evals': int(len(data)),
            'n_in_band': int(in_band.sum()),
            'in_band_fraction': float(in_band.mean()),
            'mean_acceptance': float(data[:, n_dims + 1].mean()),
            'projection_seconds': round(t_proj - t0, 1),
            'walker_seconds': round(t_walk - t_proj, 1),
        }
        with open(f"umbrella_{func_name}{suffix}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("UMBRELLA_SUMMARY", json.dumps(summary), flush=True)

    terminate_workers(comm, myrank=rank)
else:
    worker_main(comm, rank)
