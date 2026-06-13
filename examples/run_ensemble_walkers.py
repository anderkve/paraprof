"""
Prototype: affine-invariant ensemble umbrella sampler (third contender).

Like run_umbrella_walkers.py the target is a log-Gaussian in lnL, but the
walkers are organised into G level shells (each an ensemble of K walkers
sharing one home level l_g), and each walker is updated with the
Goodman-Weare affine-invariant stretch move using a partner from the same
shell:

    theta' = theta_j + z (theta_k - theta_j),   z ~ g(z) prop 1/sqrt(z) on [1/a, a]
    accept with min(1, z^(d-1) pi_g(theta') / pi_g(theta_k))

Affine invariance handles stretched/curved geometries (Rosenbrock valleys)
with no step size, gradient, or covariance tuning. Shells are updated
red/black so each sub-sweep is a batch of independent evaluations for the
worker pool. Shells are warm-started from scan-log points near their level
(projection info -> no climb burn-in).

    mpiexec -n <ncores> python run_ensemble_walkers.py <func> <roi> \\
        [--grid G] [--n-levels L] [--walkers-per-level K] \\
        [--eval-budget B] [--sigma-frac F] [--label NAME]

Outputs ensemble_<func>[_label].csv ([params..., lnL, accepted, level]) and
ensemble_<func>[_label]_summary.json.
"""
import argparse
import collections
import json
import time

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, get_test_function, read_samples, run_all_projections,
    set_log_level, terminate_workers, worker_main,
)
from paraprof.volume import ProjectionEnvelope, generate_anchors

set_log_level('WARNING')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
A_STRETCH = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('func')
    p.add_argument('roi', type=float)
    p.add_argument('--grid', type=int, default=20)
    p.add_argument('--n-levels', type=int, default=20)
    p.add_argument('--walkers-per-level', type=int, default=24)
    p.add_argument('--eval-budget', type=int, default=None)
    p.add_argument('--sigma-frac', type=float, default=None,
                   help='umbrella sigma / roi; default = one shell spacing')
    p.add_argument('--partner-level-window', type=float, default=0.0,
                   help='lnL window for a stretch partner home level; '
                        '0 = strict shells, <0 = full pool')
    p.add_argument('--label', default='')
    return p.parse_args()


args = parse_args()
func_name = args.func
roi_threshold = args.roi
roi_volume = roi_threshold + 2.0
suffix = f"_{args.label}" if args.label else ""
np.random.seed(20260613)

log_likelihood, bounds, _ = get_test_function(func_name)
bounds = np.asarray(bounds, dtype=float)
n_dims = len(bounds)
lo, extent = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

PROJECTIONS = [{'dims': [i, j], 'grid_points': [args.grid, args.grid]}
               for i in range(n_dims) for j in range(i + 1, n_dims)]


def umbrella_logpi(lnl, level, sigma):
    return -np.inf if not np.isfinite(lnl) else -0.5 * ((lnl - level) / sigma) ** 2


def draw_z(rng, n):
    u = rng.uniform(size=n)
    return ((A_STRETCH - 1.0) * u + 1.0) ** 2 / A_STRETCH


if rank == 0:
    comm.bcast((log_likelihood, None), root=0)
    t0 = time.time()
    scan_file = f"samples_{func_name}_ens.csv"

    with ProfileProjector(
        target_func=log_likelihood, bounds=bounds, projections=PROJECTIONS,
        roi_threshold=roi_threshold, pop_per_grid_point=3,
        n_initial_optimizations=60, samples_output_file=scan_file,
    ) as sampler:
        results = run_all_projections(comm=comm, sampler=sampler,
                                      projections=PROJECTIONS,
                                      save_plots=False, myrank=rank)
        n_projection_evals = sampler.target_calls
        global_max = sampler.global_max_target_val
        sampler._flush_samples_buffer()
        t_proj = time.time()

        G = args.n_levels
        K = args.walkers_per_level + (args.walkers_per_level % 2)  # even
        N = G * K
        eval_budget = args.eval_budget or n_projection_evals
        sigma = (args.sigma_frac * roi_volume if args.sigma_frac
                 else roi_volume / G)
        band_lo = global_max - roi_volume
        # Shell levels: centres uniform in DeltalnL across the band.
        shell_dlnl = (np.arange(G) + 0.5) / G * roi_volume
        shell_level = global_max - shell_dlnl

        envelope = ProjectionEnvelope.from_projection_results(
            results, global_max, n_dims)
        rng = np.random.default_rng(7)
        fallback = generate_anchors(envelope, bounds, N, roi_volume,
                                    seed=1).anchors

        # Warm-start each shell from scan-log points near its level.
        scan = read_samples(scan_file)
        s_lnl = scan[:, n_dims]
        s_ok = np.isfinite(s_lnl)
        s_p, s_lnl = scan[s_ok, :n_dims], s_lnl[s_ok]

        theta = np.empty((N, n_dims))
        level = np.empty(N)
        fb = 0
        for g in range(G):
            cand = np.flatnonzero(np.abs(s_lnl - shell_level[g]) <= sigma)
            if len(cand) >= K:
                pick = s_p[rng.choice(cand, K, replace=False)]
            elif len(cand) > 0:
                pick = s_p[rng.choice(cand, K, replace=True)]
            else:
                pick = fallback[fb:fb + K]
                fb += K
            pick = pick + 0.01 * extent * rng.standard_normal((K, n_dims))
            theta[g * K:(g + 1) * K] = pick
            level[g * K:(g + 1) * K] = shell_level[g]

        # Global red/black parity; a stretch partner is drawn from the frozen
        # half within a home-level window of the walker (w shells either side;
        # w=0 -> strict shells, w=G -> the whole pool). Home level is fixed,
        # so windowing partner choice on it preserves detailed balance.
        shell_idx = np.repeat(np.arange(G), K)
        parity = np.arange(N) % 2
        spacing = roi_volume / G
        w = G if args.partner_level_window < 0 else int(
            round(args.partner_level_window / spacing))
        idx_by = [[np.flatnonzero((shell_idx == g) & (parity == p))
                   for p in (0, 1)] for g in range(G)]
        partner_pool = [[np.concatenate([idx_by[g2][p]
                         for g2 in range(max(0, g - w), min(G, g + w + 1))])
                         for p in (0, 1)] for g in range(G)]
        active_by_parity = [np.flatnonzero(parity == p) for p in (0, 1)]

        workers = [r for r in range(comm.Get_size()) if r != rank]
        records = []

        def eval_batch(items):
            """items: list of (idx, params) -> {idx: (lnL, params)}; logs all."""
            out = {}
            free = list(workers)
            queue = collections.deque(items)
            pending, recvd = [], 0
            while recvd < len(items):
                while free and queue:
                    idx, params = queue.popleft()
                    wr = free.pop()
                    pending.append(comm.isend(
                        {'params': params, 'context': {'idx': idx}}, dest=wr))
                r = comm.recv(source=MPI.ANY_SOURCE)
                free.append(r['context']['worker_rank'])
                out[r['context']['idx']] = (r['target_val'],
                                            np.asarray(r['params'], float))
                recvd += 1
            if pending:
                MPI.Request.Waitall(pending)
            return out

        # Seed evaluation for every walker.
        seed = eval_batch([(i, theta[i].copy()) for i in range(N)])
        lnl_cur = np.empty(N)
        for i, (val, p) in seed.items():
            lnl_cur[i] = val
            records.append((p, val, 1, level[i]))
        evals = N

        n_sweeps = max(eval_budget // N - 1, 1)
        for _ in range(n_sweeps):
            if evals >= eval_budget:
                break
            for p_a in (0, 1):
                p_f = 1 - p_a
                # One stretch proposal per active walker.
                items, meta = [], {}
                for k in active_by_parity[p_a]:
                    k = int(k)
                    pool = partner_pool[shell_idx[k]][p_f]
                    j = int(pool[rng.integers(len(pool))])
                    z = float(draw_z(rng, 1)[0])
                    prop = theta[j] + z * (theta[k] - theta[j])
                    meta[k] = z
                    items.append((k, prop))
                res = eval_batch(items)
                evals += len(items)
                for k, (lnl, p) in res.items():
                    z = meta[k]
                    records.append((p, lnl, 0, level[k]))
                    log_acc = ((n_dims - 1) * np.log(z)
                               + umbrella_logpi(lnl, level[k], sigma)
                               - umbrella_logpi(lnl_cur[k], level[k], sigma))
                    if np.log(rng.uniform()) < log_acc:
                        theta[k] = p
                        lnl_cur[k] = lnl
                        records[-1] = (p, lnl, 1, level[k])
        t_walk = time.time()

        data = np.array([list(p) + [lnl, acc, lev]
                         for p, lnl, acc, lev in records])
        np.savetxt(f"ensemble_{func_name}{suffix}.csv", data, delimiter=',')
        ib = np.isfinite(data[:, n_dims]) & (data[:, n_dims] >= band_lo)
        summary = {
            'function': func_name, 'label': args.label,
            'roi_threshold': roi_threshold, 'roi_volume': roi_volume,
            'sigma': sigma, 'grid': args.grid, 'global_max': float(global_max),
            'n_levels': G, 'walkers_per_level': K, 'n_walkers': N,
            'partner_level_window': args.partner_level_window,
            'partner_window_shells': int(w),
            'n_sweeps': int(n_sweeps),
            'n_projection_evals': int(n_projection_evals),
            'eval_budget': int(eval_budget),
            'n_ensemble_evals': int(len(data)),
            'n_in_band': int(ib.sum()),
            'in_band_fraction': float(ib.mean()),
            'mean_acceptance': float(data[:, n_dims + 1].mean()),
            'projection_seconds': round(t_proj - t0, 1),
            'walker_seconds': round(t_walk - t_proj, 1),
        }
        with open(f"ensemble_{func_name}{suffix}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("ENSEMBLE_SUMMARY", json.dumps(summary), flush=True)

    terminate_workers(comm, myrank=rank)
else:
    worker_main(comm, rank)
