"""
Per-projection (compounding) measurement for the cross-projection smooth-certify
trigger, on all six 2D projections of Himmelblau-4D.

Replaces ``run_scan`` with a manual projection loop so it can snapshot the
cumulative target_calls and cumulative pool-only-certified count *after each
projection*. The driver subprocesses both arms (neighbour-only / both triggers)
over several seeds and prints a per-projection table: if the second trigger
compounds, its activity (pool-only cells) and the marginal call saving should
grow as the cross-projection pool fills over projections 1..6.

Inner run:
    mpiexec -n <ncores> python examples/run_pool_trigger_compound.py \\
        --target himmelblau_4d --mode both --seed 750123 --out /tmp/c.json
Driver (no mpiexec):
    python examples/run_pool_trigger_compound.py --drive \\
        --target himmelblau_4d --ncores 4 --reps 5
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def _inner(args):
    from mpi4py import MPI
    from paraprof import (
        ProfileProjector, run_projection, terminate_workers, worker_main,
        get_test_function, set_log_level,
    )
    from run_smooth_certify_benchmark import TARGET_KWARGS, _make_projections

    set_log_level('WARNING')
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    np.random.seed(args.seed)

    log_likelihood, bounds, _ = get_test_function(args.target)
    cfg = TARGET_KWARGS[args.target]
    projections = _make_projections(cfg['dims'])

    if myrank != 0:
        worker_main(comm, myrank)
        return

    advanced_config = dict(cfg['advanced_config'] or {})
    advanced_config['de'] = {'smooth_certify': True}
    with ProfileProjector(
        target_func=log_likelihood, bounds=bounds, projections=projections,
        advanced_config=advanced_config, **cfg['kwargs'],
    ) as sampler:
        sampler.de_pool_certify_trigger = (args.mode == 'both')
        comm.bcast((sampler.target_func, sampler.grad_func), root=0)

        per_proj = []
        for proj_idx, pc in enumerate(projections):
            if proj_idx > 0:
                sampler._reset_for_new_projection(pc)
            run_projection(comm=comm, sampler=sampler, projection_config=pc,
                           skip_init_opt_on_warm_start=(proj_idx > 0), myrank=0)
            per_proj.append({
                'cum_calls': int(sampler.target_calls),
                'cum_pool_only': int(sampler.de_cells_certified_pool_only),
                'cum_certified': int(sampler.de_cells_smooth_certified),
            })
        terminate_workers(comm, myrank=0)

    with open(args.out, 'w') as f:
        json.dump({'mode': args.mode, 'seed': args.seed, 'per_proj': per_proj}, f)
    print(f"[{args.target}/{args.mode}/seed{args.seed}] "
          f"calls={per_proj[-1]['cum_calls']} pool_only={per_proj[-1]['cum_pool_only']}")


def _deltas(per_proj, key):
    """Per-projection (non-cumulative) values from a cumulative series."""
    prev = 0
    out = []
    for p in per_proj:
        out.append(p[key] - prev)
        prev = p[key]
    return out


def _drive(args):
    seeds = [750123 + 1000 * k for k in range(args.reps)]
    runs = {'neighbour': [], 'both': []}
    with tempfile.TemporaryDirectory() as td:
        for mode in ('neighbour', 'both'):
            for seed in seeds:
                out = os.path.join(td, f'{mode}_{seed}.json')
                env = dict(os.environ)
                env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
                env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
                subprocess.run(
                    ['mpiexec', '--oversubscribe', '-n', str(args.ncores),
                     sys.executable, os.path.abspath(__file__),
                     '--target', args.target, '--mode', mode,
                     '--seed', str(seed), '--out', out],
                    check=True, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out) as f:
                    runs[mode].append(json.load(f))

    n_proj = len(runs['both'][0]['per_proj'])
    # Per-projection call deltas, averaged over seeds.
    nb_calls = np.array([_deltas(r['per_proj'], 'cum_calls') for r in runs['neighbour']])
    bo_calls = np.array([_deltas(r['per_proj'], 'cum_calls') for r in runs['both']])
    pool_only = np.array([_deltas(r['per_proj'], 'cum_pool_only') for r in runs['both']])

    print("=" * 78)
    print(f"Compounding test: {args.target}, {n_proj} 2D projections, reps={args.reps}")
    print("smooth_certify on in both arms; 'both' adds the cross-projection trigger")
    print("=" * 78)
    print(f"{'proj':>4} {'neigh calls':>13} {'both calls':>12} {'saving':>10} "
          f"{'saving%':>8} {'pool-only cells':>16}")
    for p in range(n_proj):
        nb = nb_calls[:, p].mean()
        bo = bo_calls[:, p].mean()
        save = nb - bo
        pct = 100 * save / nb if nb else 0
        print(f"{p+1:>4} {nb:>13,.0f} {bo:>12,.0f} {save:>10,.0f} "
              f"{pct:>7.1f}% {pool_only[:, p].mean():>16.1f}")

    nb_tot = nb_calls.sum(axis=1).mean()
    bo_tot = bo_calls.sum(axis=1).mean()
    print("-" * 78)
    print(f"{'TOT':>4} {nb_tot:>13,.0f} {bo_tot:>12,.0f} {nb_tot-bo_tot:>10,.0f} "
          f"{100*(nb_tot-bo_tot)/nb_tot:>7.1f}% {pool_only.sum(axis=1).mean():>16.1f}")
    print("\nCompounding => the saving% and pool-only column should trend UP with proj.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive', action='store_true')
    parser.add_argument('--target', default='himmelblau_4d')
    parser.add_argument('--mode', choices=['neighbour', 'both'])
    parser.add_argument('--seed', type=int, default=750123)
    parser.add_argument('--out')
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--reps', type=int, default=5)
    args = parser.parse_args()
    if args.drive:
        _drive(args)
    else:
        _inner(args)


if __name__ == '__main__':
    main()
