"""
Benchmark for the surrogate-gradient feature (lbfgsb.surrogate_gradient) on a
high-nuisance target, where finite-difference gradients are the dominant cost.

Target: Himmelblau-4D base + N Gaussian nuisances (shift coupling), one 2D POI
projection. Projecting on the 2 POIs profiles out (2 POIs + N nuisances) dims,
so each forward-FD gradient costs that many evaluations -- the regime the
surrogate targets. optimization_method='lbfgsb' so the L-BFGS-B path (and its
gradients) is the workhorse.

The driver runs both modes (FD only / surrogate on) over several seeds and
reports: FD evals the surrogate avoided, net target calls, and ROI grid quality
(deficit vs the elementwise-max reference) so a call win is only counted if
quality holds.

Inner:  mpiexec -n <ncores> python examples/run_surrogate_gradient_benchmark.py \\
            --nuis 8 --mode surrogate --seed 750123 --out /tmp/s.json
Driver: python examples/run_surrogate_gradient_benchmark.py --drive --nuis 8 --ncores 4 --reps 4
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from run_allow_early_de_exit_replicate_study import _reference_grids, _run_quality, _fmt


GRID = 25
ROI_THRESHOLD = 4.0


def _inner(args):
    from mpi4py import MPI
    from paraprof import (
        ProfileProjector, run_scan, worker_main, get_test_function, set_log_level,
    )
    from paraprof.nuisance_wrapper import create_nuisance_wrapped_function

    set_log_level('WARNING')
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    np.random.seed(args.seed)

    base_func, base_bounds, _ = get_test_function('himmelblau_4d')
    func, bounds, _ = create_nuisance_wrapped_function(
        base_func, base_bounds, n_poi=4, n_nuisance=args.nuis,
        coupling_mode='shift', constraint_sigma=0.5,
    )
    bounds = np.asarray(bounds, dtype=float)
    projections = [{'dims': [0, 1], 'grid_points': [GRID, GRID],
                    'optimization_method': 'lbfgsb'}]

    if myrank != 0:
        worker_main(comm, myrank)
        return

    with ProfileProjector(
        target_func=func, bounds=bounds, projections=projections,
        roi_threshold=ROI_THRESHOLD, n_initial_optimizations=60, lbfgsb_max_iter=40,
        advanced_config={'lbfgsb': {'surrogate_gradient': args.mode == 'surrogate'}},
    ) as sampler:
        results = run_scan(comm=comm, sampler=sampler, projections=projections,
                           save_plots=False, myrank=0)
        saved = sampler.target_calls_saved_by_surrogate_gradient
        total = sampler.target_calls
        gmax = sampler.global_max_target_val

    grid_pts = [GRID + 1, GRID + 1]
    grid = np.full(grid_pts, np.nan)
    for idx, sol in results[0]['coarse_solution'].get('solutions', {}).items():
        grid[tuple(idx)] = sol['likelihood']
    with open(args.out, 'w') as f:
        json.dump({'mode': args.mode, 'seed': args.seed, 'total_calls': int(total),
                   'saved_fd': int(saved), 'global_max': float(gmax),
                   'projections': [{'coarse_grid_values': grid.tolist()}]}, f)
    print(f"[nuis{args.nuis}/{args.mode}/seed{args.seed}] calls={total} saved_fd={saved} gmax={gmax:.4f}")


def _drive(args):
    seeds = [750123 + 1000 * k for k in range(args.reps)]
    runs = {'baseline': [], 'surrogate': []}
    with tempfile.TemporaryDirectory() as td:
        for mode in ('baseline', 'surrogate'):
            for seed in seeds:
                out = os.path.join(td, f'{mode}_{seed}.json')
                env = dict(os.environ)
                env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
                env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
                subprocess.run(
                    ['mpiexec', '--oversubscribe', '-n', str(args.ncores), sys.executable,
                     os.path.abspath(__file__), '--nuis', str(args.nuis), '--mode', mode,
                     '--seed', str(seed), '--out', out],
                    check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(out) as f:
                    runs[mode].append(json.load(f))

    refs = _reference_grids(runs['baseline'] + runs['surrogate'])
    defs = {m: [_run_quality(r, refs, ROI_THRESHOLD)[0] for r in runs[m]] for m in runs}
    calls = {m: [r['total_calls'] for r in runs[m]] for m in runs}
    saved = [r['saved_fd'] for r in runs['surrogate']]

    bm, sm = np.mean(calls['baseline']), np.mean(calls['surrogate'])
    print("=" * 78)
    print(f"Surrogate-gradient benchmark: Himmelblau-4D + {args.nuis} nuisances "
          f"({2 + args.nuis} profiled dims), {GRID}x{GRID}, reps={args.reps}")
    print("=" * 78)
    print(f"  FD evals avoided by surrogate (per scan): {_fmt(saved)}")
    print(f"  target calls: baseline {_fmt(calls['baseline'])}  surrogate {_fmt(calls['surrogate'])}"
          f"  ({100*(sm-bm)/bm:+.1f}%)")
    print(f"  ROI mean deficit: baseline {_fmt(defs['baseline'])}  surrogate {_fmt(defs['surrogate'])}")
    print(f"  global_max: baseline {np.mean([r['global_max'] for r in runs['baseline']]):.4f}"
          f"  surrogate {np.mean([r['global_max'] for r in runs['surrogate']]):.4f}")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drive', action='store_true')
    p.add_argument('--nuis', type=int, default=8)
    p.add_argument('--mode', choices=['baseline', 'surrogate'])
    p.add_argument('--seed', type=int, default=750123)
    p.add_argument('--out')
    p.add_argument('--ncores', type=int, default=4)
    p.add_argument('--reps', type=int, default=4)
    args = p.parse_args()
    _drive(args) if args.drive else _inner(args)


if __name__ == '__main__':
    main()
