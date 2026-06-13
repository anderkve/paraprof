"""
Large-scale volume-sampling test driver.

Runs all six 2D projections of a 4D test function, then a volume-sampling
stage with the stage `roi_threshold` set to the projection `roi_threshold`
+ 2 (reaching into the shell).

By default the volume stage is given the same compute as the scan: the
total evaluation budget (`eval_budget`) equals the projection evaluation
count, and `n_anchors` (the stratification resolution, a separate knob) is
a third of that so the uniform probe layer uses ~1/3 of the allowance and
the anchored search + interior walk spend the rest.

The defaults can be overridden to explore other regimes (e.g. an
unlimited budget with a generous interior walk, to push the realized
depth distribution toward uniform-in-lnL):

    mpiexec -n <ncores> python run_volume_scale_test.py <func> <roi> \\
        [--grid G] [--label NAME] [--n-anchors N] \\
        [--eval-budget B]  (B<=0 means unlimited) \\
        [--interior-steps S]

e.g.  mpiexec -n 4 python run_volume_scale_test.py himmelblau_4d 4.0
      mpiexec -n 4 python run_volume_scale_test.py himmelblau_4d 4.0 \\
          --label tuned --n-anchors 5000 --eval-budget 0 --interior-steps 24

Outputs carry the optional `--label` suffix so several configurations can
coexist: samples_<func>[_label].csv (the phase-tagged total log),
volume_<func>[_label].csv (the curated representatives), and
scale_summary_<func>[_label].json.
"""
import argparse
import json
import time

import numpy as np
from mpi4py import MPI

from paraprof import (
    ProfileProjector, get_test_function, run_all_projections,
    run_volume_sampling, set_log_level, terminate_workers, worker_main,
)
from paraprof.volume import normalize_volume_config

set_log_level('INFO')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('func')
    p.add_argument('roi', type=float)
    p.add_argument('--grid', type=int, default=20)
    p.add_argument('--label', default='')
    p.add_argument('--n-anchors', type=int, default=None,
                   help='default: projection_evals // 3')
    p.add_argument('--eval-budget', type=int, default=None,
                   help='default: projection_evals; <=0 means unlimited')
    p.add_argument('--interior-steps', type=int, default=None,
                   help='default: the volume config default (8)')
    return p.parse_args()


args = parse_args()
func_name = args.func
roi_threshold = args.roi
grid = args.grid
suffix = f"_{args.label}" if args.label else ""

np.random.seed(20260613)

log_likelihood, bounds, _ = get_test_function(func_name)

# All six 2D projections of the 4D space.
PROJECTIONS = [
    {'dims': [i, j], 'grid_points': [grid, grid]}
    for i in range(4) for j in range(i + 1, 4)
]

samples_file = f"samples_{func_name}{suffix}.csv"
volume_file = f"volume_{func_name}{suffix}.csv"

if rank == 0:
    comm.bcast((log_likelihood, None), root=0)
    t0 = time.time()

    with ProfileProjector(
        target_func=log_likelihood,
        bounds=bounds,
        projections=PROJECTIONS,
        roi_threshold=roi_threshold,
        pop_per_grid_point=3,
        n_initial_optimizations=60,
        samples_output_file=samples_file,
    ) as sampler:

        results = run_all_projections(
            comm=comm, sampler=sampler, projections=PROJECTIONS,
            save_plots=False, myrank=rank,
        )
        n_projection_evals = sampler.target_calls
        t_proj = time.time()

        # Resolve the volume-stage knobs (defaults give the scan-matched
        # budget; CLI flags override to explore other regimes).
        if args.eval_budget is None:
            eval_budget = n_projection_evals
        else:
            eval_budget = None if args.eval_budget <= 0 else args.eval_budget
        n_anchors = (args.n_anchors if args.n_anchors is not None
                     else max(n_projection_evals // 3, 1))

        cfg = {
            'roi_threshold': roi_threshold + 2.0,
            'n_anchors': n_anchors,
            'eval_budget': eval_budget,
            'output_file': volume_file,
        }
        if args.interior_steps is not None:
            cfg['interior_steps'] = args.interior_steps
        sampler.volume_sampling_config = normalize_volume_config(
            cfg, roi_threshold=roi_threshold)

        vol = run_volume_sampling(comm, sampler, results, myrank=rank)
        sampler._flush_samples_buffer()
        t_vol = time.time()

        total_evals = sampler.target_calls
        summary = {
            'function': func_name,
            'label': args.label,
            'projection_roi_threshold': roi_threshold,
            'volume_roi_threshold': roi_threshold + 2.0,
            'grid': grid,
            'n_projection_evals': int(n_projection_evals),
            'volume_eval_budget': (None if eval_budget is None
                                   else int(eval_budget)),
            'volume_n_anchors': int(n_anchors),
            'interior_steps': sampler.volume_sampling_config['interior_steps'],
            'n_volume_evals': int(total_evals - n_projection_evals),
            'total_evals': int(total_evals),
            'global_max': float(sampler.global_max_target_val),
            'projection_seconds': round(t_proj - t0, 1),
            'volume_seconds': round(t_vol - t_proj, 1),
        }
        if not vol.get('skipped'):
            stats = vol['stats']
            summary['volume_stats'] = {
                'n_anchors': int(stats['n_anchors']),
                'evals_used': int(stats['evals_used']),
                'n_covered': int(stats['n_covered']),
                'n_covered_probe': int(stats['n_covered_probe']),
                'n_covered_search': int(stats['n_covered_search']),
                'n_projected': int(stats['n_projected']),
                'n_holes': int(stats['n_holes']),
                'n_unbudgeted': int(stats['n_unbudgeted']),
                'prefilter_acceptance': stats['prefilter_acceptance'],
                'probe_acceptance': stats['probe_acceptance'],
                'volume_estimate': stats['volume_estimate'],
            }
        else:
            summary['volume_skipped_reason'] = vol.get('reason')

        with open(f"scale_summary_{func_name}{suffix}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print("SCALE_SUMMARY", json.dumps(summary), flush=True)

    terminate_workers(comm, myrank=rank)
else:
    worker_main(comm, rank)
