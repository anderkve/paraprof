"""
Volume-sampling benchmark: the anchored-search funnel vs. cheaper baselines.

Target: the 4D Rosenbrock log-likelihood — its ROI is a thin curved valley,
the geometry where plain rejection sampling wastes the most evaluations and
the tier-3 anchored search earns its keep.

After one profile scan (2D projection on dims [0, 1]), the volume stage runs
twice on the same scan knowledge:

1. ``probe-only`` (``search='none'``): harvest + uniform probes inside the
   projection envelope. This doubles as the *envelope-rejection* baseline.
2. ``full funnel``: harvest + probes + anchored L-BFGS-B searches.

Reported per variant:

- in-band points obtained and stage evaluations spent (evals per point);
- coverage: the fraction of a brute-force reference ROI sample that lies
  within the coverage radius (bounds-scaled) of an output in-band point;
- for context, the cost of *naive* box rejection (1 / true acceptance,
  measured by vectorized brute force — no MPI evaluations spent).

Run with MPI:

    mpiexec -n <ncores> python -m mpi4py volume_sampling_benchmark.py [n_anchors]

Required: at least 2 MPI ranks (1 master + 1+ workers). The ``-m mpi4py``
launcher makes an uncaught exception on any rank abort the whole job
instead of deadlocking the surviving ranks in MPI_Finalize.
"""
import json
import os
import sys
import tempfile

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from paraprof import (
    ProfileProjector, run_all_projections, run_volume_sampling,
    terminate_workers, worker_main, get_test_function, set_log_level,
)
from paraprof.volume import normalize_volume_config

set_log_level('WARNING')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

N_ANCHORS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROI_THRESHOLD = 4.0
N_REFERENCE = 2000

log_likelihood, bounds, _ = get_test_function('rosenbrock_4d')
bounds = np.asarray(bounds, dtype=float)  # get_test_function returns a list

PROJECTIONS = [
    {'dims': [0, 1], 'grid_points': [40, 40]},
]


def rosenbrock_logl_vec(x):
    """Vectorized rosenbrock_nd for the brute-force reference set."""
    return -0.1 * np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0
        + (1 - x[:, :-1]) ** 2.0,
        axis=1,
    )


def brute_force_reference(band_lo, rng, n_target):
    """Uniform in-ROI reference points by vectorized box rejection (offline,
    no worker evaluations). Returns (points, true box acceptance)."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    points = []
    n_drawn = 0
    n_kept = 0
    while n_kept < n_target:
        draws = rng.uniform(lo, hi, size=(2_000_000, len(bounds)))
        logls = rosenbrock_logl_vec(draws)
        keep = logls >= band_lo
        points.append(draws[keep])
        n_drawn += len(draws)
        n_kept += int(np.count_nonzero(keep))
    return np.vstack(points)[:n_target], n_kept / n_drawn


def coverage_metrics(reference, in_band_points, radius):
    """(fraction of reference points within `radius` of an output point,
    mean reference-to-nearest-output distance), both bounds-scaled. The
    mean distance keeps discriminating after the fraction saturates."""
    if len(in_band_points) == 0:
        return 0.0, float('inf')
    lo, extent = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    tree = cKDTree((in_band_points - lo) / extent)
    dists, _ = tree.query((reference - lo) / extent)
    return float(np.mean(dists <= radius)), float(np.mean(dists))


def summarize(label, vol, evals, reference):
    rows_in_band = vol['rep_points'][
        np.isin(vol['anchor_status'], ['covered', 'projected'])]
    stats = vol['stats']
    cov, mean_dist = coverage_metrics(reference, rows_in_band,
                                      stats['coverage_radius'])
    n_points = len(rows_in_band)
    return {
        'variant': label,
        'in_band_points': n_points,
        'stage_evals': evals,
        'evals_per_point': evals / n_points if n_points else float('inf'),
        'coverage': cov,
        'mean_ref_dist': mean_dist,
        'n_covered': stats['n_covered'],
        'n_projected': stats['n_projected'],
        'n_holes': stats['n_holes'],
        'probe_acceptance': stats['probe_acceptance'],
    }


def master(workdir):
    with ProfileProjector(
        target_func=log_likelihood,
        bounds=bounds,
        projections=PROJECTIONS,
        roi_threshold=ROI_THRESHOLD,
        samples_output_file=os.path.join(workdir, 'samples.csv'),
        # Placeholder; each variant below installs its own config before
        # calling run_volume_sampling directly.
        volume_sampling=None,
    ) as sampler:
        comm.bcast((sampler.target_func, sampler.grad_func), root=0)
        results = run_all_projections(
            comm=comm, sampler=sampler, projections=PROJECTIONS,
            save_plots=False, myrank=myrank,
        )
        scan_evals = sampler.target_calls

        print(f"\nScan done: {scan_evals} evaluations. "
              f"Building the brute-force reference set...", flush=True)
        rng = np.random.default_rng(20260610)
        band_lo = sampler.global_max_target_val - ROI_THRESHOLD
        reference, box_acceptance = brute_force_reference(
            band_lo, rng, N_REFERENCE)

        rows = []
        for label, overrides in [
            ('probe-only', {'search': 'none'}),
            ('full funnel', {}),
        ]:
            cfg = {'n_anchors': N_ANCHORS,
                   'output_file': os.path.join(workdir, f'{label}.csv')}
            cfg.update(overrides)
            sampler.volume_sampling_config = normalize_volume_config(
                cfg, ROI_THRESHOLD)
            evals_before = sampler.target_calls
            vol = run_volume_sampling(comm, sampler, results, myrank=myrank)
            rows.append(summarize(label, vol,
                                  sampler.target_calls - evals_before,
                                  reference))

        print(f"\n=== Volume-sampling benchmark: rosenbrock_4d, "
              f"n_anchors={N_ANCHORS} ===")
        print(f"True box acceptance (brute force): {box_acceptance:.3e} "
              f"-> naive rejection needs ~{1.0 / box_acceptance:.0f} "
              f"evals per in-band point")
        header = (f"{'variant':<14}{'in-band':>9}{'evals':>8}"
                  f"{'evals/pt':>10}{'coverage':>10}{'meandist':>10}"
                  f"{'holes':>7}")
        print(header)
        print('-' * len(header))
        for r in rows:
            print(f"{r['variant']:<14}{r['in_band_points']:>9}"
                  f"{r['stage_evals']:>8}{r['evals_per_point']:>10.1f}"
                  f"{r['coverage']:>10.3f}{r['mean_ref_dist']:>10.4f}"
                  f"{r['n_holes']:>7}")
        print(f"\nDetails: {json.dumps(rows, indent=2)}")
        print(f"Output files in {workdir}", flush=True)


if myrank == 0:
    try:
        master(tempfile.mkdtemp(prefix='volume_benchmark_'))
    finally:
        # Always reached, even on an exception: without it the workers
        # would wait forever and the job would deadlock in MPI_Finalize.
        terminate_workers(comm, myrank)
else:
    worker_main(comm, myrank)
