"""
Benchmark: volume-stage samples vs. the samples the profiling stage collects.

The motivation for the volume-sampling feature is that a profile scan's
samples concentrate on the *profile surfaces* — for a projection on
(x0, x1), the profiled dims sit at their conditional optima — so they are
not representative of the full good-fit volume. This benchmark makes that
quantitative and visual on one scan:

1. Run a profile scan (one 2D projection) and keep its in-band samples.
2. Run the ROI volume-sampling stage on the same scan knowledge.
3. Compare three same-budget sets against a brute-force uniform in-ROI
   reference: ALL in-band scan samples, the best-spread (farthest-point)
   subset of them at the volume set's size, and the volume set itself.

Reported per set: coverage of the reference within the stage's coverage
radius, mean reference-to-nearest-sample distance (bounds-scaled), and the
per-dimension spread relative to the reference. Two figures are saved:
the sample clouds in the projected (x0, x1) and profiled (x2, x3) planes,
and the logL distributions.

Run with MPI:

    mpiexec -n <ncores> python -m mpi4py volume_vs_profile_benchmark.py \\
        [himmelblau_4d|rosenbrock_4d|sphere_4d|sphere_6d|sphere_8d] \\
        [n_points] [interior_steps]

The sphere targets (logL = -|x|^2 on [-5, 5]^n, ROI = a ball) are the
clean dimension-scaling cases: as n grows, ever fewer of the scan's
by-product evaluations land in the band away from the profile surface,
so recycled samples thin out while the volume stage keeps its per-anchor
guarantee.

Required: at least 2 MPI ranks. The ``-m mpi4py`` launcher makes an
uncaught exception on any rank abort the whole job instead of deadlocking
the surviving ranks in MPI_Finalize.
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
    terminate_workers, worker_main, get_test_function, read_samples,
    set_log_level,
)
from paraprof.volume import normalize_volume_config

set_log_level('WARNING')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

FUNC_NAME = sys.argv[1] if len(sys.argv) > 1 else 'himmelblau_4d'
N_POINTS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
INTERIOR_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ROI_THRESHOLD = 4.0
N_REFERENCE = 4000

def himmelblau_4d_vec(x):
    t1 = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    t2 = (x[:, 2] ** 2 + x[:, 3] - 11) ** 2 + (x[:, 2] + x[:, 3] ** 2 - 7) ** 2
    return -0.05 * (t1 + t2)


def rosenbrock_4d_vec(x):
    return -0.1 * np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0
        + (1 - x[:, :-1]) ** 2.0,
        axis=1,
    )


def sphere(p):
    return -float(np.sum(np.asarray(p) ** 2))


def sphere_vec(x):
    return -np.sum(x ** 2, axis=1)


def resolve_target(name):
    """(scalar target, bounds, vectorized target, projection config)."""
    if name.startswith('sphere_'):
        n_dims = int(name.split('_')[1].rstrip('d'))
        proj = {'dims': [0, 1], 'grid_points': [20, 20],
                'optimization_method': 'lbfgsb'}
        return sphere, np.array([[-5.0, 5.0]] * n_dims), sphere_vec, proj
    func, bnds, _ = get_test_function(name)
    vec = {'himmelblau_4d': himmelblau_4d_vec,
           'rosenbrock_4d': rosenbrock_4d_vec}[name]
    return func, np.asarray(bnds, dtype=float), vec, \
        {'dims': [0, 1], 'grid_points': [40, 40]}


log_likelihood, bounds, logl_vec, projection = resolve_target(FUNC_NAME)
LO, EXTENT = bounds[:, 0], bounds[:, 1] - bounds[:, 0]

PROJECTIONS = [projection]


def scale(points):
    return (points - LO) / EXTENT


def brute_force_reference(band_lo, rng, n_target):
    """Uniform in-ROI points by vectorized box rejection (offline)."""
    points, n_drawn, n_kept = [], 0, 0
    while n_kept < n_target:
        draws = rng.uniform(LO, bounds[:, 1], size=(2_000_000, len(bounds)))
        keep = logl_vec(draws) >= band_lo
        points.append(draws[keep])
        n_drawn += len(draws)
        n_kept += int(np.count_nonzero(keep))
    return np.vstack(points)[:n_target], n_kept / n_drawn


def farthest_point_subset(points, n):
    """Greedy maximin subset (bounds-scaled distances): the best-spread
    selection a user could recycle from existing samples."""
    pts = scale(points)
    if len(pts) <= n:
        return points
    # Deterministic start: the point closest to the cloud centroid.
    start = int(np.argmin(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))
    chosen = [start]
    min_d2 = np.sum((pts - pts[start]) ** 2, axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_d2))
        chosen.append(nxt)
        np.minimum(min_d2, np.sum((pts - pts[nxt]) ** 2, axis=1), out=min_d2)
    return points[chosen]


def set_metrics(label, points, reference, radius):
    tree = cKDTree(scale(points))
    dists, _ = tree.query(scale(reference))
    ref_std = np.std(scale(reference), axis=0)
    return {
        'set': label,
        'n': len(points),
        'coverage': float(np.mean(dists <= radius)),
        'mean_ref_dist': float(np.mean(dists)),
        # Spread per dim relative to the true in-ROI spread (1.0 = matches).
        'rel_spread': [round(float(s / r), 3) for s, r in
                       zip(np.std(scale(points), axis=0), ref_std)],
    }


def make_figures(scan_in_band, volume_pts, reference, gmax, band_lo,
                 scan_logls, volume_logls, out_prefix):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures.")
        return []

    paths = []

    # --- Sample clouds in the projected and the profiled planes ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [
        (0, 1, 'projected dims'),
        (2, 3, 'profiled dims'),
    ]
    for row, (di, dj, kind) in enumerate(panels):
        ax = axes[row][0]
        ax.hist2d(scan_in_band[:, di], scan_in_band[:, dj], bins=90,
                  range=[bounds[di], bounds[dj]], cmap='viridis',
                  norm=matplotlib.colors.LogNorm())
        ax.set_title(f"profiling-stage in-band samples, {kind} "
                     f"(n={len(scan_in_band)})")
        ax = axes[row][1]
        ax.scatter(reference[:, di], reference[:, dj], s=3, c='lightgray',
                   label=f'uniform ROI reference (n={len(reference)})')
        ax.scatter(volume_pts[:, di], volume_pts[:, dj], s=8, c='tab:orange',
                   edgecolors='black', linewidths=0.2,
                   label=f'volume samples (n={len(volume_pts)})')
        ax.set_title(f"volume-stage samples, {kind}")
        ax.legend(loc='upper right', fontsize=7)
        for ax in axes[row]:
            ax.set_xlim(bounds[di])
            ax.set_ylim(bounds[dj])
            ax.set_xlabel(f"$x_{{{di}}}$")
            ax.set_ylabel(f"$x_{{{dj}}}$")
    fig.suptitle(f"{FUNC_NAME}: profiling-stage vs volume-stage samples"
                 + (f" (interior_steps={INTERIOR_STEPS})"
                    if INTERIOR_STEPS else ""))
    fig.tight_layout()
    path = f"{out_prefix}_clouds{'_interior' if INTERIOR_STEPS else ''}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    # --- logL distributions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(band_lo - gmax, 0.0, 50)
    ref_logls = logl_vec(reference)
    for vals, label, color in [
        (scan_logls, 'profiling-stage in-band samples', 'tab:blue'),
        (volume_logls, 'volume samples', 'tab:orange'),
        (ref_logls, 'uniform ROI reference', 'gray'),
    ]:
        ax.hist(np.asarray(vals) - gmax, bins=bins, density=True,
                histtype='step', linewidth=2, label=label, color=color)
    ax.set_xlabel(r'$\ln L - \ln L_\mathrm{max}$')
    ax.set_ylabel('density')
    ax.set_title(f"{FUNC_NAME}: logL distribution within the ROI band")
    ax.legend()
    fig.tight_layout()
    path = f"{out_prefix}_logl{'_interior' if INTERIOR_STEPS else ''}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    return paths


def master(workdir):
    with ProfileProjector(
        target_func=log_likelihood,
        bounds=bounds,
        projections=PROJECTIONS,
        roi_threshold=ROI_THRESHOLD,
        samples_output_file=os.path.join(workdir, 'samples.csv'),
        volume_sampling=None,  # installed below, after the scan snapshot
    ) as sampler:
        comm.bcast((sampler.target_func, sampler.grad_func), root=0)
        results = run_all_projections(
            comm=comm, sampler=sampler, projections=PROJECTIONS,
            save_plots=False, myrank=myrank,
        )
        scan_calls = sampler.target_calls
        print(f"\nScan done: {scan_calls} evaluations.", flush=True)

        sampler.volume_sampling_config = normalize_volume_config(
            {'mode': 'roi', 'n_points': N_POINTS,
             'interior_steps': INTERIOR_STEPS,
             'output_file': os.path.join(workdir, 'volume.csv')},
            ROI_THRESHOLD)
        vol = run_volume_sampling(comm, sampler, results, myrank=myrank)
        stats = vol['stats']
        radius = stats['coverage_radius']
        gmax = sampler.global_max_target_val
        band_lo = gmax - ROI_THRESHOLD
        sampler._flush_samples_buffer()

        # The profiling-stage samples are the rows logged before the
        # volume stage started (one row per target call, in order).
        all_rows = read_samples(os.path.join(workdir, 'samples.csv'))
        scan_rows = all_rows[:scan_calls]
        in_band = scan_rows[:, -1] >= band_lo
        scan_in_band = scan_rows[in_band, :-1]
        scan_logls = scan_rows[in_band, -1]

        resolved = np.isin(vol['anchor_status'], ['covered', 'projected'])
        volume_pts = vol['rep_points'][resolved]
        volume_logls = vol['rep_logls'][resolved]

        print("Building the brute-force reference set...", flush=True)
        reference, box_acceptance = brute_force_reference(
            band_lo, np.random.default_rng(20260610), N_REFERENCE)

        maximin = farthest_point_subset(scan_in_band, len(volume_pts))
        rows = [
            set_metrics('scan in-band (all)', scan_in_band, reference, radius),
            set_metrics('scan maximin subset', maximin, reference, radius),
            set_metrics('volume samples', volume_pts, reference, radius),
        ]

        print(f"\n=== {FUNC_NAME}: profiling-stage vs volume-stage samples "
              f"(interior_steps={INTERIOR_STEPS}) ===")
        print(f"Scan: {scan_calls} evals -> {len(scan_in_band)} in-band "
              f"samples. Volume stage: {stats['evals_used']} evals -> "
              f"{len(volume_pts)} samples "
              f"(covered {stats['n_covered']}, projected "
              f"{stats['n_projected']}, holes {stats['n_holes']}).")
        print(f"True box acceptance: {box_acceptance:.3e}; coverage radius "
              f"{radius:.4f} (scaled). rel_spread: per-dim std / reference "
              f"std (1.0 = matches the true ROI spread).")
        header = (f"{'set':<22}{'n':>8}{'coverage':>10}{'meandist':>10}"
                  f"  rel_spread")
        print(header)
        print('-' * (len(header) + 18))
        for r in rows:
            print(f"{r['set']:<22}{r['n']:>8}{r['coverage']:>10.3f}"
                  f"{r['mean_ref_dist']:>10.4f}  {r['rel_spread']}")
        print(f"\nDetails: {json.dumps(rows, indent=2)}")

        figures = make_figures(scan_in_band, volume_pts, reference, gmax,
                               band_lo, scan_logls, volume_logls,
                               os.path.join(workdir, FUNC_NAME))
        for path in figures:
            print(f"Figure: {path}")
        print(f"Output files in {workdir}", flush=True)


if myrank == 0:
    try:
        master(tempfile.mkdtemp(prefix='volume_vs_profile_'))
    finally:
        # Always reached, even on an exception: without it the workers
        # would wait forever and the job would deadlock in MPI_Finalize.
        terminate_workers(comm, myrank)
else:
    worker_main(comm, myrank)
