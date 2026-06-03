"""
Replicate study for the smooth-certification feature (``de.smooth_certify``).

A single A/B run cannot settle whether the feature is a robust win because some
targets' scans are nondeterministic (MPI ordering -> run-to-run variation). This
driver runs many independent replicates (distinct master seeds) of each target
in both modes and reports paired-with-noise distributions so a real effect can
be separated from that noise.

Two axes are measured per run:

* **Target calls** -- the efficiency axis. Reported as mean +/- std per mode and
  a Mann-Whitney U test on the per-run totals.

* **ROI quality** -- using the one-sided structure of profiling: every grid value
  is a lower bound on the truth ``L(theta) = max_phi f(theta, phi)``, so the
  elementwise max over *all* runs (both modes, all seeds) is the tightest known
  lower bound and serves as the reference grid. Each run's per-cell deficit
  ``reference - value >= 0`` then measures how far it falls short. We report,
  per run, the mean ROI deficit and the ROI coverage fraction (covered ROI cells
  / reference ROI cells), and a Mann-Whitney U test between modes. A robust win
  is a calls reduction with the deficit/coverage distributions statistically
  indistinguishable between modes.

Usage:
    python examples/run_smooth_certify_replicate_study.py \\
        --targets himmelblau_4d rosenbrock_4d rastrigin_4d \\
        --ncores 4 --reps 6
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - scipy is a core dep, but stay graceful
    mannwhitneyu = None


BASE_SEED = 750123


def _run(target, mode, seed, ncores, out_path):
    env = dict(os.environ)
    env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    script = os.path.join(os.path.dirname(__file__),
                          'run_smooth_certify_benchmark.py')
    cmd = ['mpiexec', '--oversubscribe', '-n', str(ncores),
           sys.executable, script,
           '--target', target, '--mode', mode,
           '--seed', str(seed), '--out', out_path]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out_path) as f:
        return json.load(f)


def _reference_grids(runs):
    """Elementwise nanmax over every run's grid, per projection index.

    By one-sidedness this is the tightest known lower bound on the true profile
    grid, used as the quality reference.
    """
    n_proj = len(runs[0]['projections'])
    refs = []
    for p in range(n_proj):
        stack = np.array([np.array(r['projections'][p]['coarse_grid_values'],
                                   dtype=float)
                          for r in runs])
        with np.errstate(invalid='ignore'):
            # Columns never covered by any run are all-NaN -> NaN reference;
            # the quality metric masks those out via ref_finite.
            ref = np.where(np.isnan(stack).all(axis=0), np.nan,
                           np.nanmax(np.where(np.isnan(stack), -np.inf, stack), axis=0))
        refs.append(ref)
    return refs


def _run_quality(run, refs, roi_threshold):
    """Mean ROI deficit and ROI coverage for one run against the references."""
    deficits = []
    covered = 0
    total_roi = 0
    for p, ref in enumerate(refs):
        val = np.array(run['projections'][p]['coarse_grid_values'], dtype=float)
        ref_finite = np.isfinite(ref)
        gmax = np.nanmax(ref[ref_finite]) if ref_finite.any() else np.nan
        roi = ref_finite & (ref > gmax - roi_threshold)
        total_roi += int(roi.sum())
        run_covered = roi & np.isfinite(val)
        covered += int(run_covered.sum())
        if run_covered.any():
            deficits.append(ref[run_covered] - val[run_covered])
    mean_deficit = (float(np.concatenate(deficits).mean())
                    if deficits else float('nan'))
    coverage = covered / total_roi if total_roi else float('nan')
    return mean_deficit, coverage


def _mwu(a, b):
    """Two-sided Mann-Whitney U p-value, or nan when unavailable/degenerate."""
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if mannwhitneyu is None or len(a) < 2 or len(b) < 2:
        return float('nan')
    if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
        return 1.0
    try:
        return float(mannwhitneyu(a, b, alternative='two-sided').pvalue)
    except ValueError:
        return float('nan')


def _fmt(vals):
    vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return "n/a"
    return f"{vals.mean():.4g} +/- {vals.std(ddof=1) if vals.size > 1 else 0:.2g}"


ROI_THRESHOLD = {
    'himmelblau_4d': 4.0,
    'rosenbrock_4d': 8.0,
    'rosenbrock_6d': 8.0,
    'rastrigin_4d': 8.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--reps', type=int, default=6,
                        help='Number of replicate seeds per (target, mode).')
    args = parser.parse_args()

    seeds = [BASE_SEED + 1000 * k for k in range(args.reps)]

    print("=" * 78)
    print(f"Smooth-certify replicate study  (reps={args.reps}, ncores={args.ncores})")
    print("=" * 78)

    report = {}
    for target in args.targets:
        roi_thr = ROI_THRESHOLD[target]
        runs = {'baseline': [], 'certify': []}
        with tempfile.TemporaryDirectory() as td:
            for mode in ('baseline', 'certify'):
                for seed in seeds:
                    out = os.path.join(td, f'{mode}_{seed}.json')
                    runs[mode].append(_run(target, mode, seed, args.ncores, out))

            all_runs = runs['baseline'] + runs['certify']
            refs = _reference_grids(all_runs)

            calls = {m: [r['projections'][-1]['cumulative_target_calls']
                         for r in runs[m]] for m in runs}
            certified = [r['cells_smooth_certified'] for r in runs['certify']]
            quality = {m: [_run_quality(r, refs, roi_thr) for r in runs[m]]
                       for m in runs}

        defs = {m: [q[0] for q in quality[m]] for m in quality}
        covs = {m: [q[1] for q in quality[m]] for m in quality}

        base_mean = np.mean(calls['baseline'])
        cert_mean = np.mean(calls['certify'])
        pct = 100.0 * (cert_mean - base_mean) / base_mean

        print(f"\n### {target}   (ROI threshold {roi_thr})")
        print(f"  calls     baseline {_fmt(calls['baseline'])}")
        print(f"            certify  {_fmt(calls['certify'])}   "
              f"({pct:+.1f}% on the mean)   MWU p={_mwu(calls['baseline'], calls['certify']):.3f}")
        print(f"  cells certified (certify runs): {_fmt(certified)}")
        print(f"  ROI mean deficit  baseline {_fmt(defs['baseline'])}")
        print(f"                    certify  {_fmt(defs['certify'])}   "
              f"MWU p={_mwu(defs['baseline'], defs['certify']):.3f}")
        print(f"  ROI coverage frac baseline {_fmt(covs['baseline'])}")
        print(f"                    certify  {_fmt(covs['certify'])}   "
              f"MWU p={_mwu(covs['baseline'], covs['certify']):.3f}")

        report[target] = {
            'seeds': seeds,
            'calls': calls, 'certified': certified,
            'roi_mean_deficit': defs, 'roi_coverage': covs,
            'calls_pct_change_mean': pct,
        }

    print("\n" + "=" * 78)
    print("Reading the table: a robust win = a clear negative calls % with a low")
    print("MWU p on calls, AND deficit/coverage MWU p well above 0.05 (modes")
    print("statistically indistinguishable on quality).")
    print("=" * 78)

    out_json = os.path.join(os.getcwd(), 'smooth_certify_replicate_report.json')
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == '__main__':
    main()
