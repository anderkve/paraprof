"""Scheduling benchmark on virtual workers (see virtual_pool.py).

Runs one 2-D projection of a 6-parameter target in the unit cube for several
worker counts and noise levels, and reports for each run

  evaluations   target calls
  wall          simulated wall time in units of one evaluation
  busy          mean number of workers evaluating
  cells         cells explored, and how many ended in status converged/optimized
  d_max         reported global maximum minus the true one
  roi err       stored cell value minus the reference profile value, over the
                cells whose reference value is inside the region of interest
                (median, worst, and the number of cells low by more than 0.01)

Targets
  gauss6  correlated Gaussian; the reference profile is exact
  fit6    four well-measured parameters, one with two modes of unequal depth
          and one weakly constrained periodic one, coupled; the reference
          profile comes from a multistart scipy optimisation per cell
  table6  fit6 with part of the dependence taken from coarse bilinear tables:
          continuous, kinked at the table nodes, no curvature inside a cell

`--noise a` adds a deterministic pseudo-random term of rms `a` to the target.

    python tools/bench_scheduling.py --workers 3,100,1000 --noise 0,5e-7
"""
import argparse
import collections
import json
import sys
import zlib

import numpy as np
from scipy.optimize import minimize

from virtual_pool import run_virtual_scan

PROJ = [3, 1]          # as a user writes it; paraprof sorts the dims internally

# ---- targets -----------------------------------------------------------------------------
_centre = np.array([0.55, 0.45, 0.50, 0.52, 0.47, 0.45])
_sig = np.array([0.20, 0.07, 0.05, 0.055, 0.04, 0.15])
_rng = np.random.default_rng(12345)
_A = np.eye(6) + 0.25 * _rng.standard_normal((6, 6))
_A /= np.linalg.norm(_A, axis=1, keepdims=True)
_M = (_A / _sig).T @ (_A / _sig)


def gauss6(x):
    d = np.asarray(x, dtype=float) - _centre
    return -295.0 - 0.5 * float(d @ _M @ d)


def fit6(x):
    """Order: phase (periodic, weak), m21, m32, t12, t13, t23 (two modes)."""
    x = np.asarray(x, dtype=float)
    ph, m21, m32, t12, t13, t23 = x
    core = np.array([(m21 - 0.45) / 0.07, (m32 - 0.50 - 0.15 * (t23 - 0.5)) / 0.05,
                     (t12 - 0.52) / 0.055 + 0.3 * (m21 - 0.45) / 0.07,
                     (t13 - 0.47 - 0.05 * np.cos(2 * np.pi * ph)) / 0.04])
    lo = -0.5 * ((t23 - 0.40) / 0.06) ** 2
    hi = -0.8 - 0.5 * ((t23 - 0.62) / 0.05) ** 2
    octant = np.logaddexp(lo, hi)
    phase = 1.2 * (np.cos(2 * np.pi * (ph - 0.65 - 0.2 * (t23 - 0.5))) - 1.0)
    return -295.0 - 0.5 * float(core @ core) + float(octant) + float(phase)


# Coarse tables for table6: nodes every 0.2 in m32 and every 0.1 in t23 and in the phase.
_T_M32 = np.linspace(0.0, 1.0, 6)
_T_T23 = np.linspace(0.0, 1.0, 11)
_T_PH = np.linspace(0.0, 1.0, 11)
_TAB_ATM = np.array([[-0.5 * ((a - 0.50) / 0.07) ** 2 - 0.5 * ((b - 0.45) / 0.10) ** 2
                      for b in _T_T23] for a in _T_M32])
_TAB_PH = np.array([[0.8 * (np.cos(2 * np.pi * (a - 0.65 - 0.2 * (b - 0.5))) - 1.0)
                     for b in _T_T23] for a in _T_PH])


def _bilinear(tab, xa, xb, a, b):
    i = int(np.clip(np.searchsorted(xa, a) - 1, 0, len(xa) - 2))
    j = int(np.clip(np.searchsorted(xb, b) - 1, 0, len(xb) - 2))
    u = (a - xa[i]) / (xa[i + 1] - xa[i])
    v = (b - xb[j]) / (xb[j + 1] - xb[j])
    return ((1 - u) * (1 - v) * tab[i, j] + u * (1 - v) * tab[i + 1, j]
            + (1 - u) * v * tab[i, j + 1] + u * v * tab[i + 1, j + 1])


def table6(x):
    """fit6 with the dependence on (m32, t23) and (phase, t23) read from coarse bilinear tables."""
    x = np.asarray(x, dtype=float)
    ph, m21, m32, t12, t13, t23 = x
    core = np.array([(m21 - 0.45) / 0.07, (m32 - 0.50 - 0.15 * (t23 - 0.5)) / 0.09,
                     (t12 - 0.52) / 0.055 + 0.3 * (m21 - 0.45) / 0.07,
                     (t13 - 0.47 - 0.05 * np.cos(2 * np.pi * ph)) / 0.04])
    lo = -0.5 * ((t23 - 0.40) / 0.12) ** 2
    hi = -0.8 - 0.5 * ((t23 - 0.62) / 0.10) ** 2
    return (-295.0 - 0.5 * float(core @ core) + float(np.logaddexp(lo, hi))
            + _bilinear(_TAB_ATM, _T_M32, _T_T23, m32, t23)
            + _bilinear(_TAB_PH, _T_PH, _T_T23, ph, t23))


TARGETS = {'gauss6': gauss6, 'fit6': fit6, 'table6': table6}


def with_noise(func, amp):
    if not amp:
        return func

    def noisy(x):
        x = np.asarray(x, dtype=float)
        u = zlib.crc32(x.tobytes()) / 2 ** 32
        return func(x) + amp * np.sqrt(12.0) * (u - 0.5)
    return noisy


# ---- reference profile ---------------------------------------------------------------------
def reference_profile(name, cells, axes, proj):
    """Profile value of the noise-free target at each explored cell.

    `proj` is the sampler's own order of the projection dims (paraprof sorts them), which is
    the order of `axes` and of the cell indices.
    """
    func = TARGETS[name]
    prof = [i for i in range(6) if i not in proj]
    ref = {}
    if name == 'gauss6':
        Mpp = _M[np.ix_(proj, proj)]
        Mpq = _M[np.ix_(proj, prof)]
        Mqq = _M[np.ix_(prof, prof)]
        S = Mpp - Mpq @ np.linalg.solve(Mqq, Mpq.T)
        for c in cells:
            d = np.array([axes[0][c[0]], axes[1][c[1]]]) - _centre[proj]
            # the unconstrained optimum of the profiled block may leave the box; fall back to
            # a bounded optimisation there
            q = -np.linalg.solve(Mqq, Mpq.T @ d) + _centre[prof]
            if np.all((q >= 0) & (q <= 1)):
                ref[c] = -295.0 - 0.5 * float(d @ S @ d)
            else:
                ref[c] = _numeric_profile(func, c, axes, proj, prof)
        return ref
    for c in cells:
        ref[c] = _numeric_profile(func, c, axes, proj, prof)
    return ref


def _numeric_profile(func, c, axes, proj, prof):
    rng = np.random.default_rng(abs(hash((int(c[0]), int(c[1])))) % (2 ** 32))
    best = -np.inf

    def neg(q):
        x = np.empty(6)
        x[proj] = [axes[0][c[0]], axes[1][c[1]]]
        x[prof] = q
        return -func(x)

    # starts in the full-parameter order (phase, m21, m32, t12, t13, t23), cut to the profiled dims
    full = [np.array([0.65, 0.45, 0.5, 0.52, 0.47, 0.40]), np.array([0.65, 0.45, 0.5, 0.52, 0.47, 0.62]),
            np.array([0.15, 0.45, 0.5, 0.52, 0.47, 0.40]), np.array([0.15, 0.45, 0.5, 0.52, 0.47, 0.62])]
    starts = [f[prof] for f in full] + [rng.uniform(0, 1, len(prof)) for _ in range(8)]
    for q0 in starts:
        r = minimize(neg, q0, method='L-BFGS-B', bounds=[(0, 1)] * len(prof),
                     options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 500})
        # a derivative-free pass afterwards: L-BFGS-B stalls at the kinks of a tabulated target
        r2 = minimize(neg, np.clip(r.x, 0, 1), method='Powell', bounds=[(0, 1)] * len(prof),
                      options={'xtol': 1e-10, 'ftol': 1e-14, 'maxiter': 20000})
        best = max(best, -r.fun, -r2.fun)
    return best


def true_maximum(name):
    func = TARGETS[name]
    if name == 'gauss6':
        return -295.0
    rng = np.random.default_rng(1)
    best = -np.inf
    for _ in range(200):
        r = minimize(lambda x: -func(x), rng.uniform(0, 1, 6), method='L-BFGS-B',
                     bounds=[(0, 1)] * 6, options={'ftol': 1e-15, 'gtol': 1e-10})
        best = max(best, -r.fun)
    return best


# ---- one run -------------------------------------------------------------------------------
def run(name, noise, n_workers, grid, method, roi, extra, seed):
    np.random.seed(seed)           # paraprof draws its LHS seeds from the global generator
    target = with_noise(TARGETS[name], noise)
    projections = [{'dims': list(PROJ), 'grid_points': [grid, grid], 'optimization_method': method}]
    kwargs = dict(roi_threshold=roi, pop_per_grid_point=3, lbfgsb_max_iter=20, lbfgsb_polish=True,
                  use_clustering=False, max_patching_waves=2, n_initial_optimizations=8)
    kwargs.update(extra)
    out = run_virtual_scan(target, [(0.0, 1.0)] * 6, projections, n_workers,
                           jitter=0.05, seed=seed, **kwargs)
    cells = sorted(out['population'])
    ref = reference_profile(name, cells, out['grid_axes'], list(out['projection_dims']))
    tmax = true_maximum(name)
    stored = np.array([out['population'][c]['best_fitness'] for c in cells])
    exact = np.array([ref[c] for c in cells])
    status = [out['population'][c]['status'] for c in cells]
    in_roi = exact > tmax - roi
    err = (stored - exact)[in_roi]
    return {
        'target': name, 'noise': noise, 'workers': n_workers, 'method': method,
        'evaluations': out['evaluations'], 'wall': out['wall'],
        'mean_busy_workers': out['mean_busy_workers'], 'utilisation': out['utilisation'],
        'cells': len(cells),
        'cells_done': sum(s in ('converged', 'optimized') for s in status),
        'roi_cells_true': int(in_roi.sum()),
        'global_max_error': out['global_max'] - tmax,
        'roi_err_median': float(np.median(err)) if len(err) else float('nan'),
        'roi_err_worst': float(err.min()) if len(err) else float('nan'),
        'roi_cells_low_0p01': int((err < -0.01).sum()),
        'evaluations_by_job_type': dict(collections.Counter(out['timeline'][3])),
        'master_us_per_result': 1e6 * out['master_seconds_per_result'],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target', default='gauss6', choices=sorted(TARGETS))
    ap.add_argument('--workers', default='3,100,1000')
    ap.add_argument('--noise', default='0')
    ap.add_argument('--grid', type=int, default=25)
    ap.add_argument('--method', default='lbfgsb', choices=['lbfgsb', 'de'])
    ap.add_argument('--roi', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--extra', default='{}',
                    help='JSON dict of extra ProfileProjector keyword arguments')
    ap.add_argument('--json', help='append one JSON line per run to this file')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    extra = json.loads(args.extra)

    head = (f"{'target':7s} {'noise':>8s} {'workers':>7s} {'evals':>7s} {'wall':>8s} {'busy':>7s} "
            f"{'util%':>6s} {'cells':>9s} {'d_max':>9s} {'roi err med/worst/#low':>26s} {'us/res':>7s}")
    print(head)
    for noise in [float(v) for v in args.noise.split(',')]:
        for nw in [int(v) for v in args.workers.split(',')]:
            r = run(args.target, noise, nw, args.grid, args.method, args.roi, extra, args.seed)
            r['label'] = args.label
            print(f"{r['target']:7s} {r['noise']:8.1e} {r['workers']:7d} {r['evaluations']:7d} "
                  f"{r['wall']:8.1f} {r['mean_busy_workers']:7.2f} {100 * r['utilisation']:6.1f} "
                  f"{r['cells_done']:4d}/{r['cells']:<4d} {r['global_max_error']:9.2e} "
                  f"{r['roi_err_median']:9.1e}/{r['roi_err_worst']:9.1e}/{r['roi_cells_low_0p01']:<4d} "
                  f"{r['master_us_per_result']:7.0f}")
            sys.stdout.flush()
            if args.json:
                with open(args.json, 'a') as f:
                    f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
