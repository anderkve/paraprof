"""Check which optimum of a profiled parameter a scan picks where the best
one changes across the projection.

The target is `octant_6d` (see `paraprof.test_functions`): s23 has two
optima, the lower and the upper octant, and which is higher depends on the
other parameters. The script computes a reference profile for each octant
with scipy, runs one projection on virtual workers (`virtual_pool.py`) and
compares the two at three points of the run: before patching, before the
suspect recheck and at the end. For grid points that end in the wrong
octant it lists what patching and the suspect recheck tried there.

    python tools/octant_lab.py --dims 4 --grid 60
    python tools/octant_lab.py --dims 0,2 --grid 40,40 --cells 4
    python tools/octant_lab.py --dims 4 --grid 60 --plant-shift -3

`--dims` uses the parameter order delta, m21, m32, s12, s13, s23. In the
maps a lower-case letter is a grid point that holds the reference value in
that octant (l lower, u upper), an upper-case letter one that holds the
other octant and is low, `?` one that is low in the right octant, `#` one
that is missing and `.` one outside the region of interest.

`--plant-shift` and `--plant` overwrite the grid just before patching with
reference solutions of a chosen octant: the reference assignment displaced
by a number of grid points per projection dimension, or one octant
everywhere. This tests patching and the suspect recheck on a known wrong
state.
"""
import argparse
import collections
import copy

import numpy as np
from scipy.optimize import minimize

import paraprof
import paraprof.master as _master
from virtual_pool import VirtualComm, _MPIShim

NAMES = ['delta', 'm21', 'm32', 's12', 's13', 's23']
TARGET, BOUNDS, PEAKS = paraprof.get_test_function('octant_6d')
TOL = 1e-3      # a grid point this far below the reference counts as low


def octant_of(u):
    """0 for the lower octant, 1 for the upper."""
    return int((0.30 + 0.40 * u[5]) * (1.0 - 0.015 - 0.015 * u[4]) > 0.5)


def reference(dims, npts):
    """Best value and point of each octant at every grid point, from scipy
    L-BFGS-B started in both octants (and at four values of delta where it
    is profiled). An octant that is no local optimum at a grid point has
    value -inf there."""
    prof = [i for i in range(6) if i not in dims]
    axes = [np.linspace(0.0, 1.0, n + 1) for n in npts]     # paraprof uses n + 1 points
    shape = tuple(len(a) for a in axes)
    f_branch = np.full(shape + (2,), -np.inf)
    x_branch = np.full(shape + (2, 6), np.nan)
    box = [(-0.5, 1.5) if i == 0 else (0.0, 1.0) for i in prof]
    for idx in np.ndindex(shape):
        def full(p):
            u = np.empty(6)
            u[prof] = p
            u[dims] = [axes[k][idx[k]] for k in range(len(dims))]
            u[0] %= 1.0
            return u

        for peak in PEAKS[::-1]:
            for d0 in ([0.1, 0.35, 0.6, 0.85] if 0 in prof else [peak[0]]):
                start = peak.copy()
                start[0] = d0
                res = minimize(lambda p: -TARGET(full(p)), start[prof], method='L-BFGS-B',
                               bounds=box, options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 500})
                u = full(res.x)
                o = octant_of(u)
                if -res.fun > f_branch[idx + (o,)]:
                    f_branch[idx + (o,)] = -res.fun
                    x_branch[idx + (o,)] = u
    return {'f_branch': f_branch, 'x_branch': x_branch,
            'f': f_branch.max(axis=-1), 'oct': f_branch.argmax(axis=-1)}


def snapshot(sampler):
    """{grid point: (value, octant, full parameters)} of the finished grid points."""
    snap = {}
    for idx, st in sampler.population.items():
        if st['status'] == 'done':
            u = sampler._construct_params(idx, sampler._best_profiled_params(idx))
            snap[tuple(int(i) for i in idx)] = (float(st['best_fitness']), octant_of(u), u)
    return snap


def plant(sampler, ref, assign):
    """Overwrite each grid point with the reference solution of octant
    ``assign(idx)``, or of the other octant where that one does not exist."""
    for idx, st in sampler.population.items():
        for b in (assign(idx), 1 - assign(idx)):
            x = ref['x_branch'][tuple(idx) + (b,)]
            if np.all(np.isfinite(x)):
                break
        else:
            continue
        k = int(np.argmax(st['fitnesses']))
        st['profiled_params'][k] = x[sampler.profiled_dims]
        st['fitnesses'][:] = -np.inf
        st['fitnesses'][k] = st['best_fitness'] = TARGET(x)
        st['status'] = 'done'


def run(dims, npts, method='lbfgsb', n_workers=16, seed=1, assign=None, ref=None,
        **projector_kwargs):
    """One projection on virtual workers. Returns the snapshots and the
    patching and suspect-recheck jobs, which keep what they tested."""
    np.random.seed(seed)
    paraprof.set_log_level('WARNING')
    projections = [{'dims': list(dims), 'grid_points': list(npts), 'optimization_method': method}]
    rec = {'snaps': {}, 'patch_jobs': [], 'suspect_jobs': []}
    kwargs = dict(roi_threshold=6.0, parameter_names=NAMES, periodic=['delta'])
    kwargs.update(projector_kwargs)
    comm = VirtualComm(n_workers, TARGET, seed=seed)
    real_mpi = _master.MPI
    _master.MPI = _MPIShim
    try:
        with paraprof.ProfileProjector(target_func=TARGET, bounds=BOUNDS,
                                       projections=copy.deepcopy(projections), **kwargs) as sampler:
            make_patch_jobs = sampler.create_patching_wave_jobs
            make_suspect_jobs = sampler.create_suspect_recheck_jobs

            def patch(wave_number, updated_points_last_wave, next_job_id):
                if wave_number == 0:
                    if assign is not None:
                        plant(sampler, ref, assign)
                    rec['snaps']['before patching'] = snapshot(sampler)
                jobs, next_job_id = make_patch_jobs(wave_number, updated_points_last_wave,
                                                    next_job_id)
                for job in jobs:
                    # the neighbour the test vector was taken from
                    src = next((n for n in sampler._get_valid_neighbors(job.grid_idx)
                                if n in sampler.population and np.array_equal(
                                    sampler._best_profiled_params(n), job.test_profiled_params)),
                               None)
                    rec['patch_jobs'].append((wave_number, job, src))
                return jobs, next_job_id

            def suspect(wave_number, updated_points_last_wave, next_job_id):
                if wave_number == 0:
                    rec['snaps']['before suspect recheck'] = snapshot(sampler)
                jobs, next_job_id = make_suspect_jobs(wave_number, updated_points_last_wave,
                                                      next_job_id)
                rec['suspect_jobs'] += [(wave_number, job) for job in jobs]
                return jobs, next_job_id

            sampler.create_patching_wave_jobs = patch
            sampler.create_suspect_recheck_jobs = suspect
            paraprof.run_scan(comm=comm, sampler=sampler, projections=projections,
                              save_plots=False, broadcast_target_func=False, myrank=0)
            rec['snaps']['end'] = snapshot(sampler)
            rec['sampler'] = sampler
            rec['calls'] = int(sampler.target_calls)
            rec['roi'] = float(sampler.roi_threshold)
            rec['initial_maxima'] = sorted(((float(m['target_val']), np.array(m['point']))
                                            for m in sampler.initial_maxima), key=lambda t: -t[0])
    finally:
        _master.MPI = real_mpi
    return rec


def compare(snap, ref, roi):
    """Counts by outcome over the reference region of interest, and
    {grid point: deficit} of the grid points that are low."""
    counts = collections.Counter()
    low = {}
    for idx in np.ndindex(ref['f'].shape):
        if ref['f'][idx] < -roi:
            continue
        if idx not in snap:
            counts['missing'] += 1
            continue
        f, o, _ = snap[idx]
        deficit = ref['f'][idx] - f
        if deficit <= TOL:
            counts['ok'] += 1
        else:
            counts['wrong octant' if o != ref['oct'][idx] else 'low, right octant'] += 1
            low[idx] = deficit
    return counts, low


def grid_map(snap, ref, roi):
    """Text map of the grid (see the module docstring). 2D: first projection
    dimension to the right, second upwards."""
    def char(idx):
        if ref['f'][idx] < -roi:
            return '.' if idx in snap else ' '
        if idx not in snap:
            return '#'
        f, o, _ = snap[idx]
        if ref['f'][idx] - f <= TOL:
            return 'lu'[o]
        return 'LU'[o] if o != ref['oct'][idx] else '?'

    shape = ref['f'].shape
    if len(shape) == 1:
        return ''.join(char((i,)) for i in range(shape[0]))
    return '\n'.join(''.join(char((i, j)) for i in range(shape[0]))
                     for j in reversed(range(shape[1])))


def report_cell(idx, rec, ref):
    """What patching and the suspect recheck tried at one grid point."""
    sampler = rec['sampler']
    end = rec['snaps']['end']
    f, o, _ = end[idx]
    print(f"\ngrid point {idx}: holds {'lu'[o]} at {f:.4f}; reference "
          f"{'lu'[ref['oct'][idx]]} at {ref['f'][idx]:.4f}")
    print("  neighbours: octant, value, value of the neighbour's vector at this grid point")
    rows = []
    for n in sampler._get_valid_neighbors(idx):
        n = tuple(int(i) for i in n)
        if n in end:
            nf, no, nu = end[n]
            rows.append((nf, n, no, TARGET(sampler._construct_params(idx, nu[sampler.profiled_dims]))))
    for nf, n, no, here in sorted(rows, reverse=True):
        print(f"    {n} {'lu'[no]} {nf:9.4f} {here:9.4f}{'   higher than the current value' if here > f else ''}")
    tests = [(w, j, src) for w, j, src in rec['patch_jobs'] if tuple(j.grid_idx) == idx]
    for w, j, src in tests:
        vec_oct = 'lu'[octant_of(sampler._construct_params(idx, j.test_profiled_params))]
        print(f"  patching wave {w}: vector of {tuple(int(i) for i in src)} ({vec_oct}) gave "
              f"{j.test_fitness:.4f} against {j.current_best_fitness:.4f}"
              f"{', taken' if j.will_update else ''}")
    if not tests:
        print("  no patching test")
    jobs = [(w, j) for w, j in rec['suspect_jobs'] if tuple(j.grid_idx) == idx]
    for w, j in jobs:
        seeds = ' '.join(f"{'lu'[octant_of(u)]}:{TARGET(u):.3f}" for u in
                         (sampler._construct_params(idx, s) for s in j.candidate_seeds))
        print(f"  suspect wave {w}: seeds {seeds}{'; best one taken' if j.will_update else ''}")
    if not jobs:
        print("  not a suspect-recheck candidate")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dims', default='4', help="projection dims, e.g. 4 or 0,2")
    ap.add_argument('--grid', default='60', help="grid points per projection dim")
    ap.add_argument('--method', default='lbfgsb', choices=['lbfgsb', 'de'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--plant-shift', default=None,
                    help="plant the reference octants displaced by this many grid points per dim")
    ap.add_argument('--plant', default=None, choices=['lower', 'upper'],
                    help="plant one octant everywhere it exists")
    ap.add_argument('--cells', type=int, default=3, help="wrong grid points to report on")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(',')]
    npts = [int(n) for n in args.grid.split(',')]
    ref = reference(dims, npts)

    assign = None
    if args.plant:
        assign = lambda idx: int(args.plant == 'upper')
    elif args.plant_shift:
        shift = [int(s) for s in args.plant_shift.split(',')]
        shape = ref['oct'].shape
        assign = lambda idx: int(ref['oct'][tuple(
            int(np.clip(i + s, 0, n - 1)) for i, s, n in zip(idx, shift, shape))])

    rec = run(dims, npts, method=args.method, n_workers=args.workers, seed=args.seed,
              assign=assign, ref=ref)

    print(f"projection {[NAMES[d] for d in dims]}, grid {npts}, {args.method}, "
          f"{rec['calls']} target calls")
    distinct = dict.fromkeys(f"{'lu'[octant_of(p)]} {f:.2f}" for f, p in rec['initial_maxima'])
    print("distinct initial maxima (octant, value): " + ', '.join(list(distinct)[:8]))
    low = {}
    for stage, snap in rec['snaps'].items():
        counts, low = compare(snap, ref, rec['roi'])
        worst = max(low.values()) if low else 0.0
        print(f"{stage + ':':24s} {dict(counts)}, largest deficit {worst:.3f}")
    n_taken = sum(j.will_update for _, j, _ in rec['patch_jobs'])
    n_seeded = sum(j.will_update for _, j in rec['suspect_jobs'])
    print(f"patching: {len(rec['patch_jobs'])} tests, {n_taken} taken; "
          f"suspect recheck: {len(rec['suspect_jobs'])} jobs, {n_seeded} taken")
    print(grid_map(rec['snaps']['end'], ref, rec['roi']))
    for idx in sorted(low, key=low.get, reverse=True)[:args.cells]:
        report_cell(idx, rec, ref)


if __name__ == '__main__':
    main()
