"""Projections over derived quantities: configuration, the cell bookkeeping and the
penalised fitness at the sampler level, and one end-to-end run under mpiexec."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import InvalidProjectionError


def _target(p):
    x = np.asarray(p, dtype=float)
    return -float(np.sum(x ** 2)) / 2.0, np.array([x[0] + x[2], x[1] + x[3]])


def _sampler(**kw):
    bounds = np.array([[-5.0, 5.0]] * 4)
    proj = {'derived': ['u', 'v'], 'ranges': [[-6.0, 6.0], [-6.0, 6.0]], 'grid_points': [12, 12]}
    return ProfileProjector(target_func=_target, bounds=bounds, projections=[proj],
                            derived_names=['u', 'v'], pop_per_grid_point=2, **kw)


class TestConfiguration:
    def test_derived_projection_needs_names(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        proj = {'derived': ['u', 'v'], 'ranges': [[-6, 6], [-6, 6]], 'grid_points': [12, 12]}
        with pytest.raises(InvalidProjectionError):
            ProfileProjector(target_func=_target, bounds=bounds, projections=[proj])

    def test_unknown_name_and_missing_ranges(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        with pytest.raises(InvalidProjectionError):
            ProfileProjector(target_func=_target, bounds=bounds, derived_names=['u', 'v'],
                             projections=[{'derived': ['u', 'w'], 'ranges': [[-6, 6], [-6, 6]], 'grid_points': [12, 12]}])
        with pytest.raises(InvalidProjectionError):
            ProfileProjector(target_func=_target, bounds=bounds, derived_names=['u', 'v'],
                             projections=[{'derived': ['u', 'v'], 'grid_points': [12, 12]}])

    def test_derived_mode_setup(self):
        s = _sampler()
        assert s.derived_mode and s.projection_dims == [] and s.profiled_dims == [0, 1, 2, 3]
        assert s.n_proj_dims == 2 and len(s.grid_axes) == 2 and len(s.grid_axes[0]) == 13
        assert s.grid_axes[0][0] == -6.0 and s.grid_axes[0][-1] == 6.0
        assert not s.de_allow_early_DE_exit
        # an input-space projection in the same list keeps the input-space behaviour
        assert s._construct_params((3, 4), np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [1.0, 2.0, 3.0, 4.0]


class TestCellBookkeeping:
    def test_derived_array_normalisation(self):
        s = _sampler()
        assert s._as_derived_array({'v': 2.0, 'u': 1.0}).tolist() == [1.0, 2.0]
        assert s._as_derived_array([1.0, 2.0]).tolist() == [1.0, 2.0]
        assert s._as_derived_array([1.0]) is None
        assert s._as_derived_array('nonsense') is None

    def test_cell_distance_and_penalty(self):
        s = _sampler()
        node = np.array([s.grid_axes[0][6], s.grid_axes[1][6]])          # the node at (0, 0)
        w = 12.0 / 12
        assert s._derived_cell_distance(node + 0.4 * w, (6, 6)) == 0.0
        assert abs(s._derived_cell_distance(node + np.array([1.5 * w, 0.0]), (6, 6)) - 1.0) < 1e-12
        res = {'target_val': -1.0, 'derived': node + np.array([1.5 * w, 0.0])}
        assert abs(s.fitness_for_cell(res, (6, 6)) - (-1.0 - s.derived_penalty_scale * s.roi_threshold)) < 1e-12
        assert s.fitness_for_cell(res, None) == -1.0
        assert s.fitness_for_cell({'target_val': -1.0, 'derived': None}, (6, 6)) == -np.inf

    def test_register_caches_derived_and_records_in_cell_best(self):
        s = _sampler()
        x = np.array([0.1, 0.2, 0.3, 0.4])
        val, d = _target(x)
        s._register_target_call(x, val, derived=d)
        assert s._cached_derived_coords(x).tolist() == d.tolist()
        cell = tuple(int(round((q + 6.0) / 12.0 * 12)) for q in d)      # nearest node on the 13-node axes
        assert s._get_grid_indices_from_point(x) == cell
        rec = s.derived_cell_best[cell]
        assert rec[0] == val and rec[1].tolist() == x.tolist()
        # a better evaluation in the same cell replaces it, a worse one does not
        y = np.array([0.3, 0.3, 0.1, 0.3])      # u = 0.4, v = 0.6 as well, higher logL
        s._register_target_call(y, _target(y)[0], derived=_target(y)[1])
        assert s.derived_cell_best[cell][0] == _target(y)[0]
        z = np.array([-0.5, 1.0, 0.9, -0.4])    # same cell, lower logL
        s._register_target_call(z, _target(z)[0], derived=_target(z)[1])
        assert s.derived_cell_best[cell][0] == _target(y)[0]
        assert s.global_max_target_val == _target(y)[0]

    def test_unknown_point_is_evaluated_on_the_master(self):
        s = _sampler()
        calls = s.target_calls
        idx = s._get_grid_indices_from_point(np.array([3.0, -3.0, 3.0, -3.0]))   # u = 6, v = -6: the corner node
        assert idx == (12, 0)
        assert s.target_calls == calls + 1

    def test_export_uses_in_cell_best(self):
        s = _sampler()
        x = np.array([0.5, 0.5, 0.5, 0.5])      # u = v = 1 -> node (7, 7)
        val, d = _target(x)
        s._register_target_call(x, val, derived=d)
        s.population[(7, 7)] = {'profiled_params': np.array([x]), 'fitnesses': np.array([val - 0.3]), 'best_fitness': val - 0.3,
                                'status': 'optimized', 'improvement_history': [], 'last_update_gen': 0, 'optimizer_state': None}
        s.population[(7, 8)] = dict(s.population[(7, 7)])     # a cell no evaluation landed in
        sol = s.export_grid_solution()
        assert (7, 7) in sol['solutions'] and sol['solutions'][(7, 7)]['likelihood'] == val
        assert sol['solutions'][(7, 7)]['derived'].tolist() == d.tolist()
        assert (7, 8) not in sol['solutions']

    def test_projection_into_cell_with_the_jacobian(self):
        s = _sampler()
        J = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])     # q is linear, J is exact
        x = np.array([0.0, 0.0, 2.0, 0.0])                            # u = 2, v = 0: cell (8, 6) covers u in [1.5, 2.5]
        lo, hi = s._cell_box((9, 6))                                  # the next cell, u in [2.5, 3.5]
        x_new = s._project_into_cell(x, np.array([2.0, 0.0]), J, (9, 6))
        q_new = _target(x_new)[1]
        assert lo[0] <= q_new[0] <= hi[0] and abs(q_new[1]) < 1e-12
        assert s._project_into_cell(x, np.array([2.0, 0.0]), J, (8, 6)) is None   # already inside
        # a seed from a neighbour cell is projected with that cell's stored Jacobian
        s._register_target_call(x, _target(x)[0], derived=np.array([2.0, 0.0]))
        s.population[(8, 6)] = {'profiled_params': np.array([x]), 'fitnesses': np.array([-2.0]), 'best_fitness': -2.0,
                                'status': 'optimized', 'improvement_history': [], 'last_update_gen': 0,
                                'optimizer_state': None, 'derived_jacobian': J}
        seed = s._seed_for_cell((8, 6), x, (9, 6))
        assert lo[0] <= _target(seed)[1][0] <= hi[0]
        # the step is shortest in box units: a narrow box is moved less than a wide one
        wide = ProfileProjector(target_func=_target, bounds=np.array([[-5.0, 5.0], [-5.0, 5.0], [-0.05, 0.05], [-5.0, 5.0]]),
                                projections=[{'derived': ['u', 'v'], 'ranges': [[-6.0, 6.0], [-6.0, 6.0]], 'grid_points': [12, 12]}],
                                derived_names=['u', 'v'], pop_per_grid_point=2)
        y = np.array([0.0, 0.0, 0.0, 0.0])
        moved = wide._project_into_cell(y, np.array([0.0, 0.0]), J, (9, 6))      # u must go to about 2.5
        assert abs(moved[2]) <= 0.05 and moved[0] > 2.4 and abs(_target(moved)[1][0] - 2.5) < 0.1
        # without a source cell the point's own cell (or a neighbour with a Jacobian) supplies it
        seed = s._seed_for_cell(None, x, (9, 6))
        assert lo[0] <= _target(seed)[1][0] <= hi[0]
        y = np.array([0.0, 0.0, 1.0, 0.0])                        # u = 1: cell (7, 6), no Jacobian of its own; (8, 6) is a neighbour
        s._register_target_call(y, _target(y)[0], derived=np.array([1.0, 0.0]))
        seed = s._seed_for_cell(None, y, (9, 6))
        assert lo[0] <= _target(seed)[1][0] <= hi[0]


def _drive(job, tasks=None):
    """Run a job to completion in-process against _target; return the number of evaluations."""
    if tasks is None:
        tasks = job.start()
    n = 0
    while tasks and not job.is_finished():
        t = tasks.pop(0)
        n += 1
        val, d = _target(t['params'])
        job.sampler._register_target_call(np.asarray(t['params'], float), val, derived=d)
        res = {'target_val': val, 'derived': d, 'params': t['params'], 'context': t['context']}
        tasks.extend(job.process_result(res))
    return n


def _cell_job(s, idx, x0):
    from paraprof.jobs.lbfgsb_job import LBFGSBJob
    x0 = np.asarray(x0, float)
    val, d = _target(x0)
    s._register_target_call(x0, val, derived=d)
    s.population[idx] = {'profiled_params': np.array([x0]), 'fitnesses': np.array([val]), 'best_fitness': val,
                         'status': 'LBFGSB_queued', 'improvement_history': [], 'last_update_gen': 0,
                         'optimizer_state': None}
    fit = s.derived_penalised_value(val, d, idx)
    return LBFGSBJob(0, 'LBFGSB', s, tuple(s.profiled_dims), x0, idx, x0, seed_history=None, start_fitness=fit)


class TestConstrainedPolish:
    """The polish of a derived-plane cell: quasi-Newton steps constrained to the
    cell through the finite-difference Jacobian, a merit-function line search,
    and the finishing evaluation at the cell's edge."""

    def test_margin_violation_and_edge_point(self):
        s = _sampler()
        lo, hi = s._cell_box((9, 6))               # u in [2.5, 3.5], v in [-0.5, 0.5]
        w = hi - lo
        assert s._derived_margin_violation(np.array([3.0, 0.0]), (9, 6), 0.02) == 0.0
        assert abs(s._derived_margin_violation(np.array([2.5, 0.0]), (9, 6), 0.02) - 0.02) < 1e-12
        assert abs(s._derived_margin_violation(np.array([2.0, 0.7]), (9, 6), 0.0) - 0.7) < 1e-12
        J = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        x = np.array([1.26, 0.0, 1.26, 0.0])       # u = 2.52: on the face of the cell shrunk by 0.02
        pt = s._edge_point(x, np.array([2.52, 0.0]), J, (9, 6), 0.02, 0.002)
        assert abs(_target(pt)[1][0] - (2.5 + 0.002 * w[0])) < 1e-9
        assert s._edge_point(np.array([1.5, 0.0, 1.5, 0.0]), np.array([3.0, 0.0]), J, (9, 6), 0.02, 0.002) is None

    def test_converges_on_the_face_from_inside(self):
        # The profile over (u, v) is -(u^2 + v^2) / 4; the maximum over the cell
        # u in [2.5, 3.5], v in [-0.5, 0.5] lies on its inner face, u = 2.5.
        s = _sampler()
        job = _cell_job(s, (9, 6), [1.5, 0.05, 1.5, 0.05])
        n = _drive(job)
        assert job.success and job.converged
        q = _target(job.current_params)[1]
        lo, hi = s._cell_box((9, 6))
        w = hi - lo
        assert lo[0] <= q[0] <= lo[0] + 0.03 * w[0] and abs(q[1]) < 0.05 * w[1]
        assert job.current_fitness > -(lo[0] + 0.02 * w[0]) ** 2 / 4 - 1e-4
        assert n < 60
        # the finishing evaluation sits just inside the true face
        nxt = job.on_finish(1)
        assert nxt is not None and nxt[0].type == 'DERIVED_SNAP'
        assert abs(_target(nxt[0].full_params)[1][0] - (lo[0] + 0.002 * w[0])) < 1e-6
        assert s.population[(9, 6)]['derived_jacobian'].shape == (2, 4)

    def test_start_outside_the_cell_is_pulled_in(self):
        s = _sampler()
        job = _cell_job(s, (9, 6), [2.0, 0.0, 2.0, 0.0])    # u = 4: one cell further out
        _drive(job)
        assert job.success
        q = _target(job.current_params)[1]
        lo, hi = s._cell_box((9, 6))
        assert lo[0] - 1e-6 <= q[0] <= hi[0] + 1e-6
        assert job.on_finish(1)[0].type == 'DERIVED_SNAP'

    def test_interior_optimum_is_unconstrained(self):
        s = _sampler()
        job = _cell_job(s, (6, 6), [0.3, -0.2, 0.1, 0.1])    # the cell around the origin
        _drive(job)
        assert job.success and job.converged
        assert np.linalg.norm(job.current_params) < 1e-3
        assert job.on_finish(1) is None                       # inside, not on a face: nothing to add


class TestSampleLog:
    def test_rows_carry_the_derived_values_and_warm_start_reads_them(self, tmp_path):
        from paraprof.sample_io import read_samples
        path = str(tmp_path / 'samples.csv')
        s = _sampler(samples_output_file=path)
        x = np.array([0.5, 0.5, 0.5, 0.5])
        val, d = _target(x)
        s._register_target_call(x, val, phase=3, derived=d)
        s._register_target_call(x + 1.0, -9.0, phase=3)                # a call without derived values
        s._flush_samples_buffer()
        s._sample_writer.close()
        rows = read_samples(path)
        assert rows.shape == (2, 4 + 2 + 2)
        assert rows[0, :4].tolist() == x.tolist() and rows[0, 4] == val and rows[0, 5] == 3.0
        assert rows[0, 6:].tolist() == d.tolist() and np.all(np.isnan(rows[1, 6:]))
        # a new derived-mode run seeds its initial maxima from the file without evaluating anything
        s2 = _sampler(warm_start_file=path)
        calls = s2.target_calls
        s2._initialize_from_warm_start_file(path)
        assert s2.target_calls == calls
        assert [m['target_val'] for m in s2.initial_maxima] == [val]
        assert s2._cached_derived_coords(x).tolist() == d.tolist()


class TestRecheckSelection:
    def test_every_roi_cell_is_rechecked_in_derived_mode(self):
        s = _sampler()
        base = {'profiled_params': np.zeros((1, 4)), 'fitnesses': np.array([0.0]), 'status': 'optimized',
                'improvement_history': [], 'last_update_gen': 0, 'optimizer_state': None}
        s.global_max_target_val = 0.0
        for idx, f in [((6, 6), 0.0), ((7, 6), -1.0), ((8, 6), -3.9), ((9, 6), -4.5)]:
            s.population[idx] = dict(base, best_fitness=f, fitnesses=np.array([f]))
        assert s._find_suspect_cells(0, []) == [(6, 6), (7, 6), (8, 6)]          # within roi_threshold = 4, best first
        assert set(s._find_suspect_cells(1, [(7, 6)])) == {(6, 6), (8, 6)}      # later waves: neighbours of the winners
        # the rechecked cells seed each other: cell (7, 6) gets its neighbours' points although they are rechecked too
        s.population[(6, 6)]['profiled_params'] = np.array([[1.0, 1.0, 1.0, 1.0]])
        s.population[(8, 6)]['profiled_params'] = np.array([[-1.0, 1.0, -1.0, 1.0]])
        jobs, _ = s.create_suspect_recheck_jobs(0, [], 0)
        job = [j for j in jobs if j.grid_idx == (7, 6)][0]
        assert len(job.candidate_seeds) >= 3


class TestRecheckPolishesOtherBasins:
    def test_distinct_seed_within_tolerance_is_polished(self):
        from paraprof.jobs.suspect_recheck_job import SuspectRecheckJob
        s = _sampler()
        own = np.array([0.5, 0.5, 0.5, 0.5])                       # u = v = 1: cell (7, 7)
        val = _target(own)[0]
        s.population[(7, 7)] = {'profiled_params': np.array([own]), 'fitnesses': np.array([val]), 'best_fitness': val,
                                'status': 'optimized', 'improvement_history': [], 'last_update_gen': 0, 'optimizer_state': None}
        near = own + np.array([0.01, -0.01, 0.0, 0.0])             # the same basin, a hair away
        other = np.array([3.0, 3.0, -2.0, -2.0])                   # u = v = 1 too, far away in input space
        worse = np.array([-3.0, -3.0, 4.0, 4.0])                   # u = v = 1, but far below the tolerance
        s.derived_recheck_max_polishes = 2                           # off by default
        job = SuspectRecheckJob(0, s, (7, 7), [own, near, other, worse], wave_number=0)
        fits = [val, val - 0.001, val - 0.5, val - 3.0]
        for i, (seed, f) in enumerate(zip(job.candidate_seeds, fits)):
            job.process_result({'target_val': f, 'derived': _target(seed)[1], 'params': seed, 'context': {'seed_idx': i}})
        chosen = job._seeds_to_polish()
        assert not job.will_update                                  # no seed beats the cell's own value
        assert len(chosen) == 1 and np.allclose(chosen[0][0], other)
        spawned, nxt = job.on_finish(5)
        assert spawned.type == 'SUSPECT_RECHECK_LBFGSB' and nxt == 6
        s.derived_recheck_max_polishes = 0
        assert job._seeds_to_polish() == [] and job.on_finish(5) is None


class TestOneDimensional:
    """A 1-D projection over one derived quantity: the same machinery with one axis."""

    def _sampler1d(self):
        bounds = np.array([[-5.0, 5.0]] * 4)
        proj = {'derived': ['u'], 'ranges': [[-6.0, 6.0]], 'grid_points': [12]}
        return ProfileProjector(target_func=_target, bounds=bounds, projections=[proj],
                                derived_names=['u', 'v'], pop_per_grid_point=2)

    def test_setup_and_cells(self):
        s = self._sampler1d()
        assert s.derived_mode and s.n_proj_dims == 1 and s.derived_idx == [0]
        assert len(s.grid_axes) == 1 and len(s.grid_axes[0]) == 13
        x = np.array([1.0, 2.0, 2.0, -1.0])            # u = 3 -> node 9; v is ignored
        s._register_target_call(x, _target(x)[0], derived=_target(x)[1])
        assert s._get_grid_indices_from_point(x) == (9,)
        lo, hi = s._cell_box((9,))
        assert lo.tolist() == [2.5] and hi.tolist() == [3.5]
        assert s._derived_cell_distance(np.array([4.0]), (9,)) == 0.25
        assert (9,) in s.derived_cell_best

    def test_polish_converges_on_the_face(self):
        # the profile over u is -u^2 / 4; the cell u in [2.5, 3.5] has its maximum at u = 2.5
        s = self._sampler1d()
        job = _cell_job(s, (9,), [1.5, 0.3, 1.5, -0.2])
        n = _drive(job)
        assert job.success and job.converged
        u = _target(job.current_params)[1][0]
        assert 2.5 <= u <= 2.5 + 0.03 and n < 60
        assert job.current_fitness > -(2.5 + 0.005) ** 2 / 4 - 1e-4
        nxt = job.on_finish(1)
        assert nxt is not None and abs(_target(nxt[0].full_params)[1][0] - (2.5 + 0.002)) < 1e-6


# --------------------------------------------------------------------------- #
# End-to-end under mpiexec: profile of a separable quadratic over (x0 + x2, x1 + x3)
# --------------------------------------------------------------------------- #
RUNNER = textwrap.dedent("""
    import json, sys
    import numpy as np
    from mpi4py import MPI
    from paraprof import ProfileProjector, run_scan, worker_main, set_log_level
    set_log_level('WARNING')

    def target(p):
        x = np.asarray(p, dtype=float)
        return -float(np.sum(x ** 2)) / 2.0, [x[0] + x[2], x[1] + x[3]]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    bounds = np.array([[-5.0, 5.0]] * 4)
    projections = [{'derived': ['u', 'v'], 'ranges': [[-6.0, 6.0], [-6.0, 6.0]], 'grid_points': [12, 12]}]
    np.random.seed(7)
    if rank == 0:
        with ProfileProjector(target_func=target, bounds=bounds, projections=projections,
                              derived_names=['u', 'v'], roi_threshold=4.0, pop_per_grid_point=3,
                              n_initial_optimizations=10, lbfgsb_max_iter=30) as sampler:
            results = run_scan(comm=comm, sampler=sampler, projections=projections, myrank=0)
            sol = results[0]['coarse_solution']
            out = {'calls': sampler.target_calls,
                   'cells': [[list(map(int, k)), float(v['likelihood'])] for k, v in sol['solutions'].items()],
                   'axes': [list(map(float, a)) for a in sol['grid_axes']]}
            print('RESULT ' + json.dumps(out))
    else:
        worker_main(comm, rank)
""")


@pytest.mark.skipif(shutil.which('mpiexec') is None, reason='mpiexec not on PATH')
def test_derived_projection_end_to_end():
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(RUNNER)
        path = f.name
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OMPI_', 'OPAL_', 'PMIX_', 'PMI_'))}
    try:
        proc = subprocess.run(['mpiexec', '--allow-run-as-root', '--oversubscribe', '-n', '4', sys.executable, path],
                              capture_output=True, text=True, timeout=300, env=env)
    finally:
        os.unlink(path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith('RESULT ')]
    assert line, proc.stdout
    out = json.loads(line[0][len('RESULT '):])
    axes = [np.array(a) for a in out['axes']]
    w = axes[0][1] - axes[0][0]
    # The profile over (u, v) is -(u^2 + v^2) / 4. A cell's value must lie between the
    # node value and the supremum over the node-centred cell, up to the polish tolerance.
    got = {tuple(k): v for k, v in out['cells']}
    roi = [(i, j) for i in range(13) for j in range(13) if -(axes[0][i] ** 2 + axes[1][j] ** 2) / 4 > -4.0]
    missing = [c for c in roi if c not in got]
    assert len(missing) <= 1, missing
    low, high = 0, 0
    for (i, j) in roi:
        if (i, j) not in got:
            continue
        node = -(axes[0][i] ** 2 + axes[1][j] ** 2) / 4
        du = max(abs(axes[0][i]) - 0.5 * w, 0.0)
        dv = max(abs(axes[1][j]) - 0.5 * w, 0.0)
        sup = -(du ** 2 + dv ** 2) / 4
        if got[(i, j)] < node - 0.05:
            low += 1
        if got[(i, j)] > sup + 1e-6:
            high += 1
    assert high == 0
    assert low <= max(1, len(roi) // 20), (low, len(roi))
