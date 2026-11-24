"""Tests for the L-BFGS-B job and its scheduling, run on virtual workers (no MPI)."""
import numpy as np
import pytest

from bench_scheduling import with_noise as noisy
from paraprof import ProfileProjector, get_logger
from paraprof.jobs import lbfgsb_job
from paraprof.jobs.lbfgsb_job import LBFGSBJob
from paraprof.jobs.noise_job import ecnoise
from paraprof.worker import evaluate_task
from virtual_pool import run_virtual_scan

CENTRE = np.array([0.45, 0.55, 0.50, 0.40])
SIGMA = np.array([0.10, 0.12, 0.05, 0.15])
_rng = np.random.default_rng(7)
_A = np.eye(4) + 0.3 * _rng.standard_normal((4, 4))
_A /= np.linalg.norm(_A, axis=1, keepdims=True)
M = (_A / SIGMA).T @ (_A / SIGMA)


def gauss4(x):
    d = np.asarray(x, dtype=float) - CENTRE
    return -10.0 - 0.5 * float(d @ M @ d)


def exact_profile(cell, axes, proj):
    prof = [i for i in range(4) if i not in proj]
    d = np.array([axes[0][cell[0]], axes[1][cell[1]]]) - CENTRE[proj]
    S = M[np.ix_(proj, proj)] - M[np.ix_(proj, prof)] @ np.linalg.solve(
        M[np.ix_(prof, prof)], M[np.ix_(prof, proj)])
    return -10.0 - 0.5 * float(d @ S @ d)


def plane(dims=(0, 1), grid=12, method='lbfgsb', **extra):
    return dict({'dims': list(dims), 'grid_points': [grid, grid], 'optimization_method': method},
                **extra)


def virtual_scan(projections, n_workers, target=gauss4, **kwargs):
    """A scan on virtual workers with the settings the tests share."""
    np.random.seed(3)
    settings = dict(jitter=0.05, seed=3, roi_threshold=4.0, n_initial_optimizations=6,
                    max_patching_waves=2, lbfgsb_max_iter=20)
    settings.update(kwargs)
    return run_virtual_scan(target, [(0.0, 1.0)] * 4, projections, n_workers, **settings)


def scan(target, n_workers, **advanced):
    return virtual_scan([plane()], n_workers, target,
                        advanced_config={'lbfgsb': advanced} if advanced else None)


def projection_errors(result):
    """Stored minus exact profile value over the cells inside the region of interest."""
    sol = result['coarse_solution']
    proj = list(sol['projection_dims'])
    err = []
    for cell, entry in sol['solutions'].items():
        ref = exact_profile(cell, sol['grid_axes'], proj)
        if ref > -10.0 - 4.0:
            err.append(entry['likelihood'] - ref)
    return np.array(err)


def roi_errors(out):
    return projection_errors(out['results'][-1])


@pytest.mark.parametrize('n_workers', [3, 40])
def test_scan_reaches_exact_profile(n_workers):
    out = scan(gauss4, n_workers)
    err = roi_errors(out)
    assert len(err) > 20
    assert np.all(err <= 1e-9)            # a stored value never exceeds the true profile
    assert err.min() > -1e-5
    assert abs(out['global_max'] + 10.0) < 1e-6
    assert all(st['status'] in ('converged', 'optimized') for st in out['population'].values())


def test_noisy_target_is_profiled_to_its_noise():
    amp = 1e-6
    out = scan(noisy(gauss4, amp), 3)
    err = roi_errors(out)
    assert err.min() > -100 * amp
    assert out['global_max'] > -10.0 - 100 * amp
    assert all(st['status'] in ('converged', 'optimized') for st in out['population'].values())


def test_more_workers_shorten_the_scan():
    few = scan(gauss4, 3)
    many = scan(gauss4, 60)
    assert many['wall'] < few['wall'] / 5.0
    assert many['mean_busy_workers'] > 10.0


@pytest.mark.parametrize('method', ['lbfgsb', 'de'])
def test_projection_without_profiled_parameters(method):
    # Every parameter is projected: each cell is one evaluation at its node.
    out = virtual_scan([{'dims': [0, 1, 2, 3], 'grid_points': [4] * 4,
                         'optimization_method': method}], 30)
    sol = out['results'][0]['coarse_solution']
    assert len(sol['solutions']) > 20
    for cell, entry in sol['solutions'].items():
        node = [axis[i] for axis, i in zip(sol['grid_axes'], cell)]
        assert entry['likelihood'] == gauss4(node)
    # The reported maximum is that of the multistart, which is not on a node.
    assert out['global_max'] >= max(e['likelihood'] for e in sol['solutions'].values())


# ---- one job, driven by hand ---------------------------------------------------------------
def make_sampler(target=gauss4, **advanced):
    cfg = {'fd_step': 1e-6}
    cfg.update(advanced)
    return ProfileProjector(target_func=target, bounds=[(0.0, 1.0)] * 4,
                            projections=[{'dims': [0, 1], 'grid_points': [4, 4]}],
                            advanced_config={'lbfgsb': cfg})


def drive(job, target, rng=None):
    """Run a job to the end. With `rng`, every evaluation the job offers ahead
    of time is evaluated as well, and results return in random order."""
    pending = list(job.start())
    while not job.is_finished():
        if rng is not None:
            pending.extend(iter(job.speculate, None))
            rng.shuffle(pending)
        assert pending, "job is waiting for a result that was never requested"
        result = evaluate_task(pending.pop(0), target, None, 1, get_logger())
        pending.extend(job.process_result(result))


def global_job(sampler, start):
    start = np.asarray(start, dtype=float)
    return LBFGSBJob(job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
                     opt_dims=(0, 1, 2, 3), start_params=start, grid_idx=None,
                     start_params_full=start)


@pytest.fixture
def no_hessian(monkeypatch):
    """Runs measure no Hessian, as runs over more than HESSIAN_MAX_DIMS parameters."""
    monkeypatch.setattr(lbfgsb_job, 'HESSIAN_MAX_DIMS', 0)


@pytest.mark.parametrize('hessian', [True, False])
def test_evaluations_sent_ahead_do_not_change_the_path(hessian, monkeypatch):
    if not hessian:
        monkeypatch.setattr(lbfgsb_job, 'HESSIAN_MAX_DIMS', 0)
    start = [0.9, 0.1, 0.8, 0.9]
    a = global_job(make_sampler(), start)
    drive(a, gauss4)
    b = global_job(make_sampler(), start)
    drive(b, gauss4, rng=np.random.default_rng(0))
    assert a.iteration == b.iteration
    np.testing.assert_allclose(a.current_params, b.current_params, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.current_params, CENTRE, atol=1e-4)


def test_first_trial_step_stays_inside_the_box(no_hessian):
    sampler = make_sampler()
    job = global_job(sampler, [0.9, 0.1, 0.8, 0.9])
    seen = []

    def target(x):
        seen.append(np.array(x, dtype=float))
        return gauss4(x)
    drive(job, target)
    pts = np.array(seen)
    # no point is evaluated twice, and none is a corner of the box
    assert len({p.tobytes() for p in pts}) == len(pts)
    assert not np.any(np.all((pts == 0.0) | (pts == 1.0), axis=1))


def test_probes_step_into_the_box_at_a_bound(no_hessian):
    # optimum outside the box in the last parameter: the run ends on the upper bound
    def target(x):
        d = np.asarray(x, dtype=float) - np.array([0.5, 0.5, 0.5, 1.3])
        return -float(d @ d) / 0.02
    sampler = make_sampler(target=target)
    job = global_job(sampler, [0.4, 0.6, 0.5, 0.7])
    seen = []

    def logged(x):
        seen.append(np.array(x, dtype=float))
        return target(x)
    drive(job, logged)
    np.testing.assert_allclose(job.current_params, [0.5, 0.5, 0.5, 1.0], atol=1e-4)
    assert job.converged
    assert len({p.tobytes() for p in seen}) == len(seen)
    assert np.all((np.array(seen) >= 0.0) & (np.array(seen) <= 1.0))


def test_endpoint_is_kept_when_no_step_improves(no_hessian):
    """On a target whose noise hides the slope, the run stops at its start
    point and reports it, instead of ending as a failure with nothing."""
    target = noisy(gauss4, 1e-3)
    sampler = make_sampler(target=target, fd_step=1e-8)
    start = CENTRE + 1e-4
    job = global_job(sampler, start)
    drive(job, target)
    assert job.success
    job.on_finish(next_job_id=1)
    assert len(sampler.initial_maxima) == 1
    assert sampler.global_max_target_val >= target(start) - 1e-12


# ---- sampler ---------------------------------------------------------------------------------
def test_grid_sizes_follow_their_dims_when_sorted():
    sampler = ProfileProjector(target_func=gauss4, bounds=[(0.0, 1.0)] * 4,
                               projections=[{'dims': [3, 1], 'grid_points': [4, 9]}])
    assert sampler.projection_dims == [1, 3]
    assert [len(a) for a in sampler.grid_axes] == [10, 5]


def test_fd_step_validation():
    from paraprof.exceptions import ConfigurationError
    for bad in (2.0, 'large', None):
        with pytest.raises(ConfigurationError):
            make_sampler(fd_step=bad)


# ---- noise estimate --------------------------------------------------------------------------
def test_ecnoise_recovers_a_known_noise_level():
    rng = np.random.default_rng(1)
    t = np.arange(9) * 1e-7
    for amp in (1e-3, 1e-6, 1e-9):
        f = 3.0 + 40.0 * t - 200.0 * t ** 2 + amp * rng.standard_normal(9)
        noise, inform = ecnoise(f)
        assert inform == 1
        assert amp / 4 < noise < amp * 4


def test_ecnoise_flags_a_dominating_trend():
    t = np.arange(9) * 0.1
    noise, inform = ecnoise(1.0 + 30.0 * t ** 3)
    assert inform == 3


def test_noise_stage_sets_the_step():
    amp = 1e-6
    np.random.seed(3)
    sampler = ProfileProjector(target_func=noisy(gauss4, amp), bounds=[(0.0, 1.0)] * 4,
                               projections=[{'dims': [0, 1], 'grid_points': [4, 4]}])
    jobs, _ = sampler.create_noise_estimate_job(0)
    job = jobs[0]
    drive(job, noisy(gauss4, amp))
    job.on_finish(1)
    assert amp / 5 < sampler.lbfgsb_noise < amp * 5
    # near sqrt(noise / curvature): well above the 1e-8 that suits double precision
    assert 1e-5 < sampler.lbfgsb_fd_step_value < 1e-3


# ---- initial points ------------------------------------------------------------------------
def seeded_scan(refine):
    seed = CENTRE + np.array([0.03, -0.02, 0.01, 0.04])     # near the maximum, on no grid node
    return virtual_scan([plane()], 3, initial_points=[seed], refine_initial_points=refine)


def test_initial_points_are_not_optimised_unless_asked():
    plain = seeded_scan(False)
    refined = seeded_scan(True)
    # Without refinement the maximum is the best point evaluated: the seed or a grid node.
    assert plain['global_max'] < -10.0 - 1e-3
    assert abs(refined['global_max'] + 10.0) < 1e-6
    assert 0 < refined['target_calls'] - plain['target_calls'] < 100
    for out in (plain, refined):
        assert roi_errors(out).min() > -1e-5


# ---- activation depth ----------------------------------------------------------------------
def test_activation_depth_keeps_the_cells_and_their_values():
    base = scan(gauss4, 200)
    deep = scan(gauss4, 200, activation_depth=3)
    # Cells activated ahead that stay outside the border are dropped: same cells, same profile.
    assert set(deep['population']) == set(base['population'])
    assert roi_errors(deep).min() > -1e-5
    assert abs(deep['global_max'] + 10.0) < 1e-6
    assert all(st['status'] in ('converged', 'optimized') for st in deep['population'].values())
    # The cells ahead cost evaluations.
    assert deep['target_calls'] > base['target_calls']


def test_activation_depth_does_nothing_without_idle_workers():
    base = scan(gauss4, 3)
    deep = scan(gauss4, 3, activation_depth=3)
    assert set(deep['population']) == set(base['population'])
    assert deep['target_calls'] < 1.05 * base['target_calls']


def test_activation_depth_validation():
    for bad in (0, -1, 1.5, True, 'two'):
        with pytest.raises(Exception, match='activation_depth'):
            make_sampler(activation_depth=bad)


# ---- projections started on idle workers (cross_projection eager_start) --------------------
PLANES = ([0, 1], [2, 3], [0, 2])


def multi_scan(eager, n_workers, method='lbfgsb', **kwargs):
    return virtual_scan([plane(d, method=method) for d in PLANES], n_workers,
                        advanced_config={'cross_projection': {'eager_start': eager}}, **kwargs)


@pytest.mark.parametrize('method', ['lbfgsb', 'de'])
def test_eager_start_gives_the_same_projections(method):
    seq = multi_scan(False, 200, method)
    eag = multi_scan(True, 200, method)
    assert len(eag['results']) == len(PLANES)
    for a, b, dims in zip(seq['results'], eag['results'], PLANES):
        assert list(b['coarse_solution']['projection_dims']) == sorted(dims)
        assert set(b['coarse_solution']['solutions']) == set(a['coarse_solution']['solutions'])
        err = projection_errors(b)
        assert len(err) >= 10
        assert err.min() > -1e-4 and err.max() <= 1e-9
    # The sampler is left holding the last projection, as after a sequential run.
    assert eag['projection_dims'] == sorted(PLANES[-1])
    assert set(eag['population']) == set(eag['results'][-1]['coarse_solution']['solutions'])
    assert eag['wall'] < 0.8 * seq['wall']


def test_eager_start_changes_little_on_a_busy_pool():
    seq = multi_scan(False, 3)
    eag = multi_scan(True, 3)
    assert eag['wall'] < 1.05 * seq['wall']
    assert eag['target_calls'] < 1.05 * seq['target_calls']


def test_eager_start_writes_each_projection_when_it_ends(monkeypatch, capsys):
    import paraprof.visualization
    plots = []

    def record(sampler, filename, plot_settings=None):
        plots.append((filename, sorted(int(d) for d in sampler.projection_dims),
                      sampler.target_calls))

    monkeypatch.setattr(paraprof.visualization, 'plot_profiles', record)
    out = multi_scan(True, 200, log_level='INFO', save_plots=True)
    log = capsys.readouterr().err

    # One plot per projection, made with the state of that projection in the sampler.
    names = [f"profile_plot_rank_0_dims_{'_'.join(map(str, d))}_coarse" for d in PLANES]
    assert sorted(name for name, _, _ in plots) == sorted(names)
    for name, dims, _ in plots:
        assert dims == sorted(PLANES[names.index(name)])

    # The first projection is written when it ends: before the last one, and with
    # evaluations of the scan still to come.
    order = [names.index(name) for name, _, _ in plots]
    calls_at_plot = {names.index(name): calls for name, _, calls in plots}
    assert order.index(0) < order.index(len(PLANES) - 1)
    assert calls_at_plot[0] < out['target_calls']
    assert out['results'][0]['metrics']['total_target_calls'] == calls_at_plot[0]
    for k in range(len(PLANES)):
        assert log.count(f"=== Completed Projection {k + 1}/{len(PLANES)} ===") == 1


def test_projections_with_refinement_are_not_started_early():
    out = virtual_scan(
        [plane([0, 1], 8), plane([2, 3], 8, grid_refinement_factor=2), plane([0, 2], 8)], 100,
        advanced_config={'cross_projection': {'eager_start': True}})
    assert len(out['results']) == 3
    assert out['results'][1]['refined_solution'] is not None
    for k in (0, 2):
        assert projection_errors(out['results'][k]).min() > -1e-4
    assert out['projection_dims'] == [0, 2]


def test_projection_state_lists_everything_the_reset_sets():
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(ProfileProjector._reset_for_new_projection)))
    assigned = set()
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for target in targets:
            for sub in ast.walk(target):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == 'self'):
                    assigned.add(sub.attr)
    assert assigned == set(ProfileProjector.PROJECTION_STATE_ATTRS)


def test_cells_ahead_do_not_outlive_the_scan_when_nothing_follows():
    # No patching: the cell loop is the last stage, and evaluations for dropped cells may
    # still be out when it ends. The master waits for them before the next projection.
    out = virtual_scan([plane(d, patch_coarse_grid=False) for d in PLANES[:2]], 200,
                       advanced_config={'lbfgsb': {'activation_depth': 3}})
    for res in out['results']:
        assert projection_errors(res).min() > -1e-4


# ---- branches: sub-ranges of a profiled parameter followed separately ----------------------
TWIN_FIRST = np.array([0.45, 0.55, 0.485, 0.30])    # the maximum of `twin`, in the first mode
SPLIT = {3: [0.5]}
CELL_JOB_TYPES = ('ACTIVATE_GRID_POINT', 'DE_GRID_POINT', 'LBFGSB', 'LBFGSB_LOOP',
                  'PATCHING_TEST', 'PATCHING_LBFGSB', 'REFINEMENT_LBFGSB',
                  'SUSPECT_RECHECK', 'SUSPECT_RECHECK_LBFGSB')


def make_twin(offset=-0.4, second_at=0.70, second_width=0.05):
    """Target with two modes in x3, at 0.30 and at `second_at`, and maximum -10.

    At x0 = 0.5 the second mode is `-offset` below the first, and it gains 5 per unit of x0:
    with the default offset it is the higher one for x0 above 0.58.
    """
    def twin(x):
        x0, x1, x2, x3 = np.asarray(x, dtype=float)
        first = -0.5 * (((x3 - 0.30) / 0.05) ** 2 + ((x2 - 0.50 - 0.3 * (x0 - 0.5)) / 0.06) ** 2)
        second = (-0.5 * (((x3 - second_at) / second_width) ** 2
                          + ((x2 - 0.45 + 0.2 * (x1 - 0.5)) / 0.06) ** 2)
                  + offset + 5.0 * (x0 - 0.5))
        level = -0.5 * (((x0 - 0.45) / 0.12) ** 2 + ((x1 - 0.55) / 0.15) ** 2)
        return -10.0 + level + float(np.logaddexp(first, second))
    return twin


twin = make_twin()


def numeric_profile(target, sol, starts, cells=None):
    """Profile value at the cells of an exported solution's grid (default: all of them): the
    best of one scipy optimisation of the profiled parameters per entry of `starts` (full
    points)."""
    from scipy.optimize import minimize
    proj, prof = list(sol['projection_dims']), list(sol['profiled_dims'])
    ref = {}
    for cell in (np.ndindex(*sol['grid_shape']) if cells is None else cells):
        x = np.empty(len(proj) + len(prof))
        x[proj] = [axis[i] for axis, i in zip(sol['grid_axes'], cell)]

        def neg(q):
            x[prof] = q
            return -target(x)
        ref[cell] = max(-minimize(neg, np.asarray(start)[prof], method='L-BFGS-B',
                                  bounds=[(0.0, 1.0)] * len(prof),
                                  options={'ftol': 1e-15, 'gtol': 1e-10}).fun
                        for start in starts)
    return ref


def branch_errors(sol, target=twin, roi=4.0, ref=None):
    """Stored minus reference profile value over the cells whose reference value is inside
    the region of interest. A cell the solution lacks counts as -inf."""
    if ref is None:
        # One start per mode of x3.
        ref = numeric_profile(target, sol, [TWIN_FIRST, [0.52, 0.55, 0.45, 0.70]])
    top = max(ref.values())
    return np.array([sol['solutions'][c]['likelihood'] - ref[c] if c in sol['solutions'] else -np.inf
                     for c in ref if ref[c] > top - roi])


def recorded(target):
    """The target, and the list its evaluated points are appended to."""
    points = []

    def func(x):
        points.append(np.array(x, dtype=float))
        return target(x)
    return func, points


def test_branches_report_the_higher_mode_in_every_cell():
    """The reproduction of the issue: `fit6` on the plane (m32, t12), where the two modes of
    the profiled t23 swap order across the grid."""
    import bench_scheduling as bench

    def fit6_scan(**kwargs):
        np.random.seed(1)
        return run_virtual_scan(
            bench.fit6, [(0.0, 1.0)] * 6,
            [{'dims': [2, 3], 'grid_points': [25, 25], 'optimization_method': 'lbfgsb'}], 40,
            jitter=0.05, seed=1, roi_threshold=6.0, lbfgsb_max_iter=20, use_clustering=False,
            max_patching_waves=2, n_initial_optimizations=8,
            advanced_config={'lbfgsb': {'ftol': 1e-4}}, **kwargs)

    single = fit6_scan()
    split = fit6_scan(branches={5: [0.51]})                  # t23: modes at 0.40 and at 0.62
    # Reference at the cells either scan explored, from a start in each mode of t23 and at
    # two values of the periodic phase.
    sols = [out['results'][0]['coarse_solution'] for out in (single, split)]
    starts = [[phase, 0.45, 0.5, 0.52, 0.47, t23] for phase in (0.65, 0.15) for t23 in (0.40, 0.62)]
    ref = numeric_profile(bench.fit6, sols[0], starts,
                          cells=sorted(set(sols[0]['solutions']) | set(sols[1]['solutions'])))

    def n_low(sol):
        err = branch_errors(sol, roi=6.0, ref=ref)
        assert len(err) > 60 and err.max() < 1e-6
        return int(np.sum(err < -0.01))

    # Without the split, part of the plane holds the lower mode.
    assert n_low(sols[0]) >= 10
    assert n_low(sols[1]) == 0
    assert split['evaluations'] < 2.0 * single['evaluations']


@pytest.mark.parametrize('method, n_workers, population',
                         [('lbfgsb', 3, 3), ('lbfgsb', 200, 3), ('lbfgsb', 40, 8), ('de', 40, 3),
                          ('de', 40, 8)])
def test_branches_on_a_target_with_two_modes(method, n_workers, population):
    # From a population of 4 the activation samples include points of the pool, which holds
    # both modes.
    out = virtual_scan([plane(method=method)], n_workers, target=twin, branches=SPLIT,
                       pop_per_grid_point=population)
    sol = out['results'][0]['coarse_solution']
    err = branch_errors(sol)
    assert len(err) > 50
    assert err.min() > -1e-4 and err.max() < 1e-6
    assert all(st['status'] in ('converged', 'optimized') for st in out['population'].values())
    assert set(out['population']) == set(sol['solutions'])

    # Each solution names its branch and is the highest of the per-branch values, and the
    # split parameter stays inside the sub-range of its branch.
    first, second = sol['branches']
    assert first['ranges'] == {3: (0.0, 0.5)} and second['ranges'] == {3: (0.5, 1.0)}
    assert len(first['solutions']) > 20 and len(second['solutions']) > 20
    assert set(sol['solutions']) == set(first['solutions']) | set(second['solutions'])
    for cell, entry in sol['solutions'].items():
        values = [b['solutions'][cell]['likelihood'] if cell in b['solutions'] else -np.inf
                  for b in sol['branches']]
        assert entry['likelihood'] == max(values)
        assert values[entry['branch']] == entry['likelihood']
    x3 = sol['profiled_dims'].index(3)
    assert all(e['profiled_params'][x3] <= 0.5 for e in first['solutions'].values())
    assert all(e['profiled_params'][x3] >= 0.5 for e in second['solutions'].values())
    assert {e['branch'] for e in sol['solutions'].values()} == {0, 1}


def test_branch_without_a_known_point_gets_its_own_optimisation():
    # The only initial point lies in the first mode, and there is no multistart.
    settings = dict(target=twin, initial_points=[TWIN_FIRST])
    single = virtual_scan([plane()], 40, **settings)
    split = virtual_scan([plane()], 40, branches=SPLIT, **settings)
    # The single run never sees the second mode.
    assert np.sum(branch_errors(single['results'][0]['coarse_solution']) < -0.01) > 10
    assert 'BRANCH_SEED' not in single['timeline'][3]
    err = branch_errors(split['results'][0]['coarse_solution'])
    assert err.min() > -1e-4 and err.max() < 1e-6
    assert 0 < split['timeline'][3].count('BRANCH_SEED') < 100


@pytest.mark.parametrize('split', [0.4, 0.5, 0.8])
def test_branch_start_does_not_rely_on_the_mirrored_point(split):
    """The second mode is narrow and not where the first one mirrors to. From the mirrored
    point an optimisation runs back to the split point, so the start is taken from the
    values along x3 instead."""
    target = make_twin(second_at=0.90, second_width=0.02)
    out = virtual_scan([plane()], 40, target=target, branches={3: [split]},
                       initial_points=[TWIN_FIRST])
    sol = out['results'][0]['coarse_solution']
    ref = numeric_profile(target, sol, [TWIN_FIRST, [0.52, 0.55, 0.45, 0.90]])
    err = branch_errors(sol, ref=ref)
    assert len(err) > 50 and err.min() > -1e-4 and err.max() < 1e-6
    assert all(len(b['solutions']) > 20 for b in sol['branches'])
    assert out['timeline'][3].count('BRANCH_SEED') < 100


def test_line_scan_picks_a_local_maximum_away_from_the_near_end():
    from paraprof.jobs.branch_scan_job import BranchScanJob, LINE_POINTS
    sampler = ProfileProjector(target_func=twin, bounds=[(0.0, 1.0)] * 4, branches=SPLIT,
                               projections=[{'dims': [0, 1], 'grid_points': [4, 4]}])
    upper = sampler.projection_branches[1]

    def scan(profile):
        job = BranchScanJob(0, sampler, upper, TWIN_FIRST)
        tasks = job.start()
        assert len(tasks) == LINE_POINTS + 1                 # the mirrored value is one of them
        for task in tasks:
            x = task['params']
            assert 0.5 <= x[3] <= 1.0 and np.array_equal(x[:3], TWIN_FIRST[:3])
            job.process_result({'target_val': profile(x[3]), 'context': task['context']})
        assert job.is_finished()
        return job.choose_start()

    # A bump inside the sub-range wins over the higher values next to the split point.
    point, value = scan(lambda t: max(-100.0 * (t - 0.3) ** 2, -8.0 - 400.0 * (t - 0.86) ** 2))
    assert abs(point[3] - 0.86) < 0.5 / LINE_POINTS and -8.1 < value <= -8.0
    # Falling all the way: the mirrored point.
    point, value = scan(lambda t: -100.0 * (t - 0.3) ** 2)
    assert point[3] == pytest.approx(0.7) and value == pytest.approx(-16.0)
    # Rising to the far end counts as a maximum there.
    point, value = scan(lambda t: 3.0 * t)
    assert point[3] == pytest.approx(1.0 - 0.25 / LINE_POINTS)
    sampler.close()


def test_confined_run_below_the_region_of_interest_is_continued():
    from paraprof.sampler import BRANCH_SEED_MAX_RUNS
    sampler = ProfileProjector(target_func=twin, bounds=[(0.0, 1.0)] * 4, branches=SPLIT,
                               projections=[{'dims': [0, 1], 'grid_points': [4, 4]}],
                               lbfgsb_max_iter=1, roi_threshold=0.5,
                               advanced_config={'lbfgsb': {'fd_step': 1e-6}})
    sampler.global_max_target_val = -10.0
    sampler.initial_maxima = [{'point': TWIN_FIRST, 'target_val': -10.0}]
    sampler.create_branch_seed_jobs(0)
    upper = sampler.projection_branches[1]

    # One iteration from far away ends below the region of interest (-10.5), not converged.
    job = sampler.branch_seed_run(0, upper, [0.95, 0.05, 0.95, 0.98])
    values = []
    while job is not None:
        first_tasks = job.start()
        if values:                                           # a continuation knows its start value
            assert all(t['context']['sub_type'] != 'LBFGS_INITIAL_F' for t in first_tasks)
            assert job.current_fitness == values[-1]
        pending = list(first_tasks)
        while not job.is_finished():
            pending.extend(job.process_result(
                evaluate_task(pending.pop(0), twin, None, 1, get_logger())))
        values.append(job.current_fitness)
        spawned = job.on_finish(job.id + 1)
        job = spawned[0] if spawned else None
    assert 2 <= len(values) <= BRANCH_SEED_MAX_RUNS
    assert values == sorted(values) and values[0] < -10.5 < values[-1]
    # The last endpoint is the initial maximum of the branch.
    assert [m['target_val'] for m in sampler.branch_seeds[1]] == [values[-1]]
    assert sampler.branch_seeds[0][0]['target_val'] == -10.0
    sampler.close()


def test_split_point_as_the_best_start_of_a_branch_is_reported(capsys):
    # No mode above x3 = 0.2 except the one at 0.30, and none below: the lower branch can
    # only start on the split point.
    out = virtual_scan([plane()], 40, target=make_twin(offset=-30.0), branches={3: [0.2]},
                       log_level='WARNING')
    assert 'lies on the split point 0.2 of parameter 3' in capsys.readouterr().err
    assert branch_errors(out['results'][0]['coarse_solution'],
                         make_twin(offset=-30.0)).min() > -1e-4


def test_population_larger_than_the_pool():
    # A quarter of the population comes from the pool. With fewer points in it, or in the
    # part of it inside a branch, the rest are random samples.
    out = virtual_scan([plane(grid=6)], 10, initial_points=[CENTRE], pop_per_grid_point=12)
    assert roi_errors(out).min() > -1e-5
    out = virtual_scan([plane()], 40, target=twin, branches={3: [0.5], 2: [0.5]},
                       pop_per_grid_point=8)
    assert branch_errors(out['results'][0]['coarse_solution']).min() > -1e-4


def test_refinement_start_points_are_kept_inside_the_branch():
    sampler = ProfileProjector(target_func=twin, bounds=[(0.0, 1.0)] * 4, branches=SPLIT,
                               projections=[{'dims': [0, 1], 'grid_points': [4, 4]}])
    just_outside = np.array([0.4, np.nextafter(0.5, 0.0)])   # (x2, x3) one rounding step low

    class Interpolator:
        def interpolate(self, coords):
            return just_outside
    sampler.refinement_interpolator = Interpolator()
    plain = sampler._get_refined_initialization_candidates(np.array([0.5, 0.5]))
    assert plain[0]['profiled_params'][1] < 0.5              # no branch: taken as it is
    sampler.enter_branch(1)
    confined = sampler._get_refined_initialization_candidates(np.array([0.5, 0.5]))
    np.testing.assert_array_equal(confined[0]['profiled_params'], [0.4, 0.5])
    sampler.close()


def test_confined_optimisation_runs_once_per_scan():
    # Without seeding from the pool the second projection runs a multistart of its own. The
    # branch found in the first projection is followed again, from the point found there.
    out = virtual_scan([plane(), plane([1, 2])], 40, target=twin, branches=SPLIT,
                       initial_points=[TWIN_FIRST],
                       advanced_config={'cross_projection': {'pool_seeded_initial_maxima': False}})
    types = out['timeline'][3]
    seed_runs = [i for i, t in enumerate(types) if t == 'BRANCH_SEED']
    assert seed_runs and max(seed_runs) < types.index('INITIAL_OPTIMIZATION')
    for res in out['results']:
        sol = res['coarse_solution']
        assert all(len(b['solutions']) > 5 for b in sol['branches'])
        assert branch_errors(sol).min() > -1e-4


def test_branch_outside_the_region_of_interest_is_not_followed():
    low = make_twin(offset=-9.0)                             # second mode: at best -19.07
    target, points = recorded(low)
    single = virtual_scan([plane()], 40, target=low)
    split = virtual_scan([plane()], 40, target=target, branches=SPLIT)
    sol = split['results'][0]['coarse_solution']
    assert sol['branches'][1]['solutions'] == {}
    assert {e['branch'] for e in sol['solutions'].values()} == {0}
    assert set(sol['solutions']) == set(single['results'][0]['coarse_solution']['solutions'])
    assert branch_errors(sol, low).min() > -1e-4

    # One optimisation looked into the second sub-range. Every evaluation for a cell is
    # inside the first.
    types = split['timeline'][3]
    assert len(types) == len(points)
    x3 = {kind: [p[3] for p, t in zip(points, types) if t == kind] for kind in set(types)}
    assert min(x3['BRANCH_SEED']) >= 0.5
    cell_x3 = [v for kind in CELL_JOB_TYPES for v in x3.get(kind, [])]
    assert len(cell_x3) > 500 and max(cell_x3) <= 0.5


def test_projection_with_the_split_parameter_as_an_axis_is_unchanged():
    runs = []
    for kwargs in ({}, {'branches': SPLIT}):
        target, points = recorded(twin)
        out = virtual_scan([plane([0, 3])], 40, target=target, **kwargs)
        runs.append((np.array(points), out['results'][0]['coarse_solution']))
    (plain_points, plain_sol), (split_points, split_sol) = runs
    assert np.array_equal(plain_points, split_points)        # same points, same order
    assert 'branches' not in split_sol
    assert all('branch' not in e for e in split_sol['solutions'].values())


def test_projection_can_set_or_drop_the_branches():
    planes = [plane(), plane([1, 2])]
    # The projection's own setting replaces the global one; an empty one turns it off.
    out = virtual_scan([dict(planes[0], branches={}), planes[1]], 40, target=twin, branches=SPLIT)
    assert 'branches' not in out['results'][0]['coarse_solution']
    assert len(out['results'][1]['coarse_solution']['branches']) == 2
    out = virtual_scan([dict(planes[0], branches={3: [0.45, 0.55]}), planes[1]], 40, target=twin)
    sol = out['results'][0]['coarse_solution']
    assert [b['ranges'][3] for b in sol['branches']] == [(0.0, 0.45), (0.45, 0.55), (0.55, 1.0)]
    assert sol['branches'][1]['solutions'] == {}             # no mode between 0.45 and 0.55
    assert branch_errors(sol).min() > -1e-4
    assert 'branches' not in out['results'][1]['coarse_solution']


@pytest.mark.parametrize('n_workers', [3, 200])
def test_branches_over_several_projections_and_with_eager_start(n_workers):
    planes = [plane([0, 1]), plane([1, 2]), plane([0, 3])]
    seq, eag = (virtual_scan(planes, n_workers, target=twin, branches=SPLIT,
                             advanced_config={'cross_projection': {'eager_start': eager}})
                for eager in (False, True))
    first, second, axis = (res['coarse_solution'] for res in seq['results'])
    for sol in (first, second):
        # The second projection takes the initial maxima of each branch from the pool.
        assert all(len(b['solutions']) > 5 for b in sol['branches'])
        err = branch_errors(sol)
        assert len(err) > 30 and err.min() > -1e-4 and err.max() < 1e-6
    assert 'branches' not in axis                            # x3 is an axis there
    assert branch_errors(axis).min() > -1e-4
    # Started on idle workers, the projections hold the same cells and values.
    for a, b in zip(seq['results'], eag['results']):
        a, b = a['coarse_solution'], b['coarse_solution']
        assert set(a['solutions']) == set(b['solutions'])
        assert ('branches' in a) == ('branches' in b)
        assert max(abs(a['solutions'][c]['likelihood'] - b['solutions'][c]['likelihood'])
                   for c in a['solutions']) < 1e-5
    # The sampler is left holding the last projection, which has no branches.
    for out in (seq, eag):
        assert out['projection_dims'] == [0, 3]
        assert set(out['population']) == set(out['results'][-1]['coarse_solution']['solutions'])
    if n_workers == 200:
        assert eag['wall'] < 0.9 * seq['wall']


@pytest.mark.parametrize('direct', [False, True])
def test_branches_with_grid_refinement(direct):
    out = virtual_scan([plane(grid=6, grid_refinement_factor=2, patch_refined_grid=True)], 40,
                       target=twin, branches=SPLIT, refinement_direct_eval=direct)
    coarse, refined = out['results'][0]['coarse_solution'], out['results'][0]['refined_solution']
    assert refined['grid_shape'] == (13, 13)
    for sol in (coarse, refined):
        assert all(len(b['solutions']) > 5 for b in sol['branches'])
        assert {e['branch'] for e in sol['solutions'].values()} == {0, 1}
    # Each branch is refined from its own coarse cells, so its fine cells start in its mode.
    err = branch_errors(refined)
    assert len(err) > 50 and err.max() < 1e-6
    assert err.min() > (-0.05 if direct else -1e-4)
    assert out['projection_dims'] == [0, 1] and set(out['population']) >= set(refined['solutions'])


def test_branch_seeds_are_read_per_branch_from_the_pool_and_the_file(tmp_path):
    """The pool and a warm-start file give one initial maximum per cell. Two points of
    different branches in one cell both have to arrive in their branch."""
    from paraprof.sample_io import create_sample_writer
    first = np.array([0.50, 0.50, 0.50, 0.30])
    second = np.array([0.51, 0.49, 0.50, 0.70])              # same cell of the plane, lower
    path = str(tmp_path / 'samples.csv')
    writer = create_sample_writer(path)
    writer.write_batch(np.array([list(first) + [-10.0, 1.0], list(second) + [-10.5, 1.0]]))
    writer.close()

    for source in ('pool', 'file'):
        sampler = ProfileProjector(target_func=twin, bounds=[(0.0, 1.0)] * 4, branches=SPLIT,
                                   projections=[{'dims': [0, 1], 'grid_points': [4, 4]}],
                                   warm_start_file=path)
        if source == 'pool':
            sampler._update_global_pool(first, -10.0, None)
            sampler._update_global_pool(second, -10.5, None)
            sampler._initialize_from_global_pool()
        else:
            sampler._initialize_from_warm_start_file(path)
        assert [m['target_val'] for m in sampler.initial_maxima] == [-10.0]
        jobs, _ = sampler.create_branch_seed_jobs(0, from_pool=source == 'pool',
                                                  from_file=source == 'file')
        assert jobs == []                                    # both branches have a start
        best = [seeds[0] for seeds in sampler.branch_seeds]
        assert [m['target_val'] for m in best] == [-10.0, -10.5]
        np.testing.assert_array_equal(best[1]['point'], second)
        # Without the second read the branch has no start, and one is looked for: around
        # the best point mirrored about the split point.
        jobs, _ = sampler.create_branch_seed_jobs(0)
        assert [job.type for job in jobs] == ['BRANCH_SEED'] and jobs[0].branch['index'] == 1
        np.testing.assert_allclose(jobs[0].reference, [0.50, 0.50, 0.50, 0.70])
        sampler.close()


def test_two_split_parameters_give_one_branch_per_combination():
    out = virtual_scan([plane()], 40, target=twin, branches={3: [0.5], 2: [0.5]})
    sol = out['results'][0]['coarse_solution']
    assert [b['ranges'] for b in sol['branches']] == [
        {2: (0.0, 0.5), 3: (0.0, 0.5)}, {2: (0.0, 0.5), 3: (0.5, 1.0)},
        {2: (0.5, 1.0), 3: (0.0, 0.5)}, {2: (0.5, 1.0), 3: (0.5, 1.0)}]
    err = branch_errors(sol)
    assert err.min() > -1e-4 and err.max() < 1e-6


def test_branches_validation():
    from paraprof.exceptions import ConfigurationError, InvalidProjectionError

    def make(branches=None, projection=None, **kwargs):
        return ProfileProjector(
            target_func=gauss4, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-2.0, 2.0)],
            projections=[dict({'dims': [0, 1], 'grid_points': [4, 4]}, **(projection or {}))],
            branches=branches, **kwargs)

    for bad in ([0.5], {3: [2.0]}, {3: [-2.0]}, {3: ['a']}, {3: None}, {4: [0.5]}, {'x3': [0.5]},
                {True: [0.5]}, {3: [float('nan')]}):
        with pytest.raises(ConfigurationError, match='branches'):
            make(bad)
    # A projection cannot split one of its own axes; the global setting skips such a projection.
    with pytest.raises(InvalidProjectionError, match='axis'):
        make(projection={'branches': {1: [0.5]}})
    assert make({1: [0.5]}).projection_branches is None

    names = ['a', 'b', 'c', 'd']
    sampler = make({'d': [1.0, -1.0, 1.0], 'c': 0.25}, parameter_names=names)
    assert sampler.branches == {3: [-1.0, 1.0], 2: [0.25]}
    assert [b['ranges'] for b in sampler.projection_branches] == [
        {2: (0.0, 0.25), 3: (-2.0, -1.0)}, {2: (0.0, 0.25), 3: (-1.0, 1.0)},
        {2: (0.0, 0.25), 3: (1.0, 2.0)}, {2: (0.25, 1.0), 3: (-2.0, -1.0)},
        {2: (0.25, 1.0), 3: (-1.0, 1.0)}, {2: (0.25, 1.0), 3: (1.0, 2.0)}]
    np.testing.assert_array_equal(sampler.projection_branches[1]['bounds'],
                                  [(0.0, 1.0), (0.0, 1.0), (0.0, 0.25), (-1.0, 1.0)])
    assert sampler.branch is None and sampler.active_bounds is sampler.bounds
    assert make({3: []}).projection_branches is None


def test_job_confined_to_part_of_a_range():
    """A run with its own bounds stays inside them and keeps the finite-difference step of
    the full range."""
    from scipy.optimize import minimize
    sampler = make_sampler()
    box = np.array([(0.0, 1.0)] * 3 + [(0.6, 1.0)])          # the maximum has x3 = 0.40
    start = np.array([0.9, 0.1, 0.8, 0.9])
    job = LBFGSBJob(job_id=0, job_type='BRANCH_SEED', sampler=sampler, opt_dims=(0, 1, 2, 3),
                    start_params=start, grid_idx=None, start_params_full=start, bounds=box)
    target, points = recorded(gauss4)
    drive(job, target, rng=np.random.default_rng(0))
    points = np.array(points)
    assert np.all((points >= box[:, 0]) & (points <= box[:, 1]))
    expected = minimize(lambda x: -gauss4(x), start, method='L-BFGS-B', bounds=box,
                        options={'ftol': 1e-15, 'gtol': 1e-10})
    assert job.converged
    np.testing.assert_allclose(job.current_params, expected.x, atol=1e-4)
    assert job.current_params[3] == 0.6
    # Gradient probe of x3 at the start point: fd_step (1e-6) of the full range, not of 0.4.
    probes = points[np.all(points[:, :3] == start[:3], axis=1) & (points[:, 3] != start[3])]
    assert np.min(np.abs(probes[:, 3] - start[3])) == pytest.approx(1e-6, rel=1e-6)
