"""Tests for the L-BFGS-B job and its scheduling, run on virtual workers (no MPI)."""
import numpy as np
import pytest

from bench_scheduling import with_noise as noisy
from paraprof import ProfileProjector, get_logger
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


def test_legacy_options_still_converge():
    out = scan(gauss4, 3, fd_step=None, hessian=False, speculative=False,
               rolling_activation=False, eager_activation=False)
    err = roi_errors(out)
    assert err.min() > -1e-5


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


@pytest.mark.parametrize('hessian', [True, False])
def test_evaluations_sent_ahead_do_not_change_the_path(hessian):
    start = [0.9, 0.1, 0.8, 0.9]
    a = global_job(make_sampler(hessian=hessian), start)
    drive(a, gauss4)
    b = global_job(make_sampler(hessian=hessian), start)
    drive(b, gauss4, rng=np.random.default_rng(0))
    assert a.iteration == b.iteration
    np.testing.assert_allclose(a.current_params, b.current_params, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.current_params, CENTRE, atol=1e-4)


def test_first_trial_step_stays_inside_the_box():
    sampler = make_sampler(hessian=False)
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


def test_probes_step_into_the_box_at_a_bound():
    # optimum outside the box in the last parameter: the run ends on the upper bound
    def target(x):
        d = np.asarray(x, dtype=float) - np.array([0.5, 0.5, 0.5, 1.3])
        return -float(d @ d) / 0.02
    sampler = make_sampler(target=target, hessian=False)
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


def test_endpoint_is_kept_when_no_step_improves():
    """On a target whose noise hides the slope, the run stops at its start
    point and reports it, instead of ending as a failure with nothing."""
    target = noisy(gauss4, 1e-3)
    sampler = make_sampler(target=target, hessian=False, fd_step=1e-8)
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
    with pytest.raises(ConfigurationError):
        make_sampler(fd_step=2.0)
    with pytest.raises(ConfigurationError):
        make_sampler(fd_step='large')
    assert make_sampler(fd_step=None).lbfgsb_fd_step_value is None


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
