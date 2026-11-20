"""Tests for the L-BFGS-B job and its scheduling, run on virtual workers (no MPI)."""
import os
import sys
import zlib

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.jobs.lbfgsb_job import LBFGSBJob
from paraprof.jobs.noise_job import ecnoise

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from virtual_pool import run_virtual_scan  # noqa: E402

CENTRE = np.array([0.45, 0.55, 0.50, 0.40])
SIGMA = np.array([0.10, 0.12, 0.05, 0.15])
_rng = np.random.default_rng(7)
_A = np.eye(4) + 0.3 * _rng.standard_normal((4, 4))
_A /= np.linalg.norm(_A, axis=1, keepdims=True)
M = (_A / SIGMA).T @ (_A / SIGMA)


def gauss4(x):
    d = np.asarray(x, dtype=float) - CENTRE
    return -10.0 - 0.5 * float(d @ M @ d)


def noisy(func, amp):
    """Add a deterministic pseudo-random term of rms `amp`."""
    def f(x):
        x = np.asarray(x, dtype=float)
        u = zlib.crc32(x.tobytes()) / 2 ** 32
        return func(x) + amp * np.sqrt(12.0) * (u - 0.5)
    return f


def exact_profile(cell, axes, proj):
    prof = [i for i in range(4) if i not in proj]
    d = np.array([axes[0][cell[0]], axes[1][cell[1]]]) - CENTRE[proj]
    S = M[np.ix_(proj, proj)] - M[np.ix_(proj, prof)] @ np.linalg.solve(
        M[np.ix_(prof, prof)], M[np.ix_(prof, proj)])
    return -10.0 - 0.5 * float(d @ S @ d)


def scan(target, n_workers, **advanced):
    np.random.seed(3)
    return run_virtual_scan(
        target, [(0.0, 1.0)] * 4,
        [{'dims': [0, 1], 'grid_points': [12, 12], 'optimization_method': 'lbfgsb'}],
        n_workers, jitter=0.05, seed=3, roi_threshold=4.0, n_initial_optimizations=6,
        max_patching_waves=2, lbfgsb_max_iter=20,
        advanced_config={'lbfgsb': advanced} if advanced else None)


def roi_errors(out):
    proj = list(out['projection_dims'])
    err = []
    for cell, st in out['population'].items():
        ref = exact_profile(cell, out['grid_axes'], proj)
        if ref > -10.0 - 4.0:
            err.append(st['best_fitness'] - ref)
    return np.array(err)


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


def evaluate(task, target):
    ctx = dict(task['context'])
    ctx['worker_rank'] = 1
    return {'target_val': float(target(task['params'])), 'params': task['params'],
            'context': ctx, 'error': None, 'user_gradient': None, 'user_gradient_error': None}


def drive(job, target, speculate=False, rng=None):
    """Run a job to the end. With `speculate`, every evaluation the job offers
    ahead of time is evaluated as well, and results return in random order."""
    pending = list(job.start())
    n = 0
    while not job.is_finished():
        if speculate:
            while True:
                extra = job.speculate()
                if extra is None:
                    break
                pending.append(extra)
            if rng is not None:
                rng.shuffle(pending)
        assert pending, "job is waiting for a result that was never requested"
        task = pending.pop(0)
        n += 1
        pending.extend(job.process_result(evaluate(task, target)))
    return n


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
    drive(b, gauss4, speculate=True, rng=np.random.default_rng(0))
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
    job = LBFGSBJob(job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
                    opt_dims=(0, 1, 2, 3), start_params=start, grid_idx=None,
                    start_params_full=start)
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
    out = scan(noisy(gauss4, amp), 3)
    assert out['evaluations'] > 0
    np.random.seed(3)
    sampler = ProfileProjector(target_func=noisy(gauss4, amp), bounds=[(0.0, 1.0)] * 4,
                               projections=[{'dims': [0, 1], 'grid_points': [4, 4]}])
    jobs, _ = sampler.create_noise_estimate_job(0)
    job = jobs[0]
    pending = list(job.start())
    while pending:
        pending.extend(job.process_result(evaluate(pending.pop(0), noisy(gauss4, amp))))
    assert job.is_finished()
    job.on_finish(1)
    assert amp / 5 < sampler.lbfgsb_noise < amp * 5
    # near sqrt(noise / curvature): well above the 1e-8 that suits double precision
    assert 1e-5 < sampler.lbfgsb_fd_step_value < 1e-3
