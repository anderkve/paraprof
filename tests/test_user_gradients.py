"""Tests for the user-supplied gradient feature.

Drives the L-BFGS-B state machine without MPI by acting as a fake worker:
each emitted task is evaluated locally, the result is fed back into the
job, and the cycle repeats until the job finishes.
"""
import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import ConfigurationError
from paraprof.jobs.lbfgsb_job import LBFGSBJob
from paraprof.worker import _normalize_user_gradient


class TestNormalizeUserGradient:
    def test_array_with_nan(self):
        arr, err = _normalize_user_gradient([1.0, np.nan, 3.0], 3)
        assert err is None
        assert np.isfinite(arr[0]) and np.isfinite(arr[2])
        assert np.isnan(arr[1])

    def test_shape_mismatch(self):
        arr, err = _normalize_user_gradient([1.0, 2.0], 3)
        assert arr is None
        assert err and "expected 3" in err

    def test_dict_partial(self):
        arr, err = _normalize_user_gradient({0: 1.5, 2: -0.5}, 4)
        assert err is None
        assert arr[0] == 1.5 and arr[2] == -0.5
        assert np.isnan(arr[1]) and np.isnan(arr[3])


class TestSamplerGradFuncArg:
    def test_rejects_non_callable(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        with pytest.raises(ConfigurationError):
            ProfileProjector(
                target_func=simple_2d_function,
                bounds=simple_bounds_2d,
                projections=[basic_projection_2d],
                grad_func="not callable",
            )


def _evaluate_task(task, target_func, grad_func, n_dims):
    """Mimic worker_main's per-task handling."""
    params = task['params']
    ctx = dict(task['context'])
    ctx['worker_rank'] = 1
    user_gradient = None
    user_gradient_error = None
    if ctx.get('compute_gradient') and grad_func is not None:
        try:
            user_gradient, user_gradient_error = _normalize_user_gradient(
                grad_func(params), n_dims)
        except Exception as e:
            user_gradient_error = f"{type(e).__name__}: {e}"
    return {
        'target_val': float(target_func(params)),
        'params': params,
        'context': ctx,
        'error': None,
        'user_gradient': user_gradient,
        'user_gradient_error': user_gradient_error,
    }


def run_initial_opt_job(sampler, start, target_func, grad_func=None,
                        max_iter=15, ftol=1e-9):
    """Drive an INITIAL_OPTIMIZATION job to completion.
    Returns (job, n_target_evals, fd_tasks_issued)."""
    sampler.lbfgsb_max_iter = max_iter
    sampler.lbfgsb_ftol = ftol
    n_dims = sampler.dims
    start = np.asarray(start, dtype=float)
    job = LBFGSBJob(
        job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
        opt_dims=tuple(range(n_dims)), start_params=start,
        grid_idx=None, start_params_full=start,
    )
    n_evals = 0
    fd_tasks = 0
    tasks = job.start()
    while tasks:
        new_tasks = []
        for t in tasks:
            if t['context'].get('sub_type') == 'LBFGS_GRADIENT':
                fd_tasks += 1
            new_tasks.extend(job.process_result(
                _evaluate_task(t, target_func, grad_func, n_dims)))
            n_evals += 1
        tasks = new_tasks
        if job.is_finished():
            break
    job.on_finish(next_job_id=1)
    return job, n_evals, fd_tasks


def _sphere(params):
    return -float(np.sum(np.asarray(params) ** 2))

def _sphere_grad(params):
    return -2.0 * np.asarray(params)


@pytest.fixture
def sphere_sampler():
    """Factory for a fresh 4-D sphere sampler per test."""
    def make(grad_func=None, gradient_method='forward'):
        return ProfileProjector(
            target_func=_sphere,
            bounds=np.array([[-5.0, 5.0]] * 4),
            projections=[{'dims': [0, 1], 'grid_points': [3, 3]}],
            grad_func=grad_func,
            advanced_config={'lbfgsb': {'gradient_method': gradient_method}},
        )
    return make


class TestLBFGSBWithUserGradient:

    START = np.array([1.5, -1.0, 0.7, -0.4])

    def test_full_user_gradient_eliminates_fd(self, sphere_sampler):
        sampler = sphere_sampler(grad_func=_sphere_grad)
        job, n_evals, fd_tasks = run_initial_opt_job(
            sampler, self.START, _sphere, grad_func=_sphere_grad)
        baseline = sphere_sampler(grad_func=None)
        _, n_evals_baseline, fd_baseline = run_initial_opt_job(
            baseline, self.START, _sphere)
        assert job.success
        assert fd_tasks == 0
        assert sampler.target_calls_saved_by_user_gradient == fd_baseline
        assert n_evals < n_evals_baseline
        np.testing.assert_allclose(job.current_params, [0, 0, 0, 0], atol=1e-3)

    def test_partial_user_gradient_fd_only_for_missing(self, sphere_sampler):
        """Array form with NaN for dims 1 and 3."""
        def partial_grad(p):
            g = np.asarray(_sphere_grad(p), dtype=float)
            g[1] = g[3] = np.nan
            return g
        sampler = sphere_sampler(grad_func=partial_grad)
        job, _, fd_tasks = run_initial_opt_job(
            sampler, self.START, _sphere, grad_func=partial_grad)
        assert job.success
        # 2 missing dims × 1 FD/dim (forward) per gradient round
        assert fd_tasks > 0 and fd_tasks % 2 == 0
        assert sampler.target_calls_saved_by_user_gradient == fd_tasks
        np.testing.assert_allclose(job.current_params, [0, 0, 0, 0], atol=1e-3)

    def test_grad_func_raise_falls_back(self, sphere_sampler):
        def bad_grad(p):
            raise RuntimeError("boom")
        sampler = sphere_sampler(grad_func=bad_grad)
        job, _, fd_tasks = run_initial_opt_job(
            sampler, self.START, _sphere, grad_func=bad_grad)
        baseline = sphere_sampler(grad_func=None)
        _, _, fd_baseline = run_initial_opt_job(baseline, self.START, _sphere)
        assert job.success
        assert fd_tasks == fd_baseline
        assert sampler.target_calls_saved_by_user_gradient == 0


def test_user_gradient_sign_convention(sphere_sampler):
    """grad_func returns ∇target_func; paraprof negates for the objective.
    A wrong sign would send L-BFGS-B uphill instead of to the origin."""
    sampler = sphere_sampler(grad_func=_sphere_grad)
    job, _, _ = run_initial_opt_job(
        sampler, np.array([2.0, 2.0, 2.0, 2.0]), _sphere,
        grad_func=_sphere_grad, max_iter=30)
    assert job.success
    np.testing.assert_allclose(job.current_params, np.zeros(4), atol=1e-3)
