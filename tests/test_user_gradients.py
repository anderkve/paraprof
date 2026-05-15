"""
Tests for the user-supplied gradient feature.

These tests drive the L-BFGS-B state machine without MPI by acting as a
"fake worker": each emitted task is evaluated locally, the result is fed
back into the job, and the cycle repeats until the job finishes. This
covers the same control flow as ``master_main`` (apart from the MPI
plumbing itself) and lets us assert exactly how many target evaluations
were performed.
"""
import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import ConfigurationError
from paraprof.jobs.lbfgsb_job import LBFGSBJob
from paraprof.worker import _normalize_user_gradient


# ---------------------------------------------------------------------------
# Worker-side normalization helper
# ---------------------------------------------------------------------------

class TestNormalizeUserGradient:
    def test_array_full(self):
        arr, err = _normalize_user_gradient([1.0, 2.0, 3.0], 3)
        assert err is None
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_array_with_nan(self):
        arr, err = _normalize_user_gradient([1.0, np.nan, 3.0], 3)
        assert err is None
        assert np.isfinite(arr[0]) and np.isfinite(arr[2])
        assert np.isnan(arr[1])

    def test_inf_is_treated_as_missing(self):
        arr, err = _normalize_user_gradient([np.inf, 2.0, -np.inf], 3)
        assert err is None
        assert np.isnan(arr[0]) and np.isnan(arr[2])
        assert arr[1] == 2.0

    def test_shape_mismatch(self):
        arr, err = _normalize_user_gradient([1.0, 2.0], 3)
        assert arr is None
        assert err and "expected 3" in err

    def test_dict_partial(self):
        arr, err = _normalize_user_gradient({0: 1.5, 2: -0.5}, 4)
        assert err is None
        assert arr[0] == 1.5 and arr[2] == -0.5
        assert np.isnan(arr[1]) and np.isnan(arr[3])

    def test_dict_invalid_key(self):
        arr, err = _normalize_user_gradient({0: 1.0, 5: 2.0}, 3)
        assert arr is None
        assert err and "out-of-range" in err

    def test_dict_non_numeric_value(self):
        arr, err = _normalize_user_gradient({0: "nope"}, 3)
        assert arr is None
        assert err and "non-numeric" in err

    def test_none_input(self):
        arr, err = _normalize_user_gradient(None, 3)
        assert arr is None and err is not None


# ---------------------------------------------------------------------------
# ProfileProjector constructor validation
# ---------------------------------------------------------------------------

class TestSamplerGradFuncArg:
    def test_default_is_none(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
        )
        assert sampler.grad_func is None
        assert sampler.target_calls_saved_by_user_gradient == 0
        assert sampler.user_gradient_errors == 0

    def test_accepts_callable(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        sampler = ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_2d],
            grad_func=lambda p: np.zeros_like(p),
        )
        assert callable(sampler.grad_func)

    def test_rejects_non_callable(self, simple_2d_function, simple_bounds_2d, basic_projection_2d):
        with pytest.raises(ConfigurationError):
            ProfileProjector(
                target_func=simple_2d_function,
                bounds=simple_bounds_2d,
                projections=[basic_projection_2d],
                grad_func="not callable",
            )


# ---------------------------------------------------------------------------
# Fake-worker driver: run an LBFGSBJob to completion outside MPI
# ---------------------------------------------------------------------------

def _evaluate_task(task, target_func, grad_func, n_dims):
    """Mimic worker_main's per-task handling for unit tests."""
    params = task['params']
    ctx = dict(task['context'])
    ctx['worker_rank'] = 1
    target_val = float(target_func(params))
    user_gradient = None
    user_gradient_error = None
    if ctx.get('compute_gradient') and grad_func is not None:
        try:
            raw = grad_func(params)
            user_gradient, user_gradient_error = _normalize_user_gradient(raw, n_dims)
        except Exception as e:
            user_gradient = None
            user_gradient_error = f"{type(e).__name__}: {e}"
    return {
        'target_val': target_val,
        'params': params,
        'context': ctx,
        'error': None,
        'user_gradient': user_gradient,
        'user_gradient_error': user_gradient_error,
    }


def run_initial_opt_job(sampler, start, target_func, grad_func=None,
                        max_iter=15, ftol=1e-9):
    """Drive an INITIAL_OPTIMIZATION job to completion via a synchronous
    fake worker. Returns (job, n_target_evals, fd_tasks_issued)."""
    sampler.lbfgsb_max_iter = max_iter
    sampler.lbfgsb_ftol = ftol
    n_dims = sampler.dims
    opt_dims = tuple(range(n_dims))
    job = LBFGSBJob(
        job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
        opt_dims=opt_dims, start_params=np.asarray(start, dtype=float),
        grid_idx=None, start_params_full=np.asarray(start, dtype=float),
    )
    n_evals = 0
    fd_tasks = 0
    tasks = job.start()
    while tasks:
        new_tasks = []
        for t in tasks:
            if t['context'].get('sub_type') == 'LBFGS_GRADIENT':
                fd_tasks += 1
            res = _evaluate_task(t, target_func, grad_func, n_dims)
            n_evals += 1
            new_tasks.extend(job.process_result(res))
        tasks = new_tasks
        if job.is_finished():
            break
    job.on_finish(next_job_id=1)
    return job, n_evals, fd_tasks


# ---------------------------------------------------------------------------
# Functional test fixtures: sphere & analytic gradient
# ---------------------------------------------------------------------------

def _sphere(params):
    return -float(np.sum(np.asarray(params) ** 2))

def _sphere_grad(params):
    return -2.0 * np.asarray(params)


@pytest.fixture
def sphere_sampler():
    """Plain 4-D sphere sampler. Each test that uses it constructs a fresh one."""
    def make(grad_func=None, gradient_method='forward'):
        bounds = np.array([[-5.0, 5.0]] * 4)
        projections = [{'dims': [0, 1], 'grid_points': [3, 3]}]
        return ProfileProjector(
            target_func=_sphere,
            bounds=bounds,
            projections=projections,
            grad_func=grad_func,
            advanced_config={'lbfgsb': {'gradient_method': gradient_method}},
        )
    return make


# ---------------------------------------------------------------------------
# Behavioural tests: counts and correctness
# ---------------------------------------------------------------------------

class TestLBFGSBWithUserGradient:

    def test_grad_func_none_is_noop(self, sphere_sampler):
        """With grad_func=None, no task ever carries compute_gradient=True
        and the FD-task count matches the legacy formula."""
        sampler = sphere_sampler(grad_func=None)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        _, _, fd_tasks_fd_only = run_initial_opt_job(sampler, start, _sphere)
        assert sampler.target_calls_saved_by_user_gradient == 0
        assert sampler.user_gradient_errors == 0
        # Sanity: forward FD on 4 dims → 4 FD tasks per gradient computation,
        # and we did at least one gradient computation.
        assert fd_tasks_fd_only > 0
        assert fd_tasks_fd_only % 4 == 0

    def test_full_user_gradient_eliminates_fd(self, sphere_sampler):
        sampler = sphere_sampler(grad_func=_sphere_grad)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        job, n_evals_user, fd_tasks_user = run_initial_opt_job(
            sampler, start, _sphere, grad_func=_sphere_grad
        )
        assert job.success
        # No FD tasks at all once the user gradient covers every dim.
        assert fd_tasks_user == 0
        # And the savings counter equals the FD tasks the legacy path would
        # have issued.
        baseline = sphere_sampler(grad_func=None)
        _, _, fd_tasks_baseline = run_initial_opt_job(baseline, start, _sphere)
        assert sampler.target_calls_saved_by_user_gradient == fd_tasks_baseline
        # And we did strictly fewer target evaluations than the baseline.
        baseline2 = sphere_sampler(grad_func=None)
        _, n_evals_baseline, _ = run_initial_opt_job(baseline2, start, _sphere)
        assert n_evals_user < n_evals_baseline
        # Final point converges to the origin (sphere's optimum).
        np.testing.assert_allclose(job.current_params, [0, 0, 0, 0], atol=1e-3)

    def test_partial_user_gradient_fd_only_for_missing(self, sphere_sampler):
        """User provides dims 0 and 2 of a 4D function as an array with NaN;
        FD must fire only for dims 1 and 3."""
        def partial_grad(params):
            g = np.asarray(_sphere_grad(params), dtype=float)
            g[1] = np.nan
            g[3] = np.nan
            return g
        sampler = sphere_sampler(grad_func=partial_grad)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        job, _, fd_tasks = run_initial_opt_job(
            sampler, start, _sphere, grad_func=partial_grad
        )
        assert job.success
        # Two FD tasks per gradient computation (one for dim 1, one for dim 3),
        # all in forward mode.
        assert fd_tasks > 0
        assert fd_tasks % 2 == 0
        # Savings = 2 dims * (1 fd-task each, forward) per gradient.
        assert sampler.target_calls_saved_by_user_gradient == fd_tasks
        np.testing.assert_allclose(job.current_params, [0, 0, 0, 0], atol=1e-3)

    def test_partial_user_gradient_dict_form(self, sphere_sampler):
        """Dict form of grad_func — only dims 0 and 2 known."""
        def partial_grad_dict(params):
            g = _sphere_grad(params)
            return {0: float(g[0]), 2: float(g[2])}
        sampler = sphere_sampler(grad_func=partial_grad_dict)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        job, _, fd_tasks = run_initial_opt_job(
            sampler, start, _sphere, grad_func=partial_grad_dict
        )
        assert job.success
        assert fd_tasks > 0
        assert sampler.target_calls_saved_by_user_gradient == fd_tasks
        np.testing.assert_allclose(job.current_params, [0, 0, 0, 0], atol=1e-3)

    def test_central_method_doubles_savings_per_dim(self, sphere_sampler):
        """In central-difference mode, each user-provided dim saves 2 FD calls."""
        sampler_user = sphere_sampler(grad_func=_sphere_grad,
                                      gradient_method='central')
        start = np.array([1.5, -1.0, 0.7, -0.4])
        _, _, fd_user = run_initial_opt_job(
            sampler_user, start, _sphere, grad_func=_sphere_grad
        )
        sampler_fd = sphere_sampler(grad_func=None, gradient_method='central')
        _, _, fd_baseline = run_initial_opt_job(sampler_fd, start, _sphere)
        assert fd_user == 0
        assert sampler_user.target_calls_saved_by_user_gradient == fd_baseline
        # And the baseline issues 2*n_dims FD tasks per gradient.
        assert fd_baseline % 8 == 0

    def test_grad_func_raise_falls_back(self, sphere_sampler):
        """A grad_func that raises must not abort the run — fall back to FD
        for every dim."""
        def bad_grad(params):
            raise RuntimeError("user grad blew up")
        sampler = sphere_sampler(grad_func=bad_grad)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        job, _, fd_tasks = run_initial_opt_job(
            sampler, start, _sphere, grad_func=bad_grad
        )
        assert job.success
        # Should fall back to FD for everything; same as the no-grad-func
        # case in terms of FD-task count.
        baseline = sphere_sampler(grad_func=None)
        _, _, fd_baseline = run_initial_opt_job(baseline, start, _sphere)
        assert fd_tasks == fd_baseline
        # Saved counter remains zero — user provided nothing usable.
        assert sampler.target_calls_saved_by_user_gradient == 0

    def test_shape_wrong_user_grad_falls_back(self, sphere_sampler):
        """If grad_func returns a wrong-shape array, fall back to FD."""
        def wrong_shape(params):
            return np.array([1.0, 2.0])  # 4D expected
        sampler = sphere_sampler(grad_func=wrong_shape)
        start = np.array([1.5, -1.0, 0.7, -0.4])
        job, _, fd_tasks = run_initial_opt_job(
            sampler, start, _sphere, grad_func=wrong_shape
        )
        assert job.success
        baseline = sphere_sampler(grad_func=None)
        _, _, fd_baseline = run_initial_opt_job(baseline, start, _sphere)
        assert fd_tasks == fd_baseline
        assert sampler.target_calls_saved_by_user_gradient == 0

    def test_result_matches_fd_baseline(self, sphere_sampler):
        """Sanity: the final optimum reached with a user gradient is the same
        (within tolerance) as the FD-only baseline."""
        start = np.array([2.0, -1.5, 1.0, -0.7])
        s_user = sphere_sampler(grad_func=_sphere_grad)
        job_user, _, _ = run_initial_opt_job(
            s_user, start, _sphere, grad_func=_sphere_grad
        )
        s_fd = sphere_sampler(grad_func=None)
        job_fd, _, _ = run_initial_opt_job(s_fd, start, _sphere)
        np.testing.assert_allclose(job_user.current_params,
                                   job_fd.current_params, atol=1e-2)
        assert abs(job_user.current_fitness - job_fd.current_fitness) < 1e-2

    def test_line_search_piggybacks_gradient(self, sphere_sampler):
        """Every line-search attempt (including backtracking) must request
        the user gradient — that's how we get a gradient at the accepted x
        without an extra round-trip."""
        sampler = sphere_sampler(grad_func=_sphere_grad)
        opt_dims = tuple(range(sampler.dims))
        start = np.array([1.0, 1.0, 1.0, 1.0])
        job = LBFGSBJob(
            job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
            opt_dims=opt_dims, start_params=start,
            grid_idx=None, start_params_full=start,
        )
        tasks = job.start()
        # Drive past the initial-f and gradient stages — full user grad
        # means we land on a line-search task immediately.
        res = _evaluate_task(tasks[0], _sphere, _sphere_grad, sampler.dims)
        tasks = job.process_result(res)
        assert len(tasks) == 1
        ls = tasks[0]
        assert ls['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        assert ls['context']['alpha'] == 1.0
        assert ls['context']['compute_gradient'] is True
        # Force a backtrack by feeding back an objective that fails Armijo.
        bad_res = _evaluate_task(ls, lambda p: -1e10, _sphere_grad, sampler.dims)
        next_tasks = job.process_result(bad_res)
        assert len(next_tasks) == 1
        assert next_tasks[0]['context']['alpha'] < 1.0
        # Backtracking attempts still carry the gradient request.
        assert next_tasks[0]['context']['compute_gradient'] is True

    def test_no_compute_gradient_flag_when_grad_func_absent(self, sphere_sampler):
        """With grad_func=None, no task ever carries compute_gradient=True."""
        sampler = sphere_sampler(grad_func=None)
        opt_dims = tuple(range(sampler.dims))
        start = np.array([1.0, 1.0, 1.0, 1.0])
        job = LBFGSBJob(
            job_id=0, job_type='INITIAL_OPTIMIZATION', sampler=sampler,
            opt_dims=opt_dims, start_params=start,
            grid_idx=None, start_params_full=start,
        )
        seen_flags = []
        tasks = job.start()
        steps = 0
        while tasks and steps < 200:
            new_tasks = []
            for t in tasks:
                seen_flags.append(t['context'].get('compute_gradient', False))
                res = _evaluate_task(t, _sphere, None, sampler.dims)
                new_tasks.extend(job.process_result(res))
                steps += 1
            tasks = new_tasks
            if job.is_finished():
                break
        assert not any(seen_flags), "Tasks should not request gradient when grad_func is None"


# ---------------------------------------------------------------------------
# Sign convention regression — explicit
# ---------------------------------------------------------------------------

def test_user_gradient_is_for_target_func_not_objective(sphere_sampler):
    """grad_func returns ∇target_func, paraprof negates internally for the
    minimization objective. If a user accidentally returned ∇(-target_func),
    L-BFGS-B would walk uphill instead of down. This test pins the sign."""
    # _sphere_grad returns d/dx(-sum x^2) = -2x, i.e. it points to origin
    # (gradient of target points "uphill" in target space). With correct
    # interpretation, the optimizer should descend the objective and reach
    # the origin.
    sampler = sphere_sampler(grad_func=_sphere_grad)
    job, _, _ = run_initial_opt_job(
        sampler, np.array([2.0, 2.0, 2.0, 2.0]), _sphere,
        grad_func=_sphere_grad, max_iter=30,
    )
    assert job.success
    np.testing.assert_allclose(job.current_params, np.zeros(4), atol=1e-3)
