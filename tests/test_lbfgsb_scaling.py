"""Curvature-scaled metric of the asynchronous L-BFGS-B job.

Driven synchronously as in test_lbfgsb_exits.py; the driver here also
records the task sequence (sub-type, line-search alpha)."""
import numpy as np
import pytest

from paraprof import ProfileProjector, get_test_function, set_log_level
from paraprof.jobs.lbfgsb_job import LBFGSBJob

set_log_level('ERROR')

W0, W1 = 1e-3, 1.0          # curvature length scales of the separable quadratic


def quad(x):
    return -0.5 * ((x[0] / W0) ** 2 + (x[1] / W1) ** 2 + x[2] ** 2)


def quad_flat(x):           # independent of x1
    return -0.5 * ((x[0] / W0) ** 2 + x[2] ** 2)


def quad_grad(x):
    return -np.array([x[0] / W0 ** 2, x[1] / W1 ** 2, x[2]])


def _sampler(f, bounds, dims=(2,), grid=20, grad_func=None):
    proj = [{'dims': list(dims), 'grid_points': [grid] * len(dims)}]
    s = ProfileProjector(target_func=f, bounds=bounds, projections=proj, grad_func=grad_func)
    s._reset_for_new_projection(proj[0])
    return s


def _job(s, f, idx, start, job_type='LBFGSB_LOOP'):
    start = np.asarray(start, float)
    full = s._construct_params(idx, start)
    return LBFGSBJob(0, job_type, s, tuple(s.profiled_dims), start, idx, full, start_fitness=float(f(full)))


def _drive(job, f, grad=None):
    """Run to completion; return the list of (sub_type, alpha or None)."""
    tasks = job.start()
    seq = []
    while tasks and not job.is_finished():
        t = tasks.pop(0)
        ctx = t['context']
        seq.append((ctx.get('sub_type'), ctx.get('alpha')))
        res = {'target_val': float(f(t['params'])), 'params': t['params'], 'context': ctx}
        if ctx.get('compute_gradient') and grad is not None:
            res['user_gradient'] = grad(t['params'])
        tasks.extend(job.process_result(res))
    return seq


BOX = [[-1.0, 1.0]] * 3


class TestCurvatureProbe:
    def test_badly_scaled_quadratic_converges_in_a_few_iterations(self):
        s = _sampler(quad, BOX)
        job = _job(s, quad, (10,), [5e-4, 0.4])
        seq = _drive(job, quad)
        subs = [a for a, _ in seq]
        assert subs.count('LBFGS_CURVATURE') == 2
        # scale within 10% of the curvature length scales
        assert job.scale_from_curvature.all()
        assert job.scale == pytest.approx([W0, W1], rel=0.1)
        # first line-search step is the full Newton step and is accepted
        i_ls = subs.index('LBFGS_LINE_SEARCH')
        assert seq[i_ls][1] == 1.0
        assert subs[i_ls + 1] != 'LBFGS_LINE_SEARCH'
        assert job.success and job.converged
        assert job.iteration <= 3
        assert np.allclose(job.current_params, 0.0, atol=1e-6)

    def test_flat_direction_keeps_the_box_width(self):
        s = _sampler(quad_flat, BOX)
        job = _job(s, quad_flat, (10,), [5e-4, 0.4])
        _drive(job, quad_flat)
        assert job.scale_from_curvature[0] and not job.scale_from_curvature[1]
        assert job.scale[0] == pytest.approx(W0, rel=0.1)
        assert job.scale[1] == job.width[1] == 2.0
        assert job.success and job.converged
        assert abs(job.current_params[0]) < 1e-6

    def test_converged_start_on_a_well_scaled_problem_is_not_probed(self):
        f, bounds, _ = get_test_function('sphere_4d')
        s = _sampler(f, bounds, dims=(0, 1))
        job = _job(s, f, (10, 10), [0.0, 0.0])
        seq = _drive(job, f)
        subs = [a for a, _ in seq]
        assert 'LBFGS_CURVATURE' not in subs
        assert len(seq) == len(s.profiled_dims) + 1
        assert job.success and job.converged
        assert not job.scale_from_curvature.any()

    def test_converged_start_on_a_stiff_direction_costs_one_probe_round(self):
        # The first gradient uses the box-scaled finite-difference step, whose
        # forward-difference bias (eps * h_ii / 2) on the stiff direction
        # looks like a gradient, so the probe is issued; the run then ends on
        # the flat-step exit after one line-search evaluation.
        s = _sampler(quad, BOX)
        job = _job(s, quad, (10,), [0.0, 0.0])
        seq = _drive(job, quad)
        subs = [a for a, _ in seq]
        assert subs.count('LBFGS_CURVATURE') == 2
        assert len(seq) == 2 * len(s.profiled_dims) + 1
        assert job.success and job.converged
        assert np.allclose(job.current_params, 0.0, atol=1e-9)

    def test_probe_steps_inward_at_a_bound(self):
        s = _sampler(quad, BOX)
        job = _job(s, quad, (10,), [0.995, 0.5])
        tasks = job.start()
        while job.status != 'NEEDS_CURVATURE':
            t = tasks.pop(0)
            tasks.extend(job.process_result({'target_val': float(quad(t['params'])), 'params': t['params'], 'context': t['context']}))
        steps = {t['context']['dim']: t['context']['step'] for t in tasks}
        assert steps[0] < 0 and steps[1] > 0
        assert all(abs(t['params'][0]) <= 1.0 for t in tasks)

    def test_history_free_step_is_capped(self):
        # Far start on a flat-ish direction: the fallback scale (box width)
        # would ask for a step of many box widths; it is capped at a quarter
        # of the box per coordinate and the run still converges.
        s = _sampler(quad, BOX)
        job = _job(s, quad, (10,), [0.0, 0.9])
        # replace the probe result by a fallback to exercise the cap
        tasks = job.start()
        while job.status != 'NEEDS_LINE_SEARCH':
            t = tasks.pop(0)
            if t['context'].get('sub_type') == 'LBFGS_CURVATURE':
                job.scale[:] = job.width      # pretend the probe failed
                job.scale_from_curvature[:] = False
                job.pending_curvature_evals -= 1
                if job.pending_curvature_evals == 0:
                    tasks.extend(job._compute_direction_and_line_search())
                continue
            tasks.extend(job.process_result({'target_val': float(quad(t['params'])), 'params': t['params'], 'context': t['context']}))
        assert np.max(np.abs(job.search_direction) / job.width) == pytest.approx(0.25)

    def test_user_gradient_probe_tasks_are_plain_evaluations(self):
        s = _sampler(quad, BOX, grad_func=quad_grad)
        job = _job(s, quad, (10,), [5e-4, 0.4])
        tasks = job.start()
        seq = []
        while tasks and not job.is_finished():
            t = tasks.pop(0)
            ctx = t['context']
            seq.append(ctx)
            res = {'target_val': float(quad(t['params'])), 'params': t['params'], 'context': ctx}
            if ctx.get('compute_gradient'):
                res['user_gradient'] = quad_grad(t['params'])
            tasks.extend(job.process_result(res))
        probes = [c for c in seq if c['sub_type'] == 'LBFGS_CURVATURE']
        assert len(probes) == 2 and not any(c['compute_gradient'] for c in probes)
        assert not any(c['sub_type'] == 'LBFGS_GRADIENT' for c in seq)
        assert job.success and job.converged
        assert np.allclose(job.current_params, 0.0, atol=1e-6)


class TestRawUnitsPreserved:
    def test_pairs_and_optimizer_state_stay_in_raw_units(self):
        f, bounds, _ = get_test_function('sphere_4d')
        s = _sampler(f, bounds, dims=(0, 1))
        job = _job(s, f, (10, 10), [2.0, -3.0])
        _drive(job, f)
        assert job.success and job.converged
        assert len(job.s_hist) >= 1
        # the first accepted step moved the point by s_0 in raw coordinates
        s0 = job.s_hist[0]
        assert np.linalg.norm(s0) > 0.1
