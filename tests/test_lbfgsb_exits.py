"""Early exits of the asynchronous L-BFGS-B job and neighbour seeding in the lbfgsb mode.

The job is driven synchronously: every task it emits is evaluated in-process and
fed back through ``process_result``, which lets the tests count target
evaluations exactly.
"""
import collections

import numpy as np
import pytest

from paraprof import ProfileProjector, get_test_function, set_log_level
from paraprof.jobs.lbfgsb_job import LBFGSBJob

set_log_level('ERROR')


def _make_sampler(name='sphere_4d', dims=(0, 1), grid=20, advanced_config=None):
    f, bounds, _ = get_test_function(name)
    proj = [{'dims': list(dims), 'grid_points': [grid] * len(dims)}]
    s = ProfileProjector(target_func=f, bounds=bounds, projections=proj,
                         advanced_config=advanced_config)
    s._reset_for_new_projection(proj[0])
    return s, f


def _drive(job, f, tasks=None):
    """Run a job to completion in-process; return (evaluations, line-search evaluations).

    ``tasks`` may be the list returned by an earlier ``job.start()`` (a job must
    not be started twice)."""
    if tasks is None:
        tasks = job.start()
    n = n_ls = 0
    while tasks and not job.is_finished():
        t = tasks.pop(0)
        n += 1
        n_ls += t['context'].get('sub_type') == 'LBFGS_LINE_SEARCH'
        res = {'target_val': float(f(t['params'])), 'params': t['params'], 'context': t['context']}
        tasks.extend(job.process_result(res))
    return n, n_ls


def _job(s, f, idx, start, job_type='LBFGSB_LOOP', seed_history=None):
    start = np.asarray(start, float)
    full = s._construct_params(idx, start)
    return LBFGSBJob(0, job_type, s, tuple(s.profiled_dims), start, idx, full,
                     seed_history=seed_history, start_fitness=float(f(full)))


class TestGradientNormExit:
    def test_converged_start_exits_after_one_gradient(self):
        s, f = _make_sampler()
        job = _job(s, f, (10, 10), [0.0, 0.0])
        n, n_ls = _drive(job, f)
        assert job.success and job.converged
        assert n_ls == 0
        assert n == len(s.profiled_dims)           # forward-difference gradient only
        assert s.lbfgsb_gtol_exits == 1

    def test_far_start_still_optimises(self):
        s, f = _make_sampler()
        job = _job(s, f, (10, 10), [2.0, -3.0])
        n, _ = _drive(job, f)
        assert job.success and job.converged
        assert np.allclose(job.current_params, 0.0, atol=1e-4)
        assert n > len(s.profiled_dims)

    def test_gtol_none_disables_test(self):
        s, f = _make_sampler(advanced_config={'lbfgsb': {'gtol': None}})
        job = _job(s, f, (10, 10), [0.0, 0.0])
        _drive(job, f)
        assert s.lbfgsb_gtol_exits == 0


class TestFlatStepExit:
    def test_rejected_zero_change_step_ends_run(self):
        # gtol disabled: the job must take a line-search step from the exact
        # optimum; f cannot change there, so the flat-step exit ends the run
        # instead of halving alpha ~34 times and reporting failure.
        s, f = _make_sampler(advanced_config={'lbfgsb': {'gtol': None}})
        job = _job(s, f, (10, 10), [0.0, 0.0])
        n, n_ls = _drive(job, f)
        assert job.success and job.converged
        assert n_ls == 1
        assert n == len(s.profiled_dims) + 1
        assert s.lbfgsb_flat_exits == 1
        assert np.allclose(job.current_params, 0.0)


class TestNeighbourSeedingInLbfgsbMode:
    def _population_entry(self, params, fitness, history=None):
        return {
            'profiled_params': np.array([params], float),
            'fitnesses': np.array([fitness], float),
            'best_fitness': float(fitness),
            'status': 'converged',
            'improvement_history': collections.deque(maxlen=3),
            'last_update_gen': 0,
            'optimizer_state': history,
        }

    def test_loop_job_tests_neighbour_and_inherits_history(self):
        s, f = _make_sampler()
        idx, nb = (10, 10), (10, 11)
        hist = {'s': [np.array([0.1, 0.0])], 'y': [np.array([0.2, 0.0])]}
        s.population[nb] = self._population_entry([0.0, 0.0], f(s._construct_params(nb, np.zeros(2))), hist)
        s.population[idx] = self._population_entry([1.5, 1.5], f(s._construct_params(idx, np.array([1.5, 1.5]))))
        s.population[idx]['status'] = 'active'
        jobs, _ = s._create_lbfgsb_jobs_for_active(0, 'LBFGSB_LOOP', 'converged')
        assert len(jobs) == 1
        job = jobs[0]
        tasks = job.start()
        assert job.status == 'NEEDS_NEIGHBOR_TEST'
        assert tasks[0]['context']['sub_type'] == 'LBFGS_NEIGHBOR_TEST'
        assert len(job.s_hist) == 0            # not yet: only once the neighbour's point is adopted
        t = tasks.pop(0)
        tasks.extend(job.process_result({'target_val': float(f(t['params'])), 'params': t['params'], 'context': t['context']}))
        assert len(job.s_hist) == 1 and np.allclose(job.s_hist[0], hist['s'][0])
        n, _ = _drive(job, f, tasks)
        assert job.success
        # the neighbour's optimum beats the cell's own start and is adopted
        assert np.allclose(job.current_params, 0.0, atol=1e-6)

    def test_loop_job_without_seedable_neighbour_starts_from_known_value(self):
        # No neighbour to test: the start value is known from the population,
        # so the job goes straight to the gradient without re-evaluating it.
        s, f = _make_sampler()
        idx = (10, 10)
        s.population[idx] = self._population_entry([0.5, 0.5], f(s._construct_params(idx, np.array([0.5, 0.5]))))
        s.population[idx]['status'] = 'active'
        jobs, _ = s._create_lbfgsb_jobs_for_active(0, 'LBFGSB_LOOP', 'converged')
        job = jobs[0]
        tasks = job.start()
        assert all(t['context']['sub_type'] == 'LBFGS_GRADIENT' for t in tasks)
        assert len(tasks) == len(s.profiled_dims)
        n, _ = _drive(job, f, tasks)
        assert job.success and np.allclose(job.current_params, 0.0, atol=1e-6)


class TestStoppingRule:
    @pytest.mark.parametrize('N, expect', [
        (4, False),   # W = 1: W(W+1)/(N-W-2) = 2/1 = 2.0 > 0.5
        (7, False),   # 2/4 = 0.5, not strictly below the threshold
        (8, True),    # 2/5 = 0.4 < 0.5
    ])
    def test_expected_undiscovered_single_basin(self, N, expect):
        s, _ = _make_sampler(advanced_config={'basin_detection': {'undiscovered_threshold': 0.5}})
        s.basin_min_starts = 0
        s.global_max_target_val = 0.0
        s.initial_optima_registry = [{'point': np.zeros(4), 'point_norm': np.zeros(4), 'target_val': 0.0, 'count': N}]
        assert s.basin_detection_should_stop(N) is expect

    def test_formula_value(self):
        s, _ = _make_sampler(advanced_config={'basin_detection': {'undiscovered_threshold': 0.5}})
        s.basin_min_starts = 0
        s.global_max_target_val = 0.0
        # W = 2 optima, N = 12 hits: 2*3/(12-2-2) = 0.75 -> keep going at 0.5
        s.initial_optima_registry = [
            {'point': np.zeros(4), 'point_norm': np.zeros(4), 'target_val': 0.0, 'count': 6},
            {'point': np.ones(4), 'point_norm': np.ones(4), 'target_val': -0.1, 'count': 6},
        ]
        assert s.basin_detection_should_stop(12) is False
        # N = 16: 6/12 = 0.5 -> not strictly below; N = 18: 6/14 = 0.43 -> stop
        s.initial_optima_registry[0]['count'] = 9
        s.initial_optima_registry[1]['count'] = 9
        assert s.basin_detection_should_stop(18) is True
