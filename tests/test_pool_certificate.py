"""
Unit tests for the cross-projection pool-certificate pass
(``cross_projection.pool_certificate``, idea 3 flavor a).

These exercise the sampler-level candidate selection and the job's
regression-proof adopt-on-improvement logic directly, without the MPI loop.
"""
import numpy as np

from paraprof import ProfileProjector
from paraprof.jobs.pool_certificate_job import PoolCertificateJob


def _make_sampler(pool_certificate=False, grid_n=10):
    n_dims = 3  # 1 projection dim + 2 profiled

    def target(p):
        return -float(np.sum(p ** 2))

    bounds = np.array([[-5.0, 5.0]] * n_dims)
    projection = {'dims': [0], 'grid_points': [grid_n]}
    return ProfileProjector(
        target_func=target, bounds=bounds, projections=[projection],
        pop_per_grid_point=2, roi_threshold=10.0,
        advanced_config={'cross_projection': {'pool_certificate': pool_certificate}},
    )


def _plant_cell(sampler, idx, phi, fitness):
    if not isinstance(idx, tuple):
        idx = (idx,)
    sampler.population[idx] = {
        'profiled_params': np.array([np.asarray(phi, float)]),
        'fitnesses': np.array([float(fitness)]),
        'best_fitness': float(fitness),
        'status': 'optimized',
        'improvement_history': [],
        'last_update_gen': 0,
        'optimizer_state': None,
    }
    sampler.global_max_target_val = max(sampler.global_max_target_val, float(fitness))


class TestConfigPlumbing:
    def test_default_off(self):
        s = _make_sampler()
        assert s.pool_certificate is False
        assert (s.pool_cert_tests, s.pool_cert_raises, s.pool_cert_gain) == (0, 0, 0.0)

    def test_opt_in(self):
        assert _make_sampler(pool_certificate=True).pool_certificate is True


class TestCandidateSelection:
    def test_disabled_returns_nothing(self):
        s = _make_sampler(pool_certificate=False)
        _plant_cell(s, (5,), [0.1, 0.1], fitness=-1.0)
        node0 = s.grid_axes[0][5]
        s._update_global_pool(np.array([node0, 0.0, 0.0]), 0.0, grid_idx=None)
        jobs, _ = s.create_pool_certificate_jobs(0)
        assert jobs == []

    def test_higher_cross_projection_point_triggers(self):
        """A pool point landing in a cell with fitness above the cell's current
        value (and ROI-competitive) produces a test job with that point's phi."""
        s = _make_sampler(pool_certificate=True)
        _plant_cell(s, (5,), [3.0, 3.0], fitness=-18.0)  # poorly converged cell
        node0 = s.grid_axes[0][5]
        # A better optimum essentially at this node: phi=(0,0), fitness ~ node0^2.
        better_fitness = -float(node0 ** 2)
        s._update_global_pool(np.array([node0, 0.0, 0.0]), better_fitness, grid_idx=None)

        jobs, _ = s.create_pool_certificate_jobs(0)
        assert len(jobs) == 1
        assert jobs[0].grid_idx == (5,)
        np.testing.assert_allclose(jobs[0].candidate_phi, [0.0, 0.0])

    def test_no_trigger_when_pool_not_better(self):
        """A pool point that does not beat the cell's current value (e.g. the
        cell's own optimum echoed back) creates no job."""
        s = _make_sampler(pool_certificate=True)
        node0 = s.grid_axes[0][5]
        fit = -float(node0 ** 2)
        _plant_cell(s, (5,), [0.0, 0.0], fitness=fit)
        s._update_global_pool(np.array([node0, 0.0, 0.0]), fit, grid_idx=None)
        jobs, _ = s.create_pool_certificate_jobs(0)
        assert jobs == []


class TestJobIsRegressionProof:
    def test_improvement_counts_and_spawns_polish(self):
        s = _make_sampler(pool_certificate=True)
        _plant_cell(s, (5,), [3.0, 3.0], fitness=-18.0)
        job = PoolCertificateJob(0, s, (5,), candidate_phi=np.array([0.0, 0.0]),
                                 pool_fitness=-1.0)
        job.start()
        assert s.pool_cert_tests == 1
        job.process_result({'target_val': -1.0, 'context': {}})  # beats -18.0
        assert s.pool_cert_raises == 1
        assert s.pool_cert_gain > 0
        spawn = job.on_finish(1)
        assert spawn is not None and spawn[0].type == 'PATCHING_LBFGSB'

    def test_no_improvement_is_a_noop(self):
        """A candidate that does not beat the current value adopts nothing and
        spawns no polish -- the pass can only ever raise a value."""
        s = _make_sampler(pool_certificate=True)
        _plant_cell(s, (5,), [0.0, 0.0], fitness=-0.5)
        job = PoolCertificateJob(0, s, (5,), candidate_phi=np.array([2.0, 2.0]),
                                 pool_fitness=-0.4)
        job.start()
        job.process_result({'target_val': -9.0, 'context': {}})  # worse than -0.5
        assert s.pool_cert_raises == 0
        assert job.on_finish(1) is None
