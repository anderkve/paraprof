"""
Tests for DE trial point pre-screening functionality.
"""
import pytest
import numpy as np


def test_de_prescreening_disabled():
    """Test that no trials are skipped when pre-screening is disabled."""
    from paraprof.jobs.de_job import DEGridPointJob

    # Create mock sampler with pre-screening disabled
    class MockSampler:
        def __init__(self):
            self.use_de_prescreening = False
            self.pop_per_grid_point = 3
            self.n_cont_dims = 2
            self.continuous_dims = [2, 3]
            self.mutation_strategy = 'current-to-rand/1'
            self.neighbor_pull_probability = 0.0
            self.memory_size = 10
            self.memory_F = np.full(10, 0.5)
            self.memory_CR = np.full(10, 0.5)
            self.direct_eval_mode = False
            self.population = {
                (0, 0): {
                    'continuous_params': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                    'fitnesses': np.array([-1.0, -2.0, -3.0])
                }
            }

        def _ensure_bounds(self, params, dims):
            return params

        def _construct_params(self, grid_idx, cont_params):
            return np.concatenate([np.array([0.0, 0.0]), cont_params])

        def _get_valid_neighbors(self, grid_idx):
            return []

    sampler = MockSampler()
    parent_pool = [
        {'continuous_params': np.array([0.1, 0.2]), 'fitness': -1.0},
        {'continuous_params': np.array([0.3, 0.4]), 'fitness': -2.0},
        {'continuous_params': np.array([0.5, 0.6]), 'fitness': -3.0},
    ]

    # Create DE job
    job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=(0, 0),
        parent_pool=parent_pool,
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )

    # Generate tasks
    np.random.seed(42)
    tasks = job.start()

    # With pre-screening disabled, should not screen out any trials
    assert job.trials_screened_out == 0, "Should not screen out trials when disabled"


def test_de_job_tracking_statistics():
    """Test that DE job correctly tracks trial generation statistics."""
    from paraprof.jobs.de_job import DEGridPointJob

    # Create mock sampler
    class MockSampler:
        def __init__(self):
            self.use_de_prescreening = False
            self.pop_per_grid_point = 5
            self.n_cont_dims = 2
            self.continuous_dims = [2, 3]
            self.mutation_strategy = 'rand/1'
            self.neighbor_pull_probability = 0.0
            self.memory_size = 10
            self.memory_F = np.full(10, 0.5)
            self.memory_CR = np.full(10, 0.5)
            self.direct_eval_mode = False
            self.de_trials_generated = 0
            self.de_trials_screened_out = 0
            self.population = {
                (0, 0): {
                    'continuous_params': np.array([
                        [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]
                    ]),
                    'fitnesses': np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
                }
            }

        def _ensure_bounds(self, params, dims):
            return params

        def _construct_params(self, grid_idx, cont_params):
            return np.concatenate([np.array([0.0, 0.0]), cont_params])

        def _get_valid_neighbors(self, grid_idx):
            return []

    sampler = MockSampler()
    parent_pool = [
        {'continuous_params': np.array([0.1, 0.2]), 'fitness': -1.0},
        {'continuous_params': np.array([0.3, 0.4]), 'fitness': -2.0},
        {'continuous_params': np.array([0.5, 0.6]), 'fitness': -3.0},
        {'continuous_params': np.array([0.7, 0.8]), 'fitness': -4.0},
    ]

    # Create DE job
    job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=(0, 0),
        parent_pool=parent_pool,
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )

    # Generate tasks
    np.random.seed(123)
    tasks = job.start()

    # Check statistics are tracked
    assert hasattr(job, 'trials_generated'), "Should track trials_generated"
    assert hasattr(job, 'trials_screened_out'), "Should track trials_screened_out"
    assert job.trials_generated >= 0, "trials_generated should be non-negative"


def test_de_prescreening_screen_method():
    """Test the _screen_trial_with_emulator method logic."""
    from paraprof.jobs.de_job import DEGridPointJob

    # Create mock sampler with pre-screening enabled
    class MockSampler:
        def __init__(self):
            self.use_de_prescreening = True
            self.emulator_confidence_threshold = 2.0
            self.emulator_min_neighbors = 10
            self.pop_per_grid_point = 1
            self.n_cont_dims = 2
            self.continuous_dims = [2, 3]
            self.roi_threshold = 3.0
            self.eval_cache = []  # Empty cache - emulator will fail to build
            self.population = {
                (0, 0): {
                    'continuous_params': np.array([[0.1, 0.2]]),
                    'fitnesses': np.array([-1.0])
                }
            }

        def _construct_params(self, grid_idx, cont_params):
            return np.concatenate([np.array([0.0, 0.0]), cont_params])

    sampler = MockSampler()

    job = DEGridPointJob(
        job_id=1,
        sampler=sampler,
        grid_idx=(0, 0),
        parent_pool=[],
        pbest_archive=[],
        successful_F_list=[],
        successful_CR_list=[]
    )

    # Test with empty cache (should always evaluate)
    trial_params = np.array([0.1, 0.2])
    full_params = np.array([0.0, 0.0, 0.1, 0.2])
    target_fitness = -1.0

    should_evaluate = job._screen_trial_with_emulator(
        trial_params, target_fitness, full_params
    )

    # With insufficient data, should always evaluate
    assert should_evaluate is True, "Should evaluate when emulator cannot be built"


def test_de_prescreening_disabled_when_sklearn_missing():
    """Test that pre-screening is disabled when sklearn is not available."""
    from paraprof.jobs.de_job import EMULATOR_AVAILABLE, DEGridPointJob

    # If sklearn is not available, pre-screening should be disabled
    if not EMULATOR_AVAILABLE:
        class MockSampler:
            def __init__(self):
                self.use_de_prescreening = True  # User enabled it
                self.pop_per_grid_point = 1

        sampler = MockSampler()

        job = DEGridPointJob(
            job_id=1,
            sampler=sampler,
            grid_idx=(0, 0),
            parent_pool=[],
            pbest_archive=[],
            successful_F_list=[],
            successful_CR_list=[]
        )

        # Should always return True (evaluate) when sklearn unavailable
        should_evaluate = job._screen_trial_with_emulator(
            np.array([0.0]), -1.0, np.array([0.0, 0.0])
        )

        assert should_evaluate is True, "Should evaluate all when sklearn unavailable"
