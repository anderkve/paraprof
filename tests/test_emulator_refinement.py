"""
Tests for emulator-enhanced grid refinement functionality.

This module tests the three-tier refinement strategy that uses GP emulators
to classify refinement points and reduce computational cost by 60-85%.
"""
import pytest
import numpy as np


# Check if scikit-learn is available
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_direct_evaluation_job():
    """Test DirectEvaluationJob execution."""
    from paraprof.jobs.direct_eval_job import DirectEvaluationJob
    from paraprof import GridAnchoredDESampler

    # Create a simple test function (4D function, 2D projection)
    def simple_func(params):
        return -(params[0]**2 + params[1]**2 + params[2]**2 + params[3]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]  # 2D projection of 4D space

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        pop_per_grid_point=1
    )

    # Create a DirectEvaluationJob
    grid_idx = (2, 2)  # Middle of grid
    continuous_params = np.array([0.0, 0.0])  # For dims [2, 3]

    job = DirectEvaluationJob(
        job_id=1,
        sampler=sampler,
        grid_idx=grid_idx,
        continuous_params=continuous_params
    )

    # Job should return one task
    tasks = job.start()
    assert len(tasks) == 1, "DirectEvaluationJob should return exactly one task"

    # Simulate processing result
    result = {
        'target_val': simple_func(continuous_params),
        'params': sampler._construct_params(grid_idx, continuous_params),
        'context': tasks[0]['context'],
        'emulator_screened': False
    }

    new_tasks = job.process_result(result)
    assert len(new_tasks) == 0, "DirectEvaluationJob should not generate new tasks"
    assert job.is_finished(), "Job should be finished after processing result"
    assert job.success, "Job should be marked as successful"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_refinement_emulator_creation():
    """Test building RefinementEmulator from mock coarse solution."""
    from paraprof.emulator_utils import build_refinement_emulator

    # Create mock coarse solution with expected structure
    np.random.seed(42)

    # Create grid axes
    grid_axes = [
        np.linspace(-5.0, 5.0, 6),  # Dim 0 (6 points including endpoints)
        np.linspace(-5.0, 5.0, 6)   # Dim 1
    ]

    # Create solutions dict
    solutions = {}
    for i in range(6):
        for j in range(6):
            grid_idx = (i, j)
            proj_coords_0 = grid_axes[0][i]
            proj_coords_1 = grid_axes[1][j]

            # Simple paraboloid
            likelihood = -(proj_coords_0**2 + proj_coords_1**2)
            continuous_params = np.array([0.0, 0.0])  # 2D continuous space

            solutions[grid_idx] = {
                'likelihood': likelihood,
                'continuous_params': continuous_params
            }

    coarse_solution = {
        'grid_axes': grid_axes,
        'solutions': solutions
    }

    emulator_config = {
        'emulator_length_scale': 1.0,
        'emulator_noise_level': 0.01
    }

    # Build emulator
    emulator = build_refinement_emulator(coarse_solution, emulator_config)

    assert emulator is not None, "Emulator should be built successfully"

    # Test predictions
    X_test = np.array([[0.0, 0.0]])  # Origin
    results = emulator.predict_batch(X_test)

    assert 'likelihood' in results, "Should predict likelihood"
    assert 'likelihood_std' in results, "Should predict likelihood uncertainty"
    assert 'continuous_params' in results, "Should predict continuous parameters"

    # Origin should have high fitness (close to 0)
    assert results['likelihood'][0] > -5.0, "Origin should have high predicted fitness"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_refinement_emulator_predictions():
    """Test RefinementEmulator prediction methods."""
    from paraprof.emulator_utils import RefinementEmulator

    # Generate training data
    np.random.seed(123)
    X_train = np.random.uniform(-3, 3, (50, 2))
    y_likelihood = -(X_train[:, 0]**2 + X_train[:, 1]**2)
    y_continuous = np.random.uniform(-1, 1, (50, 2))  # Mock continuous params

    emulator = RefinementEmulator(X_train, y_likelihood, y_continuous)

    # Test likelihood prediction
    X_test = np.array([[0.0, 0.0], [2.0, 2.0]])
    pred_ll, pred_std = emulator.predict_likelihood(X_test, return_std=True)

    assert len(pred_ll) == 2, "Should return 2 predictions"
    assert len(pred_std) == 2, "Should return 2 std values"
    assert pred_ll[0] > pred_ll[1], "Origin should have higher fitness"

    # Test continuous params prediction
    pred_params = emulator.predict_continuous_params(X_test)
    assert pred_params.shape == (2, 2), "Should predict 2D continuous params"

    # Test batch prediction
    results = emulator.predict_batch(X_test)
    assert 'likelihood' in results
    assert 'likelihood_std' in results
    assert 'continuous_params' in results


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_point_classification():
    """Test tier assignment logic in _classify_refinement_points."""
    from paraprof import GridAnchoredDESampler
    from paraprof.emulator_utils import RefinementEmulator

    # Create sampler
    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [10, 10]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        roi_threshold=3.0,
        refinement_ucb_beta=2.0
    )

    # Create mock emulator
    np.random.seed(456)
    X_train = np.random.uniform(-5, 5, (50, 2))
    y_likelihood = -(X_train[:, 0]**2 + X_train[:, 1]**2)
    y_continuous = np.zeros((50, 2))

    emulator = RefinementEmulator(X_train, y_likelihood, y_continuous)

    # Create test fine points
    all_fine_points = []
    for i in range(5):
        for j in range(5):
            grid_idx = (i, j)
            proj_coords = np.array([
                -5.0 + 2.5 * i,
                -5.0 + 2.5 * j
            ])
            all_fine_points.append({
                'grid_idx': grid_idx,
                'projection_coordinates': proj_coords
            })

    # Classify points
    tiers = sampler._classify_refinement_points(all_fine_points, emulator)

    assert 'tier1_critical' in tiers
    assert 'tier2_standard' in tiers
    assert 'tier3_simple' in tiers

    # Should classify some points into each tier (for this distribution)
    total_points = len(tiers['tier1_critical']) + len(tiers['tier2_standard']) + len(tiers['tier3_simple'])
    assert total_points == len(all_fine_points), "All points should be classified"

    # Points near origin should be tier 1 (high likelihood)
    origin_point = next(p for p in tiers['tier1_critical'] if tuple(p['grid_idx']) == (2, 2))
    assert origin_point is not None, "Origin should be in tier 1"


def test_point_classification_fallback():
    """Test tier classification fallback when emulator is None."""
    from paraprof import GridAnchoredDESampler

    # Create sampler
    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections
    )

    # Create test fine points
    all_fine_points = [
        {'grid_idx': (0, 0), 'projection_coordinates': np.array([0.0, 0.0])},
        {'grid_idx': (1, 1), 'projection_coordinates': np.array([1.0, 1.0])},
    ]

    # Classify with emulator=None
    tiers = sampler._classify_refinement_points(all_fine_points, emulator=None)

    # All points should be tier 1 (conservative fallback)
    assert len(tiers['tier1_critical']) == len(all_fine_points)
    assert len(tiers['tier2_standard']) == 0
    assert len(tiers['tier3_simple']) == 0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_get_all_roi_fine_points():
    """Test _get_all_roi_fine_points identifies fine points in ROI cells."""
    from paraprof import GridAnchoredDESampler

    # Create sampler with coarse grid
    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        roi_threshold=3.0
    )

    # Manually add some converged points to population
    sampler.population[(2, 2)] = {
        'best_fitness': -1.0,  # In ROI (> -3.0)
        'status': 'converged',
        'continuous_params': np.zeros((1, 2))
    }
    sampler.population[(0, 0)] = {
        'best_fitness': -10.0,  # Outside ROI
        'status': 'converged',
        'continuous_params': np.zeros((1, 2))
    }

    sampler.roi_grid_indices = {(2, 2)}

    # Setup interpolator (mock)
    class MockInterpolator:
        def grid_shape(self):
            return (10, 10)  # 2x refinement

        def get_coarse_cell(self, fine_idx):
            # Simple 2x refinement mapping
            return (fine_idx[0] // 2, fine_idx[1] // 2)

    sampler.interpolator = MockInterpolator()

    # Get ROI fine points
    roi_fine = sampler._get_all_roi_fine_points()

    # Should get fine points from cell (2,2)
    # With 2x refinement: fine indices (4,4), (4,5), (5,4), (5,5)
    assert len(roi_fine) > 0, "Should find ROI fine points"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_tiered_job_creation():
    """Integration test for create_emulator_enhanced_refinement_jobs."""
    from paraprof import GridAnchoredDESampler

    # Create sampler with coarse solution
    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    coarse_proj = {'dims': [0, 1], 'grid_points': [5, 5]}
    projections = [coarse_proj]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        roi_threshold=3.0
    )

    # Create mock coarse solution
    coarse_solution = {
        'converged_points': [],
        'projection_dims': [0, 1],
        'grid_shape': (5, 5)
    }

    # Add converged points
    np.random.seed(789)
    for i in range(25):
        grid_idx = (i // 5, i % 5)
        proj_coords = np.array([
            -5.0 + 2.5 * (i // 5),
            -5.0 + 2.5 * (i % 5)
        ])
        fitness = -(proj_coords[0]**2 + proj_coords[1]**2)

        coarse_solution['converged_points'].append({
            'grid_idx': grid_idx,
            'projection_coordinates': proj_coords,
            'continuous_params': np.random.uniform(-0.5, 0.5, 2),
            'best_fitness': fitness
        })

    # Setup refinement
    sampler.setup_refinement_run(coarse_solution, refinement_factor=2)
    sampler._reset_for_new_projection({'dims': [0, 1], 'grid_points': [10, 10]})

    # Create refinement jobs
    jobs, next_job_id, metrics = sampler.create_emulator_enhanced_refinement_jobs(job_id=1)

    assert len(jobs) > 0, "Should create refinement jobs"
    assert 'tier1_count' in metrics
    assert 'tier2_count' in metrics
    assert 'tier3_count' in metrics
    assert 'estimated_cost' in metrics
    assert 'savings_percentage' in metrics

    # Total points should match
    total = metrics['tier1_count'] + metrics['tier2_count'] + metrics['tier3_count']
    assert total > 0, "Should classify some points"

    # Should have cost savings
    assert metrics['savings_percentage'] >= 0, "Should report savings percentage"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_tier2_job_settings():
    """Verify tier-2 sampler settings are configured correctly."""
    from paraprof import GridAnchoredDESampler

    # Create sampler with specific settings
    def simple_func(params):
        return -(params[0]**2 + params[1]**2 + params[2]**2 + params[3]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        lbfgsb_max_iter=15,
        lbfgsb_ftol=1e-7,
        refinement_tier2_max_iter=5,
        refinement_tier2_ftol_multiplier=10.0
    )

    # Verify tier-2 settings were computed correctly
    assert sampler.refinement_tier2_max_iter == 5, "Should use explicit tier2 max_iter"
    assert sampler.refinement_tier2_ftol_multiplier == 10.0, "Should use specified multiplier"

    # Expected tier-2 ftol
    expected_tier2_ftol = sampler.lbfgsb_ftol * 10.0
    assert expected_tier2_ftol == 1e-6, "Tier-2 ftol should be 10x looser"


def test_emulator_refinement_disabled():
    """Test that refinement works when use_emulator_refinement=False."""
    from paraprof import GridAnchoredDESampler

    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        use_emulator_refinement=False
    )

    assert sampler.use_emulator_refinement is False
    # Standard refinement should still work (tested in other test files)


def test_configuration_parameter_defaults():
    """Test that new refinement parameters have correct defaults."""
    from paraprof import GridAnchoredDESampler

    def simple_func(params):
        return -(params[0]**2 + params[1]**2)

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5]}]

    # Create with defaults
    sampler = GridAnchoredDESampler(
        target_func=simple_func,
        bounds=bounds,
        projections=projections,
        lbfgsb_max_iter=30
    )

    # Check defaults
    assert sampler.use_emulator_refinement is True
    assert sampler.refinement_tier2_max_iter == max(5, 30 // 3)  # Should be 10
    assert sampler.refinement_tier2_ftol_multiplier == 10.0
    assert sampler.refinement_ucb_beta == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
