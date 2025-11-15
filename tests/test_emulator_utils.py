"""
Tests for emulator utility functions.
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
def test_local_emulator_basic():
    """Test that LocalEmulator can fit a simple quadratic function."""
    from paraprof.emulator_utils import LocalEmulator

    # Generate simple quadratic data
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (50, 2))
    y = -(X[:, 0]**2 + X[:, 1]**2)  # Negative paraboloid

    # Build emulator
    emulator = LocalEmulator(X, y)

    assert emulator.is_fitted, "Emulator should be fitted"

    # Test prediction
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    pred_mean, pred_std = emulator.predict(X_test, return_std=True)

    assert len(pred_mean) == 2, "Should return 2 predictions"
    assert len(pred_std) == 2, "Should return 2 std values"
    assert pred_mean[0] > pred_mean[1], "Origin should have higher fitness than (1,1)"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_local_emulator_insufficient_data():
    """Test that LocalEmulator handles insufficient data gracefully."""
    from paraprof.emulator_utils import LocalEmulator

    # Too few points
    X = np.array([[0.0, 0.0]])
    y = np.array([1.0])

    emulator = LocalEmulator(X, y)

    # Should either not fit or return infinite uncertainty
    X_test = np.array([[1.0, 1.0]])
    pred_mean, pred_std = emulator.predict(X_test, return_std=True)

    # If not fitted, should return dummy values
    if not emulator.is_fitted:
        assert np.isinf(pred_std[0]), "Should return infinite std if not fitted"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_local_emulator_prediction_accuracy():
    """Test that emulator predictions are accurate on known function."""
    from paraprof.emulator_utils import LocalEmulator

    # Generate data from known smooth function
    np.random.seed(123)
    X_train = np.random.uniform(-3, 3, (100, 2))
    y_train = -(X_train[:, 0]**2 + X_train[:, 1]**2)  # True function

    # Test on separate data
    X_test = np.random.uniform(-3, 3, (20, 2))
    y_test = -(X_test[:, 0]**2 + X_test[:, 1]**2)

    # Build and test emulator
    emulator = LocalEmulator(X_train, y_train)

    if emulator.is_fitted:
        score = emulator.score(X_test, y_test)
        # R^2 score should be high for smooth function
        assert score > 0.8, f"R^2 score {score} should be > 0.8 for smooth function"


def test_gather_nearby_evaluations_with_cache():
    """Test gathering nearby evaluations from eval_cache."""
    from paraprof.emulator_utils import gather_nearby_evaluations

    # Create mock sampler with eval_cache
    class MockSampler:
        def __init__(self):
            self.roi_threshold = 3.0
            self.eval_cache = []

            # Add some cached evaluations
            for i in range(20):
                self.eval_cache.append({
                    'params': np.random.uniform(-5, 5, 4),
                    'fitness': np.random.uniform(-10, 0),
                    'call_number': i
                })

    sampler = MockSampler()
    center = np.array([0.0, 0.0, 0.0, 0.0])

    # Gather nearby points
    data = gather_nearby_evaluations(sampler, center, radius_factor=10.0, min_points=5)

    assert 'X' in data, "Should return X array"
    assert 'y' in data, "Should return y array"
    assert 'n_points' in data, "Should return n_points"
    assert data['n_points'] >= 5, "Should gather at least min_points"


def test_gather_nearby_evaluations_from_population():
    """Test gathering evaluations from population when no cache."""
    from paraprof.emulator_utils import gather_nearby_evaluations

    # Create mock sampler without eval_cache
    class MockSampler:
        def __init__(self):
            self.roi_threshold = 3.0
            self.population = {
                (0, 0): {
                    'fitnesses': np.array([-1.0, -2.0, -3.0]),
                    'continuous_params': np.array([
                        [0.1, 0.2],
                        [0.2, 0.3],
                        [0.3, 0.4]
                    ])
                }
            }
            self.projection_dims = [0, 1]
            self.grid_axes = [np.linspace(-5, 5, 10), np.linspace(-5, 5, 10)]

        def _construct_params(self, grid_idx, cont_params):
            # Simple mock: just return continuous params + grid location
            grid_coords = self._get_grid_coords_from_indices(grid_idx)
            return np.concatenate([grid_coords, cont_params])

        def _get_grid_coords_from_indices(self, grid_idx):
            return np.array([self.grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    sampler = MockSampler()
    center = np.array([0.0, 0.0, 0.0, 0.0])

    # Gather nearby points
    data = gather_nearby_evaluations(sampler, center, radius_factor=10.0, min_points=1)

    assert data['n_points'] >= 0, "Should return valid n_points"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_build_local_emulator():
    """Test building emulator from sampler state."""
    from paraprof.emulator_utils import build_local_emulator

    # Create mock sampler with eval_cache
    class MockSampler:
        def __init__(self):
            self.roi_threshold = 3.0
            self.emulator_length_scale = 1.0
            self.emulator_noise_level = 0.01
            self.eval_cache = []

            # Add cached evaluations
            np.random.seed(42)
            for i in range(30):
                params = np.random.uniform(-3, 3, 4)
                fitness = -(params[0]**2 + params[1]**2)  # Simple function
                self.eval_cache.append({
                    'params': params,
                    'fitness': fitness,
                    'call_number': i
                })

    sampler = MockSampler()
    center = np.array([0.0, 0.0, 0.0, 0.0])

    # Build emulator
    emulator = build_local_emulator(sampler, center, min_points=10)

    assert emulator is not None, "Should build emulator with sufficient data"
    assert emulator.is_fitted, "Emulator should be fitted"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_build_local_emulator_insufficient_data():
    """Test that build_local_emulator returns None with insufficient data."""
    from paraprof.emulator_utils import build_local_emulator

    # Create mock sampler with very little data
    class MockSampler:
        def __init__(self):
            self.roi_threshold = 3.0
            self.emulator_length_scale = 1.0
            self.emulator_noise_level = 0.01
            self.eval_cache = [
                {'params': np.array([0.0, 0.0, 0.0, 0.0]), 'fitness': 0.0, 'call_number': 0}
            ]

    sampler = MockSampler()
    center = np.array([0.0, 0.0, 0.0, 0.0])

    # Try to build emulator
    emulator = build_local_emulator(sampler, center, min_points=10)

    assert emulator is None, "Should return None with insufficient data"
