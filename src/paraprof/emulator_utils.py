"""
Emulator utilities for sample-efficient optimization.

This module provides Gaussian Process (GP) emulator functionality to reduce
the number of expensive likelihood evaluations by predicting trial point fitness.
"""
import numpy as np
from .logger import get_logger

logger = get_logger()

# Check if scikit-learn is available
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not found. Emulator features disabled. "
        "Install with: pip install scikit-learn>=1.3.0"
    )


class LocalEmulator:
    """
    Local Gaussian Process emulator for likelihood prediction.

    Builds a GP from nearby evaluations to predict fitness of new points
    without expensive likelihood evaluations.
    """

    def __init__(self, X, y, length_scale=1.0, noise_level=0.01):
        """
        Initialize and fit a local GP emulator.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input points
        y : np.ndarray, shape (n_samples,)
            Training target values (fitness/likelihood)
        length_scale : float, optional
            RBF kernel length scale (default: 1.0, auto-optimized)
        noise_level : float, optional
            White noise level for GP (default: 0.01)
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot create emulator: scikit-learn not installed")
            self.is_fitted = False
            return

        self.X = X
        self.y = y
        self.is_fitted = False

        # Build GP with RBF kernel + white noise
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )

        try:
            self.gp.fit(X, y)
            self.is_fitted = True
            logger.debug(f"GP emulator fitted with {len(X)} points")
        except Exception as e:
            logger.warning(f"GP fit failed: {e}. Emulator disabled.")
            self.is_fitted = False

    def predict(self, X_test, return_std=True):
        """
        Predict fitness at test points.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test points
        return_std : bool, optional
            Whether to return uncertainty estimates (default: True)

        Returns
        -------
        mean : np.ndarray, shape (n_test,)
            Predicted fitness values
        std : np.ndarray, shape (n_test,), optional
            Prediction uncertainty (standard deviation)
        """
        if not self.is_fitted:
            # Emulator not fitted, return dummy predictions
            n_test = len(X_test) if len(X_test.shape) > 1 else 1
            if return_std:
                return np.zeros(n_test), np.inf * np.ones(n_test)
            else:
                return np.zeros(n_test)

        return self.gp.predict(X_test, return_std=return_std)

    def score(self, X_test, y_test):
        """
        Calculate R² score on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test input points
        y_test : np.ndarray
            Test target values

        Returns
        -------
        float
            R² score (-inf if not fitted)
        """
        if not self.is_fitted:
            return -np.inf
        return self.gp.score(X_test, y_test)


def gather_nearby_evaluations(sampler, center_params, radius_factor=2.0, min_points=10):
    """
    Gather evaluated points near a center location.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    center_params : np.ndarray
        Center point to gather neighbors around
    radius_factor : float, optional
        Radius in units of ROI threshold (default: 2.0)
    min_points : int, optional
        Minimum points to return (expands radius if needed, default: 10)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'X': np.ndarray of parameter vectors
        - 'y': np.ndarray of fitness values
        - 'n_points': int number of points
    """
    # Start with eval cache if available
    if hasattr(sampler, 'eval_cache') and sampler.eval_cache:
        all_params = np.array([e['params'] for e in sampler.eval_cache])
        all_fitness = np.array([e['fitness'] for e in sampler.eval_cache])
    else:
        # Fallback: gather from population
        all_params = []
        all_fitness = []

        for grid_idx, state in sampler.population.items():
            for i in range(len(state['fitnesses'])):
                cont_params = state['continuous_params'][i]
                full_params = sampler._construct_params(grid_idx, cont_params)
                all_params.append(full_params)
                all_fitness.append(state['fitnesses'][i])

        if not all_params:
            return {
                'X': np.empty((0, len(center_params))),
                'y': np.empty(0),
                'n_points': 0
            }

        all_params = np.array(all_params)
        all_fitness = np.array(all_fitness)

    # Calculate distances from center
    distances = np.linalg.norm(all_params - center_params, axis=1)

    # Adaptive radius to ensure minimum points
    radius = radius_factor * sampler.roi_threshold
    nearby_mask = distances <= radius

    # Expand radius if too few points
    expansion_count = 0
    while np.sum(nearby_mask) < min_points and expansion_count < 10:
        radius *= 1.5
        nearby_mask = distances <= radius
        expansion_count += 1

    nearby_params = all_params[nearby_mask]
    nearby_fitness = all_fitness[nearby_mask]

    logger.debug(
        f"Gathered {len(nearby_params)} points within radius {radius:.2e} "
        f"of center (target: {min_points})"
    )

    return {
        'X': nearby_params,
        'y': nearby_fitness,
        'n_points': len(nearby_params)
    }


def build_local_emulator(sampler, center_params, min_points=10):
    """
    Build a local GP emulator around a center point.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    center_params : np.ndarray
        Center point for local emulator
    min_points : int, optional
        Minimum points required to build emulator (default: 10)

    Returns
    -------
    LocalEmulator or None
        Fitted emulator, or None if insufficient data or sklearn unavailable
    """
    if not SKLEARN_AVAILABLE:
        return None

    # Gather nearby evaluations
    data = gather_nearby_evaluations(sampler, center_params, min_points=min_points)

    if data['n_points'] < min_points:
        logger.debug(f"Insufficient data for emulator: {data['n_points']} < {min_points}")
        return None

    # Get emulator hyperparameters from sampler
    length_scale = getattr(sampler, 'emulator_length_scale', 1.0)
    noise_level = getattr(sampler, 'emulator_noise_level', 0.01)

    # Build emulator
    emulator = LocalEmulator(data['X'], data['y'], length_scale, noise_level)

    if not emulator.is_fitted:
        return None

    return emulator
