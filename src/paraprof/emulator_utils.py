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
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
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

        n_dims = X.shape[1]

        # Build GP with RBF kernel
        # kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-08, 1e8)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e5))
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-08, 1e8)) * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e4), nu=1.5)
        # kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-08, 1e8)) * Matern(length_scale=[length_scale]*n_dims, length_scale_bounds=[(1e-2, 1e4)]*n_dims, nu=1.5)

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=0,
            alpha=1e-6,
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


def gather_nearby_evaluations(sampler, center_params, radius_factor=2.0, min_points=10, max_points=None):
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
    max_points : int, optional
        Maximum points to return (selects closest if exceeded, default: None = no limit)

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

    # Cap at max_points if specified (select closest points)
    if max_points is not None and len(nearby_params) > max_points:
        # Sort by distance to prioritize closer points
        nearby_distances = distances[nearby_mask]
        sorted_indices = np.argsort(nearby_distances)

        # Take closest max_points
        selected_indices = sorted_indices[:max_points]
        nearby_params = nearby_params[selected_indices]
        nearby_fitness = nearby_fitness[selected_indices]

        logger.debug(
            f"Gathered {len(selected_indices)} points (capped from {np.sum(nearby_mask)}) "
            f"within radius {radius:.2e}"
        )
    else:
        logger.debug(
            f"Gathered {len(nearby_params)} points within radius {radius:.2e} "
            f"of center (target: {min_points})"
        )

    return {
        'X': nearby_params,
        'y': nearby_fitness,
        'n_points': len(nearby_params)
    }


def build_local_emulator(sampler, center_params, min_points=10, max_points=None):
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
    max_points : int, optional
        Maximum points to use for training (default: None = no limit)
        Limits GP training time by capping dataset size

    Returns
    -------
    LocalEmulator or None
        Fitted emulator, or None if insufficient data or sklearn unavailable
    """
    if not SKLEARN_AVAILABLE:
        return None

    # Gather nearby evaluations
    data = gather_nearby_evaluations(sampler, center_params, min_points=min_points, max_points=max_points)

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


def prepare_emulator_cache_for_worker(sampler, center_params, min_points=10, max_points=None):
    """
    Prepare evaluation cache data to send to worker for emulator-based pre-screening.

    This function gathers nearby evaluations and packages them in a compact format
    for transmission to workers. Workers will use this data to build local GP
    emulators and perform trial pre-screening independently.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance with eval_cache
    center_params : np.ndarray
        Center point for local emulator (typically the trial point)
    min_points : int, optional
        Minimum points required (default: 10)
    max_points : int, optional
        Maximum points to send (default: None = no limit)
        Limits data transfer size and GP training time

    Returns
    -------
    dict or None
        Dictionary with keys for worker-side emulator building:
        - 'X': np.ndarray of parameter vectors
        - 'y': np.ndarray of fitness values
        - 'length_scale': float, RBF kernel length scale
        - 'noise_level': float, GP noise level
        - 'confidence_threshold': float, UCB beta parameter
        Returns None if insufficient data or emulator disabled
    """
    # Check if pre-screening is enabled
    if not getattr(sampler, 'use_de_prescreening', False):
        return None

    if not SKLEARN_AVAILABLE:
        return None

    # Gather nearby evaluations
    data = gather_nearby_evaluations(sampler, center_params, min_points=min_points, max_points=max_points)

    if data['n_points'] < min_points:
        logger.debug(f"Insufficient data for worker emulator: {data['n_points']} < {min_points}")
        return None

    # Package data with hyperparameters
    return {
        'X': data['X'],
        'y': data['y'],
        'length_scale': getattr(sampler, 'emulator_length_scale', 1.0),
        'noise_level': getattr(sampler, 'emulator_noise_level', 0.01),
        'confidence_threshold': getattr(sampler, 'emulator_confidence_threshold', 2.0)
    }


def build_emulator_from_cache(cache_data):
    """
    Build a GP emulator from pre-gathered cache data (worker-side).

    This function is designed to be called on worker processes that receive
    pre-packaged evaluation cache data from the master.

    Parameters
    ----------
    cache_data : dict
        Dictionary with keys:
        - 'X': np.ndarray, training inputs
        - 'y': np.ndarray, training targets
        - 'length_scale': float
        - 'noise_level': float

    Returns
    -------
    LocalEmulator or None
        Fitted emulator, or None if sklearn unavailable or fit fails
    """
    if not SKLEARN_AVAILABLE:
        return None

    if cache_data is None:
        return None

    X = cache_data['X']
    y = cache_data['y']
    length_scale = cache_data.get('length_scale', 1.0)
    noise_level = cache_data.get('noise_level', 0.01)

    # Build emulator
    emulator = LocalEmulator(X, y, length_scale, noise_level)

    if not emulator.is_fitted:
        return None

    return emulator
