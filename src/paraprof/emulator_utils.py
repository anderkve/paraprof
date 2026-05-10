"""
Emulator utilities for sample-efficient optimization.

This module provides Gaussian Process (GP) emulator functionality to reduce
the number of expensive likelihood evaluations by predicting trial point fitness.
"""
import numpy as np
from scipy.optimize import minimize
from .logger import get_logger

logger = get_logger()

# Check if scikit-learn is available
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not found. Emulator features disabled. "
        "Install with: pip install scikit-learn>=1.3.0"
    )


# ============================================================================
# Gaussian Process Emulator Constants
# ============================================================================

# Kernel configuration
GP_CONSTANT_KERNEL_VALUE = 1.0
"""Initial constant multiplier value for GP kernel"""

GP_CONSTANT_VALUE_BOUNDS_MIN = 1e-8
"""Lower bound for constant kernel value optimization"""

GP_CONSTANT_VALUE_BOUNDS_MAX = 1e8
"""Upper bound for constant kernel value optimization"""

GP_LENGTH_SCALE_BOUNDS_MIN = 1e-4
"""Lower bound for Matern kernel length scale optimization"""

GP_LENGTH_SCALE_BOUNDS_MAX = 1e4
"""Upper bound for Matern kernel length scale optimization"""

GP_MATERN_NU = 1.5
"""Smoothness parameter for Matern kernel (nu=1.5 gives once-differentiable functions)"""

# GP optimizer configuration
GP_OPTIMIZER_MAX_ITER = 20_000
"""Maximum iterations for L-BFGS-B hyperparameter optimization"""

GP_ALPHA_NOISE = 1e-6
"""Noise regularization parameter (alpha) for GP fitting"""

GP_N_RESTARTS = 0
"""Number of random restarts for hyperparameter optimization"""


class LocalEmulator:
    """
    Local Gaussian Process emulator for likelihood prediction.

    Builds a GP from nearby evaluations to predict fitness of new points
    without expensive likelihood evaluations.
    """

    def __init__(self, X, y, length_scale=1.0, noise_level=0.01):
        """
        Initialize and fit a local GP emulator with input standardization.

        Training inputs are standardized (z-score normalization) to improve
        GP performance by making the kernel isotropic. Each local emulator
        uses its own scaler fitted to its training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input points (will be standardized internally)
        y : np.ndarray, shape (n_samples,)
            Training target values (fitness/likelihood, not scaled)
        length_scale : float, optional
            Matern kernel length scale in standardized units (default: 1.0)
            Note: After standardization, length_scale ~ 1.0 is typically good
        noise_level : float, optional
            White noise level for GP (default: 0.01)
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot create emulator: scikit-learn not installed")
            self.is_fitted = False
            return

        self.X_raw = X  # Store raw data for reference
        self.y = y
        self.is_fitted = False
        self.X_scaler = None

        n_dims = X.shape[1]

        # Standardize input features (z-score normalization)
        # Each local emulator scales based on its own training data distribution
        self.X_scaler = StandardScaler()
        try:
            X_scaled = self.X_scaler.fit_transform(X)
        except Exception as e:
            logger.warning(f"Input standardization failed: {e}. Using raw inputs.")
            X_scaled = X
            self.X_scaler = None

        # Build GP with Matern kernel
        # After standardization, a single isotropic length scale works well
        kernel = ConstantKernel(
            constant_value=GP_CONSTANT_KERNEL_VALUE,
            constant_value_bounds=(GP_CONSTANT_VALUE_BOUNDS_MIN, GP_CONSTANT_VALUE_BOUNDS_MAX)
        ) * Matern(
            length_scale=length_scale,
            length_scale_bounds=(GP_LENGTH_SCALE_BOUNDS_MIN, GP_LENGTH_SCALE_BOUNDS_MAX),
            nu=GP_MATERN_NU
        )


        # Define own optimizer
        def optimizer(obj_func, x0, bounds):
            res = minimize(
                obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
                options={"maxiter": GP_OPTIMIZER_MAX_ITER}
            )
            return res.x, res.fun

        # Create GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=GP_N_RESTARTS,
            alpha=GP_ALPHA_NOISE,
            optimizer=optimizer,
        )

        try:
            # Fit GP on standardized inputs
            self.gp.fit(X_scaled, y)
            self.is_fitted = True
            logger.debug(
                f"GP emulator fitted with {len(X)} points (inputs standardized)"
            )
        except Exception as e:
            logger.warning(f"GP fit failed: {e}. Emulator disabled.")
            self.is_fitted = False

    def predict(self, X_test, return_std=True):
        """
        Predict fitness at test points.

        Test points are automatically standardized using the same scaler
        fitted to the training data.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_test, n_features)
            Test points (in original parameter space)
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

        # Reshape if needed (only if 1D)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # Standardize test points using training scaler
        if self.X_scaler is not None:
            try:
                X_test_scaled = self.X_scaler.transform(X_test)
            except Exception as e:
                logger.warning(f"Test point standardization failed: {e}. Using raw inputs.")
                X_test_scaled = X_test
        else:
            # Scaler not fitted (fallback to raw inputs)
            X_test_scaled = X_test

        return self.gp.predict(X_test_scaled, return_std=return_std)

    def score(self, X_test, y_test):
        """
        Calculate R² score on test data.

        Test points are automatically standardized using the same scaler
        fitted to the training data.

        Parameters
        ----------
        X_test : np.ndarray
            Test input points (in original parameter space)
        y_test : np.ndarray
            Test target values

        Returns
        -------
        float
            R² score (-inf if not fitted)
        """
        if not self.is_fitted:
            return -np.inf

        # Standardize test points using training scaler
        if self.X_scaler is not None:
            try:
                X_test_scaled = self.X_scaler.transform(X_test)
            except Exception as e:
                logger.warning(f"Test point standardization failed: {e}. Using raw inputs.")
                X_test_scaled = X_test
        else:
            X_test_scaled = X_test

        return self.gp.score(X_test_scaled, y_test)


def gather_nearby_evaluations(sampler, center_params, radius_factor=2.0, min_points=10, max_points=None, grid_idx=None):
    """
    Gather evaluated points near a center location.

    Uses local per-grid-point caches for efficient gathering. If grid_idx is provided,
    gathers from that grid point and its neighbors. Otherwise, falls back to global cache.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    center_params : np.ndarray
        Center point to gather neighbors around
    radius_factor : float, optional
        Radius in units of ROI threshold (default: 2.0)
    min_points : int, optional
        Minimum points to return (expands radius if needed, default: 10)
    max_points : int, optional
        Maximum points to return (selects closest if exceeded, default: None = no limit)
    grid_idx : tuple, optional
        Grid index for local cache gathering (default: None = use global cache)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'X': np.ndarray of parameter vectors
        - 'y': np.ndarray of fitness values
        - 'n_points': int number of points
    """
    all_params = []
    all_fitness = []

    # Get expected dimensionality from center_params
    expected_ndim = len(center_params)

    # Strategy 1: Use local caches if grid_idx provided
    if grid_idx is not None:
        # Gather from center grid point's cache
        if grid_idx in sampler.local_eval_caches:
            for e in sampler.local_eval_caches[grid_idx]:
                params = np.asarray(e['params'])
                if len(params) == expected_ndim:
                    all_params.append(params)
                    all_fitness.append(e['fitness'])

        # Gather from neighboring grid points' caches
        neighbors = sampler._get_valid_neighbors(grid_idx)
        for neighbor_idx in neighbors:
            if neighbor_idx in sampler.local_eval_caches:
                for e in sampler.local_eval_caches[neighbor_idx]:
                    params = np.asarray(e['params'])
                    if len(params) == expected_ndim:
                        all_params.append(params)
                        all_fitness.append(e['fitness'])

        logger.debug(
            f"Gathered {len(all_params)} points from local cache (grid {grid_idx} + neighbors)"
        )

    # Strategy 2: Use global cache if no grid_idx or local caches insufficient
    if len(all_params) < min_points and sampler.global_eval_cache:
        for e in sampler.global_eval_cache:
            params = np.asarray(e['params'])
            if len(params) == expected_ndim:
                all_params.append(params)
                all_fitness.append(e['fitness'])
        logger.debug(
            f"Added {len(sampler.global_eval_cache)} points from global cache"
        )

    # Strategy 3: Ultimate fallback - gather from population
    if len(all_params) < min_points:
        for gidx, state in sampler.population.items():
            for i in range(len(state['fitnesses'])):
                cont_params = state['continuous_params'][i]
                full_params = sampler._construct_params(gidx, cont_params)
                full_params = np.asarray(full_params)
                if len(full_params) == expected_ndim:
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

    # If we already have enough points from local caches, just cap at max_points
    # This avoids expensive distance calculations on the full dataset
    if len(all_params) <= max_points if max_points else True:
        # We have few enough points, return all
        return {
            'X': all_params,
            'y': all_fitness,
            'n_points': len(all_params)
        }

    # We have too many points - select closest to center
    if max_points is not None and len(all_params) > max_points:
        # Use squared distances to avoid expensive sqrt (argpartition only needs ordering)
        distances_squared = np.sum((all_params - center_params)**2, axis=1)
        closest_indices = np.argpartition(distances_squared, max_points)[:max_points]
        all_params = all_params[closest_indices]
        all_fitness = all_fitness[closest_indices]

        logger.debug(
            f"Selected {max_points} closest points from {len(distances_squared)} total"
        )

    return {
        'X': all_params,
        'y': all_fitness,
        'n_points': len(all_params)
    }


def build_local_emulator(sampler, center_params, min_points=10, max_points=None, grid_idx=None):
    """
    Build a local GP emulator around a center point.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    center_params : np.ndarray
        Center point for local emulator
    min_points : int, optional
        Minimum points required to build emulator (default: 10)
    max_points : int, optional
        Maximum points to use for training (default: None = no limit)
        Limits GP training time by capping dataset size
    grid_idx : tuple, optional
        Grid index for local cache gathering (default: None)

    Returns
    -------
    LocalEmulator or None
        Fitted emulator, or None if insufficient data or sklearn unavailable
    """
    if not SKLEARN_AVAILABLE:
        return None

    # Gather nearby evaluations (using local caches if grid_idx provided)
    data = gather_nearby_evaluations(
        sampler,
        center_params,
        min_points=min_points,
        max_points=max_points,
        grid_idx=grid_idx
    )

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


def prepare_emulator_cache_for_worker(sampler, center_params, min_points=10, max_points=None, grid_idx=None):
    """
    Prepare evaluation cache data to send to worker for emulator-based pre-screening.

    This function gathers nearby evaluations and packages them in a compact format
    for transmission to workers. Workers will use this data to build local GP
    emulators and perform trial pre-screening independently.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance with eval_cache
    center_params : np.ndarray
        Center point for local emulator (typically the trial point)
    min_points : int, optional
        Minimum points required (default: 10)
    max_points : int, optional
        Maximum points to send (default: None = no limit)
        Limits data transfer size and GP training time
    grid_idx : tuple, optional
        Grid index for local cache gathering (default: None)

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
    if not getattr(sampler, 'use_emulator', False):
        return None

    if not SKLEARN_AVAILABLE:
        return None

    # Gather nearby evaluations (using local caches if grid_idx provided)
    data = gather_nearby_evaluations(
        sampler,
        center_params,
        min_points=min_points,
        max_points=max_points,
        grid_idx=grid_idx
    )

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
