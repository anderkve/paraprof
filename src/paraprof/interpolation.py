"""
Grid interpolation utilities for refinement runs.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .logger import get_logger

logger = get_logger()

# Check if scikit-learn is available for GP interpolation
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available. MultiGPInterpolator will not be available.")


class GridInterpolator:
    """
    Interpolates continuous parameter values from a coarse grid solution
    to predict good starting points for a refined grid.

    This class creates separate interpolators for each continuous parameter
    dimension, allowing prediction of optimal continuous parameters at any
    point in the projection space.
    """

    def __init__(self, coarse_grid_solution):
        """
        Initializes the interpolator from a coarse grid solution.

        Parameters
        ----------
        coarse_grid_solution : dict
            Dictionary returned by GridAnchoredDESampler.export_grid_solution()
            containing:
            - 'grid_axes': List of arrays defining grid coordinates
            - 'projection_dims': List of projection dimension indices
            - 'continuous_dims': List of continuous dimension indices
            - 'solutions': Dict mapping grid_idx -> solution dict
            - 'grid_shape': Tuple of grid dimensions
        """
        self.grid_axes = coarse_grid_solution['grid_axes']
        self.projection_dims = coarse_grid_solution['projection_dims']
        self.continuous_dims = coarse_grid_solution['continuous_dims']
        self.grid_shape = coarse_grid_solution['grid_shape']
        self.solutions = coarse_grid_solution['solutions']

        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

        # Build interpolators for each continuous parameter
        self._build_interpolators()

        # Cache for likelihood interpolator (built on first use)
        self._likelihood_interpolator = None


    def _build_interpolators(self):
        """
        Constructs RegularGridInterpolator objects for each continuous parameter.

        For each continuous dimension, creates a dense grid of values from the
        sparse coarse solution, then builds an interpolator.
        """
        self.interpolators = []

        if self.n_cont_dims == 0:
            # No continuous dimensions to interpolate
            return

        # For each continuous parameter dimension
        for cont_dim_idx in range(self.n_cont_dims):
            # Create a dense grid filled with NaN
            values_grid = np.full(self.grid_shape, np.nan)

            # Fill in the known values from coarse solutions
            for grid_idx, solution in self.solutions.items():
                continuous_params = solution['continuous_params']
                values_grid[grid_idx] = continuous_params[cont_dim_idx]

            # Check if we have enough data for interpolation
            valid_mask = ~np.isnan(values_grid)
            n_valid = np.sum(valid_mask)

            if n_valid == 0:
                logger.warning(f" No valid data for continuous dim {cont_dim_idx}. "
                      f"Interpolator will return NaN.")
                # Create dummy interpolator that returns NaN
                interpolator = None
            elif n_valid < 2**self.n_proj_dims:
                logger.warning(f" Sparse data ({n_valid} points) for continuous dim {cont_dim_idx}. "
                      f"Using nearest neighbor interpolation.")
                # Use nearest neighbor for sparse data
                interpolator = RegularGridInterpolator(
                    self.grid_axes,
                    values_grid,
                    method='nearest',
                    bounds_error=False,
                    fill_value=None  # Will extrapolate using nearest
                )
            else:
                # Use linear interpolation with nearest neighbor extrapolation
                interpolator = RegularGridInterpolator(
                    self.grid_axes,
                    values_grid,
                    method='linear',
                    bounds_error=False,
                    fill_value=None  # Will extrapolate using nearest
                )

            self.interpolators.append(interpolator)


    def interpolate(self, projection_coords):
        """
        Interpolates continuous parameter values at given projection coordinates.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        np.ndarray or None
            Interpolated continuous parameter values, shape (n_cont_dims,)
            Returns None if no continuous dimensions exist
        """
        if self.n_cont_dims == 0:
            return None

        projection_coords = np.asarray(projection_coords)

        # Ensure projection_coords is in the right shape for interpolation
        # RegularGridInterpolator expects shape (n_points, n_dims)
        coords_for_interp = projection_coords.reshape(1, -1)

        interpolated_params = np.zeros(self.n_cont_dims)

        for i, interpolator in enumerate(self.interpolators):
            if interpolator is None:
                # No valid data, return NaN
                interpolated_params[i] = np.nan
            else:
                try:
                    interpolated_params[i] = interpolator(coords_for_interp)[0]
                except Exception as e:
                    logger.warning(f"Warning: Interpolation failed for continuous dim {i}: {e}")
                    interpolated_params[i] = np.nan

        return interpolated_params


    def get_interpolated_likelihood(self, projection_coords):
        """
        Interpolates the likelihood value at given projection coordinates.

        Uses a cached interpolator for efficiency when called multiple times.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        float
            Interpolated likelihood value
        """
        # Build likelihood interpolator on first use and cache it
        if self._likelihood_interpolator is None:
            likelihood_grid = np.full(self.grid_shape, np.nan)

            for grid_idx, solution in self.solutions.items():
                likelihood_grid[grid_idx] = solution['likelihood']

            # Create interpolator for likelihood
            self._likelihood_interpolator = RegularGridInterpolator(
                self.grid_axes,
                likelihood_grid,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

        projection_coords = np.asarray(projection_coords).reshape(1, -1)
        return self._likelihood_interpolator(projection_coords)[0]


    def get_coverage_fraction(self):
        """
        Returns the fraction of grid points that have valid solutions.

        Returns
        -------
        float
            Fraction of grid points with solutions (0 to 1)
        """
        total_grid_points = np.prod(self.grid_shape)
        valid_points = len(self.solutions)
        return valid_points / total_grid_points if total_grid_points > 0 else 0.0


    def __repr__(self):
        """String representation of the interpolator."""
        coverage = self.get_coverage_fraction()
        return (f"GridInterpolator(grid_shape={self.grid_shape}, "
                f"n_continuous_dims={self.n_cont_dims}, "
                f"coverage={coverage:.1%})")


class MultiGPInterpolator:
    """
    Interpolates optimal continuous parameters using multiple low-dimensional GPs.

    Instead of modeling the high-dimensional likelihood function directly, this class
    trains one GP per continuous parameter to predict optimal values as a function of
    projection coordinates only. This dramatically reduces dimensionality and allows
    efficient training without additional likelihood evaluations.

    For a 2D projection with 2 continuous dimensions:
    - Trains 2 separate 2D GPs (one per continuous parameter)
    - Each GP predicts: optimal_θ_cont[i] = f_i(θ_projection)
    - Uses existing coarse grid optima as training data (zero extra evaluations)

    Advantages over linear interpolation:
    - Captures non-linear variation of optimal parameters
    - Provides uncertainty estimates for adaptive refinement
    - Handles noisy/sparse data more robustly

    Advantages over full high-dimensional GP:
    - Much faster training (low-dimensional GPs)
    - No extra likelihood evaluations needed
    - Natural parallelization (train GPs independently)
    """

    def __init__(self, coarse_grid_solution, kernel_type='matern', length_scale=None,
                 noise_level=1e-5, normalize_inputs=True):
        """
        Initialize the multi-GP interpolator from a coarse grid solution.

        Parameters
        ----------
        coarse_grid_solution : dict
            Dictionary returned by GridAnchoredDESampler.export_grid_solution()
            containing:
            - 'grid_axes': List of arrays defining grid coordinates
            - 'projection_dims': List of projection dimension indices
            - 'continuous_dims': List of continuous dimension indices
            - 'solutions': Dict mapping grid_idx -> solution dict
            - 'grid_shape': Tuple of grid dimensions
        kernel_type : str, optional
            Type of GP kernel to use. Options:
            - 'rbf': Radial Basis Function (smooth)
            - 'matern': Matérn kernel (less smooth, more robust)
            Default: 'matern'
        length_scale : float or array-like, optional
            Initial length scale for the kernel. If None, uses adaptive default
            based on grid spacing. Can be scalar or array of length n_proj_dims.
        noise_level : float, optional
            White noise level for GP (regularization). Default: 1e-5
        normalize_inputs : bool, optional
            Whether to normalize input coordinates to [0, 1]. Recommended for
            numerical stability. Default: True
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "MultiGPInterpolator requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

        self.grid_axes = coarse_grid_solution['grid_axes']
        self.projection_dims = coarse_grid_solution['projection_dims']
        self.continuous_dims = coarse_grid_solution['continuous_dims']
        self.grid_shape = coarse_grid_solution['grid_shape']
        self.solutions = coarse_grid_solution['solutions']

        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.normalize_inputs = normalize_inputs

        # Compute normalization parameters if needed
        if self.normalize_inputs:
            self._compute_normalization_params()

        # Determine length scale if not provided
        if length_scale is None:
            self.length_scale = self._estimate_length_scale()
        else:
            self.length_scale = length_scale

        # Train GPs (one per continuous dimension)
        self.gps = []
        self.training_stats = []
        self._train_gps()

        logger.info(f"MultiGPInterpolator: Trained {self.n_cont_dims} GPs on {len(self.solutions)} coarse grid points")


    def _compute_normalization_params(self):
        """Compute parameters for normalizing projection coordinates to [0, 1]."""
        self.proj_mins = np.array([axis.min() for axis in self.grid_axes])
        self.proj_maxs = np.array([axis.max() for axis in self.grid_axes])
        self.proj_ranges = self.proj_maxs - self.proj_mins

        # Handle edge case of constant dimensions
        self.proj_ranges = np.where(self.proj_ranges > 0, self.proj_ranges, 1.0)


    def _normalize_coords(self, coords):
        """Normalize projection coordinates to [0, 1]."""
        if not self.normalize_inputs:
            return coords
        return (coords - self.proj_mins) / self.proj_ranges


    def _estimate_length_scale(self):
        """
        Estimate a reasonable length scale based on grid spacing.

        Uses the median grid spacing as a heuristic. In normalized coordinates,
        this becomes a fraction of the total range.
        """
        # Compute typical grid spacing for each dimension
        spacings = []
        for axis in self.grid_axes:
            if len(axis) > 1:
                spacing = np.median(np.diff(axis))
                spacings.append(spacing)
            else:
                spacings.append(1.0)

        if self.normalize_inputs:
            # In normalized coords, median spacing becomes spacing/range
            normalized_spacings = np.array(spacings) / self.proj_ranges
            # Use 2-3x the grid spacing as length scale (allows smooth interpolation)
            length_scale = 2.0 * np.median(normalized_spacings)
        else:
            # Use 2-3x median spacing in original coordinates
            length_scale = 2.0 * np.median(spacings)

        logger.debug(f"  Estimated GP length scale: {length_scale:.4f}")
        return length_scale


    def _create_kernel(self):
        """Create a GP kernel based on configuration."""
        if self.kernel_type == 'rbf':
            # RBF kernel (infinitely differentiable, very smooth)
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-3, 1e2)
            )
        elif self.kernel_type == 'matern':
            # Matérn kernel with ν=2.5 (twice differentiable, less smooth than RBF)
            # More robust to non-smooth optimal parameter surfaces
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-3, 1e2),
                nu=2.5
            )
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        # Add white noise for numerical stability
        kernel = kernel + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e-1))

        return kernel


    def _train_gps(self):
        """Train one GP per continuous parameter dimension."""
        if self.n_cont_dims == 0:
            logger.info("  No continuous dimensions to train GPs for")
            return

        # Extract training data
        X_train_list = []
        y_trains = [[] for _ in range(self.n_cont_dims)]

        for grid_idx, solution in self.solutions.items():
            # Get grid coordinates in projection space
            grid_coords = np.array([
                self.grid_axes[i][grid_idx[i]]
                for i in range(self.n_proj_dims)
            ])

            X_train_list.append(grid_coords)

            # Extract optimal continuous parameters
            cont_params = solution['continuous_params']
            for i in range(self.n_cont_dims):
                y_trains[i].append(cont_params[i])

        X_train = np.array(X_train_list)
        n_train = len(X_train)

        if n_train == 0:
            logger.warning("  No training data available for GPs")
            return

        # Normalize inputs if requested
        if self.normalize_inputs:
            X_train = self._normalize_coords(X_train)

        logger.info(f"  Training {self.n_cont_dims} GPs with {n_train} samples each...")

        # Train one GP per continuous dimension
        for i in range(self.n_cont_dims):
            y_train = np.array(y_trains[i])

            # Create kernel
            kernel = self._create_kernel()

            # Create and train GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,  # Multiple restarts for better hyperparameter optimization
                alpha=self.noise_level,  # Additional noise for numerical stability
                normalize_y=True  # Normalize targets for numerical stability
            )

            # Fit GP
            gp.fit(X_train, y_train)

            self.gps.append(gp)

            # Store training statistics
            log_marginal_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)
            stats = {
                'cont_dim_idx': i,
                'n_train': n_train,
                'y_mean': np.mean(y_train),
                'y_std': np.std(y_train),
                'log_marginal_likelihood': log_marginal_likelihood,
                'kernel': str(gp.kernel_)
            }
            self.training_stats.append(stats)

            logger.debug(f"    GP {i}: log_marginal_likelihood={log_marginal_likelihood:.2f}, "
                        f"y_range=[{y_train.min():.3e}, {y_train.max():.3e}]")


    def interpolate(self, projection_coords, return_std=True):
        """
        Predict optimal continuous parameters at given projection coordinates.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)
        return_std : bool, optional
            If True, also return prediction uncertainties. Default: True

        Returns
        -------
        continuous_params : np.ndarray
            Predicted optimal continuous parameters, shape (n_cont_dims,)
        uncertainties : np.ndarray or None
            GP prediction standard deviations for each dimension, shape (n_cont_dims,)
            Only returned if return_std=True, otherwise None
        """
        if self.n_cont_dims == 0:
            return None if not return_std else (None, None)

        projection_coords = np.asarray(projection_coords).reshape(1, -1)

        # Normalize if needed
        if self.normalize_inputs:
            projection_coords = self._normalize_coords(projection_coords)

        continuous_params = np.zeros(self.n_cont_dims)
        uncertainties = np.zeros(self.n_cont_dims) if return_std else None

        # Query each GP
        for i, gp in enumerate(self.gps):
            if return_std:
                mean, std = gp.predict(projection_coords, return_std=True)
                continuous_params[i] = mean[0]
                uncertainties[i] = std[0]
            else:
                mean = gp.predict(projection_coords, return_std=False)
                continuous_params[i] = mean[0]

        if return_std:
            return continuous_params, uncertainties
        else:
            return continuous_params


    def get_max_uncertainty(self, projection_coords):
        """
        Get the maximum uncertainty across all continuous dimensions.

        Useful for deciding whether to run L-BFGS-B refinement.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        float
            Maximum standard deviation across all continuous dimensions
        """
        _, uncertainties = self.interpolate(projection_coords, return_std=True)
        if uncertainties is None:
            return 0.0
        return np.max(uncertainties)


    def get_coverage_fraction(self):
        """
        Returns the fraction of grid points that have valid solutions.

        Returns
        -------
        float
            Fraction of grid points with solutions (0 to 1)
        """
        total_grid_points = np.prod(self.grid_shape)
        valid_points = len(self.solutions)
        return valid_points / total_grid_points if total_grid_points > 0 else 0.0


    def __repr__(self):
        """String representation of the interpolator."""
        coverage = self.get_coverage_fraction()
        return (f"MultiGPInterpolator(grid_shape={self.grid_shape}, "
                f"n_continuous_dims={self.n_cont_dims}, "
                f"n_gps={len(self.gps)}, "
                f"kernel={self.kernel_type}, "
                f"coverage={coverage:.1%})")
