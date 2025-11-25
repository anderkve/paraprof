"""
Grid interpolation utilities for refinement runs.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .logger import get_logger

logger = get_logger()

# Check if scikit-learn is available for GP interpolation and clustering
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available. MultiGPInterpolator and clustering features will not be available.")


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
            kernel = C(1.0, (1e-3, 1e6)) * RBF(
                length_scale=[self.length_scale] * self.n_proj_dims,
                length_scale_bounds=[(1e-3, 1e6)] * self.n_proj_dims
            )
        elif self.kernel_type == 'matern':
            # Matérn kernel with ν=1.5 (once differentiable, less smooth than RBF)
            # More robust to non-smooth optimal parameter surfaces
            kernel = C(1.0, (1e-3, 1e6)) * Matern(
                length_scale=[self.length_scale] * self.n_proj_dims,
                length_scale_bounds=[(1e-4, 1e2)] * self.n_proj_dims,
                nu=0.5
            )
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        # Add white noise for numerical stability
        # kernel = kernel + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e-1))

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
                n_restarts_optimizer=1,  # Multiple restarts for better hyperparameter optimization
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


    def interpolate(self, projection_coords, return_std=False):
        """
        Predict optimal continuous parameters at given projection coordinates.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)
        return_std : bool, optional
            If True, also return prediction uncertainties. Default: False

        Returns
        -------
        continuous_params : np.ndarray or None
            Predicted optimal continuous parameters, shape (n_cont_dims,)
            Returns None if no continuous dimensions exist
        uncertainties : np.ndarray or None
            GP prediction standard deviations for each dimension, shape (n_cont_dims,)
            Only returned if return_std=True
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


    def get_interpolated_likelihood(self, projection_coords):
        """
        Interpolates the likelihood value at given projection coordinates using GP.

        Trains a GP model for likelihood values on first call and caches it for
        subsequent calls. This enables efficient pre-screening of fine grid points
        during refinement.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        float
            Interpolated likelihood value at the given coordinates
        """
        # Build likelihood GP on first use and cache it
        if not hasattr(self, '_likelihood_gp') or self._likelihood_gp is None:
            logger.debug("  Building likelihood GP for interpolation...")

            # Extract training data from coarse grid solutions
            X_train_list = []
            y_train_list = []

            for grid_idx, solution in self.solutions.items():
                # Get grid coordinates in projection space
                grid_coords = np.array([
                    self.grid_axes[i][grid_idx[i]]
                    for i in range(self.n_proj_dims)
                ])
                X_train_list.append(grid_coords)
                y_train_list.append(solution['likelihood'])

            if len(X_train_list) == 0:
                logger.warning("  No training data available for likelihood GP")
                self._likelihood_gp = None
                return np.nan

            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)

            # Normalize inputs if needed
            if self.normalize_inputs:
                X_train = self._normalize_coords(X_train)

            # Create kernel for likelihood GP
            kernel = self._create_kernel()

            # Create and train likelihood GP
            self._likelihood_gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=1,
                alpha=self.noise_level,
                normalize_y=True  # Normalize targets for numerical stability
            )

            self._likelihood_gp.fit(X_train, y_train)

            log_marginal_likelihood = self._likelihood_gp.log_marginal_likelihood(
                self._likelihood_gp.kernel_.theta
            )
            logger.debug(f"  Likelihood GP: log_marginal_likelihood={log_marginal_likelihood:.2f}, "
                        f"n_train={len(y_train)}")

        # If GP training failed, return NaN
        if self._likelihood_gp is None:
            return np.nan

        # Predict likelihood at query point
        projection_coords = np.asarray(projection_coords).reshape(1, -1)

        # Normalize if needed
        if self.normalize_inputs:
            projection_coords = self._normalize_coords(projection_coords)

        # Get prediction
        try:
            predicted_likelihood = self._likelihood_gp.predict(projection_coords, return_std=False)
            return float(predicted_likelihood[0])
        except Exception as e:
            logger.warning(f"  Likelihood GP prediction failed: {e}")
            return np.nan


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


# =============================================================================
# Clustering-Based Boundary Detection for Refinement
# =============================================================================

def cluster_coarse_grid_by_modes(coarse_solution, method='dbscan',
                                  eps=None, min_samples=None,
                                  eps_multiplier=3.0,
                                  distance_threshold=1.0, n_clusters=2,
                                  include_likelihood=False, likelihood_weight=0.1,
                                  include_projection_coords=True, projection_weight=1.0):
    """
    Cluster coarse grid points by their optimal continuous parameters (and optionally likelihood).
    Each cluster represents a distinct mode/optimum being tracked in the continuous parameter space.

    This clustering enables detection of discontinuity boundaries where the optimal continuous
    parameters change abruptly, typically because optimization has switched from tracking one
    mode to another.

    Parameters
    ----------
    coarse_solution : dict
        Exported coarse grid solution containing:
        - 'solutions': Dict mapping grid_idx -> solution dict
        - 'grid_shape': Tuple of grid dimensions
        - 'grid_axes': List of arrays defining grid coordinates
        - 'continuous_dims': List of continuous dimension indices
        - 'projection_dims': List of projection dimension indices
    method : str, optional
        Clustering method: 'dbscan', 'hierarchical', or 'kmeans' (default: 'dbscan')
    eps : float, optional
        DBSCAN eps parameter (maximum distance for neighbors).
        If None, auto-estimated from data (default: None)
    min_samples : int, optional
        DBSCAN min_samples parameter (minimum cluster size).
        If None, defaults to max(2, n_continuous_dims) (default: None)
    eps_multiplier : float, optional
        Multiplier for auto-estimated DBSCAN eps (default: 3.0)
        Higher values create fewer, larger clusters. Only used when eps=None.
    distance_threshold : float, optional
        Hierarchical clustering distance threshold (default: 1.0)
    n_clusters : int, optional
        K-means number of clusters (default: 2)
    include_likelihood : bool, optional
        If True, include log-likelihood as an additional feature for clustering.
        Generally should be False (default) to cluster only by continuous parameter
        smoothness. Likelihood is still used for selecting best cluster at boundaries.
        (default: False)
    likelihood_weight : float, optional
        Relative weight for likelihood feature compared to continuous parameters.
        Only used if include_likelihood=True (default: 0.1)
    include_projection_coords : bool, optional
        If True, include projection (gridded) parameter coordinates as features.
        This provides spatial context, helping DBSCAN recognize that nearby grid points
        with similar continuous parameters form connected clusters. Highly recommended.
        (default: True)
    projection_weight : float, optional
        Relative weight for projection coordinates compared to continuous parameters.
        Higher values emphasize spatial connectivity. (default: 1.0)

    Returns
    -------
    cluster_labels : dict
        Maps grid_idx (tuple) -> cluster_label (int)
        Label -1 indicates noise points (for DBSCAN)
    cluster_info : dict
        Statistics and metadata about clusters:
        - 'cluster_sizes': dict mapping cluster_id -> number of points
        - 'cluster_centers': dict mapping cluster_id -> mean continuous params
        - 'cluster_spreads': dict mapping cluster_id -> std continuous params
        - 'n_noise': number of noise points (label -1)
        - 'unique_labels': set of all cluster labels
        - 'method': clustering method used
    boundary_points : set
        Grid indices (tuples) at cluster boundaries (neighbors in different clusters)

    Examples
    --------
    >>> cluster_labels, cluster_info, boundary_points = cluster_coarse_grid_by_modes(
    ...     coarse_solution, method='dbscan'
    ... )
    >>> print(f"Found {len(cluster_info['unique_labels'])} clusters")
    >>> print(f"Boundary points: {len(boundary_points)}")
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "Clustering requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    # Extract features from coarse solutions
    grid_indices = []
    feature_list = []
    grid_axes = coarse_solution['grid_axes']
    n_proj_dims = len(grid_axes)

    for grid_idx in sorted(coarse_solution['solutions'].keys()):
        solution = coarse_solution['solutions'][grid_idx]
        cont_params = solution['continuous_params']

        # Start with continuous parameters
        features = list(cont_params)

        # Add projection coordinates if requested
        if include_projection_coords:
            # Get the actual parameter values at this grid point
            proj_coords = [grid_axes[i][grid_idx[i]] for i in range(n_proj_dims)]
            # Weight projection coordinates
            proj_coords_weighted = [c * projection_weight for c in proj_coords]
            features.extend(proj_coords_weighted)

        # Add likelihood if requested
        if include_likelihood:
            likelihood = solution['likelihood']
            features.append(likelihood * likelihood_weight)

        grid_indices.append(grid_idx)
        feature_list.append(np.array(features))

    if len(feature_list) == 0:
        logger.warning("No coarse solutions to cluster")
        return {}, {}, set()

    features_array = np.array(feature_list)
    n_points, n_features = features_array.shape
    n_cont_dims = len(coarse_solution['continuous_dims'])

    logger.info(f"Clustering {n_points} coarse grid points by modes...")

    # Build feature description
    feature_desc = f"{n_cont_dims} continuous params"
    if include_projection_coords:
        feature_desc += f" + {n_proj_dims} projection coords (weight={projection_weight})"
    if include_likelihood:
        feature_desc += f" + likelihood (weight={likelihood_weight})"
    logger.info(f"  Features: {feature_desc}")

    # Normalize features for clustering (critical for stability)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # Perform clustering
    if method == 'dbscan':
        # DBSCAN: Density-based, finds arbitrary shaped clusters, auto-determines number
        if min_samples is None:
            min_samples = max(2, n_cont_dims)

        # Auto-estimate eps if not provided
        if eps is None:
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(features_scaled)
            distances, _ = nn.kneighbors(features_scaled)

            # Use 90th percentile as base estimate
            eps_base = np.percentile(distances[:, -1], 90)

            # Apply multiplier to be more permissive
            # This prevents over-segmentation from small variations in continuous params
            # Default factor of 3 works well empirically - creates clusters for distinct modes
            # while allowing smooth variation within a mode
            eps = eps_base * eps_multiplier

            logger.info(f"  Auto-estimated DBSCAN eps = {eps:.3f} (base: {eps_base:.3f}, multiplier: {eps_multiplier})")

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(features_scaled)
        logger.info(f"  DBSCAN parameters: eps={eps:.3f}, min_samples={min_samples}")

    elif method == 'hierarchical':
        # Hierarchical clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
        labels = clustering.fit_predict(features_scaled)
        logger.info(f"  Hierarchical clustering: distance_threshold={distance_threshold}")

    elif method == 'kmeans':
        # K-means: need to specify number of clusters
        from sklearn.cluster import KMeans
        clustering = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = clustering.fit_predict(features_scaled)
        logger.info(f"  K-means: n_clusters={n_clusters}")

    else:
        raise ValueError(f"Unknown clustering method: {method}. "
                        f"Must be 'dbscan', 'hierarchical', or 'kmeans'")

    # Create mapping from grid_idx to cluster label
    cluster_labels = {grid_idx: int(label) for grid_idx, label in zip(grid_indices, labels)}

    # Compute cluster statistics (only on continuous params, not likelihood)
    cont_params_array = np.array([solution['continuous_params']
                                   for solution in coarse_solution['solutions'].values()])
    cluster_info = _compute_cluster_statistics(
        cluster_labels, cont_params_array, grid_indices, coarse_solution, method
    )

    # Identify boundary points
    boundary_points = _identify_cluster_boundaries(
        cluster_labels, coarse_solution['grid_shape']
    )

    # Log results
    n_clusters_found = len(cluster_info['cluster_sizes'])
    n_noise = cluster_info['n_noise']
    logger.info(f"Clustering complete:")
    logger.info(f"  Clusters found: {n_clusters_found}")
    if n_noise > 0:
        logger.info(f"  Noise points: {n_noise} ({100*n_noise/n_points:.1f}%)")
    logger.info(f"  Boundary points: {len(boundary_points)} ({100*len(boundary_points)/n_points:.1f}%)")

    for cluster_id in sorted(cluster_info['cluster_sizes'].keys()):
        size = cluster_info['cluster_sizes'][cluster_id]
        logger.info(f"    Cluster {cluster_id}: {size} points ({100*size/n_points:.1f}%)")

    return cluster_labels, cluster_info, boundary_points


def _compute_cluster_statistics(cluster_labels, cont_params_array,
                                 grid_indices, coarse_solution, method):
    """
    Compute statistics for each cluster.

    Parameters
    ----------
    cluster_labels : dict
        Maps grid_idx -> cluster_label
    cont_params_array : np.ndarray
        Array of continuous parameters, shape (n_points, n_cont_dims)
    grid_indices : list
        List of grid indices (in same order as cont_params_array)
    coarse_solution : dict
        Coarse grid solution
    method : str
        Clustering method used

    Returns
    -------
    dict
        Cluster statistics
    """
    unique_labels = set(cluster_labels.values())

    cluster_sizes = {}
    cluster_centers = {}
    cluster_spreads = {}
    cluster_max_likelihoods = {}

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise points

        # Find all points in this cluster
        mask = np.array([cluster_labels[grid_idx] == cluster_id for grid_idx in grid_indices])
        cluster_params = cont_params_array[mask]
        cluster_grid_indices = [grid_indices[i] for i, m in enumerate(mask) if m]

        # Get likelihoods for this cluster
        cluster_likelihoods = [coarse_solution['solutions'][idx]['likelihood']
                               for idx in cluster_grid_indices]

        cluster_sizes[cluster_id] = len(cluster_params)
        cluster_centers[cluster_id] = np.mean(cluster_params, axis=0)
        cluster_spreads[cluster_id] = np.std(cluster_params, axis=0)
        cluster_max_likelihoods[cluster_id] = max(cluster_likelihoods)

    n_noise = sum(1 for label in cluster_labels.values() if label == -1)

    return {
        'cluster_sizes': cluster_sizes,
        'cluster_centers': cluster_centers,
        'cluster_spreads': cluster_spreads,
        'cluster_max_likelihoods': cluster_max_likelihoods,
        'n_noise': n_noise,
        'unique_labels': unique_labels,
        'method': method
    }


def _identify_cluster_boundaries(cluster_labels, grid_shape):
    """
    Find grid points at cluster boundaries.
    A point is at a boundary if any neighbor belongs to a different cluster.

    Parameters
    ----------
    cluster_labels : dict
        Maps grid_idx -> cluster_label
    grid_shape : tuple
        Shape of the grid

    Returns
    -------
    set
        Grid indices at boundaries
    """
    boundary_points = set()

    for grid_idx, my_cluster in cluster_labels.items():
        # Get neighbors in the grid
        neighbors = _get_grid_neighbors(grid_idx, grid_shape)

        # Check if any neighbor has a different cluster label
        for neighbor_idx in neighbors:
            if neighbor_idx in cluster_labels:
                neighbor_cluster = cluster_labels[neighbor_idx]
                if neighbor_cluster != my_cluster:
                    # Found a boundary
                    boundary_points.add(grid_idx)
                    break

    return boundary_points


def _get_grid_neighbors(grid_idx, grid_shape):
    """
    Get valid neighbor indices in the grid (using 6-connectivity for efficiency).

    Parameters
    ----------
    grid_idx : tuple
        Grid index
    grid_shape : tuple
        Shape of the grid

    Returns
    -------
    list
        List of neighbor grid indices (tuples)
    """
    neighbors = []

    # grid_idx is already a tuple, use it directly
    for dim in range(len(grid_shape)):
        for delta in [-1, +1]:
            neighbor_coords = list(grid_idx)
            neighbor_coords[dim] += delta

            # Check bounds
            if 0 <= neighbor_coords[dim] < grid_shape[dim]:
                neighbors.append(tuple(neighbor_coords))

    return neighbors


def get_cluster_based_initialization(fine_grid_coords, coarse_solution,
                                     cluster_labels, cluster_info,
                                     boundary_points, interpolator):
    """
    Get initialization parameters for a fine grid point using cluster-aware interpolation.

    For fine grid points near cluster boundaries, this function extrapolates continuous
    parameters from all nearby clusters and returns multiple candidate parameter sets
    to be evaluated. The caller should evaluate the likelihood at each candidate and
    select the best one.

    Parameters
    ----------
    fine_grid_coords : array-like
        Projection coordinates of the fine grid point, shape (n_proj_dims,)
    coarse_solution : dict
        Coarse grid solution
    cluster_labels : dict
        Maps grid_idx -> cluster_label
    cluster_info : dict
        Cluster statistics
    boundary_points : set
        Grid indices at cluster boundaries
    interpolator : GridInterpolator or MultiGPInterpolator
        Interpolator for continuous parameters

    Returns
    -------
    candidates : list of dict
        List of candidate parameter sets. Each dict contains:
        - 'continuous_params': np.ndarray - continuous parameter values
        - 'cluster_id': int or None - source cluster ID
        - 'method': str - how these params were generated
    """
    # Find k nearest coarse grid points
    k = 6
    k_nearest_indices, k_nearest_distances = _find_k_nearest_coarse_points(
        fine_grid_coords, coarse_solution, k=k
    )

    # Check if any of the nearest points are at boundaries
    nearby_boundaries = [idx for idx in k_nearest_indices if idx in boundary_points]

    if len(nearby_boundaries) == 0:
        # Not near a boundary - use standard interpolation
        continuous_params = interpolator.interpolate(fine_grid_coords)
        return [{
            'continuous_params': continuous_params,
            'cluster_id': None,
            'method': 'interpolated'
        }]

    # Near a boundary - determine which clusters are involved
    nearby_clusters = set()
    for idx in k_nearest_indices:
        if idx in cluster_labels:
            cluster_id = cluster_labels[idx]
            if cluster_id != -1:  # Exclude noise points
                nearby_clusters.add(cluster_id)

    if len(nearby_clusters) <= 1:
        # Only one cluster nearby - use standard interpolation
        continuous_params = interpolator.interpolate(fine_grid_coords)
        return [{
            'continuous_params': continuous_params,
            'cluster_id': list(nearby_clusters)[0] if nearby_clusters else None,
            'method': 'interpolated'
        }]

    # Multiple clusters nearby - generate candidates from each cluster
    candidates = []

    # Always add interpolated version as first candidate
    interpolated_params = interpolator.interpolate(fine_grid_coords)
    candidates.append({
        'continuous_params': interpolated_params,
        'cluster_id': None,
        'method': 'interpolated'
    })

    # For each nearby cluster, extrapolate continuous parameters
    for cluster_id in nearby_clusters:
        # Find all nearby coarse points in this cluster
        cluster_points = []
        cluster_distances = []
        cluster_params = []

        for idx, dist in zip(k_nearest_indices, k_nearest_distances):
            if idx in cluster_labels and cluster_labels[idx] == cluster_id:
                cluster_points.append(idx)
                cluster_distances.append(dist)
                cluster_params.append(coarse_solution['solutions'][idx]['continuous_params'])

        if len(cluster_points) > 0:
            # Use inverse-distance weighted average for smooth interpolation within the cluster
            cluster_params = np.array(cluster_params)
            cluster_distances = np.array(cluster_distances)

            # Avoid division by zero for exact matches
            cluster_distances = np.maximum(cluster_distances, 1e-10)

            # Inverse distance weights
            weights = 1.0 / cluster_distances
            weights /= np.sum(weights)

            # Weighted average
            continuous_params = np.sum(cluster_params * weights[:, np.newaxis], axis=0)

            candidates.append({
                'continuous_params': continuous_params,
                'cluster_id': cluster_id,
                'method': 'cluster_extrapolated',
                'n_points_used': len(cluster_points)
            })

    return candidates


def _find_k_nearest_coarse_points(fine_coords, coarse_solution, k=6):
    """
    Find k nearest coarse grid points to fine grid coordinates.

    Parameters
    ----------
    fine_coords : array-like
        Projection coordinates of fine grid point, shape (n_proj_dims,)
    coarse_solution : dict
        Coarse grid solution
    k : int, optional
        Number of nearest neighbors to find (default: 6)

    Returns
    -------
    k_nearest_indices : list
        List of k nearest grid indices (tuples)
    k_nearest_distances : list
        Corresponding distances
    """
    coarse_coords_list = []
    coarse_indices = []

    grid_axes = coarse_solution['grid_axes']

    for grid_idx in coarse_solution['solutions'].keys():
        # Get projection coordinates of this coarse grid point
        proj_coords = np.array([grid_axes[i][grid_idx[i]] for i in range(len(grid_axes))])
        coarse_coords_list.append(proj_coords)
        coarse_indices.append(grid_idx)

    if len(coarse_coords_list) == 0:
        return [], []

    coarse_coords_array = np.array(coarse_coords_list)
    fine_coords = np.asarray(fine_coords)

    # Compute distances
    distances = np.linalg.norm(coarse_coords_array - fine_coords, axis=1)

    # Get k nearest (or all if fewer than k)
    k_actual = min(k, len(distances))
    k_nearest_mask = np.argsort(distances)[:k_actual]
    k_nearest_indices = [coarse_indices[i] for i in k_nearest_mask]
    k_nearest_distances = [distances[i] for i in k_nearest_mask]

    return k_nearest_indices, k_nearest_distances
