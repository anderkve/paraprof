"""
Grid interpolation utilities for refinement runs.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .logger import get_logger

logger = get_logger()

# Check if scikit-learn is available for clustering
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available. Clustering features will not be available.")


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
            Dictionary returned by ProfileProjector.export_grid_solution()
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

        # Pre-compute coarse grid coordinates for performance
        self._coarse_coords_cache = {}
        for grid_idx in self.solutions.keys():
            proj_coords = np.array([self.grid_axes[i][grid_idx[i]]
                                   for i in range(len(self.grid_axes))])
            self._coarse_coords_cache[grid_idx] = proj_coords


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
    interpolator : GridInterpolator
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
    # Use interpolator's cached coordinates if available
    k = 6
    coords_cache = getattr(interpolator, '_coarse_coords_cache', None)
    k_nearest_indices, k_nearest_distances = _find_k_nearest_coarse_points(
        fine_grid_coords, coarse_solution, k=k, coords_cache=coords_cache
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


def _find_k_nearest_coarse_points(fine_coords, coarse_solution, k=6, coords_cache=None):
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
    coords_cache : dict, optional
        Pre-computed coordinates cache {grid_idx: proj_coords}

    Returns
    -------
    k_nearest_indices : list
        List of k nearest grid indices (tuples)
    k_nearest_distances : list
        Corresponding distances
    """
    coarse_coords_list = []
    coarse_indices = []

    # Use cache if provided, otherwise compute
    if coords_cache is not None:
        for grid_idx, proj_coords in coords_cache.items():
            coarse_coords_list.append(proj_coords)
            coarse_indices.append(grid_idx)
    else:
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
