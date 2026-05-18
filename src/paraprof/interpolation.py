"""Grid interpolation utilities for refinement runs."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .logger import get_logger

logger = get_logger()

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.debug("scikit-learn not available. Clustering features will not be available.")


class GridInterpolator:
    """Interpolate coarse-grid profiled-param values to seed a refined-grid run.

    Builds one RegularGridInterpolator per profiled dimension over the
    sparse coarse solution. Takes the dict returned by
    ``ProfileProjector.export_grid_solution()``.
    """

    def __init__(self, coarse_grid_solution):
        self.grid_axes = coarse_grid_solution['grid_axes']
        self.projection_dims = coarse_grid_solution['projection_dims']
        self.profiled_dims = coarse_grid_solution['profiled_dims']
        self.grid_shape = coarse_grid_solution['grid_shape']
        self.solutions = coarse_grid_solution['solutions']

        self.n_proj_dims = len(self.projection_dims)
        self.n_prof_dims = len(self.profiled_dims)

        # Build interpolators for each profiled parameter
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
        """Build one RegularGridInterpolator per profiled-param dimension."""
        self.interpolators = []
        if self.n_prof_dims == 0:
            return

        for prof_dim_idx in range(self.n_prof_dims):
            values_grid = np.full(self.grid_shape, np.nan)
            for grid_idx, solution in self.solutions.items():
                values_grid[grid_idx] = solution['profiled_params'][prof_dim_idx]

            n_valid = int(np.sum(~np.isnan(values_grid)))

            if n_valid == 0:
                logger.warning(f" No valid data for profiled dim {prof_dim_idx}. "
                               f"Interpolator will return NaN.")
                interpolator = None
            elif n_valid < 2**self.n_proj_dims:
                logger.warning(f" Sparse data ({n_valid} points) for profiled dim {prof_dim_idx}. "
                               f"Using nearest neighbor interpolation.")
                interpolator = RegularGridInterpolator(
                    self.grid_axes, values_grid,
                    method='nearest', bounds_error=False, fill_value=None,
                )
            else:
                # fill_value=NaN so callers can detect out-of-bounds; we clip
                # query points to the grid in _clip_to_grid first.
                interpolator = RegularGridInterpolator(
                    self.grid_axes, values_grid,
                    method='linear', bounds_error=False, fill_value=np.nan,
                )

            self.interpolators.append(interpolator)


    def _clip_to_grid(self, coords):
        """Clip query coordinates into the grid domain.

        Linear RegularGridInterpolator extrapolates wildly outside the
        grid; clipping forces nearest-neighbour behaviour at the boundary.
        """
        coords = np.asarray(coords, dtype=float).copy()
        for i, axis in enumerate(self.grid_axes):
            coords[..., i] = np.clip(coords[..., i], axis[0], axis[-1])
        return coords


    def interpolate(self, projection_coords):
        """Interpolated profiled params at the given projection coords, or None."""
        if self.n_prof_dims == 0:
            return None

        projection_coords = self._clip_to_grid(np.asarray(projection_coords))
        coords_for_interp = projection_coords.reshape(1, -1)

        interpolated_params = np.zeros(self.n_prof_dims)
        for i, interpolator in enumerate(self.interpolators):
            if interpolator is None:
                interpolated_params[i] = np.nan
                continue
            try:
                interpolated_params[i] = interpolator(coords_for_interp)[0]
            except Exception as e:
                logger.warning(f"Warning: Interpolation failed for profiled dim {i}: {e}")
                interpolated_params[i] = np.nan

        return interpolated_params


    def get_interpolated_likelihood(self, projection_coords):
        """Interpolated likelihood at given projection coords. Lazily built and cached."""
        if self._likelihood_interpolator is None:
            likelihood_grid = np.full(self.grid_shape, np.nan)
            for grid_idx, solution in self.solutions.items():
                likelihood_grid[grid_idx] = solution['likelihood']
            self._likelihood_interpolator = RegularGridInterpolator(
                self.grid_axes, likelihood_grid,
                method='linear', bounds_error=False, fill_value=np.nan,
            )

        projection_coords = self._clip_to_grid(np.asarray(projection_coords)).reshape(1, -1)
        return self._likelihood_interpolator(projection_coords)[0]


    def get_coverage_fraction(self):
        """Fraction of grid points with valid solutions, in [0, 1]."""
        total_grid_points = np.prod(self.grid_shape)
        valid_points = len(self.solutions)
        return valid_points / total_grid_points if total_grid_points > 0 else 0.0


    def __repr__(self):
        coverage = self.get_coverage_fraction()
        return (f"GridInterpolator(grid_shape={self.grid_shape}, "
                f"n_profiled_dims={self.n_prof_dims}, "
                f"coverage={coverage:.1%})")


# --- Clustering-based boundary detection for refinement ---

def cluster_coarse_grid_by_modes(coarse_solution, method='dbscan',
                                  eps=None, min_samples=None,
                                  eps_multiplier=3.0,
                                  distance_threshold=1.0, n_clusters=2,
                                  include_likelihood=False, likelihood_weight=0.1,
                                  include_projection_coords=True, projection_weight=1.0):
    """Cluster coarse-grid cells by their optimal profiled parameters.

    Each cluster represents a distinct mode being tracked in profiled-param
    space. Returns ``(cluster_labels, cluster_info, boundary_points)``:
    ``cluster_labels`` maps grid_idx -> label (-1 = DBSCAN noise),
    ``cluster_info`` carries sizes/centers/spreads, and ``boundary_points``
    are cells whose neighbours sit in a different cluster.

    Including projection coords as features (the default) gives DBSCAN
    spatial context so connected components with similar profiled params
    are kept together.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "Clustering requires scikit-learn. "
            "Install with: pip install scikit-learn"
        )

    grid_indices = []
    feature_list = []
    grid_axes = coarse_solution['grid_axes']
    n_proj_dims = len(grid_axes)

    for grid_idx in sorted(coarse_solution['solutions'].keys()):
        solution = coarse_solution['solutions'][grid_idx]
        features = list(solution['profiled_params'])

        if include_projection_coords:
            proj_coords = [grid_axes[i][grid_idx[i]] for i in range(n_proj_dims)]
            features.extend(c * projection_weight for c in proj_coords)

        if include_likelihood:
            features.append(solution['likelihood'] * likelihood_weight)

        grid_indices.append(grid_idx)
        feature_list.append(np.array(features))

    if len(feature_list) == 0:
        logger.warning("No coarse solutions to cluster")
        return {}, {}, set()

    features_array = np.array(feature_list)
    n_points, _ = features_array.shape
    n_prof_dims = len(coarse_solution['profiled_dims'])

    logger.info(f"Clustering {n_points} coarse grid points by modes...")

    feature_desc = f"{n_prof_dims} profiled params"
    if include_projection_coords:
        feature_desc += f" + {n_proj_dims} projection coords (weight={projection_weight})"
    if include_likelihood:
        feature_desc += f" + likelihood (weight={likelihood_weight})"
    logger.info(f"  Features: {feature_desc}")

    features_scaled = StandardScaler().fit_transform(features_array)

    if method == 'dbscan':
        if min_samples is None:
            min_samples = max(2, n_prof_dims)
        if eps is None:
            nn = NearestNeighbors(n_neighbors=min_samples)
            nn.fit(features_scaled)
            distances, _ = nn.kneighbors(features_scaled)
            eps_base = np.percentile(distances[:, -1], 90)
            # Multiplier prevents over-segmentation from small variations in
            # profiled params; ~3 keeps distinct modes apart while letting a
            # single mode vary smoothly.
            eps = eps_base * eps_multiplier
            logger.info(f"  Auto-estimated DBSCAN eps = {eps:.3f} (base: {eps_base:.3f}, multiplier: {eps_multiplier})")

        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit_predict(features_scaled)
        logger.info(f"  DBSCAN parameters: eps={eps:.3f}, min_samples={min_samples}")

    elif method == 'hierarchical':
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='ward',
        ).fit_predict(features_scaled)
        logger.info(f"  Hierarchical clustering: distance_threshold={distance_threshold}")

    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(features_scaled)
        logger.info(f"  K-means: n_clusters={n_clusters}")

    else:
        raise ValueError(f"Unknown clustering method: {method}. "
                        f"Must be 'dbscan', 'hierarchical', or 'kmeans'")

    cluster_labels = {grid_idx: int(label) for grid_idx, label in zip(grid_indices, labels)}

    prof_params_array = np.array([solution['profiled_params']
                                   for solution in coarse_solution['solutions'].values()])
    cluster_info = _compute_cluster_statistics(
        cluster_labels, prof_params_array, grid_indices, coarse_solution, method
    )
    boundary_points = _identify_cluster_boundaries(
        cluster_labels, coarse_solution['grid_shape']
    )

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


def _compute_cluster_statistics(cluster_labels, prof_params_array,
                                 grid_indices, coarse_solution, method):
    """Sizes, centers, spreads, and max-likelihood per cluster."""
    unique_labels = set(cluster_labels.values())

    cluster_sizes = {}
    cluster_centers = {}
    cluster_spreads = {}
    cluster_max_likelihoods = {}

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        mask = np.array([cluster_labels[grid_idx] == cluster_id for grid_idx in grid_indices])
        cluster_params = prof_params_array[mask]
        cluster_grid_indices = [grid_indices[i] for i, m in enumerate(mask) if m]
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
    """Grid cells with at least one neighbour in a different cluster."""
    boundary_points = set()
    for grid_idx, my_cluster in cluster_labels.items():
        for neighbor_idx in _get_grid_neighbors(grid_idx, grid_shape):
            if neighbor_idx in cluster_labels and cluster_labels[neighbor_idx] != my_cluster:
                boundary_points.add(grid_idx)
                break
    return boundary_points


def _get_grid_neighbors(grid_idx, grid_shape):
    """6-connectivity in-grid neighbours of ``grid_idx``."""
    neighbors = []
    for dim in range(len(grid_shape)):
        for delta in [-1, +1]:
            neighbor_coords = list(grid_idx)
            neighbor_coords[dim] += delta
            if 0 <= neighbor_coords[dim] < grid_shape[dim]:
                neighbors.append(tuple(neighbor_coords))
    return neighbors


def get_cluster_based_initialization(fine_grid_coords, coarse_solution,
                                     cluster_labels, cluster_info,
                                     boundary_points, interpolator):
    """Candidate profiled-param sets for a fine grid point, cluster-aware.

    Near a cluster boundary, returns one candidate per nearby cluster
    (via inverse-distance weighting within each cluster) plus the plain
    interpolated candidate. Elsewhere, returns just the interpolated one.
    """
    k = 6
    coords_cache = getattr(interpolator, '_coarse_coords_cache', None)
    k_nearest_indices, k_nearest_distances = _find_k_nearest_coarse_points(
        fine_grid_coords, coarse_solution, k=k, coords_cache=coords_cache
    )

    nearby_boundaries = [idx for idx in k_nearest_indices if idx in boundary_points]

    if len(nearby_boundaries) == 0:
        return [{
            'profiled_params': interpolator.interpolate(fine_grid_coords),
            'cluster_id': None,
            'method': 'interpolated'
        }]

    nearby_clusters = {
        cluster_labels[idx] for idx in k_nearest_indices
        if idx in cluster_labels and cluster_labels[idx] != -1
    }

    if len(nearby_clusters) <= 1:
        return [{
            'profiled_params': interpolator.interpolate(fine_grid_coords),
            'cluster_id': next(iter(nearby_clusters), None),
            'method': 'interpolated'
        }]

    candidates = [{
        'profiled_params': interpolator.interpolate(fine_grid_coords),
        'cluster_id': None,
        'method': 'interpolated'
    }]

    for cluster_id in nearby_clusters:
        cluster_distances = []
        cluster_params = []
        for idx, dist in zip(k_nearest_indices, k_nearest_distances):
            if idx in cluster_labels and cluster_labels[idx] == cluster_id:
                cluster_distances.append(dist)
                cluster_params.append(coarse_solution['solutions'][idx]['profiled_params'])

        if not cluster_params:
            continue

        cluster_params = np.array(cluster_params)
        cluster_distances = np.maximum(np.array(cluster_distances), 1e-10)
        weights = 1.0 / cluster_distances
        weights /= weights.sum()
        candidates.append({
            'profiled_params': np.sum(cluster_params * weights[:, np.newaxis], axis=0),
            'cluster_id': cluster_id,
            'method': 'cluster_extrapolated',
            'n_points_used': len(cluster_params),
        })

    return candidates


def _find_k_nearest_coarse_points(fine_coords, coarse_solution, k=6, coords_cache=None):
    """k nearest coarse-grid cells to ``fine_coords`` as (indices, distances)."""
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
