"""
Prediction models for speculative parallel evaluation.

This module contains strategies for predicting which grid points are likely
to be activated next, enabling speculative pre-computation.
"""
import numpy as np
from .logger import get_logger

logger = get_logger()


def predict_neighbor_activations(sampler, active_grid_indices):
    """
    Predict neighbors likely to be activated based on active point improvements.

    Strategy: Grid points adjacent to high-improvement active points are
    likely to be activated soon by dynamic activation logic.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance with current state
    active_grid_indices : set
        Set of currently active grid indices

    Returns
    -------
    list of (grid_idx, priority, reason)
        Predictions sorted by priority (higher = more likely)
    """
    predictions = []

    for grid_idx in active_grid_indices:
        if grid_idx not in sampler.population:
            continue

        state = sampler.population[grid_idx]

        # Calculate improvement rate
        if len(state.get('improvement_history', [])) > 0:
            improvement_rate = np.mean(state['improvement_history'])
        else:
            improvement_rate = 0.0

        # High improvement = likely to activate neighbors
        if improvement_rate > sampler.dynamic_activation_improvement_threshold:
            for neighbor_idx in sampler._get_valid_neighbors(grid_idx):
                if neighbor_idx not in sampler.active_grid_indices and \
                   neighbor_idx not in sampler.pending_activation_indices:

                    # Estimate neighbor fitness via interpolation
                    est_fitness = _estimate_neighbor_fitness(sampler, neighbor_idx)

                    # Priority based on improvement rate and estimated fitness
                    priority = improvement_rate * max(0, est_fitness + 1e10)
                    predictions.append((neighbor_idx, priority, 'neighbor_proximity'))

    return predictions


def predict_wavefront_extrapolation(sampler, active_grid_indices):
    """
    Extrapolate activation wavefront direction to predict next ring.

    Strategy: Fit movement direction of active set, predict points in the
    extrapolated direction.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance with current state
    active_grid_indices : set
        Set of currently active grid indices

    Returns
    -------
    list of (grid_idx, priority, reason)
        Predictions sorted by priority (higher = more likely)
    """
    if len(active_grid_indices) < 3:
        return []  # Need at least 3 points to extrapolate

    predictions = []

    # Get initial activation indices (if available)
    if not hasattr(sampler, '_initial_activation_indices') or \
       len(sampler._initial_activation_indices) == 0:
        # Use current active indices as baseline
        return []

    # Calculate center of mass of active points
    active_coords = [sampler._grid_idx_to_coords(idx) for idx in active_grid_indices]
    com = np.mean(active_coords, axis=0)

    # Calculate wavefront direction (away from initial activation)
    initial_coords = [sampler._grid_idx_to_coords(idx)
                     for idx in sampler._initial_activation_indices]
    initial_com = np.mean(initial_coords, axis=0)

    direction = com - initial_com
    direction_norm_val = np.linalg.norm(direction)

    if direction_norm_val < 1e-10:
        return []

    direction_norm = direction / direction_norm_val

    # Predict next ring of points in this direction
    for grid_idx in active_grid_indices:
        coords = np.array(sampler._grid_idx_to_coords(grid_idx))

        # Extrapolate one step in wavefront direction
        next_coords = coords + direction_norm
        next_coords_rounded = np.round(next_coords).astype(int)

        # Check if valid and not already activated
        if sampler._coords_in_bounds(next_coords_rounded):
            next_idx = sampler._coords_to_grid_idx(next_coords_rounded)

            if next_idx not in sampler.active_grid_indices and \
               next_idx not in sampler.pending_activation_indices:

                # Priority based on source point's fitness
                priority = sampler.population[grid_idx]['best_fitness']
                predictions.append((next_idx, priority, 'wavefront_extrapolation'))

    return predictions


def predict_by_neighbor_interpolation(sampler, active_grid_indices):
    """
    Predict high-likelihood points via neighbor fitness interpolation.

    Strategy: If interpolated fitness from neighbors exceeds ROI threshold,
    the point is likely to be activated.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance with current state
    active_grid_indices : set
        Set of currently active grid indices

    Returns
    -------
    list of (grid_idx, priority, reason)
        Predictions sorted by priority (higher = more likely)
    """
    predictions = []
    roi_likelihood_threshold = sampler.global_max_target_val - sampler.roi_threshold

    # Get all inactive neighbors of active points
    candidates = set()
    for grid_idx in active_grid_indices:
        for neighbor_idx in sampler._get_valid_neighbors(grid_idx):
            if neighbor_idx not in sampler.active_grid_indices and \
               neighbor_idx not in sampler.pending_activation_indices:
                candidates.add(neighbor_idx)

    for candidate_idx in candidates:
        # Interpolate fitness from activated neighbors
        neighbor_fitnesses = []
        for neighbor_idx in sampler._get_valid_neighbors(candidate_idx):
            if neighbor_idx in sampler.population:
                neighbor_fitnesses.append(sampler.population[neighbor_idx]['best_fitness'])

        if len(neighbor_fitnesses) >= 2:
            # Average of top 2 neighbors
            neighbor_fitnesses.sort(reverse=True)
            interpolated_fitness = np.mean(neighbor_fitnesses[:2])

            if interpolated_fitness > roi_likelihood_threshold:
                priority = interpolated_fitness
                predictions.append((candidate_idx, priority, 'neighbor_interpolation'))

    return predictions


def rank_speculation_targets(predictions, max_targets=None):
    """
    Rank and prioritize speculation targets.

    Combines predictions from multiple strategies, removes duplicates,
    and returns top-ranked targets.

    Parameters
    ----------
    predictions : list of (grid_idx, priority, reason)
        Raw predictions from multiple strategies
    max_targets : int, optional
        Maximum number of targets to return

    Returns
    -------
    list of (grid_idx, priority, reason)
        Sorted and deduplicated predictions
    """
    if not predictions:
        return []

    # Deduplicate: keep highest priority for each grid_idx
    target_dict = {}
    for grid_idx, priority, reason in predictions:
        if grid_idx not in target_dict or priority > target_dict[grid_idx][0]:
            target_dict[grid_idx] = (priority, reason)

    # Convert back to list and sort by priority
    ranked = [(idx, pri, reason) for idx, (pri, reason) in target_dict.items()]
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Limit to max_targets if specified
    if max_targets is not None:
        ranked = ranked[:max_targets]

    return ranked


def _estimate_neighbor_fitness(sampler, grid_idx):
    """
    Estimate fitness at a grid point based on activated neighbors.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    grid_idx : int
        Grid index to estimate fitness for

    Returns
    -------
    float
        Estimated fitness (or -inf if no neighbors available)
    """
    neighbor_fitnesses = []

    for neighbor_idx in sampler._get_valid_neighbors(grid_idx):
        if neighbor_idx in sampler.population:
            neighbor_fitnesses.append(sampler.population[neighbor_idx]['best_fitness'])

    if not neighbor_fitnesses:
        return -np.inf

    # Use average of all activated neighbors
    return np.mean(neighbor_fitnesses)
