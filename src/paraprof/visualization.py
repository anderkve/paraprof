"""
Visualization utilities for profile likelihood plots.
Supports 1D, 2D, and N-D projections.
"""
import numpy as np
import itertools
from .logger import get_logger

logger = get_logger()


def plot_profiles(sampler, filename, plot_settings=None):
    """
    Generates and saves profile likelihood plots for any dimensionality.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance containing the profile likelihood grid
    filename : str
        Output filename (without extension)
    plot_settings : dict, optional
        Plot settings with keys:
        - 'dpi': int (default: 300)
        - 'filetype': str (default: 'png')
        - 'slice_mode': str (default: 'max') - for 3D+: 'max' or 'all'
        - 'vmin': float (default: -4.0) - colorbar minimum
        - 'vmax': float (default: 0.0) - colorbar maximum
        - 'contour_levels': list (default: [-3.0, -1.0]) - contour levels
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("\nMatplotlib not found. Skipping visualization.")
        return

    # Set default plot settings
    if plot_settings is None:
        plot_settings = {}

    # Route to appropriate plotting function based on dimensionality
    if sampler.n_proj_dims == 1:
        _plot_1d_profile(sampler, filename, plot_settings)
    elif sampler.n_proj_dims == 2:
        _plot_2d_profile(sampler, filename, plot_settings)
    elif sampler.n_proj_dims >= 3:
        _plot_nd_profile(sampler, filename, plot_settings)
    else:
        logger.info(f"Invalid projection dimensions: {sampler.n_proj_dims}")


def _plot_1d_profile(sampler, filename, plot_settings):
    """
    Plots a 1D profile likelihood as a line plot.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    filename : str
        Output filename (without extension)
    plot_settings : dict
        Plot settings
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')

    # Extract 1D profile
    grid_axis = sampler.grid_axes[0]
    profile_1d = np.full(len(grid_axis), np.nan)

    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_1d[grid_idx[0]] = fitness

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    dim = sampler.projection_dims[0]

    # Plot profile
    valid_mask = ~np.isnan(profile_1d)
    ax.plot(grid_axis[valid_mask], profile_1d[valid_mask], 'b-', linewidth=2, label='Profile Likelihood')
    ax.scatter(grid_axis[valid_mask], profile_1d[valid_mask], c='blue', s=20, zorder=5)

    # Mark active points
    active_points = []
    active_likelihoods = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
            coords = sampler._get_grid_coords_from_indices(grid_idx)
            active_points.append(coords[0])
            active_likelihoods.append(state['best_fitness'])

    if active_points:
        ax.scatter(active_points, active_likelihoods, c='red', s=50,
                   marker='o', edgecolor='black', linewidth=1.5,
                   label='Active DE Points', zorder=10)

    # Add confidence level lines
    if not np.all(np.isnan(profile_1d)):
        max_likelihood = np.nanmax(profile_1d)
        for delta, label in [(-1.0, '68% CL'), (-4.0, '95% CL')]:
            level = max_likelihood + delta
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.7, label=label)

    ax.set_xlabel(f'Parameter {dim}', fontsize=12)
    ax.set_ylabel('Log Likelihood', fontsize=12)
    ax.set_title(f'1D Profile Likelihood for Parameter {dim}', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    # Save the plot
    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved 1D plot to: {output_filename}")


def _plot_2d_profile(sampler, filename, plot_settings):
    """
    Plots a 2D profile likelihood as a heatmap with contours.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    filename : str
        Output filename (without extension)
    plot_settings : dict
        Plot settings
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    vmin = plot_settings.get('vmin', -4.0)
    vmax = plot_settings.get('vmax', 0.0)
    contour_levels = plot_settings.get('contour_levels', [-3.0, -1.0])

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(7, 6),
                            gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.0})
    ax = axes[0]

    dim1, dim2 = sampler.projection_dims

    # Create 2D grid from sparse dict
    profile_2d = np.full(sampler.grid_shape, -np.inf)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_2d[grid_idx] = fitness

    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    masked_profile = np.ma.masked_where(profile_2d == -np.inf, profile_2d)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='0.75')

    im = ax.imshow(masked_profile.T, extent=extent, aspect='equal', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    # Add white contour lines
    X, Y = np.meshgrid(sampler.grid_axes[0], sampler.grid_axes[1])
    ax.contour(X, Y, masked_profile.T, levels=contour_levels, colors='white', linewidths=1.0)

    # Mark active points
    active_points = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
            coords = sampler._get_grid_coords_from_indices(grid_idx)
            active_points.append(coords)

    if active_points:
        active_points = np.array(active_points)
        ax.scatter(active_points[:, 0], active_points[:, 1], c='cyan', s=3,
                   edgecolor='black', lw=0.5, label='Active DE Points')

    ax.set_title(f'Profile likelihood for parameters {sampler.projection_dims}')
    ax.set_xlabel(f'Parameter {dim1}')
    ax.set_ylabel(f'Parameter {dim2}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    cax = axes[1]
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log Likelihood')

    fig.tight_layout()

    # Save the plot
    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved 2D plot to: {output_filename}")


def _plot_nd_profile(sampler, filename, plot_settings):
    """
    Plots a 3D+ profile likelihood as 2D slices.

    For N-dimensional grids (N >= 3), creates 2D slice plots showing
    all pairwise projections through the maximum likelihood point.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    filename : str
        Output filename (without extension)
    plot_settings : dict
        Plot settings with optional 'slice_mode':
        - 'max': slices through maximum likelihood point (default)
        - 'all': marginalized projections over all other dimensions
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    slice_mode = plot_settings.get('slice_mode', 'max')
    vmin = plot_settings.get('vmin', -4.0)
    vmax = plot_settings.get('vmax', 0.0)

    n_dims = sampler.n_proj_dims
    dims = sampler.projection_dims

    # Find maximum likelihood point
    max_likelihood = -np.inf
    max_grid_idx = None
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        if fitness > max_likelihood:
            max_likelihood = fitness
            max_grid_idx = grid_idx

    if max_grid_idx is None:
        logger.info("No profile likelihood data found. Skipping plot.")
        return

    # Generate all pairwise dimension combinations
    dim_pairs = list(itertools.combinations(range(n_dims), 2))
    n_pairs = len(dim_pairs)

    # Create subplot grid
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (dim_i, dim_j) in enumerate(dim_pairs):
        ax = axes[idx]

        if slice_mode == 'max':
            # Extract 2D slice through maximum likelihood point
            profile_slice = _extract_2d_slice(sampler, dim_i, dim_j, max_grid_idx)
        else:  # slice_mode == 'all' (marginalized)
            # Marginalize over all other dimensions
            profile_slice = _marginalize_to_2d(sampler, dim_i, dim_j)

        # Plot the slice
        extent = [sampler.grid_axes[dim_i][0], sampler.grid_axes[dim_i][-1],
                  sampler.grid_axes[dim_j][0], sampler.grid_axes[dim_j][-1]]

        masked_profile = np.ma.masked_where(profile_slice == -np.inf, profile_slice)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='0.75')

        im = ax.imshow(masked_profile.T, extent=extent, aspect='equal',
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        # Add contours
        X, Y = np.meshgrid(sampler.grid_axes[dim_i], sampler.grid_axes[dim_j])
        ax.contour(X, Y, masked_profile.T, levels=[-3.0, -1.0],
                  colors='white', linewidths=1.0)

        # Mark the maximum point (if in slice mode)
        if slice_mode == 'max':
            max_coords = sampler._get_grid_coords_from_indices(max_grid_idx)
            ax.scatter([max_coords[dim_i]], [max_coords[dim_j]],
                      c='red', s=100, marker='*', edgecolor='white',
                      linewidth=1.5, label='Global Max', zorder=10)

        ax.set_xlabel(f'Param {dims[dim_i]}')
        ax.set_ylabel(f'Param {dims[dim_j]}')
        ax.set_title(f'Dims {dims[dim_i]}-{dims[dim_j]}')
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add colorbar for each subplot
        fig.colorbar(im, ax=ax, orientation='vertical', label='Log L', fraction=0.046)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    mode_str = "Max Slice" if slice_mode == 'max' else "Marginalized"
    fig.suptitle(f'{n_dims}D Profile Likelihood - {mode_str} Projections\n'
                 f'Dimensions: {dims}', fontsize=14, y=0.995)
    fig.tight_layout()

    # Save the plot
    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved {n_dims}D plot ({n_pairs} slices) to: {output_filename}")


def _extract_2d_slice(sampler, dim_i, dim_j, anchor_idx):
    """
    Extracts a 2D slice through the N-D grid at a fixed anchor point.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    dim_i, dim_j : int
        Dimension indices to extract (in range 0 to n_proj_dims-1)
    anchor_idx : tuple
        Grid index of the anchor point for the slice

    Returns
    -------
    np.ndarray
        2D array of shape (grid_points[dim_i], grid_points[dim_j])
    """
    shape_i = sampler.grid_shape[dim_i]
    shape_j = sampler.grid_shape[dim_j]
    slice_2d = np.full((shape_i, shape_j), -np.inf)

    # Iterate through all grid points and extract those matching the anchor
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        # Check if all dimensions except dim_i and dim_j match the anchor
        matches_anchor = True
        for k in range(sampler.n_proj_dims):
            if k != dim_i and k != dim_j:
                if grid_idx[k] != anchor_idx[k]:
                    matches_anchor = False
                    break

        if matches_anchor:
            slice_2d[grid_idx[dim_i], grid_idx[dim_j]] = fitness

    return slice_2d


def _marginalize_to_2d(sampler, dim_i, dim_j):
    """
    Marginalizes the N-D profile likelihood to a 2D projection.

    Takes the maximum likelihood over all other dimensions for each
    (dim_i, dim_j) grid point pair.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance
    dim_i, dim_j : int
        Dimension indices to project onto

    Returns
    -------
    np.ndarray
        2D array of shape (grid_points[dim_i], grid_points[dim_j])
    """
    shape_i = sampler.grid_shape[dim_i]
    shape_j = sampler.grid_shape[dim_j]
    marginalized_2d = np.full((shape_i, shape_j), -np.inf)

    # For each (i, j) pair, find the maximum likelihood across all other dims
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        i_idx = grid_idx[dim_i]
        j_idx = grid_idx[dim_j]

        if fitness > marginalized_2d[i_idx, j_idx]:
            marginalized_2d[i_idx, j_idx] = fitness

    return marginalized_2d
