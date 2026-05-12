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
    sampler : ProfileProjector
        The sampler instance containing the profile likelihood grid
    filename : str
        Output filename (without extension)
    plot_settings : dict, optional
        Plot settings with keys:
        - 'dpi': int (default: 300)
        - 'filetype': str (default: 'png')
        - 'slice_mode': str (default: 'max') - for 3D+: 'max' or 'all'
        - 'vmin': float (default: -4.0) - colorbar minimum (relative to best-fit log L)
        - 'vmax': float (default: 0.0) - colorbar maximum (relative to best-fit log L)
        - 'contour_levels': list (default: [-3.0, -1.0]) - contour levels for the
          2D and N-D log-likelihood plots, expressed relative to the best-fit log L
        - 'plot_profiled_params': bool (default: True) - plot optimal profiled parameter values
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

    # Plot profiled parameter values if enabled and profiled parameters exist
    plot_profiled = plot_settings.get('plot_profiled_params', True)
    if plot_profiled and sampler.n_prof_dims > 0:
        plot_profiled_parameters(sampler, filename, plot_settings)


def _plot_1d_profile(sampler, filename, plot_settings):
    """
    Plots a 1D profile likelihood as a line plot.

    Parameters
    ----------
    sampler : ProfileProjector
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

    # Locate the best-fit grid point.
    valid_mask = ~np.isnan(profile_1d)
    best_fit_x = None
    best_fit_loglike = None
    if np.any(valid_mask):
        best_idx = int(np.nanargmax(profile_1d))
        best_fit_x = grid_axis[best_idx]
        best_fit_loglike = float(profile_1d[best_idx])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(axis='both', labelsize=20)

    dim = sampler.projection_dims[0]

    # Plot profile
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

    # Mark the best-fit point with a white star with a black border.
    if best_fit_x is not None:
        ax.scatter([best_fit_x], [best_fit_loglike], c='white', s=240,
                   marker='*', edgecolor='black', linewidth=1.5,
                   label='Best fit', zorder=11)

    # Add confidence level lines
    if best_fit_loglike is not None:
        for delta, label in [(-1.0, '68% CL'), (-4.0, '95% CL')]:
            level = best_fit_loglike + delta
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.7, label=label)

    title = f'1D Profile Likelihood for Parameter {dim}'
    if best_fit_x is not None:
        title += (f'\nBest fit: param {dim} = {best_fit_x:.3e}, '
                  f'log L = {best_fit_loglike:.3e}')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f'Parameter {dim}', fontsize=24)
    ax.set_ylabel('Log Likelihood', fontsize=24)
    ax.legend(fontsize=18)
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
    sampler : ProfileProjector
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
    fig, axes = plt.subplots(1, 2, figsize=(9, 8),
                            gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.0})
    ax = axes[0]
    ax.tick_params(axis='both', labelsize=15)

    dim1, dim2 = sampler.projection_dims

    # Create 2D grid from sparse dict
    profile_2d = np.full(sampler.grid_shape, -np.inf)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_2d[grid_idx] = fitness

    # Locate the best-fit grid point.
    finite_mask = np.isfinite(profile_2d)
    best_fit_idx = None
    best_fit_loglike = 0.0
    best_fit_coords = None
    if np.any(finite_mask):
        flat_idx = int(np.argmax(np.where(finite_mask, profile_2d, -np.inf)))
        best_fit_idx = np.unravel_index(flat_idx, profile_2d.shape)
        best_fit_loglike = float(profile_2d[best_fit_idx])
        best_fit_coords = sampler._get_grid_coords_from_indices(best_fit_idx)

    # Express the profile relative to the best-fit log-likelihood so that the
    # default colorbar range (and contour levels) are meaningful regardless of
    # the absolute offset of the log-likelihood.
    delta_profile_2d = np.where(finite_mask, profile_2d - best_fit_loglike, -np.inf)

    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    masked_profile = np.ma.masked_where(~finite_mask, delta_profile_2d)

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

    # Mark the best-fit point with a white star with a black border.
    if best_fit_coords is not None:
        ax.scatter([best_fit_coords[0]], [best_fit_coords[1]],
                   c='white', s=120, marker='*', edgecolor='black',
                   linewidth=1.0, label='Best fit', zorder=11)

    title = f'Profile likelihood for parameters {sampler.projection_dims}'
    if best_fit_coords is not None:
        title += (f'\nBest fit: param {dim1} = {best_fit_coords[0]:.3e}, '
                  f'param {dim2} = {best_fit_coords[1]:.3e}, '
                  f'log L = {best_fit_loglike:.3e}')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(f'Parameter {dim1}', fontsize=15)
    ax.set_ylabel(f'Parameter {dim2}', fontsize=15)
    ax.legend(fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)

    cax = axes[1]
    cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                        label=r'$\log L - \log L_{\mathrm{best\text{-}fit}}$')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(r'$\log L - \log L_{\mathrm{best\text{-}fit}}$', fontsize=15)

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
    sampler : ProfileProjector
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
    contour_levels = plot_settings.get('contour_levels', [-3.0, -1.0])

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

        # Express the slice relative to the best-fit log-likelihood so that
        # the colorbar range and contour levels are independent of the
        # absolute log-likelihood offset.
        finite_mask = np.isfinite(profile_slice)
        profile_slice = np.where(finite_mask, profile_slice - max_likelihood, -np.inf)

        # Plot the slice
        extent = [sampler.grid_axes[dim_i][0], sampler.grid_axes[dim_i][-1],
                  sampler.grid_axes[dim_j][0], sampler.grid_axes[dim_j][-1]]

        masked_profile = np.ma.masked_where(~finite_mask, profile_slice)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='0.75')

        im = ax.imshow(masked_profile.T, extent=extent, aspect='equal',
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        # Add contours
        X, Y = np.meshgrid(sampler.grid_axes[dim_i], sampler.grid_axes[dim_j])
        ax.contour(X, Y, masked_profile.T, levels=contour_levels,
                  colors='white', linewidths=1.0)

        # Mark the best-fit point with a white star with a black border.
        max_coords = sampler._get_grid_coords_from_indices(max_grid_idx)
        ax.scatter([max_coords[dim_i]], [max_coords[dim_j]],
                  c='white', s=120, marker='*', edgecolor='black',
                  linewidth=1.0, label='Best fit', zorder=10)

        ax.set_xlabel(f'Param {dims[dim_i]}')
        ax.set_ylabel(f'Param {dims[dim_j]}')
        ax.set_title(f'Dims {dims[dim_i]}-{dims[dim_j]}')
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add colorbar for each subplot
        fig.colorbar(im, ax=ax, orientation='vertical',
                     label=r'$\log L - \log L_{\mathrm{best\text{-}fit}}$',
                     fraction=0.046)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    mode_str = "Max Slice" if slice_mode == 'max' else "Marginalized"
    max_coords = sampler._get_grid_coords_from_indices(max_grid_idx)
    best_fit_str = ', '.join(f'param {dims[k]} = {max_coords[k]:.3e}'
                             for k in range(n_dims))
    fig.suptitle(f'{n_dims}D Profile Likelihood - {mode_str} Projections\n'
                 f'Dimensions: {dims}\n'
                 f'Best fit: {best_fit_str}, log L = {max_likelihood:.3e}',
                 fontsize=12, y=0.995)
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
    sampler : ProfileProjector
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
    sampler : ProfileProjector
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


def plot_profiled_parameters(sampler, base_filename, plot_settings=None):
    """
    Generates plots showing optimal profiled parameter values across the projection space.

    Creates one plot per profiled parameter dimension, visualizing which parameter
    values were selected by the profiling at each point in the projection grid.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance containing the profile likelihood grid
    base_filename : str
        Base output filename (without extension)
    plot_settings : dict, optional
        Plot settings (same as plot_profiles)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("\nMatplotlib not found. Skipping profiled parameter visualization.")
        return

    if sampler.n_prof_dims == 0:
        logger.debug("No profiled parameters to plot.")
        return

    # Set default plot settings
    if plot_settings is None:
        plot_settings = {}

    # Route to appropriate plotting function based on projection dimensionality
    if sampler.n_proj_dims == 1:
        _plot_1d_profiled_params(sampler, base_filename, plot_settings)
    elif sampler.n_proj_dims == 2:
        _plot_2d_profiled_params(sampler, base_filename, plot_settings)
    elif sampler.n_proj_dims >= 3:
        _plot_nd_profiled_params(sampler, base_filename, plot_settings)


def _plot_1d_profiled_params(sampler, base_filename, plot_settings):
    """
    Plots optimal profiled parameter values for 1D projections.

    Creates a multi-panel figure with one subplot per profiled parameter,
    showing how the optimal value varies across the projection dimension.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    base_filename : str
        Base output filename (without extension)
    plot_settings : dict
        Plot settings
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')

    # Extract profiled parameter values at each grid point
    grid_axis = sampler.grid_axes[0]
    n_prof = sampler.n_prof_dims

    # Create figure with subplots for each profiled parameter
    fig, axes = plt.subplots(n_prof, 1, figsize=(10, 4 * n_prof))
    if n_prof == 1:
        axes = [axes]

    proj_dim = sampler.projection_dims[0]

    for prof_idx in range(n_prof):
        ax = axes[prof_idx]
        prof_param_values = np.full(len(grid_axis), np.nan)

        # Extract optimal profiled parameter values from population
        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue  # No profiled params in direct eval mode
            elif state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_values[grid_idx[0]] = profiled_params[prof_idx]

        # Get the actual profiled dimension index in full parameter space
        prof_dim = sampler.profiled_dims[prof_idx]

        # Plot the profiled parameter values
        valid_mask = ~np.isnan(prof_param_values)
        ax.plot(grid_axis[valid_mask], prof_param_values[valid_mask], 'b-', linewidth=2)
        ax.scatter(grid_axis[valid_mask], prof_param_values[valid_mask], c='blue', s=30, zorder=5)

        # Add parameter bounds as horizontal lines
        param_bounds = sampler.bounds[prof_dim]
        ax.axhline(y=param_bounds[0], color='red', linestyle='--', alpha=0.5, label='Bounds')
        ax.axhline(y=param_bounds[1], color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel(f'Parameter {proj_dim} (projection)', fontsize=12)
        ax.set_ylabel(f'Optimal parameter {prof_dim}', fontsize=12)
        ax.set_title(f'Optimal profiled parameter {prof_dim} vs projection dimension', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    # Save the plot
    output_filename = f"{base_filename}_profiled_params.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved profiled parameter plot to: {output_filename}")


def _plot_2d_profiled_params(sampler, base_filename, plot_settings):
    """
    Plots optimal profiled parameter values for 2D projections.

    Creates one heatmap per profiled parameter showing the optimal value
    at each point in the 2D projection grid.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    base_filename : str
        Base output filename (without extension)
    plot_settings : dict
        Plot settings
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    n_prof = sampler.n_prof_dims

    dim1, dim2 = sampler.projection_dims

    # Create one plot per profiled parameter
    for prof_idx in range(n_prof):
        # Create 2D grid for this profiled parameter
        prof_param_grid = np.full(sampler.grid_shape, np.nan)

        # Extract optimal values from population
        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue  # No profiled params in direct eval mode
            elif state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_grid[grid_idx] = profiled_params[prof_idx]

        # Get the actual profiled dimension index in full parameter space
        prof_dim = sampler.profiled_dims[prof_idx]
        param_bounds = sampler.bounds[prof_dim]

        # Create figure with colorbar
        fig, axes = plt.subplots(1, 2, figsize=(7, 6),
                                gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.0})
        ax = axes[0]

        extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
                  sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

        masked_grid = np.ma.masked_where(np.isnan(prof_param_grid), prof_param_grid)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='0.75')

        # Use parameter bounds for colorbar limits
        vmin = param_bounds[0]
        vmax = param_bounds[1]

        im = ax.imshow(masked_grid.T, extent=extent, aspect='equal', origin='lower',
                       cmap=cmap, vmin=vmin, vmax=vmax)

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

        ax.set_title(f'Optimal profiled parameter {prof_dim}')
        ax.set_xlabel(f'Parameter {dim1}')
        ax.set_ylabel(f'Parameter {dim2}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        cax = axes[1]
        fig.colorbar(im, cax=cax, orientation='vertical', label=f'Parameter {prof_dim} Value')

        fig.tight_layout()

        # Save the plot
        output_filename = f"{base_filename}_profiled_param_{prof_dim}.{filetype}"
        fig.savefig(output_filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved profiled parameter {prof_dim} plot to: {output_filename}")


def _plot_nd_profiled_params(sampler, base_filename, plot_settings):
    """
    Plots optimal profiled parameter values for 3D+ projections.

    For N-dimensional grids (N >= 3), creates 2D slice plots for each
    profiled parameter, showing all pairwise projections through the
    maximum likelihood point.

    Parameters
    ----------
    sampler : ProfileProjector
        The sampler instance
    base_filename : str
        Base output filename (without extension)
    plot_settings : dict
        Plot settings
    """
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    slice_mode = plot_settings.get('slice_mode', 'max')
    n_prof = sampler.n_prof_dims
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
        logger.info("No profile likelihood data found. Skipping profiled parameter plots.")
        return

    # Generate all pairwise dimension combinations
    dim_pairs = list(itertools.combinations(range(n_dims), 2))
    n_pairs = len(dim_pairs)

    # Create one figure per profiled parameter
    for prof_idx in range(n_prof):
        prof_dim = sampler.profiled_dims[prof_idx]
        param_bounds = sampler.bounds[prof_dim]

        # Create N-D grid for this profiled parameter
        prof_param_grid = np.full(sampler.grid_shape, np.nan)

        # Extract optimal values from population
        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue
            elif state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_grid[grid_idx] = profiled_params[prof_idx]

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
                param_slice = _extract_2d_param_slice(prof_param_grid, dim_i, dim_j,
                                                      max_grid_idx, n_dims)
            else:  # marginalized
                param_slice = _marginalize_param_to_2d(prof_param_grid, dim_i, dim_j)

            # Plot the slice
            extent = [sampler.grid_axes[dim_i][0], sampler.grid_axes[dim_i][-1],
                      sampler.grid_axes[dim_j][0], sampler.grid_axes[dim_j][-1]]

            masked_slice = np.ma.masked_where(np.isnan(param_slice), param_slice)

            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='0.75')

            im = ax.imshow(masked_slice.T, extent=extent, aspect='equal',
                          origin='lower', cmap=cmap, vmin=param_bounds[0], vmax=param_bounds[1])

            # Mark the maximum likelihood point (if in slice mode)
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
            fig.colorbar(im, ax=ax, orientation='vertical',
                        label=f'Param {prof_dim}', fraction=0.046)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        mode_str = "Max Slice" if slice_mode == 'max' else "Marginalized"
        fig.suptitle(f'Optimal profiled parameter {prof_dim} - {mode_str} projections\n'
                     f'Projection dimensions: {dims}', fontsize=14, y=0.995)
        fig.tight_layout()

        # Save the plot
        output_filename = f"{base_filename}_profiled_param_{prof_dim}.{filetype}"
        fig.savefig(output_filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved profiled parameter {prof_dim} plot to: {output_filename}")


def _extract_2d_param_slice(param_grid, dim_i, dim_j, anchor_idx, n_dims):
    """
    Extracts a 2D slice of profiled parameter values through an N-D grid.

    Parameters
    ----------
    param_grid : np.ndarray
        N-D array of profiled parameter values
    dim_i, dim_j : int
        Dimension indices to extract
    anchor_idx : tuple
        Grid index of the anchor point for the slice
    n_dims : int
        Number of projection dimensions

    Returns
    -------
    np.ndarray
        2D array of shape (grid_points[dim_i], grid_points[dim_j])
    """
    shape_i = param_grid.shape[dim_i]
    shape_j = param_grid.shape[dim_j]
    slice_2d = np.full((shape_i, shape_j), np.nan)

    # Iterate through the grid and extract matching points
    for idx in np.ndindex(param_grid.shape):
        # Check if all dimensions except dim_i and dim_j match the anchor
        matches_anchor = True
        for k in range(n_dims):
            if k != dim_i and k != dim_j:
                if idx[k] != anchor_idx[k]:
                    matches_anchor = False
                    break

        if matches_anchor:
            slice_2d[idx[dim_i], idx[dim_j]] = param_grid[idx]

    return slice_2d


def _marginalize_param_to_2d(param_grid, dim_i, dim_j):
    """
    Marginalizes an N-D parameter grid to a 2D projection.

    For each (dim_i, dim_j) grid point pair, takes the median value
    across all other dimensions (to get a representative value).

    Parameters
    ----------
    param_grid : np.ndarray
        N-D array of profiled parameter values
    dim_i, dim_j : int
        Dimension indices to project onto

    Returns
    -------
    np.ndarray
        2D array of shape (grid_points[dim_i], grid_points[dim_j])
    """
    shape_i = param_grid.shape[dim_i]
    shape_j = param_grid.shape[dim_j]
    marginalized_2d = np.full((shape_i, shape_j), np.nan)

    # For each (i, j) pair, collect all values across other dims
    for i_idx in range(shape_i):
        for j_idx in range(shape_j):
            values = []
            for idx in np.ndindex(param_grid.shape):
                if idx[dim_i] == i_idx and idx[dim_j] == j_idx:
                    val = param_grid[idx]
                    if not np.isnan(val):
                        values.append(val)

            if values:
                # Use median as representative value
                marginalized_2d[i_idx, j_idx] = np.median(values)

    return marginalized_2d
