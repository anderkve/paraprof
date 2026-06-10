"""
Visualization utilities for profile likelihood plots.
Supports 1D, 2D, and N-D projections.
"""
import os
import numpy as np
import itertools
from .logger import get_logger

logger = get_logger()


def plot_profiles(sampler, filename, plot_settings=None):
    """Save profile-likelihood plots for any projection dimensionality.

    ``plot_settings`` keys (all optional): ``dpi`` (300), ``filetype`` ('png'),
    ``slice_mode`` ('max' or 'all', 3D+), ``vmin``/``vmax`` (colorbar range
    relative to best-fit log L; -4.0/0.0), ``contour_levels``
    (default ``[-3.0, -1.0]``), ``plot_profiled_params`` (True),
    ``output_dir`` (default ``'.'``; created automatically if missing).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("\nMatplotlib not found. Skipping visualization.")
        return

    if plot_settings is None:
        plot_settings = {}

    output_path = _resolve_output_path(filename, plot_settings)

    if sampler.n_proj_dims == 1:
        _plot_1d_profile(sampler, output_path, plot_settings)
    elif sampler.n_proj_dims == 2:
        _plot_2d_profile(sampler, output_path, plot_settings)
    elif sampler.n_proj_dims >= 3:
        _plot_nd_profile(sampler, output_path, plot_settings)
    else:
        logger.info(f"Invalid projection dimensions: {sampler.n_proj_dims}")

    if plot_settings.get('plot_profiled_params', True) and sampler.n_prof_dims > 0:
        plot_profiled_parameters(sampler, filename, plot_settings)


def _resolve_output_path(filename, plot_settings):
    """Join ``filename`` with ``plot_settings['output_dir']`` and ensure the dir exists."""
    output_dir = plot_settings.get('output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _plot_1d_profile(sampler, filename, plot_settings):
    """1D profile likelihood as a line plot."""
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')

    grid_axis = sampler.grid_axes[0]
    profile_1d = np.full(len(grid_axis), np.nan)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_1d[grid_idx[0]] = fitness

    valid_mask = ~np.isnan(profile_1d)
    best_fit_x = None
    best_fit_loglike = None
    if np.any(valid_mask):
        best_idx = int(np.nanargmax(profile_1d))
        best_fit_x = grid_axis[best_idx]
        best_fit_loglike = float(profile_1d[best_idx])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.tick_params(axis='both', labelsize=20)

    dim = sampler.projection_dims[0]

    ax.plot(grid_axis[valid_mask], profile_1d[valid_mask], 'b-', linewidth=2, label='Profile Likelihood')
    ax.scatter(grid_axis[valid_mask], profile_1d[valid_mask], c='blue', s=20, zorder=5)

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

    if best_fit_x is not None:
        ax.scatter([best_fit_x], [best_fit_loglike], c='white', s=240,
                   marker='*', edgecolor='black', linewidth=1.5,
                   label='Best fit', zorder=11)

    # Wilks (1 DOF): ΔlogL = -0.5 -> 68% CL, -1.92 -> 95% CL.
    if best_fit_loglike is not None:
        for delta, label in [(-0.5, '68% CL'), (-1.92, '95% CL')]:
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

    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved 1D plot to: {output_filename}")


def _plot_2d_profile(sampler, filename, plot_settings):
    """2D profile likelihood as a heatmap with contours."""
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    vmin = plot_settings.get('vmin', -4.0)
    vmax = plot_settings.get('vmax', 0.0)
    contour_levels = plot_settings.get('contour_levels', [-3.0, -1.0])

    fig, axes = plt.subplots(1, 2, figsize=(9, 8),
                            gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.0})
    ax = axes[0]
    ax.tick_params(axis='both', labelsize=15)

    dim1, dim2 = sampler.projection_dims

    profile_2d = np.full(sampler.grid_shape, -np.inf)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_2d[grid_idx] = fitness

    finite_mask = np.isfinite(profile_2d)
    best_fit_idx = None
    best_fit_loglike = 0.0
    best_fit_coords = None
    if np.any(finite_mask):
        flat_idx = int(np.argmax(np.where(finite_mask, profile_2d, -np.inf)))
        best_fit_idx = np.unravel_index(flat_idx, profile_2d.shape)
        best_fit_loglike = float(profile_2d[best_fit_idx])
        best_fit_coords = sampler._get_grid_coords_from_indices(best_fit_idx)

    # Use ΔlogL = logL - logL_best so colorbar and contour levels are
    # independent of the absolute log-likelihood offset.
    delta_profile_2d = np.where(finite_mask, profile_2d - best_fit_loglike, -np.inf)

    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    masked_profile = np.ma.masked_where(~finite_mask, delta_profile_2d)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='0.75')

    im = ax.imshow(masked_profile.T, extent=extent, aspect='equal', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(sampler.grid_axes[0], sampler.grid_axes[1])
    ax.contour(X, Y, masked_profile.T, levels=contour_levels, colors='white', linewidths=1.0)

    active_points = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
            coords = sampler._get_grid_coords_from_indices(grid_idx)
            active_points.append(coords)

    if active_points:
        active_points = np.array(active_points)
        ax.scatter(active_points[:, 0], active_points[:, 1], c='cyan', s=3,
                   edgecolor='black', lw=0.5, label='Active DE Points')

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

    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved 2D plot to: {output_filename}")


def _plot_nd_profile(sampler, filename, plot_settings):
    """3D+ profile likelihood as a grid of 2D slices over all dimension pairs.

    ``slice_mode='max'`` (default) slices through the global max; ``'all'``
    marginalizes (max) over the other dims.
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

    max_likelihood = -np.inf
    max_grid_idx = None
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        if fitness > max_likelihood:
            max_likelihood = fitness
            max_grid_idx = grid_idx

    if max_grid_idx is None:
        logger.info("No profile likelihood data found. Skipping plot.")
        return

    dim_pairs = list(itertools.combinations(range(n_dims), 2))
    n_pairs = len(dim_pairs)

    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (dim_i, dim_j) in enumerate(dim_pairs):
        ax = axes[idx]

        if slice_mode == 'max':
            profile_slice = _extract_2d_slice(sampler, dim_i, dim_j, max_grid_idx)
        else:
            profile_slice = _marginalize_to_2d(sampler, dim_i, dim_j)

        finite_mask = np.isfinite(profile_slice)
        profile_slice = np.where(finite_mask, profile_slice - max_likelihood, -np.inf)

        extent = [sampler.grid_axes[dim_i][0], sampler.grid_axes[dim_i][-1],
                  sampler.grid_axes[dim_j][0], sampler.grid_axes[dim_j][-1]]

        masked_profile = np.ma.masked_where(~finite_mask, profile_slice)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='0.75')

        im = ax.imshow(masked_profile.T, extent=extent, aspect='equal',
                      origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        X, Y = np.meshgrid(sampler.grid_axes[dim_i], sampler.grid_axes[dim_j])
        ax.contour(X, Y, masked_profile.T, levels=contour_levels,
                  colors='white', linewidths=1.0)

        max_coords = sampler._get_grid_coords_from_indices(max_grid_idx)
        ax.scatter([max_coords[dim_i]], [max_coords[dim_j]],
                  c='white', s=120, marker='*', edgecolor='black',
                  linewidth=1.0, label='Best fit', zorder=10)

        ax.set_xlabel(f'Param {dims[dim_i]}')
        ax.set_ylabel(f'Param {dims[dim_j]}')
        ax.set_title(f'Dims {dims[dim_i]}-{dims[dim_j]}')
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.colorbar(im, ax=ax, orientation='vertical',
                     label=r'$\log L - \log L_{\mathrm{best\text{-}fit}}$',
                     fraction=0.046)

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

    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved {n_dims}D plot ({n_pairs} slices) to: {output_filename}")


def _extract_2d_slice(sampler, dim_i, dim_j, anchor_idx):
    """2D slice through the N-D profile grid at a fixed anchor index."""
    shape_i = sampler.grid_shape[dim_i]
    shape_j = sampler.grid_shape[dim_j]
    slice_2d = np.full((shape_i, shape_j), -np.inf)

    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        matches_anchor = True
        for k in range(sampler.n_proj_dims):
            if k != dim_i and k != dim_j and grid_idx[k] != anchor_idx[k]:
                matches_anchor = False
                break
        if matches_anchor:
            slice_2d[grid_idx[dim_i], grid_idx[dim_j]] = fitness

    return slice_2d


def _marginalize_to_2d(sampler, dim_i, dim_j):
    """2D projection of the N-D profile via max-likelihood over the other dims."""
    shape_i = sampler.grid_shape[dim_i]
    shape_j = sampler.grid_shape[dim_j]
    marginalized_2d = np.full((shape_i, shape_j), -np.inf)

    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        i_idx = grid_idx[dim_i]
        j_idx = grid_idx[dim_j]
        if fitness > marginalized_2d[i_idx, j_idx]:
            marginalized_2d[i_idx, j_idx] = fitness

    return marginalized_2d


def plot_profiled_parameters(sampler, base_filename, plot_settings=None):
    """Save plots of the optimal profiled-param values across the projection grid.

    One plot per profiled dimension. ``plot_settings`` is the same dict
    accepted by :func:`plot_profiles`.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("\nMatplotlib not found. Skipping profiled parameter visualization.")
        return

    if sampler.n_prof_dims == 0:
        logger.debug("No profiled parameters to plot.")
        return

    if plot_settings is None:
        plot_settings = {}

    base_path = _resolve_output_path(base_filename, plot_settings)

    if sampler.n_proj_dims == 1:
        _plot_1d_profiled_params(sampler, base_path, plot_settings)
    elif sampler.n_proj_dims == 2:
        _plot_2d_profiled_params(sampler, base_path, plot_settings)
    elif sampler.n_proj_dims >= 3:
        _plot_nd_profiled_params(sampler, base_path, plot_settings)


def _plot_1d_profiled_params(sampler, base_filename, plot_settings):
    """Per-profiled-param line plots vs the 1D projection dim."""
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')

    grid_axis = sampler.grid_axes[0]
    n_prof = sampler.n_prof_dims

    fig, axes = plt.subplots(n_prof, 1, figsize=(10, 4 * n_prof))
    if n_prof == 1:
        axes = [axes]

    proj_dim = sampler.projection_dims[0]

    for prof_idx in range(n_prof):
        ax = axes[prof_idx]
        prof_param_values = np.full(len(grid_axis), np.nan)

        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue
            if state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_values[grid_idx[0]] = profiled_params[prof_idx]

        prof_dim = sampler.profiled_dims[prof_idx]

        valid_mask = ~np.isnan(prof_param_values)
        ax.plot(grid_axis[valid_mask], prof_param_values[valid_mask], 'b-', linewidth=2)
        ax.scatter(grid_axis[valid_mask], prof_param_values[valid_mask], c='blue', s=30, zorder=5)

        param_bounds = sampler.bounds[prof_dim]
        ax.axhline(y=param_bounds[0], color='red', linestyle='--', alpha=0.5, label='Bounds')
        ax.axhline(y=param_bounds[1], color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel(f'Parameter {proj_dim} (projection)', fontsize=12)
        ax.set_ylabel(f'Optimal parameter {prof_dim}', fontsize=12)
        ax.set_title(f'Optimal profiled parameter {prof_dim} vs projection dimension', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    output_filename = f"{base_filename}_profiled_params.{filetype}"
    fig.savefig(output_filename, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved profiled parameter plot to: {output_filename}")


def _plot_2d_profiled_params(sampler, base_filename, plot_settings):
    """One heatmap per profiled param showing its optimal value on the 2D grid."""
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    n_prof = sampler.n_prof_dims

    dim1, dim2 = sampler.projection_dims

    for prof_idx in range(n_prof):
        prof_param_grid = np.full(sampler.grid_shape, np.nan)

        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue
            if state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_grid[grid_idx] = profiled_params[prof_idx]

        prof_dim = sampler.profiled_dims[prof_idx]
        param_bounds = sampler.bounds[prof_dim]

        fig, axes = plt.subplots(1, 2, figsize=(7, 6),
                                gridspec_kw={'width_ratios': [20, 1], 'wspace': 0.0})
        ax = axes[0]

        extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
                  sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

        masked_grid = np.ma.masked_where(np.isnan(prof_param_grid), prof_param_grid)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='0.75')

        im = ax.imshow(masked_grid.T, extent=extent, aspect='equal', origin='lower',
                       cmap=cmap, vmin=param_bounds[0], vmax=param_bounds[1])

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

        output_filename = f"{base_filename}_profiled_param_{prof_dim}.{filetype}"
        fig.savefig(output_filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved profiled parameter {prof_dim} plot to: {output_filename}")


def _plot_nd_profiled_params(sampler, base_filename, plot_settings):
    """One figure per profiled param: 2D slice plots for all dimension pairs."""
    import matplotlib.pyplot as plt

    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    slice_mode = plot_settings.get('slice_mode', 'max')
    n_prof = sampler.n_prof_dims
    n_dims = sampler.n_proj_dims
    dims = sampler.projection_dims

    max_likelihood = -np.inf
    max_grid_idx = None
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        if fitness > max_likelihood:
            max_likelihood = fitness
            max_grid_idx = grid_idx

    if max_grid_idx is None:
        logger.info("No profile likelihood data found. Skipping profiled parameter plots.")
        return

    dim_pairs = list(itertools.combinations(range(n_dims), 2))
    n_pairs = len(dim_pairs)

    for prof_idx in range(n_prof):
        prof_dim = sampler.profiled_dims[prof_idx]
        param_bounds = sampler.bounds[prof_dim]

        prof_param_grid = np.full(sampler.grid_shape, np.nan)
        for grid_idx, state in sampler.population.items():
            if sampler.direct_eval_mode:
                continue
            if state['status'] in ['converged', 'optimized']:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                prof_param_grid[grid_idx] = profiled_params[prof_idx]

        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_pairs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (dim_i, dim_j) in enumerate(dim_pairs):
            ax = axes[idx]

            if slice_mode == 'max':
                param_slice = _extract_2d_param_slice(prof_param_grid, dim_i, dim_j,
                                                      max_grid_idx, n_dims)
            else:
                param_slice = _marginalize_param_to_2d(prof_param_grid, dim_i, dim_j)

            extent = [sampler.grid_axes[dim_i][0], sampler.grid_axes[dim_i][-1],
                      sampler.grid_axes[dim_j][0], sampler.grid_axes[dim_j][-1]]

            masked_slice = np.ma.masked_where(np.isnan(param_slice), param_slice)

            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='0.75')

            im = ax.imshow(masked_slice.T, extent=extent, aspect='equal',
                          origin='lower', cmap=cmap, vmin=param_bounds[0], vmax=param_bounds[1])

            if slice_mode == 'max':
                max_coords = sampler._get_grid_coords_from_indices(max_grid_idx)
                ax.scatter([max_coords[dim_i]], [max_coords[dim_j]],
                          c='red', s=100, marker='*', edgecolor='white',
                          linewidth=1.5, label='Global Max', zorder=10)

            ax.set_xlabel(f'Param {dims[dim_i]}')
            ax.set_ylabel(f'Param {dims[dim_j]}')
            ax.set_title(f'Dims {dims[dim_i]}-{dims[dim_j]}')
            ax.grid(True, linestyle='--', alpha=0.3)

            fig.colorbar(im, ax=ax, orientation='vertical',
                        label=f'Param {prof_dim}', fraction=0.046)

        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        mode_str = "Max Slice" if slice_mode == 'max' else "Marginalized"
        fig.suptitle(f'Optimal profiled parameter {prof_dim} - {mode_str} projections\n'
                     f'Projection dimensions: {dims}', fontsize=14, y=0.995)
        fig.tight_layout()

        output_filename = f"{base_filename}_profiled_param_{prof_dim}.{filetype}"
        fig.savefig(output_filename, dpi=dpi)
        plt.close(fig)
        logger.info(f"Saved profiled parameter {prof_dim} plot to: {output_filename}")


def _extract_2d_param_slice(param_grid, dim_i, dim_j, anchor_idx, n_dims):
    """2D slice of the profiled-param grid at a fixed anchor index."""
    shape_i = param_grid.shape[dim_i]
    shape_j = param_grid.shape[dim_j]
    slice_2d = np.full((shape_i, shape_j), np.nan)

    for idx in np.ndindex(param_grid.shape):
        matches_anchor = True
        for k in range(n_dims):
            if k != dim_i and k != dim_j and idx[k] != anchor_idx[k]:
                matches_anchor = False
                break
        if matches_anchor:
            slice_2d[idx[dim_i], idx[dim_j]] = param_grid[idx]

    return slice_2d


def _marginalize_param_to_2d(param_grid, dim_i, dim_j):
    """2D projection of the profiled-param grid via median over the other dims."""
    shape_i = param_grid.shape[dim_i]
    shape_j = param_grid.shape[dim_j]
    marginalized_2d = np.full((shape_i, shape_j), np.nan)

    for i_idx in range(shape_i):
        for j_idx in range(shape_j):
            values = []
            for idx in np.ndindex(param_grid.shape):
                if idx[dim_i] == i_idx and idx[dim_j] == j_idx:
                    val = param_grid[idx]
                    if not np.isnan(val):
                        values.append(val)
            if values:
                marginalized_2d[i_idx, j_idx] = np.median(values)

    return marginalized_2d


def plot_volume_samples(volume_result, dims, filename, plot_settings=None,
                        grid_solution=None, parameter_names=None):
    """Scatter the volume-sampling stage's tagged samples over two parameters.

    ``volume_result`` is the dict from ``run_volume_sampling`` /
    ``sampler.volume_stage_result`` (must not be a skipped result);
    ``dims`` is the pair of parameter indices to use as axes. Samples are
    colored by their provenance tag (harvest / probe / search / hole
    closest-approach). Pass an ``export_grid_solution()`` dict whose
    ``projection_dims`` equal ``dims`` as ``grid_solution`` to draw the
    profile likelihood (relative to its maximum) underneath.

    ``plot_settings`` keys (all optional): ``dpi`` (300), ``filetype``
    ('png'), ``vmin``/``vmax`` (background range relative to best-fit
    log L; -4.0/0.0), ``output_dir`` ('.').
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("\nMatplotlib not found. Skipping visualization.")
        return

    from .volume import TAG_HARVEST, TAG_HOLE, TAG_PROBE, TAG_SEARCH, \
        assemble_volume_rows

    if volume_result.get('skipped'):
        logger.info("Volume stage was skipped; nothing to plot.")
        return

    if plot_settings is None:
        plot_settings = {}
    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')
    output_path = _resolve_output_path(filename, plot_settings)

    rows = assemble_volume_rows(volume_result)
    d0, d1 = dims

    fig, ax = plt.subplots(figsize=(8, 7))

    # Optional profile-likelihood background from a matching 2D projection.
    if grid_solution is not None:
        proj_dims = list(grid_solution['projection_dims'])
        if proj_dims != sorted([d0, d1]):
            logger.warning(
                f"grid_solution projects dims {proj_dims}, not {sorted(dims)}; "
                f"skipping the background."
            )
        else:
            axes_ = grid_solution['grid_axes']
            shape = grid_solution['grid_shape']
            grid = np.full(shape, np.nan)
            for idx, sol in grid_solution['solutions'].items():
                grid[idx] = sol['likelihood']
            ref = np.nanmax(grid) if np.isfinite(np.nanmax(grid)) else 0.0
            vmin = plot_settings.get('vmin', -4.0)
            vmax = plot_settings.get('vmax', 0.0)
            # Transpose so axis 0 of the grid runs along x.
            mesh = ax.pcolormesh(
                axes_[0], axes_[1], (grid - ref).T,
                vmin=vmin, vmax=vmax, cmap='viridis', shading='nearest',
            )
            fig.colorbar(mesh, ax=ax,
                         label=r'$\ln L - \ln L_\mathrm{max}$ (profile)')

    styles = [
        (TAG_HARVEST, 'harvest', 'tab:green', 'o'),
        (TAG_PROBE, 'probe (uniform subset)', 'tab:cyan', 's'),
        (TAG_SEARCH, 'search', 'tab:orange', 'o'),
        (TAG_HOLE, 'hole closest approach', 'tab:red', 'x'),
    ]
    for tag, label, color, marker in styles:
        mask = rows[:, -1] == tag if len(rows) else np.zeros(0, dtype=bool)
        if not mask.any():
            continue
        # Unfilled markers ('x') reject edgecolors.
        edge = {} if marker == 'x' else {'edgecolors': 'black',
                                         'linewidths': 0.3}
        ax.scatter(rows[mask, d0], rows[mask, d1], s=14, c=color,
                   marker=marker,
                   label=f"{label} ({int(np.count_nonzero(mask))})",
                   zorder=3, **edge)

    names = parameter_names or {}
    ax.set_xlabel(names.get(d0, f"$x_{{{d0}}}$") if isinstance(names, dict)
                  else names[d0])
    ax.set_ylabel(names.get(d1, f"$x_{{{d1}}}$") if isinstance(names, dict)
                  else names[d1])
    ax.set_title(f"Volume samples (mode: {volume_result['mode']})")
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out = f"{output_path}_volume_{d0}_{d1}.{filetype}"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info(f"Volume-sample plot saved to {out}")
