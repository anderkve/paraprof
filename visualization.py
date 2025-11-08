"""
Visualization utilities for profile likelihood plots.
"""
import numpy as np


def plot_profiles(sampler, filename, plot_settings=None):
    """
    Generates and saves the 2D profile likelihood plot.

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
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
        return

    # Set default plot settings
    if plot_settings is None:
        plot_settings = {}
    dpi = plot_settings.get('dpi', 300)
    filetype = plot_settings.get('filetype', 'png')

    if sampler.n_proj_dims != 2:
        print('Plotting only supported for 2D projections. Skipping plot.')
        return

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [10, 1]})
    ax = axes[0]

    dim1, dim2 = sampler.projection_dims

    # --- Create 2D grid from sparse dict ---
    profile_2d = np.full(sampler.grid_shape, -np.inf)
    for grid_idx, fitness in sampler.profile_likelihood_grid.items():
        profile_2d[grid_idx] = fitness
    # ---

    extent = [sampler.grid_axes[0][0], sampler.grid_axes[0][-1],
              sampler.grid_axes[1][0], sampler.grid_axes[1][-1]]

    plot_baseline = sampler.global_max_target_val
    vmin = plot_baseline - 3.0
    vmax = plot_baseline

    masked_profile = np.ma.masked_where(profile_2d == -np.inf, profile_2d)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='white')

    im = ax.imshow(masked_profile.T, extent=extent, aspect='equal', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    active_points = []
    for grid_idx, state in sampler.population.items():
        if state.get('status') == 'active':
             coords = sampler._get_grid_coords_from_indices(grid_idx)
             active_points.append(coords)

    if active_points:
        active_points = np.array(active_points)
        ax.scatter(active_points[:, 0], active_points[:, 1], c='cyan', s=3,
                   edgecolor='black', lw=0.5, label='Active DE Points')

    ax.set_title(f'Profile Likelihood (Gen: {sampler.current_generation}, Dims: {sampler.projection_dims})')
    ax.set_xlabel(f'Parameter {dim1}')
    ax.set_ylabel(f'Parameter {dim2}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    cax = axes[1]
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log Likelihood')

    fig.tight_layout()

    # Save the plot
    output_filename = f"{filename}.{filetype}"
    fig.savefig(output_filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {output_filename}")
