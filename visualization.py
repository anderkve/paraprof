"""
Visualization utilities for profile likelihood plots.
"""
import numpy as np


def plot_profiles(sampler, fig, axes):
    """
    Generates and displays the 2D profile likelihood plot.

    Parameters
    ----------
    sampler : GridAnchoredDESampler
        The sampler instance containing the profile likelihood grid
    fig : matplotlib.figure.Figure
        The figure to plot on
    axes : list of matplotlib.axes.Axes
        List containing [main_axis, colorbar_axis]
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")
        return

    ax = axes[0]
    ax.clear()

    if sampler.n_proj_dims != 2:
        ax.text(0.5, 0.5, 'Plotting only supported for 2D projections.',
                horizontalalignment='center', verticalalignment='center')
        fig.canvas.draw()
        plt.pause(0.01)
        return

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

    im = ax.imshow(masked_profile.T, extent=extent, aspect='auto', origin='lower',
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

    if sampler.initial_maxima:
        peaks = np.array([m['point'] for m in sampler.initial_maxima])
        ax.plot(peaks[:, dim1], peaks[:, dim2], 'r*', markersize=10,
                label='Found Maxima', markeredgecolor='k')

    ax.set_title(f'Profile Likelihood (Gen: {sampler.current_generation}, Dims: {sampler.projection_dims})')
    ax.set_xlabel(f'Parameter {dim1}')
    ax.set_ylabel(f'Parameter {dim2}')
    if ax.get_legend() is None: # Avoid duplicate legends
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    cax = axes[1]
    cax.clear()
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log Likelihood')

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)
