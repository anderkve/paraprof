"""
Generate the publication-quality profile-likelihood showcase plots that the
README displays at the top of the page.

Reads the ``.npz`` grids produced by ``run_showcase_scan.py``, renders one
high-resolution PNG per (function, projection) combination (one 1D
projection and two 2D projections per function) using a LaTeX-style serif
math font, and writes a JSON table summarising the total target-function
evaluation count for each test function.

Usage::

    python examples/make_showcase_plots.py
"""
import argparse
import json
import os

import numpy as np

# Functions and human-readable display strings used for plot titles. Order
# here controls the row order in the README grid.
SHOWCASE_ORDER = [
    ('himmelblau_4d', 'Himmelblau 4D'),
    ('rosenbrock_4d', 'Rosenbrock 4D'),
    ('ackley_4d',     'Ackley 4D'),
]


def setup_matplotlib():
    """Configure matplotlib for a clean LaTeX-style look without needing real LaTeX."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'text.usetex': False,           # avoid needing a TeX install
        'font.family': 'serif',
        'font.serif': ['STIX Two Text', 'STIXGeneral', 'Times New Roman',
                       'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.top': True,
        'ytick.right': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    return plt


def plot_1d(plt, axis, logL, proj_dim, title_name, out_path, dpi):
    """Render a single 1D profile-likelihood plot."""
    axis = np.asarray(axis)
    logL = np.asarray(logL)

    valid = np.isfinite(logL)
    if not valid.any():
        raise ValueError(f"No valid 1D likelihood values for {title_name}.")

    max_logL = np.max(logL[valid])
    delta_logL = logL - max_logL
    best_x = axis[np.argmax(np.where(valid, logL, -np.inf))]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(axis[valid], delta_logL[valid], color='#1f3a93',
            linewidth=2.2, label=r'Profile likelihood', zorder=3)

    # 1-DOF Wilks confidence levels (delta(-2 log L) = 1, 3.84).
    for delta_chi2, label_text, ls in [(1.0, r'$68\%$ CL', '--'),
                                       (3.84, r'$95\%$ CL', ':')]:
        ax.axhline(-0.5 * delta_chi2, color='0.35', linestyle=ls,
                   linewidth=1.1, label=label_text, zorder=2)

    ax.scatter([best_x], [0.0], marker='*', s=180,
               facecolor='white', edgecolor='black', linewidths=1.1,
               zorder=4, label=r'Best fit')

    ax.set_xlabel(rf'$x_{{{proj_dim}}}$')
    ax.set_ylabel(r'$\Delta \log L = \log L - \log L_{\max}$')
    ax.set_title(f'{title_name}: 1D profile for $x_{{{proj_dim}}}$')

    # Sensible y-range that emphasises the high-likelihood region without
    # squashing the structure to a single pixel.
    y_min = max(-6.0, float(np.nanmin(delta_logL[valid]) - 0.2))
    ax.set_ylim(y_min, 0.4)
    ax.set_xlim(axis[0], axis[-1])
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              edgecolor='0.6')

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_2d(plt, axis_x, axis_y, logL, dims, title_name, out_path, dpi):
    """Render a single 2D profile-likelihood heat-map with CL contours."""
    axis_x = np.asarray(axis_x)
    axis_y = np.asarray(axis_y)
    logL = np.asarray(logL)
    dims = list(dims)

    finite = np.isfinite(logL)
    if not finite.any():
        raise ValueError(f"No valid 2D likelihood values for {title_name} ({dims}).")
    max_logL = np.max(logL[finite])
    delta = np.where(finite, logL - max_logL, np.nan)

    flat_idx = int(np.nanargmax(np.where(finite, delta, -np.inf)))
    bi, bj = np.unravel_index(flat_idx, delta.shape)
    best_x = axis_x[bi]
    best_y = axis_y[bj]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('0.85')
    masked = np.ma.masked_invalid(delta)

    vmin, vmax = -8.0, 0.0
    extent = [axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]]
    im = ax.imshow(masked.T, extent=extent, origin='lower', aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    # 2-DOF Wilks: delta(-2 log L) = 2.30 (68%), 6.18 (95%) -> delta logL.
    X, Y = np.meshgrid(axis_x, axis_y, indexing='ij')
    contour_levels = [-6.18 / 2, -2.30 / 2]
    contour_labels = {-6.18 / 2: r'$95\%$', -2.30 / 2: r'$68\%$'}
    cs = ax.contour(X, Y, np.where(finite, delta, np.nan),
                    levels=sorted(contour_levels),
                    colors=['white', 'white'],
                    linewidths=[1.0, 1.6],
                    linestyles=['--', '-'])
    fmt = {lv: contour_labels[lv] for lv in cs.levels}
    ax.clabel(cs, inline=True, fontsize=10, fmt=fmt)

    ax.scatter([best_x], [best_y], marker='*', s=200,
               facecolor='white', edgecolor='black', linewidths=1.2,
               zorder=10)

    ax.set_xlabel(rf'$x_{{{dims[0]}}}$')
    ax.set_ylabel(rf'$x_{{{dims[1]}}}$')
    ax.set_title(f'{title_name}: 2D profile for $(x_{{{dims[0]}}}, x_{{{dims[1]}}})$')
    ax.set_xlim(axis_x[0], axis_x[-1])
    ax.set_ylim(axis_y[0], axis_y[-1])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r'$\Delta \log L = \log L - \log L_{\max}$')
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='examples/example_plots/showcase/data',
                        help='Directory holding the .npz/.json outputs of run_showcase_scan.py.')
    parser.add_argument('--out-dir', default='examples/example_plots/showcase',
                        help='Directory for the rendered showcase PNGs and summary table.')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    plt = setup_matplotlib()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for name, display in SHOWCASE_ORDER:
        npz_path = os.path.join(args.data_dir, f'{name}.npz')
        json_path = os.path.join(args.data_dir, f'{name}.json')
        if not os.path.exists(npz_path):
            print(f"[skip] {name}: {npz_path} not found")
            continue

        data = np.load(npz_path, allow_pickle=False)

        out_1d = os.path.join(args.out_dir, f'{name}_1d.png')
        out_2d_a = os.path.join(args.out_dir, f'{name}_2d_a.png')
        out_2d_b = os.path.join(args.out_dir, f'{name}_2d_b.png')

        plot_1d(plt,
                axis=data['axis_1d'],
                logL=data['likelihood_1d'],
                proj_dim=int(data['proj_dim_1d']),
                title_name=display,
                out_path=out_1d,
                dpi=args.dpi)
        plot_2d(plt,
                axis_x=data['axis_2d_a_x'],
                axis_y=data['axis_2d_a_y'],
                logL=data['likelihood_2d_a'],
                dims=np.asarray(data['proj_dims_2d_a']).tolist(),
                title_name=display,
                out_path=out_2d_a,
                dpi=args.dpi)
        plot_2d(plt,
                axis_x=data['axis_2d_b_x'],
                axis_y=data['axis_2d_b_y'],
                logL=data['likelihood_2d_b'],
                dims=np.asarray(data['proj_dims_2d_b']).tolist(),
                title_name=display,
                out_path=out_2d_b,
                dpi=args.dpi)

        with open(json_path) as f:
            scan_summary = json.load(f)
        summary.append({
            'function': name,
            'display_name': display,
            'total_target_calls': scan_summary['total_target_calls'],
            'plot_1d': os.path.relpath(out_1d, '.'),
            'plot_2d_a': os.path.relpath(out_2d_a, '.'),
            'plot_2d_b': os.path.relpath(out_2d_b, '.'),
        })
        print(f"[ok]   {display:>14}  total target evaluations: {scan_summary['total_target_calls']:>10,}")

    out_summary = os.path.join(args.out_dir, 'summary.json')
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_summary}")


if __name__ == '__main__':
    main()
