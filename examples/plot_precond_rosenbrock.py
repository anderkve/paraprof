"""
Rosenbrock-4d coverage of the umbrella walkers under three configs, to show
what actually closes the stiff-valley gap: chain length vs preconditioning.

    python plot_precond_rosenbrock.py

Reads umbrella_rosenbrock_4d{,_iso_long,_precond_long}.csv and writes
precond_rosenbrock.png: projections (0,1) and (2,3), one column per config,
in-band points coloured by lnL.
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

s = json.load(open("umbrella_rosenbrock_4d_summary.json"))
gmax, roi = s['global_max'], s['roi_volume']
band_lo = gmax - roi

CONFIGS = [('', 'isotropic, 30 steps'),
           ('_iso_long', 'isotropic, 120 steps'),
           ('_precond_long', 'preconditioned, 120 steps')]
PAIRS = [(0, 1), (2, 3)]
rng = np.random.default_rng(0)

fig, axes = plt.subplots(len(PAIRS), len(CONFIGS),
                         figsize=(5.0 * len(CONFIGS), 4.6 * len(PAIRS)))
for col, (suffix, label) in enumerate(CONFIGS):
    d = read_samples(f"umbrella_rosenbrock_4d{suffix}.csv")
    lnl = d[:, 4]
    ib = np.isfinite(lnl) & (lnl >= band_lo)
    P, L = d[ib, :4], lnl[ib]
    idx = (rng.choice(len(P), 40000, replace=False) if len(P) > 40000
           else np.arange(len(P)))
    for row, (i, j) in enumerate(PAIRS):
        ax = axes[row, col]
        sc = ax.scatter(P[idx, i], P[idx, j], c=L[idx], s=4, cmap='viridis',
                        vmin=band_lo, vmax=gmax, linewidths=0, rasterized=True)
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        if row == 0:
            ax.set_title(f"{label}\n(n_in_band={int(ib.sum()):,})")
        if col == 0:
            ax.set_ylabel(f"dims ({i},{j})\n\nx{j}")
cb = fig.colorbar(sc, ax=axes, shrink=0.6, label='lnL')
fig.suptitle("rosenbrock_4d umbrella coverage: chain length is the main "
             "lever; preconditioning adds a modest gain", fontsize=13)
fig.savefig("precond_rosenbrock.png", dpi=150, bbox_inches='tight')
print("wrote precond_rosenbrock.png")
