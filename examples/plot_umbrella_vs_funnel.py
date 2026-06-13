"""
Compare the umbrella-walker prototype against the volume-sampling funnel.

Reads:
  volume_<func>.csv               funnel representatives ([params, lnL, tag])
  umbrella_<func>.csv             umbrella evals ([params, lnL, acc, level])
  umbrella_<func>_summary.json    global_max / roi_volume

Writes:
  lnl_overlay_umbrella.png        lnL density: funnel vs umbrella vs uniform
  spacefill_<func>.png            6 projections, funnel (top) vs umbrella
                                  (bottom), coloured by lnL
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

FUNCS = ['himmelblau_4d', 'rosenbrock_4d']
N_DIMS = 4
rng = np.random.default_rng(0)


def load(func):
    s = json.load(open(f"umbrella_{func}_summary.json"))
    gmax, roi = s['global_max'], s['roi_volume']
    band_lo = gmax - roi

    fun = read_samples(f"volume_{func}.csv")          # [params, lnL, tag]
    f_in = fun[:, -1] != 3                             # drop hole rows
    fun_p, fun_l = fun[f_in, :N_DIMS], fun[f_in, N_DIMS]

    umb = read_samples(f"umbrella_{func}.csv")         # [params, lnL, acc, lvl]
    u_in = np.isfinite(umb[:, N_DIMS]) & (umb[:, N_DIMS] >= band_lo)
    umb_p, umb_l = umb[u_in, :N_DIMS], umb[u_in, N_DIMS]
    return gmax, roi, band_lo, (fun_p, fun_l), (umb_p, umb_l)


# --- Figure A: lnL distribution overlay ---
fig, axes = plt.subplots(1, len(FUNCS), figsize=(7.5 * len(FUNCS), 4.8))
for ax, func in zip(np.atleast_1d(axes), FUNCS):
    gmax, roi, band_lo, (_, fun_l), (_, umb_l) = load(func)
    edges = np.linspace(band_lo, gmax, 25)
    ax.hist(fun_l, bins=edges, density=True, histtype='step', lw=2,
            color='tab:red', label=f'funnel reps (n={len(fun_l):,})')
    ax.hist(umb_l, bins=edges, density=True, histtype='step', lw=2,
            color='tab:blue', label=f'umbrella walkers (n={len(umb_l):,})')
    ax.axhline(1.0 / roi, ls='--', color='k', lw=1.2, label='uniform-in-lnL')
    ax.set_xlabel('lnL')
    ax.set_ylabel('density')
    ax.set_title(f'{func}: lnL distribution of in-band samples')
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("lnl_overlay_umbrella.png", dpi=150)
print("wrote lnl_overlay_umbrella.png")


# --- Figures B/C: space-filling, funnel vs umbrella, coloured by lnL ---
def subsample(n, cap):
    idx = np.arange(n)
    return rng.choice(idx, cap, replace=False) if n > cap else idx


PAIRS = [(0, 1), (0, 2), (2, 3)]
for func in FUNCS:
    gmax, roi, band_lo, (fun_p, fun_l), (umb_p, umb_l) = load(func)
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(5.2 * len(PAIRS), 9.2))
    for row, (label, P, L, color_lo) in enumerate([
            ('funnel representatives', fun_p, fun_l, band_lo),
            ('umbrella walkers', umb_p, umb_l, band_lo)]):
        si = subsample(len(P), 40000)
        for col, (i, j) in enumerate(PAIRS):
            ax = axes[row, col]
            sc = ax.scatter(P[si, i], P[si, j], c=L[si], s=3, cmap='viridis',
                            vmin=band_lo, vmax=gmax, linewidths=0,
                            rasterized=True)
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"x{j}")
            if col == 0:
                ax.set_ylabel(f"{label}\n\nx{j}")
            ax.set_title(f"dims ({i}, {j})")
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, label='lnL')
    fig.suptitle(f"{func}: space-filling coloured by lnL — "
                 f"funnel (top) vs umbrella (bottom)", fontsize=14)
    fig.savefig(f"spacefill_{func}.png", dpi=150, bbox_inches='tight')
    print(f"wrote spacefill_{func}.png")
