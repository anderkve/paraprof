"""
Three-way comparison: funnel vs umbrella (isotropic) vs affine-invariant
ensemble. Writes:

  lnl_three_way.png        lnL density per method, both functions
  spacefill_ensemble_rosenbrock.png   Rosenbrock valley: umbrella vs ensemble
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


def in_band(file, gmax, band_lo, drop_tag=None):
    d = read_samples(file)
    lnl = d[:, N_DIMS]
    m = np.isfinite(lnl) & (lnl >= band_lo)
    if drop_tag is not None:
        m &= d[:, -1] != drop_tag
    return d[m, :N_DIMS], lnl[m]


# --- lnL density overlay ---
fig, axes = plt.subplots(1, len(FUNCS), figsize=(7.5 * len(FUNCS), 4.8))
for ax, func in zip(np.atleast_1d(axes), FUNCS):
    s = json.load(open(f"umbrella_{func}_summary.json"))
    gmax, roi = s['global_max'], s['roi_volume']
    band_lo = gmax - roi
    edges = np.linspace(band_lo, gmax, 25)
    for name, file, tag, color in [
            ('funnel reps', f"volume_{func}.csv", 3, 'tab:red'),
            ('umbrella (isotropic)', f"umbrella_{func}.csv", None, 'tab:green'),
            ('ensemble (affine-inv.)', f"ensemble_{func}.csv", None, 'tab:blue')]:
        _, l = in_band(file, gmax, band_lo, tag)
        ax.hist(l, bins=edges, density=True, histtype='step', lw=2,
                color=color, label=f'{name} (n={len(l):,})')
    ax.axhline(1.0 / roi, ls='--', color='k', lw=1.2, label='uniform-in-lnL')
    ax.set_xlabel('lnL')
    ax.set_ylabel('density')
    ax.set_title(f'{func}: lnL distribution of in-band samples')
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("lnl_three_way.png", dpi=150)
print("wrote lnl_three_way.png")


# --- Rosenbrock coverage: umbrella vs ensemble ---
func = 'rosenbrock_4d'
s = json.load(open(f"umbrella_{func}_summary.json"))
gmax, roi = s['global_max'], s['roi_volume']
band_lo = gmax - roi
PAIRS = [(0, 1), (2, 3)]
fig, axes = plt.subplots(2, 2, figsize=(11, 9.2))
for col, (name, file) in enumerate([('umbrella (isotropic)',
                                     f"umbrella_{func}.csv"),
                                    ('ensemble (affine-invariant)',
                                     f"ensemble_{func}.csv")]):
    P, L = in_band(file, gmax, band_lo)
    idx = rng.choice(len(P), 40000, replace=False) if len(P) > 40000 else np.arange(len(P))
    for row, (i, j) in enumerate(PAIRS):
        ax = axes[row, col]
        sc = ax.scatter(P[idx, i], P[idx, j], c=L[idx], s=4, cmap='viridis',
                        vmin=band_lo, vmax=gmax, linewidths=0, rasterized=True)
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        if row == 0:
            ax.set_title(f"{name}\n(n_in_band={len(P):,})")
cb = fig.colorbar(sc, ax=axes, shrink=0.6, label='lnL')
fig.suptitle("rosenbrock_4d valley coverage (coloured by lnL): "
             "umbrella vs affine-invariant ensemble", fontsize=13)
fig.savefig("spacefill_ensemble_rosenbrock.png", dpi=150, bbox_inches='tight')
print("wrote spacefill_ensemble_rosenbrock.png")
