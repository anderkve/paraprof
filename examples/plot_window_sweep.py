"""
Partner-level-window sweep for the ensemble sampler: does pooling all
walkers (ignoring shells) hurt? Overlays the in-band lnL distribution for
strict shells vs full pool, per function, and prints the metric table.

    python plot_window_sweep.py
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

FUNCS = ['himmelblau_4d', 'rosenbrock_4d']
WINS = [('w0', 'strict shells', 'tab:purple'),
        ('full', 'full pool', 'tab:orange')]

fig, axes = plt.subplots(1, len(FUNCS), figsize=(7.5 * len(FUNCS), 4.8))
for ax, func in zip(np.atleast_1d(axes), FUNCS):
    gmax = json.load(open(f"ensemble_{func}_w0_summary.json"))['global_max']
    roi = json.load(open(f"ensemble_{func}_w0_summary.json"))['roi_volume']
    band_lo = gmax - roi
    edges = np.linspace(band_lo, gmax, 25)
    for wl, name, color in WINS:
        d = read_samples(f"ensemble_{func}_{wl}.csv")
        lnl = d[:, 4]
        l = lnl[np.isfinite(lnl) & (lnl >= band_lo)]
        ax.hist(l, bins=edges, density=True, histtype='step', lw=2,
                color=color, label=f'{name} (n={len(l):,})')
    ax.axhline(1.0 / roi, ls='--', color='k', lw=1.2, label='uniform-in-lnL')
    ax.set_xlabel('lnL')
    ax.set_ylabel('density')
    ax.set_title(f'{func}: lnL distribution — strict shells vs full pool')
    ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("lnl_strict_vs_full.png", dpi=150)
print("wrote lnl_strict_vs_full.png")
