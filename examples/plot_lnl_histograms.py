"""
Histogram the volume-sampling representatives in lnL, to check how close
the stage came to a uniform-in-lnL distribution.

    python plot_lnl_histograms.py

Reads the curated representative files volume_<func>.csv (in-band rows,
tags 0=harvest / 1=probe / 2=search) and the run's global max from
scale_summary_<func>.json. A uniform-in-lnL target would be flat across
the band [global_max - roi_threshold, global_max]; the dashed line shows
that reference. Bars are stacked by tier so the probe layer
(uniform-in-volume) and the interior-walk search layer (uniform-in-lnL
target) are distinguishable.
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

FUNCS = ['himmelblau_4d', 'rosenbrock_4d']
TAGS = [(0, 'harvest', 'tab:green'),
        (1, 'probe (uniform-in-volume)', 'tab:orange'),
        (2, 'search / interior walk', 'tab:blue')]
N_BINS = 24

fig, axes = plt.subplots(1, len(FUNCS), figsize=(7.0 * len(FUNCS), 5.0))

for ax, func in zip(np.atleast_1d(axes), FUNCS):
    summary = json.load(open(f"scale_summary_{func}.json"))
    gmax = summary['global_max']
    roi = summary['volume_roi_threshold']

    rows = read_samples(f"volume_{func}.csv")
    logl = rows[:, -2]
    tag = rows[:, -1].astype(int)
    inband = tag != 3            # drop hole closest-approach rows (not in-band)

    edges = np.linspace(gmax - roi, gmax, N_BINS + 1)
    series, colors, labels = [], [], []
    for t, label, color in TAGS:
        sel = inband & (tag == t)
        series.append(logl[sel])
        colors.append(color)
        labels.append(f"{label}  (n={int(sel.sum()):,})")

    ax.hist(series, bins=edges, stacked=True, color=colors, label=labels,
            edgecolor='white', linewidth=0.2)

    n_total = int(inband.sum())
    uniform = n_total / N_BINS
    ax.axhline(uniform, ls='--', color='k', lw=1.2,
               label=f"uniform-in-lnL ({n_total:,} reps)")

    ax.set_xlabel("lnL")
    ax.set_ylabel(f"representatives per bin (width {roi / N_BINS:.2f})")
    ax.set_title(f"{func}: volume-sampling representatives vs lnL\n"
                 f"band [{gmax - roi:.0f}, {gmax:.0f}], "
                 f"roi_threshold = {roi:g}")
    ax.legend(fontsize=9, loc='upper left')

fig.tight_layout()
out = "lnl_histograms.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
