"""
Compare the lnL distribution of the volume-sampling representatives between
the scan-matched-budget config and a walk-heavy "tuned" config (unlimited
budget, more interior steps), to see how close each comes to uniform-in-lnL.

    python plot_lnl_comparison.py

Reads volume_<func>[_tuned].csv and scale_summary_<func>[_tuned].json and
writes lnl_comparison.png: a row per function, a column per config, each a
stacked histogram in lnL by tier (harvest / probe / search) with the
uniform-in-lnL reference (dashed).
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

FUNCS = ['himmelblau_4d', 'rosenbrock_4d']
CONFIGS = [('', 'scan-matched budget'), ('_tuned', 'tuned: unlimited budget, walk-heavy')]
TAGS = [(0, 'harvest', 'tab:green'),
        (1, 'probe (uniform-in-volume)', 'tab:orange'),
        (2, 'search (boundary + interior walk)', 'tab:blue')]
N_BINS = 24

fig, axes = plt.subplots(len(FUNCS), len(CONFIGS),
                         figsize=(8.0 * len(CONFIGS), 4.6 * len(FUNCS)))

for r, func in enumerate(FUNCS):
    for c, (suffix, cfg_label) in enumerate(CONFIGS):
        ax = axes[r, c]
        summary = json.load(open(f"scale_summary_{func}{suffix}.json"))
        gmax = summary['global_max']
        roi = summary['volume_roi_threshold']

        rows = read_samples(f"volume_{func}{suffix}.csv")
        logl = rows[:, -2]
        tag = rows[:, -1].astype(int)
        inband = tag != 3

        edges = np.linspace(gmax - roi, gmax, N_BINS + 1)
        series, colors, labels = [], [], []
        for t, label, color in TAGS:
            sel = inband & (tag == t)
            series.append(logl[sel])
            colors.append(color)
            labels.append(f"{label}  (n={int(sel.sum()):,})")
        ax.hist(series, bins=edges, stacked=True, color=colors, label=labels,
                edgecolor='white', linewidth=0.2)

        counts, _ = np.histogram(logl[inband], bins=edges)
        n_total = int(inband.sum())
        ax.axhline(n_total / N_BINS, ls='--', color='k', lw=1.2,
                   label='uniform-in-lnL')
        edge_top = counts[0] / max(counts[-1], 1)  # deepest / shallowest bin

        ax.set_xlabel("lnL")
        ax.set_ylabel(f"reps per bin (width {roi / N_BINS:.2f})")
        ax.set_title(f"{func} — {cfg_label}\n"
                     f"{n_total:,} reps, edge/top = {edge_top:.0f}x")
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle("Volume-sampling representatives vs lnL: how close to "
             "uniform-in-lnL?", fontsize=14, y=1.0)
fig.tight_layout()
out = "lnl_comparison.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
