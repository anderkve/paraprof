"""
Scatter the total sample set (projection scan + volume sampling) in all
six 2D projections of a 4D run produced by run_volume_scale_test.py.

    python plot_total_samples.py <func>

Reads the phase-tagged log ``samples_<func>.csv`` (every evaluation from
both stages) and writes ``total_samples_<func>.png``. Points are coloured
by stage so the contrast between the profile-surface concentration of the
scan and the volume-filling of the sampling stage is visible.
"""
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from paraprof import read_samples  # noqa: E402

func = sys.argv[1]
roi_volume = float(sys.argv[2]) if len(sys.argv) > 2 else None

samples = read_samples(f"samples_{func}.csv")
n_dims = samples.shape[1] - 2
params = samples[:, :n_dims]
phase = samples[:, -1].astype(int)

scan = phase <= 3        # phases 0-3: initial, scan, refine, suspect
probe = phase == 4       # volume sampling: anchor probes (uniform subset)
search = phase == 5      # volume sampling: anchored search + interior walk
volume = probe | search

rng = np.random.default_rng(0)


def subsample(mask, cap):
    idx = np.flatnonzero(mask)
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    return idx


pairs = [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]
ncols = 3
nrows = int(np.ceil(len(pairs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))

for ax, (i, j) in zip(axes.flat, pairs):
    vi = subsample(volume, 80000)
    ax.scatter(params[vi, i], params[vi, j], s=2, c='tab:orange',
               alpha=0.18, linewidths=0, rasterized=True,
               label='volume sampling')
    si = subsample(scan, 50000)
    ax.scatter(params[si, i], params[si, j], s=2, c='tab:blue',
               alpha=0.45, linewidths=0, rasterized=True,
               label='projection scan')
    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"x{j}")
    ax.set_title(f"projection dims ({i}, {j})")

for ax in axes.flat[len(pairs):]:
    ax.set_visible(False)

fig.subplots_adjust(top=0.84, bottom=0.07, left=0.06, right=0.98,
                    hspace=0.33, wspace=0.26)

handles, labels = axes.flat[0].get_legend_handles_labels()
for h in handles:
    h.set_alpha(1.0)
    h.set_sizes([28])
fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.90),
           markerscale=1.0, fontsize=12, frameon=False)

title = (f"{func}: total sample set across all 2D projections "
         f"({len(samples):,} evaluations)")
if roi_volume is not None:
    title += f"   |   volume-sampling roi_threshold = {roi_volume:g}"
fig.suptitle(title, y=0.965, fontsize=14)
out = f"total_samples_{func}.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}  ({len(samples):,} total points: "
      f"{scan.sum():,} scan, {probe.sum():,} probe, {search.sum():,} search)")
