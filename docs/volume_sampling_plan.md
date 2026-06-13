# Volume sampling: design and architecture

This document describes the design and implementation of the volume-sampling
stage. For user documentation see the README's "Volume sampling" section.

## Goal

After the user-requested projections complete, an optional stage collects a
**stratified, well-spread set of samples** in full parameter space inside the
stage's region of interest: `{logL > logL_max − roi_threshold}`. The stage's
`roi_threshold` defaults to the projection's but can be set larger to also
reach into the shell around the projection ROI (the band is always one-sided,
running all the way up to the global max). Motivation: profile-likelihood scans
concentrate samples on low-dimensional profile surfaces (measure zero in the ROI
volume), but global-fit users also want representative points throughout the
good-fit volume — and just outside it, to understand why neighboring regions fail.

## Design decisions

- **Stratified coverage, not uniform density.** A minimum-spacing (Poisson-disk
  style) anchor set represents every feature at the resolution scale regardless
  of its volume — thin strips and small islands included — which uniform
  (volume-weighted) sampling would starve.
- **Three-tier funnel**, each tier cheaper per point than the next:
  1. *Harvest* — cover anchors from existing samples (zero evaluations).
  2. *Direct probe* — one evaluation at each anchor point.
  3. *Anchored search* — for anchors whose probe failed: minimize distance to
     the anchor subject to the band constraint, via L-BFGS-B on a hinged
     objective, warm-started from the nearest known in-band point.
- **Projection-envelope prefilter.** Converged profile grids are rigorous upper
  bounds: a point whose projection falls in a below-threshold (or
  never-activated) cell of *any* computed projection cannot be in the band.
  Anchors are drawn only inside the intersection of these cylinder sets.
- **A uniform subset for free.** Anchors are scrambled-Sobol draws over the
  bounds box, filtered by the prefilter; tier-2 probes are therefore uniform
  over the prefilter region. The probes that land in-band form an unbiased
  uniform-on-ROI subset, and their acceptance fraction yields an unbiased
  **ROI volume estimate**.
- **Search/report decoupling.** The hinge penalty steers tier-3 search only;
  every recorded sample carries its true `logL`, so band membership is always
  re-derivable, including after global-max drift.
- **Hole detection.** An anchor whose tier-3 search ends without ever reaching
  the band is classified as a hole; the closest-approach point is reported.

## Configuration

```python
volume_sampling = {
    'roi_threshold': None,         # ΔlnL band depth; None = projection's roi_threshold
    'n_anchors': 1000,             # number of stratified anchor points
    'min_spacing': None,           # Poisson-disk radius (bounds-scaled); None = auto
    'eval_budget': None,           # hard cap on stage evaluations (None = unlimited)
    'search': 'lbfgsb',            # tier-3 method: 'lbfgsb' or 'none'
    'probe_all_anchors': True,     # probe harvest-covered anchors (uniform subset)
    'search_max_iter': None,       # per-anchor L-BFGS-B cap (None = lbfgsb_max_iter)
    'interior_steps': 8,           # post-entry walk steps (0 = off)
    'depth_law': 'uniform_dlnl',   # walk depth target: 'volume', 'uniform_dlnl', 'uniform_sigma'
    'harvest_files': None,         # extra sample files for tier 1
    'output_file': 'volume_samples.csv',
    'summary_file': None,          # None = derived from output_file
}
```

## Architecture

### Projection-envelope prefilter (`ProjectionEnvelope`)

Built from `export_grid_solution()`-style records and the final global maximum.
For each point and each stored projection, maps `point[projection_dims]` to a
grid cell (same arithmetic as `_get_grid_indices_from_point`); rejects if the
cell was never activated (presumed outside the ROI, matching the scan's expansion
logic) or its profile value is below `global_max − roi_threshold`. Never-activated
cells are stored as `−∞`. If any projection covers the full parameter space
(direct-eval mode), the prefilter already covers the ROI and the stage is skipped.

### Anchor generation (`generate_anchors`)

Scrambled Sobol draws over the bounds box (in power-of-2 batches) filtered by
the envelope until `n_anchors` anchors pass. The draw count and acceptance
fraction are recorded for the volume estimate. With `min_spacing`, accepted
anchors are thinned Poisson-disk style. Coverage radius = `min_spacing` when
set, else the median nearest-neighbor distance among the anchors.

### Harvest tier (`harvest_existing_samples`)

Streams past samples batch by batch (memory O(n_anchors)); each in-band sample
is assigned to its nearest anchor only (one sample never covers several anchors,
preserving stratification). Per anchor the closest sample wins, higher logL
breaking ties. The sample format is `[params..., logL, phase]`; logL is read at
column `n_dims` and the phase column is ignored.

### Probe and search jobs

**`VolumeProbeJob`**: one evaluation per anchor, results recorded unconditionally
in `probed`/`probe_logls`. In-band probes become the anchor's representative at
distance 0. With `probe_all_anchors=True` (the default) every anchor is probed
regardless of harvest coverage, keeping the probe set uniform for the volume
estimate.

**`VolumeSearchJob`**: subclass of `LBFGSBJob`. The objective is computed
master-side: `-(dist² + κ·v²)` with `dist` the bounds-scaled distance to the
anchor and `v = max(0, band_lo − logL)`. `κ = SEARCH_PENALTY_STRENGTH /
roi_threshold²` so a full-threshold violation costs one unit of scaled
distance². In-band evaluations get a fully analytic gradient; out-of-band
evaluations chain-rule a user `grad_func` or fall back to FD. The job ends at
the first in-band covering point (``hit``), or at termination as ``projected``
(in-band but beyond the coverage radius) or ``hole``.

**Interior walk** (`interior_steps > 0`): after entering the band, a covering
hit launches a depth-targeted walk that marches along the inward aim direction
(toward the nearest known deep point, falling back to the entry–anchor ray),
bisects to land near the drawn depth target, then spends remaining budget on
tangential randomization along the iso-likelihood shell. All moves are capped at
the coverage radius (for hit entries) or `1.5×entry_dist` (for projected
entries). The depth target is drawn adaptively at stage level so under-filled
depth bins (reachable by only a few anchors) are retried until satisfied.

### Outputs

**Volume sample file** (`output_file`): rows `[params..., logL, tag]`, one per
anchor with an in-band representative (covered or projected), tagged 0
(harvested), 1 (probe — the uniform subset), or 2 (search result), plus tag-3
closest-approach rows for hole anchors (NOT in-band; diagnostic).

**JSON summary** (`summary_file`): run config, band edges, per-status anchor
counts, prefilter and probe acceptance fractions, ROI volume estimate with
binomial uncertainty, global-max drift, and whether the uniform subset is valid.

The main `samples_output_file` (the phase-tagged per-evaluation log) is
unaffected; volume-stage evaluations land there as ordinary rows tagged with
`PHASE_VOLUME_PROBE` or `PHASE_VOLUME_SEARCH`.
