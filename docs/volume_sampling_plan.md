# Implementation plan: stratified ROI/shell volume sampling

Status: design/plan — no implementation yet.

## Goal

After the user-requested projections complete, an optional run stage collects a
**stratified, well-spread set of samples** in full input-parameter space, either
inside the region of interest (`mode='roi'`: `logL > logL_max − roi_threshold`) or
in a shell around it (`mode='shell'`: `logL` between two thresholds). Motivation:
profile-likelihood scans concentrate samples on low-dimensional profile surfaces
(measure zero in the ROI volume), but global-fit users also want representative
points throughout the good-fit volume — and just outside it, to understand from the
per-point computations *why* neighboring regions fail.

Design decisions agreed in discussion:

- **Stratified coverage, not uniform density.** A minimum-spacing (Poisson-disk
  style) set represents every feature at the resolution scale regardless of its
  volume — thin strips and small islands included — which uniform (volume-weighted)
  sampling starves. Uniform-on-ROI MCMC is rejected for v1 (mixing, disconnected
  components, unverifiable uniformity).
- **Three-tier funnel**, each tier strictly cheaper per point than the next:
  1. *Harvest* — cover anchors from existing samples (zero evaluations).
  2. *Direct probe* — one evaluation at each anchor point.
  3. *Anchored search* — for anchors whose probe failed: minimize distance to the
     anchor subject to the band constraint, via L-BFGS-B on a hinged objective,
     warm-started from the nearest known in-band point.
- **Projection-envelope prefilter.** The converged profile grids are rigorous upper
  bounds: a point whose projection falls in a below-threshold (or never-activated)
  cell of *any* computed projection cannot be in the band. Anchors are drawn only
  inside the intersection of these cylinder sets.
- **A uniform subset for free.** Anchors are scrambled-Sobol draws over the box,
  filtered by the prefilter; tier-2 probes are therefore (randomized-QMC) uniform
  over the prefilter region, and the probes that land in-band — before any tier-3
  projection — are tagged as an honest uniform-on-ROI subset. The acceptance
  fraction yields an unbiased **ROI volume estimate** as a summary statistic.
- **Search/report decoupling** (carried over from the anchored-DE discussion): the
  hinge penalty steers tier-3 search only; every recorded sample carries its true
  `logL`, so band membership is always re-derivable, including after global-max
  drift.
- **Hole detection**: an anchor whose tier-3 search converges with residual
  distance well above the anchor spacing marks a hole in the ROI; the
  closest-approach point is reported.

Relation to other planned work: `docs/derived_projections_plan.md` is deferred.
This feature needs **none** of its plumbing — anchors, distances, and constraints
all live in input space, which the master already has.

## Configuration surface

Constructor argument on `ProfileProjector` (validated up front like other config):

```python
volume_sampling = {
    'mode': 'roi',              # 'roi' or 'shell'
    'shell_threshold': 25.0,    # outer ΔlnL; inner edge is roi_threshold (shell mode)
    'n_points': 1000,           # target number of anchors
    'min_spacing': None,        # optional Poisson-disk radius (bounds-scaled units);
                                # None = whatever spacing n_points Sobol anchors give
    'eval_budget': None,        # hard cap on stage evaluations (None = unlimited)
    'search': 'lbfgsb',         # tier-3 method: 'lbfgsb' or 'none' (probe-only)
    'probe_all_anchors': True,  # probe even harvest-covered anchors (uniform subset)
    'search_max_iter': None,    # per-anchor L-BFGS-B cap (None = lbfgsb_max_iter)
    'harvest_files': None,      # extra sample files for tier 1 (besides this run's)
    'output_file': 'volume_samples.csv',   # provenance-tagged output (csv/h5)
    'summary_file': None,       # JSON summary; None = derive from output_file
}
```

`run_all_projections` runs the stage after the projection loop when
`volume_sampling` is set. A standalone `run_volume_sampling(comm, sampler)` entry
point is also exposed (usable after `run_all_projections` returns, same process).

---

## Phase 1 — Projection-envelope prefilter

Master-side only; no MPI, no workers.

1. **Retain projection knowledge.** After each projection completes,
   `run_all_projections` already collects per-projection results; ensure each
   retained record carries what the prefilter needs (all present in
   `export_grid_solution` output): `projection_dims`, `grid_axes`, and per-cell
   best `likelihood`. Refinement runs contribute their finest grid.

2. **`ProjectionEnvelope` class** (new module, e.g. `volume.py`). Built at
   volume-stage start from the retained records plus the *final*
   `global_max_target_val` — per-cell ROI membership is recomputed here, not
   frozen per projection, so later-projection improvements to the global max are
   honored. Vectorized `test(points, threshold_delta)`: for each point and each
   stored projection, map `point[projection_dims]` to a cell index (same
   arithmetic as `_get_grid_indices_from_point`, on stored axes); reject if the
   cell has no stored solution (never activated ⇒ presumed below threshold,
   matching the scan's own expansion logic) or its profile value is below
   `global_max − threshold_delta`.

3. **Threshold semantics.** `mode='roi'` filters with `roi_threshold`. Shell mode
   filters with the *outer* threshold only — profile values upper-bound `logL`, so
   high-profile cells cannot exclude shell membership (the point underneath may
   still be low-likelihood); the inner cut is enforced by actual evaluations in
   tiers 2/3.

4. **Degenerate-case detection.** If any retained projection has
   `len(projection_dims) == n_dims` (direct-eval mode), the grid already covers
   the full space: log a clear "volume sampling adds only resolution here" notice
   and skip the stage (overridable flag if someone insists).

Tests: cylinder-intersection correctness on synthetic grids; never-activated-cell
rejection; shell one-sidedness; final-global-max recomputation; degenerate-case
skip.

## Phase 2 — Anchor generation and harvest tier

Master-side only.

1. **Anchors.** Scrambled Sobol over the bounds box, filtered through the
   envelope, drawn until `n_points` anchors are accepted (with a draw cap and a
   warning if the prefilter acceptance is so low the cap is hit — that itself is a
   useful diagnostic). Record the raw draw count and the prefilter acceptance
   fraction (feeds the volume estimate). If `min_spacing` is set, thin the
   accepted anchors Poisson-disk style (greedy, in bounds-scaled coordinates).

2. **Harvest.** Stream all available past samples — this run's
   `samples_output_file` plus any `harvest_files` — via
   `sample_io.iter_sample_batches`. For each sample inside the band (true `logL`
   test, both edges), find its nearest anchor (scipy `cKDTree` over anchors,
   built once); if within the coverage radius (the anchor spacing), record it as
   that anchor's candidate, keeping the closest (tie-break: higher `logL`).
   Memory stays O(n_anchors), independent of sample-file size. If no sample file
   exists, skip tier 1 with a log note.

3. Each anchor ends Phase 2 in one of two states: `covered(sample)` or
   `uncovered`, plus a warm-start hint: the nearest known in-band sample at any
   distance (for tier 3).

Tests: KD-tree assignment and tie-breaking; band test on both edges; streaming
over both file formats; spacing thinning; bounds-scaling of distances.

## Phase 3 — Probe and search jobs, orchestration

1. **`VolumeProbeJob`** — modeled directly on `InitialPointEvalJob` (batch of
   single evaluations through the normal master/result loop, so
   `_register_target_call` and the global pool see everything). Probes all anchors
   by default (`probe_all_anchors=True`): this is what makes the in-band probe
   results a clean uniform subset and the volume estimate unbiased — harvest then
   only saves tier-3 work, never silently biases tier 2. With
   `probe_all_anchors=False`, covered anchors are skipped (cheaper; uniform
   subset and volume estimate are then reported as unavailable).

2. **`VolumeSearchJob`** — subclass of `LBFGSBJob` with `grid_idx=None`
   (passthrough mode) and `opt_dims = all dims`. The override point: the job's
   objective is computed master-side from the raw `target_val`, so workers are
   untouched. Objective (minimized):

   `dist²(θ, anchor) + κ·hinge(θ)²`, with `hinge = max(0, band_lo − logL)` plus,
   in shell mode, `max(0, logL − band_hi)`; distances in bounds-scaled units; κ
   auto-set so a hinge violation of `roi_threshold` costs ~1 unit of scaled
   distance² (advanced-config `volume.penalty_strength` multiplier, default 1.0).
   Gradient: the distance term is analytic; the hinge term needs ∇logL, which
   reuses the existing finite-difference machinery and the `grad_func` piggyback
   unchanged (chain rule applied master-side). Warm start: the anchor's nearest
   in-band sample (Phase 2); fall back to the probe point.

   Termination per anchor: first evaluation that lands in-band *and* within the
   coverage radius ends the job successfully (we need a covering point, not a
   stationary point); otherwise run to `search_max_iter` and classify:
   - in-band but far from anchor → accepted as `projected` (boundary-projected);
   - never in-band → `hole`, with the closest-approach point (min hinge) recorded.

3. **Orchestration.** New stage list run by the existing `master_main` loop
   mechanics: `VOLUME_PROBE` (one batch job) → `VOLUME_SEARCH` (per-anchor jobs
   for probe-failed anchors, batched to keep workers saturated, skipped when
   `search='none'`). Reuse the dispatch/queue/result plumbing; volume stages
   never touch `population`/grid state, so they can run with the last
   projection's state still loaded. `eval_budget` enforced at job-creation time
   (anchors processed in Sobol order; unprocessed anchors reported as
   `unbudgeted`). Global-max drift: if any stage evaluation beats
   `global_max_target_val`, update it, log loudly, and recompute all band
   classifications at output time from stored `logL` values (membership is never
   frozen); if the drift exceeds a tolerance (default: report-only), warn that
   the projections themselves may warrant a re-run.

Tests: probe bookkeeping; hinged objective and analytic+FD gradient assembly;
early in-band termination; hole vs projected classification; budget cut-off;
drift reclassification. Job-level tests run without MPI (synthetic results fed to
`process_result`, as in the existing suite).

## Phase 4 — Outputs

1. **Volume sample file** (`output_file`, csv/h5 via `sample_io`): rows
   `[params..., logL, tag]` — one extra float column encoding provenance:
   `0=harvested, 1=probe (uniform subset), 2=projected, 3=hole closest-approach`.
   The main `samples_output_file` format is untouched (stage evaluations land
   there as ordinary rows). `read_samples` already handles arbitrary widths;
   document the layout.

2. **JSON summary** (`summary_file`): mode and thresholds; anchor counts per
   outcome (covered / probe-hit / projected / hole / unbudgeted); prefilter
   acceptance; **ROI volume estimate** = box volume × prefilter acceptance ×
   probe acceptance, with a binomial uncertainty; achieved spacing statistics;
   global-max drift if any; whether the uniform subset is valid
   (`probe_all_anchors`).

3. **End-of-run log summary** mirroring the JSON essentials, in the style of the
   existing master summary block.

## Phase 5 — Docs, examples, tests, benchmarks

1. Example script: 4D test function (existing `test_functions.py`), all-pairs 2D
   projections, then ROI volume sampling; a second variant with `mode='shell'`.
   Visualization helper: scatter of volume samples colored by provenance over a
   2D projection's contour plot.
2. Validation test: nD Gaussian logL where ROI volume and uniformity are known —
   assert the volume estimate within tolerance and the uniform-subset tag's
   statistical behavior; a two-island target to exercise multi-component coverage;
   a target with an interior hole to exercise hole detection.
3. Benchmark: coverage (fraction of true-ROI Sobol reference points within the
   coverage radius of an output sample) and evaluations-per-accepted-point vs a
   pure-rejection baseline, on a thin curved ROI (Rosenbrock) — this quantifies
   what tier 3 buys.
4. README section, CHANGELOG entry, docstrings. Document plainly: output is
   stratified coverage, *not* a uniform draw (except the tagged probe subset);
   boundary enrichment of `projected` points; holes/islands missed by the
   original scan are inherited.

---

## Risks and open questions

- **Prefilter looseness with few projections**: with a single 2D projection of a
  high-dimensional space, prefilter acceptance may be tiny and tier 3 does most of
  the work. The acceptance diagnostics make this visible; cheap internal 1D
  profiles to tighten the envelope are a possible v2, not v1.
- **High-dimension expectations**: covering at spacing r needs ~(1/r)^d points;
  docs must frame the deliverable as representative diversity, not dense filling.
- **Tier-3 objective conditioning**: the hinge creates a kink at the band edge;
  L-BFGS-B handles one-sided kinks adequately in practice, but if benchmarks show
  stalling, a squared-hinge smoothing parameter is the first knob (already
  squared in the formulation above).
- **`probe_all_anchors` cost**: probing harvest-covered anchors costs up to
  `n_points` extra evaluations; defaulted on for statistical cleanliness, with the
  off-switch documented as trading the uniform subset away.

## Suggested sequencing

Phases 1–2 are pure master-side logic with fast unit tests and no MPI. Phase 3 is
the substance (two job classes + stage wiring). Phase 4 is small. A working
end-to-end prototype = Phases 1–3 with the file outputs stubbed; run the Phase 5
benchmarks before freezing κ and the termination defaults.
