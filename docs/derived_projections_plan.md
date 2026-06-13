# Implementation plan: profile likelihood projections in derived-parameter planes

Status: deferred — kept for future work. The volume-sampling feature
(`volume_sampling_plan.md`) is being pursued first; it shares none of this
plan's plumbing (anchors and constraints live in input space there).

## Goal

Add support for user-requested 1D/2D profile likelihood projections over **derived
quantities** (e.g. physical masses) that paraprof cannot express analytically — it
only learns their values from the target function's output at evaluated points.

Core mechanism (agreed in design discussion):

- Lay the grid over the derived quantities. Each DE individual is **anchored** to a
  grid cell: its selection fitness is `logL(θ) − κ·dist²(d(θ), cell_center)`, with
  the distance measured in cell-width units. The center pull keeps the population
  stratified over the derived plane instead of piling up at shared good corners.
- **Search/report decoupling**: the penalized fitness drives DE selection only. The
  *reported* profile value per cell is the best `logL` among all evaluations whose
  derived values actually landed in that cell (hard membership for bookkeeping, soft
  anchoring for dynamics). This makes the results penalty-bias-free: κ affects
  efficiency, never correctness. The plotted quantity is the per-cell max, the
  natural definition for binned derived-plane profiles.
- **κ auto-scaling**: per-axis distances are scaled by cell width; κ is set so that
  sitting one cell width from the center costs a fixed fraction of `roi_threshold`.
  Not a required user knob. A mild per-individual κ ramp handles individuals that
  fail to enter their cell.
- Cells whose center is unreachable converge to a closest-approach point; a residual
  distance above ~a cell width marks the cell **unattainable** (distinct from
  low-likelihood in output and plots).

The same machinery doubles as a **top-up mode**: after a normal input-space scan,
bin existing samples in the derived plane and run anchored DE only on
under-converged ROI-adjacent cells.

Related but out of scope here: the space-filling ROI/shell volume-sampling stage.
It shares Phase 1 (derived values in the sample flow) and should reuse that design,
but is planned separately.

---

## Phase 1 — Plumbing: derived values through the system

Prerequisite for everything else; independently useful (derived values in sample
files enable post-hoc derived-plane binning today).

1. **Target function protocol.** Accept both forms:
   - current: `target_func(params) -> float`
   - new: `target_func(params) -> (float, derived)` where `derived` is a
     `{name: value}` dict or an array matching a declared name order.

   New constructor argument `derived_names: list[str]` declares the names and fixes
   the column order. Validation: tuple returns without `derived_names` (and vice
   versa) raise `ConfigurationError` at first evaluation, with a clear message.

2. **Worker (`worker.py`).** Normalize the return into the result dict as
   `result['derived']`: float array of length `n_derived`, NaN for
   missing/non-finite entries (logged like target errors, but non-fatal — a NaN
   derived value just means the point can't be cell-classified). Bare-scalar
   targets produce `derived=None`. Follow the existing dual-form precedent used for
   the `(target_func, grad_func)` bcast payload.

3. **Master result loop (`master.py`) and registration (`sampler.py`).**
   `_register_target_call(params, target_val)` gains a `derived` argument; the
   sample buffer rows become `[params..., derived..., logL]`
   (width `n_dims + n_derived + 1`).

4. **Sample I/O (`sample_io.py`).** Column layout `[params..., derived..., logL]`.
   HDF5: store `derived_names` and `n_dims` as dataset attrs. CSV: stays headerless;
   document the layout (and consider an optional `#`-comment header line, skipped on
   read). `read_samples`/`combine_samples` are width-agnostic already; the
   `warm_start_file` reader must slice `params = row[:n_dims]`, `logL = row[-1]`
   and accept both old (`n_dims+1`) and new widths, erroring on anything else.

5. **Global pool.** Pool entries gain the derived vector so derived-space proximity
   seeding (Phase 4) and the proximity-pool cache can operate in derived
   coordinates.

6. **Pass-throughs.** `nuisance_wrapper.py` forwards derived values unchanged.
   GAMBIT plugin: return observables from the likelihood evaluation (they are
   computed anyway).

Tests: protocol validation, worker normalization (dict/array/NaN/missing), sample
file round-trip in both formats and both widths, warm-start from both widths,
`combine_samples` width-mismatch error.

## Phase 2 — Derived projection spec and grid state

1. **Projection spec.** New form alongside `{'dims': ...}`:

   ```python
   {'derived': ['m1', 'm2'], 'grid_points': [50, 50],
    'range': [[lo1, hi1], [lo2, hi2]]}   # or 'range': 'auto'
   ```

   Name resolution against `derived_names` (analogous to `_resolve_dims`). A
   projection is either input-dims or derived — no mixing in v1.

2. **`_reset_for_new_projection`.** Derived mode sets `self.derived_mode = True`,
   `projection_derived_idx` (indices into the derived vector), grid axes over the
   derived ranges, and — crucially — `profiled` search space = **all** `n_dims`
   input dimensions (`_construct_params` is bypassed; trial points are full input
   vectors). The input-dims path is untouched.

3. **Cell membership.** A derived-mode variant of `_get_grid_indices_from_point`
   that maps a result's *derived values* (not its input coords) to a cell index,
   returning `None` for points outside the grid range or with NaN derived values.

4. **Ranges.** v1 policy: explicit ranges, or `'auto'` = min/max of derived values
   seen during the initial optimization stage, padded by a margin factor (default
   e.g. 20%). The grid is fixed once laid; later samples that spill outside are
   still logged and counted, and a spill fraction is reported at the end so the
   user can rerun with wider ranges. No mid-run regridding in v1.

## Phase 3 — Anchored DE job

New job type `ANCHORED_DE_GRID_POINT` (own class next to `DEGridPointJob`; the
input-space DE job stays untouched — the two differ in trial construction, fitness,
and bookkeeping, and sharing a base class buys little).

1. **State.** `population[grid_idx]` keeps the same shape as today but
   `profiled_params[i]` are full `n_dims` vectors and each state carries the
   anchor (cell center in derived coords) plus, per individual, the latest derived
   values and raw `logL`. `fitnesses` hold the **penalized** fitness (selection);
   a separate per-cell record `cell_best = {'full_params', 'derived', 'logL'}`
   holds the best *within-cell* hit (reporting).

2. **Fitness.** `penalized = logL − κ · Σ_k ((d_k − c_k)/w_k)²` with `w_k` = cell
   width on derived axis k. Default `κ = penalty_strength * roi_threshold` with
   `penalty_strength` defaulting to ~1.0 (advanced-config key, not a headline knob).
   Per-individual ramp: if an individual has not produced a within-cell hit after
   `N` generations (advanced-config, default e.g. 10), multiply its κ by a factor
   (default e.g. 2, capped) until it does.

3. **Selection and bookkeeping.** Trial vs target compared on the *target's* anchor,
   standard 1-to-1 DE selection. Independently of selection, **every** result in
   derived mode is routed through a new
   `sampler._register_derived_cell_hit(full_params, derived, logL)` from the master
   result loop, so any evaluation opportunistically improves whichever cell it lands
   in (the derived-mode analog of `_update_global_pool`, which it also feeds).
   `profile_likelihood_grid[idx]` mirrors `cell_best[idx]['logL']`.

4. **Donor mixing.** Reuse the existing `neighbor_pull_probability` mechanism with
   neighbors on the *derived* grid. Parent-pool donors: with probability
   `anchored_de.local_donor_prob` (default high, e.g. 0.8) draw donors from cells
   within a k-ring of the target's anchor; otherwise from the global parent pool.
   Cross-anchor donors are informative (they encode how θ changes as d moves), but
   far-anchor donors are mostly destructive — this is the knob that controls that
   trade-off, and the first thing to tune in benchmarks.

5. **Convergence and polish.** Improvement history tracks the penalized fitness,
   same window logic as `DEGridPointJob.on_finish`. L-BFGS-B polish of the
   penalized objective is possible only via finite differences over all `n_dims`
   (no gradient of d): support it but default `lbfgsb_polish=False` in derived
   mode, with a log note. `allow_early_DE_exit` is disabled in derived mode in v1
   (its smooth-interior heuristic assumes fixed projection coords).

6. **Unattainable cells.** At convergence with no within-cell hit: if the best
   residual scaled distance exceeds a threshold (default 0.5 cell widths), set
   `status='unattainable'` and record the closest-approach point. Unattainable
   cells terminate expansion like below-threshold cells and are rendered
   distinctly (Phase 6).

## Phase 4 — Activation, expansion, patching, suspect recheck in derived mode

1. **Initial activation.** Bin all available samples (warm-start file, initial
   optimization stage, pool) into derived cells. Activate occupied cells inside the
   ROI band; seed each cell's population with its best within-cell sample plus
   diverse extras (nearest-in-derived-space pool points, LHS over inputs), reusing
   the `activation_mix_ratios` structure. The unconstrained initial-optimization
   stage runs as today — it establishes `global_max_target_val` and produces the
   seeds (and the `'auto'` ranges).

2. **Dynamic activation.** When a cell converges above threshold, activate
   neighbor cells seeded from its best full params — the anchor moves one cell
   over and the penalty pulls the seed outward (continuation). Same frontier logic
   as `create_dynamic_activation_jobs`, keyed on derived-grid indices.

3. **Patching.** A neighbor's best θ generally does *not* land in my cell, so the
   input-space "test neighbor params directly" shortcut doesn't apply. Derived-mode
   patching wave = short anchored-DE burst per candidate cell with neighbor best-θ
   vectors injected as population members. Same wave/baseline-improvement loop in
   `master_main`.

4. **Suspect recheck.** The discontinuity metric runs on `cell_best['full_params']`
   across the derived-grid neighborhood — unchanged MAD logic. Multivaluedness
   (disconnected θ-branches mapping to the same cell with different logL) is *the*
   failure mode in derived mode, so this pass matters more here: keep it enabled by
   default and include a multivalued target in the benchmarks (see Testing).

## Phase 5 — Master wiring and top-up mode

1. **Stages.** Derived-mode stage list:
   `INITIAL_POINTS_EVAL? → INITIAL_OPTIMIZATION → DERIVED_ACTIVATION →
   ANCHORED_DE_LOOP → PATCHING_WAVES → SUSPECT_RECHECK_WAVES`.
   Implement `ANCHORED_DE_LOOP` as a sibling of `DE_LOOP` (same generation loop,
   different job factory) rather than overloading `DE_LOOP` with mode switches.
   Refinement runs (`setup_refinement_run`) are **not supported** for derived
   projections in v1 — clear `ConfigurationError`.

2. **Top-up behavior.** Falls out of activation: a cell whose binned best is
   already above threshold and whose seeds are dense enough starts with a warm,
   nearly-converged population, so its anchored-DE cost is a short convergence
   check. Optional accelerator `derived.topup_accept_existing` (default False): mark
   such cells `'optimized'` immediately when their `cell_best` is within a
   tolerance of the neighborhood trend, skipping DE entirely — patching and suspect
   recheck still get a chance to flag them. Conservative default because binned
   maxima are lower bounds with no per-cell convergence guarantee.

3. **Result loop.** In derived mode the master routes every result through
   `_register_derived_cell_hit` (Phase 3.3) in addition to
   `_register_target_call`. MPI payloads grow by one small float array per result —
   negligible.

## Phase 6 — Output, visualization, docs, examples

1. `export_grid_solution`: include derived axes/names, `cell_best`, closest-approach
   info, and unattainable statuses.
2. `visualization.py`: derived-name axis labels; unattainable cells rendered
   distinctly from never-activated cells (e.g. hatched vs white).
3. End-of-run summary: spill fraction outside the derived grid range, count of
   unattainable cells, count of cells with no within-cell hit.
4. Examples: a test function with nontrivial analytic derived quantities, e.g.
   `m1 = sqrt(θ0² + θ1²)`, `m2 = θ2·θ3` over a multimodal logL — including a
   **multivalued** case (two θ-branches per (m1, m2) cell with different logL).
   Example script + README section + CHANGELOG entry. GAMBIT plugin example YAML
   updated to declare observables.

---

## Testing and benchmark strategy

- Unit tests per phase as listed above; anchored-DE selection math and κ scaling
  testable without MPI (jobs are plain objects fed synthetic results, as in the
  existing test suite).
- Integration: 2D derived projection of a 4D test function where the derived-plane
  profile is computable by brute force on a dense grid; assert per-cell agreement
  within tolerance, and that no ROI cells are missing (the gap-filling claim).
- Baseline comparison: same target, post-hoc binning of an input-space scan's
  samples vs. the anchored-DE scan — quantifies the raggedness this feature
  removes and the evaluation cost it adds.
- Multivalued target: verify suspect recheck recovers the better branch.
- Unattainable region: derived quantities with a forbidden region (e.g.
  `m1 ≥ m2` impossible) — assert cells are marked unattainable, not low-likelihood.

## Risks and open questions (tracked, not blockers)

- **Donor-mixing scope** (`local_donor_prob`, k-ring size): the main efficiency
  knob; settle defaults by benchmark, not by argument.
- **Per-cell cost**: curved-valley landscapes (logL ridge ∩ d-level-set tube) will
  need more generations than input-space profiling; budget expectations go in the
  docs.
- **Grid-range misjudgment**: v1 answer is the spill report + rerun; mid-run grid
  extension is a possible v2.
- **Polish in derived mode**: FD-based L-BFGS-B on the penalized objective may be
  worth enabling by default later if benchmarks show DE stalls near convergence.
- **Sample-format compatibility**: the width change is the one external surface;
  both widths must round-trip through `warm_start_file` forever.

## Suggested sequencing

Phases are ordered by dependency; 1 and 2 are small, 3+4 carry the algorithmic
substance, 5 is wiring, 6 is polish. Phase 1 merges independently (useful on its
own). A working end-to-end prototype needs 1–5 with patching/suspect-recheck
stubbed; benchmarks should run before freezing the Phase 3 defaults.
