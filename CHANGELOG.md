# Changelog

All notable changes to ParaProf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Volume-sampling groundwork (phases 1–2 of `docs/volume_sampling_plan.md`)** —
  master-side building blocks for the upcoming post-projection stage that
  collects a stratified, well-spread sample set inside the ROI (or in a shell
  around it). New `volume.py` module: `ProjectionEnvelope` (necessary-condition
  prefilter built from the converged projection grids — a point whose projection
  falls in a below-threshold or never-activated cell of any computed projection
  provably cannot be in the band), scrambled-Sobol anchor generation filtered
  through the envelope (with prefilter-acceptance bookkeeping for the ROI volume
  estimate), and the tier-1 harvest that covers anchors from already-evaluated
  samples by streaming existing sample files. New `ProfileProjector` argument
  `volume_sampling` (config dict, validated at construction; default None =
  stage disabled). The stage itself — probe and search jobs, orchestration,
  outputs — lands with phases 3–4.

- **Early exit from the DE search on smooth cells** (`advanced_config['de']['allow_early_DE_exit']`,
  **on by default**). Every active grid cell normally spends at least
  `de.convergence_window` DE generations confirming convergence. When a fresh
  cell's in-population neighbours agree on the profiled argmax (and the neighbour
  warm-start was the best activation seed), the local argmax field is smooth, so
  the cell runs a **single** DE generation then goes straight to the L-BFGS-B
  polish. That generation still runs, so the exit is *measured*: a cell that
  improves keeps going. A replicate study
  (`examples/run_allow_early_de_exit_replicate_study.py`, 6–8 seeds per mode,
  scoring ROI quality with a noise-robust one-sided deficit metric) shows a
  robust win with no measurable quality cost on unimodal-inner targets —
  Himmelblau-4D −13.7% and Rosenbrock-4D −10.9% target calls (both Mann–Whitney
  *p* < 0.01), ROI deficit indistinguishable from baseline. Set
  `allow_early_DE_exit=False` for a genuinely multimodal-inner target
  (Rastrigin-4D: deficit up, coverage 68%→59%, *p* ≤ 0.02), where one DE
  generation under-explores the modes. Reuses existing data only (no new
  evaluations); adds the `sampler.de_cells_skipped` diagnostic counter.

- **`n_optima` prior** — optional `ProfileProjector` argument giving the number
  of optima the target has *globally*; use only when confident it has one or a
  few. It steers the initial-optimization basin-detection stage: a known
  **maximum** stops the rolling multistart once that many distinct optima are
  found — the global maximum is then necessarily among them, so the
  `basin_detection.min_starts` floor is skipped (`n_optima=1` stops after the
  first converged start) — while a known **minimum** keeps the stage running
  until at least that many are found. Pass an `int` (exact) or
  `{'min': int, 'max': int}`. On genuinely multimodal targets, where the
  Bayesian rule needs ~`W²` repeat hits to enumerate `W` modes and so runs to
  the `n_initial_optimizations` cap, this saves substantially (Himmelblau-4D,
  16 modes: ~62% fewer target calls, all modes found, identical global
  maximum); on unimodal targets at adequate convergence the rule already stops
  early, so the prior is a no-op there.
- **Pluggable sample file formats with an HDF5 binary option.** Sample I/O now
  goes through a format layer (`paraprof.sample_io`) that dispatches on the
  file extension: `.csv` (text, default) or `.h5`/`.hdf5` (HDF5 binary, ~half
  the size and faster, via the optional `h5py` extra — `pip install
  paraprof[hdf5]`). Both `samples_output_file` and `warm_start_file` accept any
  supported format. New public helpers: `read_samples`/`write_samples`
  (one-shot load/save), `combine_samples` (stream-merges files, mixing
  formats), and `create_sample_writer`.
- **Basin detection for the initial-optimization stage**, on by default.
  Replaces the fixed all-at-once batch of Latin-hypercube global L-BFGS-B
  starts with a *rolling* multistart: each converged optimum is clustered
  online into a registry of distinct optima, and a Boender-Rinnooy Kan Bayesian
  stopping rule (restricted to ROI-competitive optima) halts the stage once the
  expected number of undiscovered ROI optima falls below
  `basin_detection.undiscovered_threshold`, aborting any still-running
  optimizations at that point. `n_initial_optimizations` is now an upper bound
  rather than a fixed count. Knobs under `advanced_config['basin_detection']`:
  `undiscovered_threshold` (set to `0` to disable early stopping), `min_starts`,
  and `batch_size` (`None` = FD-aware auto: ≈ `n_workers` / the per-gradient
  finite-difference fan-out). The clustering tolerance is a fixed internal
  constant, not user-tunable. New sampler state `initial_optima_registry` plus
  the `register_initial_optimum`, `basin_detection_roi_stats`,
  `basin_detection_should_stop`, and `resolve_initial_opt_batch_size` helpers.
- **Convergence-gated basin registration.** Only initial-optimization runs that
  actually converge (terminate on the L-BFGS-B function tolerance, not by
  exhausting `lbfgsb_max_iter`) are counted as distinct optima for the stopping
  rule. A truncated descent still updates the global maximum, the solution
  pool, and `initial_maxima` — it just doesn't mint a spurious basin. This
  restores the Boender-Rinnooy Kan rule's precondition (local searches run to
  convergence); without it, a stiff target under a too-small `lbfgsb_max_iter`
  registered many pseudo-optima strewn along a valley (Rosenbrock-4D: 29 for a
  one-optimum function), inflating `W` so the rule ran to the cap. Gating
  collapses that to the true count (Rosenbrock 29→1, Himmelblau-4D 17→16) with
  no change to adequate-budget behaviour or grid quality, and makes `n_optima`
  wait for a genuinely converged optimum. Always on; no knob.

### Changed
- **Updated default settings.** `roi_threshold` now defaults to `4.0` (was
  `3.0`); `de.allow_early_DE_exit` now defaults to `True` (was off);
  `suspect_recheck.max_waves` now defaults to `10` (was `3`); and
  `suspect_recheck.polish_threshold` now defaults to `1e-3` (was `1e-4`).
- The default `n_initial_optimizations` is now a generous safe ceiling,
  `min(400, 50 * n_dims)`, since the stopping rule controls the actual spend. An
  explicit `n_initial_optimizations` overrides the default.
- **User-supplied gradient support** via the new `grad_func` constructor
  argument on `ProfileProjector`. When provided, the L-BFGS-B paths
  request `grad_func(params)` from workers alongside the target evaluation
  and skip the finite-difference evaluations the user covered. Partial
  gradients are supported: return a length-`n_dims` array with `NaN` for
  unknown components, or a `{dim_index: value}` dict for known components.
  Any uncovered dim still falls back to the configured
  `lbfgsb.gradient_method` (forward or central FD). DE paths are
  unchanged. New counters: `sampler.target_calls_saved_by_user_gradient`
  (FD calls avoided) and `sampler.user_gradient_errors` (grad_func
  failures that triggered FD fallback); both appear in the end-of-run
  summary. Sign convention: `grad_func` returns the gradient of the
  function being MAXIMIZED. The MPI broadcast format for `run_scan` is
  now a `(target_func, grad_func)` tuple; the worker still accepts the
  legacy bare-callable form so manual broadcasts continue to work.
- **Suspect-cell recheck** pass, on by default. Runs after standard patching
  to catch grid cells (including contiguous strips) that converged to a
  wrong optimum in the profiled dimensions but slipped past the
  fitness-only patching filter. Flags cells whose profiled-parameter
  vector is far (robust MAD threshold) from its neighbour-median, then
  re-evaluates each flagged cell against a small, diverse set of seeds
  (non-suspect neighbours, an extended Chebyshev-radius ring, and
  proximity samples from the cross-projection global pool); seeds that
  beat the cell by more than a tolerance trigger an L-BFGS-B polish.
  Subsequent waves propagate from updated cells the same way patching
  does, so a fix at the boundary of a stuck strip carries inward.
  Configurable via the new ``suspect_recheck`` sub-dict of
  ``advanced_config`` (see ``ProfileProjector`` docstring); also exposed
  as ``sampler.suspect_recheck_enabled`` etc. The whole pass is a no-op
  on perfectly-smooth surfaces.
- **Cross-projection knowledge transfer** for multi-projection scans, on by
  default. When `run_all_projections` runs more than one projection, the
  later projections automatically reuse evaluations from the earlier ones
  via two surgical hooks on the existing in-memory `global_solution_pool`:
  - On every projection after the first, `initial_maxima` are seeded from
    the pool (mapped onto the new projection's grid, best per cell, ROI
    filtered); when this fires the master skips the
    `n_initial_optimizations` global L-BFGS-B starts that would otherwise
    rediscover known maxima.
  - At every grid-cell activation, one random LHS slot in the population
    is replaced with the highest-fitness past evaluation whose
    projection-dim coordinates are closest to the cell.
  Both hooks are no-ops on the first projection and when the pool is
  empty. They are toggleable via the new ``cross_projection`` sub-dict
  of ``advanced_config``::

      ProfileProjector(..., advanced_config={
          'cross_projection': {
              'proximity_warm_start': False,        # disable per-cell hook
              'pool_seeded_initial_maxima': False,  # disable initial_maxima seeding
          },
      })

  and surface as ``sampler.proximity_warm_start`` /
  ``sampler.pool_seeded_initial_maxima`` instance attributes after
  construction (used by the benchmark for A/B testing).
  Benchmarks (`examples/run_proximity_warm_start_benchmark*.py`) on the
  full 6-projection 50x50 scans of Himmelblau-4D and Rosenbrock-4D show
  ~10% and ~50% reductions in target-function calls, respectively;
  Rosenbrock-6D (15 projections) drops 64%, Rastrigin-4D drops 70%.
- `global_pool_size` now auto-scales with target dimensionality:
  `clip(n_dims * DEFAULT_GLOBAL_POOL_PER_DIM, DEFAULT_GLOBAL_POOL_SIZE,
  DEFAULT_GLOBAL_POOL_MAX)`. The 4-D default is unchanged (10 000
  entries); higher-D scans get a proportionally larger pool so that the
  cross-projection knowledge accumulated by the hooks above isn't evicted
  in scans with many projections (`C(n_dims, 2)` grows quadratically in
  `n_dims`). The cap at `DEFAULT_GLOBAL_POOL_MAX` (100 000 entries)
  bounds master-side memory and the per-projection proximity-cache
  rebuild cost at very high `n_dims`.
- `ProfileProjector(warm_start_file=...)` — dedicated path for reading
  warm-start samples, separate from `samples_output_file`. Previously, the
  master would implicitly warm-start from `samples_output_file`; that
  coupling is gone. To preserve the old behaviour, set
  `warm_start_file=samples_output_file`.
- `InitialPointEvalJob` (`paraprof.jobs.InitialPointEvalJob`) — user-supplied
  `initial_points` are now evaluated through the standard master/result loop
  instead of a hand-rolled send/recv block. They are recorded in
  `samples_output_file` and `global_solution_pool` like every other
  evaluation. A new `INITIAL_POINTS_EVAL` stage runs before
  `INITIAL_OPTIMIZATION`.
- Simplified example scripts `examples/run_himmelblau_4d_simple.py` and
  `examples/run_rosenbrock_4d_simple.py` showing the trimmed user API.
- New module-level constants in `sampler.py` for the DE/activation/patching
  knobs that benchmarking showed do not affect ROI grid quality.
- **Host-framework integration API** for embedding paraprof inside an
  external master/worker MPI loop (e.g. as a ScannerBit plugin):
  - `worker_main(comm, myrank, target_func=None)` accepts a pre-supplied
    target function instead of waiting on `comm.bcast`. Useful when the
    target function is a bound method that cannot be pickled.
  - `ProfileProjector(parameter_names=...)` enables projection `dims` to
    be specified by parameter name (string), in addition to integer index.
  - New `paraprof.run_scan(comm, sampler, projections, ...,
    broadcast_target_func=True)` convenience helper that bundles the
    target-function broadcast, `run_all_projections`, and `terminate_workers`
    into a single master-side call.
- `GAMBIT_plugin/gambit_paraprof.py`: ScannerBit plugin that exposes
  paraprof as a scanner inside GAMBIT, plus `GAMBIT_plugin/README.md` and
  a `paraprof_example.yaml` snippet.

### Changed
- **Trimmed the user-tunable surface by roughly 2/3.** After the cleanup
  the constructor exposes 8 core kwargs and `advanced_config` exposes
  ~7 keys across 4 sub-dicts (was ~30 keys across 8 sub-dicts).
- `lbfgsb_polish` constructor argument now stored as `self.lbfgsb_polish`
  (was inconsistently stored as `self.lbfgsb_refinement`).
- README updated: corrected per-projection key names, added an
  `advanced_config` reference table, refreshed the Quick Start example
  to use the context-manager pattern, refreshed the Project Structure
  layout, and pointed users at the new `_simple` example scripts.

### Removed
- **CMA-ES optimization path.** Benchmarking showed it took ~21x more
  target evaluations than DE/L-BFGS-B for the same answer on the test
  suite. Deleted `jobs/cmaes_job.py`, the `CMAES_LOOP` master stage, the
  `create_cmaes_generation_jobs` sampler method, all `cmaes_*`
  attributes, and the `cmaes` advanced_config sub-dict.
  `optimization_method` is now restricted to `{'de', 'lbfgsb'}`.
- **Coordinate-descent refinement path.** It only saved a handful of
  calls vs L-BFGS-B refinement and added another job class. Deleted
  `jobs/cd_job.py`, the `use_cd_refinement` constructor argument, the
  `cd_*` attributes, and the `cd` advanced_config sub-dict. Refinement
  now always uses L-BFGS-B.
- **GP emulator pre-screening.** Off by default; the path was slow on
  the small benchmark and 16 of 17 emulator tests were failing on main.
  Deleted `emulator_utils.py`, the `use_emulator` constructor argument,
  all `emulator_*` attributes, the `emulator` advanced_config sub-dict,
  the per-grid-point and global eval caches that only fed the GP, and
  the worker-side pre-screening block. `pyproject.toml`'s optional
  `[emulator]` extra was renamed to `[clustering]` (scikit-learn is
  still optionally used by refinement clustering).
- **Six low-signal advanced_config knobs hidden** (moved to module-level
  constants, defaults unchanged): `de.mutation_strategy`,
  `de.pbest_fraction`, `de.neighbor_pull_probability`, `global_pool_size`,
  `patching.n_neighbors`, `activation.mix_ratios`. Gold-standard grid
  comparisons on Rosenbrock-4D showed each had no measurable effect on
  ROI grid quality.
- Stale `test_cd_refinement.py` at the repo root (it imported the
  pre-rename class `GridAnchoredDESampler` and was non-functional).
- Stale tests covering the removed features: `test_de_prescreening.py`,
  `test_emulator_utils.py`, `test_emulator_refinement.py`, and a
  `test_mutation_strategy_validation` test that tested a non-existent
  constructor kwarg.

### Fixed
- 1-D profile plots: confidence-level lines were drawn at ΔlogL = -1.0
  ("68% CL") and -4.0 ("95% CL"). Those deltas don't match the Wilks
  1-DOF mapping; the correct values are -0.5 and -1.92. Updated the
  defaults so the labels match the geometry. The 2-D contour defaults
  ([-1, -3]) were already correct for 2 DOF and are unchanged.
- `DEGridPointJob` could hang the master if `parent_pool` was below the
  three-parent minimum (or below two in the `current-to-pbest/1` branch):
  the per-individual `continue` did not decrement `evals_remaining`, so
  the job never marked itself finished. The upstream guard in
  `create_de_generation_jobs` made this practically unreachable, but the
  job is now defensively safe on its own.
- Initial-point evaluations no longer bypass `_register_target_call` /
  `_update_global_pool`, so they now appear in `samples_output_file` and
  in the global solution pool. The stray `sampler.global_best_params`
  assignment (never initialised, never read) was removed in the same
  refactor.
- `ConfigurationError`, `InvalidBoundsError`, and `InvalidProjectionError`
  were imported only inside `__init__` but referenced from
  `_reset_for_new_projection`; promoted to module scope. Without this
  fix, passing an unknown `optimization_method` raised `NameError`
  instead of `ConfigurationError`.
- `__del__` now guards against partially-initialised objects, removing
  a noisy `AttributeError: '_file_closed'` that surfaced as an unraisable
  exception in pytest output.
- README documented several per-projection keys that did not exist in
  the code (`enable_refinement`, `refinement_factor`, `patching_coarse`,
  `patching_refined`, `lbfgsb_refinement`); replaced with the actual keys.

## [1.0.0] - 2025-11-11

### Added
- Initial release of ParaProf
- Grid-Anchored Differential Evolution sampler
- MPI-based parallel execution
- Support for 1D, 2D, and N-D profile likelihood projections
- Automatic grid refinement with interpolation
- Multiple DE mutation strategies
- L-BFGS-B local optimization
- Patching algorithm for gradient-based refinement
- Comprehensive benchmark test functions
- Visualization tools for profile likelihood plots
- Direct evaluation mode for full-dimensional grids
- Warm-start capability across projections
- Global solution pool for cross-projection knowledge transfer

[Unreleased]: https://github.com/anderkve/paraprof/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/anderkve/paraprof/releases/tag/v1.0.0
