# Changelog

All notable changes to ParaProf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
