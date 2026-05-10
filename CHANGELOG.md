# Changelog

All notable changes to ParaProf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Simplified example scripts `examples/run_himmelblau_4d_simple.py` and
  `examples/run_rosenbrock_4d_simple.py` showing the trimmed user API.
- New module-level constants in `sampler.py` for the DE/activation/patching
  knobs that benchmarking showed do not affect ROI grid quality.

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
