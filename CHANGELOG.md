# Changelog

All notable changes to ParaProf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Emulator-based DE trial pre-screening**: Reduces DE evaluations by 30-50% using Gaussian Process emulators
  - New module: `src/paraprof/emulator_utils.py` for GP-based fitness prediction
  - New sampler parameters: `use_de_prescreening`, `emulator_confidence_threshold`, `emulator_min_neighbors`, `emulator_length_scale`, `emulator_noise_level`
  - Evaluation cache for GP training data with smart pruning (keeps best + recent)
  - Upper Confidence Bound (UCB) acquisition function for trial filtering
  - Comprehensive test suite in `tests/test_emulator_utils.py` and `tests/test_de_prescreening.py`
  - Optional dependency: scikit-learn >= 1.3.0
- Modern package infrastructure with `pyproject.toml`
- MIT License
- Reorganized to src/ layout for better package structure
- Comprehensive test suite with pytest
- GitHub Actions CI workflow for automated testing
- Type hints marker (`py.typed`) for mypy support
- CHANGELOG.md for tracking changes
- Enhanced .gitignore for project-specific files

### Changed
- Modified `DEGridPointJob` to support emulator-based pre-screening of trial points
- Extended `_register_target_call()` to populate evaluation cache for emulator training
- Moved all source code to `src/paraprof/` directory
- Updated imports to use proper package structure
- Examples now import from installed `paraprof` package
- Minimum Python version set to 3.10+

### Fixed
- Package now properly pip-installable
- Import paths corrected for package structure

### Performance
- Typical 2D projection: ~30-50% reduction in DE evaluations when emulator pre-screening is enabled
- Maintains solution quality (max likelihood difference < 1e-6)

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
