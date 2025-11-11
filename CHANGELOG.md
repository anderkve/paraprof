# Changelog

All notable changes to ParaProf will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern package infrastructure with `pyproject.toml`
- MIT License
- Reorganized to src/ layout for better package structure
- Comprehensive test suite with pytest
- GitHub Actions CI workflow for automated testing
- Type hints marker (`py.typed`) for mypy support
- CHANGELOG.md for tracking changes
- Enhanced .gitignore for project-specific files

### Changed
- Moved all source code to `src/paraprof/` directory
- Updated imports to use proper package structure
- Examples now import from installed `paraprof` package
- Minimum Python version set to 3.10+

### Fixed
- Package now properly pip-installable
- Import paths corrected for package structure

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
