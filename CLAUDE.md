# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ParaProf is a high-performance Python package for computing profile likelihood projections using parallelized grid-anchored differential evolution (DE). It uses MPI-based master-worker parallelization to explore complex parameter spaces efficiently.

## Running MPI-based Code

**Critical**: This codebase requires MPI execution. Always use `mpiexec` with appropriate process counts:

```bash
# Run examples (typically needs 2+ processes)
mpiexec -n 4 python examples/run_himmelblau_4d.py

# Run tests (some require MPI)
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=src/paraprof --cov-report=term-missing
```

**Important environment variable**: Set `OMP_NUM_THREADS=1` when running MPI code to avoid thread oversubscription:
```bash
OMP_NUM_THREADS=1 mpiexec -n 4 python your_script.py
```

## Installation and Dependencies

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[viz]"          # Visualization (matplotlib)
pip install -e ".[emulator]"     # GP-based optimization (scikit-learn)
pip install -e ".[dev]"          # Development tools
pip install -e ".[all]"          # Everything
```

## Testing

```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/paraprof --cov-report=term-missing

# Linting
ruff check src/
black --check src/

# Type checking
mypy src/paraprof
```

## Architecture

### Master-Worker Execution Model

ParaProf uses an **asynchronous master-worker architecture** with MPI:

- **Rank 0 (Master)**: Orchestrates workflow, manages job queues, tracks convergence, holds all state
- **Rank 1+ (Workers)**: Stateless function evaluators, receive tasks and return results

This design ensures workers never hold state and can be terminated/restarted freely.

### Core Components

1. **GridAnchoredDESampler** (`sampler.py`)
   - Central state manager holding all grid data, populations, fitness values
   - Configuration container for algorithm parameters
   - **NOT** an executor - state is modified by jobs and master

2. **Master Process** (`master.py`)
   - `master_main()`: Main state machine coordinating the workflow
   - `run_projection()`: High-level API for running projections with optional refinement
   - `run_all_projections()`: Convenience wrapper for multiple projections
   - Job queue management and worker task distribution

3. **Worker Process** (`worker.py`)
   - `worker_main()`: Simple event loop waiting for tasks
   - Receives `(params, context)`, evaluates `target_func(params)`, returns result
   - Optional emulator-based trial pre-screening for efficiency

4. **Job System** (`jobs/`)
   - **Base class**: `Job` - asynchronous multi-step operations
   - Jobs break work into tasks (function evaluations) and process results iteratively
   - Key job types:
     - `LBFGSBJob`: Global optimization using L-BFGS-B
     - `DEGridPointJob`: Differential evolution at grid points
     - `ActivationJob`: Expand region of interest by activating neighbors
     - `PatchingTestJob`: Wave-based gradient refinement across grid
     - `CoordinateDescentJob`: Coordinate descent refinement
     - `CMAESGridPointJob`: CMA-ES optimization (experimental)

### Algorithm Workflow

1. **Initial Global Optimization** (`LBFGSBJob`)
   - Find promising starting points using L-BFGS-B
   - Multiple random initializations to explore parameter space

2. **Grid Population** (`DEGridPointJob`)
   - Anchor DE populations at grid points in projection dimensions
   - Optimize continuous (non-projected) parameters at each anchor

3. **Adaptive Activation** (`ActivationJob`)
   - Dynamically activate grid points near high-likelihood regions
   - Uses ROI threshold (chi-squared units) to focus computation

4. **Patching** (`PatchingTestJob`)
   - Wave-based propagation of improvements across grid
   - Tests if neighboring solutions can improve current grid points
   - Helps escape local optima and smooth the likelihood surface

5. **Grid Refinement** (`interpolation.py`)
   - Optional increase in grid resolution
   - Uses linear interpolation for warm-starting refined grid
   - Controlled by `enable_refinement` and `refinement_factor` in projection config

### Nuisance Parameter Framework

**NuisanceParameterWrapper** (`nuisance_wrapper.py`) augments test functions with constrained nuisance parameters:

- Separates parameters into POI (parameters of interest) and nuisance parameters
- Parameter ordering: `[poi_0, ..., poi_n, nuis_0, ..., nuis_m]`
- Coupling modes: 'shift', 'scale', 'rotation' - how nuisance parameters affect POI
- Mimics realistic physics scenarios with Gaussian-constrained systematics

This is primarily for testing/benchmarking with realistic complexity.

## Key Configuration Parameters

### Sampler Parameters
- `pop_per_grid_point`: Population size per grid point (default: 1)
- `n_initial_optimizations`: Number of global L-BFGS-B runs (default: 20)
- `roi_threshold`: Region of interest threshold in χ² units (default: 3.0)
- `convergence_threshold`: DE convergence threshold (default: 1e-5)
- `max_patching_waves`: Maximum patching iterations (default: 10)

### Projection Configuration
Each projection dict can specify:
- `dims`: List of parameter indices to project (required)
- `grid_points`: Grid resolution per dimension (required)
- `optimization_method`: 'de' or 'lbfgsb' (default: 'de')
- `lbfgsb_refinement`: Enable L-BFGS-B after DE (default: True)
- `patching_coarse`: Enable patching on coarse grid (default: True)
- `enable_refinement`: Enable grid refinement (default: False)
- `refinement_factor`: Refinement multiplier (default: 2)
- `patching_refined`: Enable patching on refined grid (default: False)

### Emulator-Based Optimization
- `use_de_prescreening`: Enable GP-based trial filtering (default: False)
- `emulator_confidence_threshold`: UCB exploration parameter (default: 2.0)
- Requires `pip install -e ".[emulator]"` for scikit-learn

Emulator pre-screening reduces function evaluations by 30-50% by predicting which trial points are promising.

## Code Style

- Line length: 100 characters (Black and Ruff configured)
- Python 3.10+ type hints encouraged but not enforced
- Tests exclude examples directory (see `pyproject.toml`)
- Logging via custom logger (`logger.py`) with MPI rank awareness

## Common Development Patterns

### Adding a New Job Type

1. Create class in `src/paraprof/jobs/` inheriting from `Job`
2. Implement `start()` - return initial task list
3. Implement `process_result()` - handle worker results, return new tasks
4. Implement `is_finished()` - check completion status
5. Optional: `on_finish()` - update sampler state, spawn child jobs
6. Register in `jobs/__init__.py`

### Adding a New Test Function

1. Add to `test_functions.py` with signature: `get_<name>_function()`
2. Return `(log_likelihood_func, bounds, peak_locations)`
3. Register in `get_test_function()` dispatcher
4. Optionally create nuisance-wrapped version in `nuisance_wrapper.py`

### MPI Communication Pattern

Workers receive tasks as:
```python
task = {'params': np.ndarray, 'context': dict}
```

Workers return:
```python
result = {'params': params, 'value': float, 'context': context}
```

The `context` dict allows jobs to track which task corresponds to which result.

## Performance Considerations

- **Function evaluations are the bottleneck** - minimize these through emulator pre-screening
- Workers are stateless - safe to oversubscribe or run on heterogeneous hardware
- Master process is lightweight - job management is O(grid_size), not O(evaluations)
- Clustering (`use_clustering=True`) helps identify multiple basins in multimodal problems

## Visualization

Automatic plotting via `visualization.py`:
- 1D: Line plots with confidence levels
- 2D: Heatmaps with contour lines
- 3D+: Pairwise 2D slices (max or marginalized)

Additional plots for continuous parameters show optimal values across projection grid.

Enable with `save_plots=True` in `run_projection()`.
