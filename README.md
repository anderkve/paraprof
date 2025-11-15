# ParaProf: Parallel Profile Likelihood Computation

[![Tests](https://github.com/anderkve/paraprof/workflows/Tests/badge.svg)](https://github.com/anderkve/paraprof/actions)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ParaProf** is a high-performance Python package for computing profile likelihood projections using parallelized grid-anchored differential evolution (DE). It efficiently explores parameter spaces by strategically placing populations on grid points and dynamically activating regions of interest.

## Key Features

- 🚀 **Parallel Execution**: MPI-based master-worker architecture for efficient parallelization
- 📊 **N-Dimensional Projections**: Supports 1D, 2D, 3D, and higher-dimensional profile likelihood grids
- 🎯 **Adaptive Sampling**: Dynamic grid activation focuses computational effort on high-likelihood regions
- 🔄 **Grid Refinement**: Interpolation-based refinement for increased resolution without full re-computation
- 🔧 **Patching Algorithm**: Wave-based gradient refinement to escape local optima
- 🧠 **Emulator-Enhanced Sampling**: Optional GP-based trial pre-screening reduces evaluations by 30-50%
- 📈 **Built-in Visualization**: Automatic plotting for 1D, 2D, and N-D projections
- 🧪 **Benchmark Suite**: Comprehensive test functions (Himmelblau, Rosenbrock, Rastrigin, etc.)
- 💾 **Warm Starting**: Reuse results across multiple projections

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

```bash
# With visualization support
pip install -e ".[viz]"

# With emulator support (recommended for 30-50% fewer evaluations)
pip install -e ".[emulator]"

# With development tools
pip install -e ".[dev]"

# With everything
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- NumPy
- SciPy
- mpi4py (requires MPI implementation like OpenMPI or MPICH)
- Matplotlib (optional, for visualization)
- scikit-learn (optional, for emulator-based optimization)

## Quick Start

### Minimal Example

```python
from mpi4py import MPI
from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Define target function and projections
log_likelihood, bounds, _ = get_test_function("himmelblau_4d")
projections = [
    {'dims': [0, 1], 'grid_points': [50, 50]},
]

if myrank == 0:
    # Master process
    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=bounds,
        projections=projections,
        pop_per_grid_point=3,
    )

    # Broadcast target function
    comm.bcast(sampler.target_func, root=0)

    # Run projections
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=projections,
        num_generations=1000,
        save_plots=True
    )

    terminate_workers(comm, myrank)
else:
    # Worker process
    worker_main(comm, myrank)
```

Run with MPI:

```bash
mpiexec -n 4 python your_script.py
```

## How It Works

### Algorithm Overview

ParaProf uses a **grid-anchored differential evolution** strategy:

1. **Grid Setup**: Parameters are projected onto a regular grid (user-specified dimensions)
2. **Initial Optimization**: Global L-BFGS-B finds starting maxima
3. **Population Initialization**: DE populations are anchored at promising grid points
4. **Adaptive Evolution**: DE optimizes continuous parameters at each grid point
5. **Dynamic Activation**: High-likelihood neighbors are automatically activated
6. **Patching**: Optional gradient-based refinement propagates improvements across the grid
7. **Refinement**: Optional grid resolution increase using interpolated warm starts

### Master-Worker Architecture

- **Master Process** (rank 0): Orchestrates workflow, manages job queues, tracks convergence
- **Worker Processes** (rank 1+): Evaluate target function in parallel, stateless execution

### Key Components

- `GridAnchoredDESampler`: Central state manager and algorithm configuration
- `master_main()`: State machine coordinating the workflow
- `worker_main()`: Simple event loop for function evaluations
- Job classes: Asynchronous multi-step operations (L-BFGS-B, DE, activation, patching)

## Examples

The `examples/` directory contains several demonstration scripts:

```bash
# 2D projection (default)
mpiexec -n 4 python examples/run_himmelblau_4d.py

# Multiple 1D projections
mpiexec -n 4 python examples/run_himmelblau_1d.py

# 3D projection
mpiexec -n 4 python examples/run_himmelblau_3d.py
```

## Configuration

### Key Parameters

- `pop_per_grid_point`: Population size per grid point (default: 1)
- `mutation_strategy`: DE mutation strategy ('current-to-rand/1', 'rand/1', 'current-to-pbest/1')
- `n_initial_optimizations`: Number of global optimizations (default: 20)
- `roi_threshold`: Region of interest threshold in χ² units (default: 3.0)
- `convergence_threshold`: DE convergence threshold (default: 1e-5)
- `max_patching_waves`: Maximum patching iterations (default: 10)
- `use_de_prescreening`: Enable emulator-based trial filtering (default: False)
- `emulator_confidence_threshold`: UCB exploration parameter (default: 2.0)

### Projection Options

Each projection can specify:

- `dims`: List of parameter indices to project
- `grid_points`: Grid resolution per dimension
- `lbfgsb`: Enable/disable L-BFGS-B refinement (default: True)
- `patching_coarse`: Enable patching on coarse grid (default: True)
- `enable_refinement`: Enable grid refinement (default: False)
- `refinement_factor`: Refinement multiplier (default: 2)
- `patching_refined`: Enable patching on refined grid (default: False)

## Visualization

ParaProf automatically generates publication-ready plots:

### 1D Profiles
- Line plot with confidence levels (68%, 95%)
- Active grid point markers

### 2D Profiles
- Heatmap with contour lines
- Customizable colorbars

### 3D+ Profiles
- Pairwise 2D slice plots
- Maximum slice or marginalized views

### Plot Settings

```python
plot_settings = {
    'dpi': 300,
    'filetype': 'png',
    'slice_mode': 'max',  # or 'all' for marginalization
    'vmin': -4.0,
    'vmax': 0.0,
}
```

## Testing

Run the test suite:

```bash
# Basic tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/paraprof --cov-report=term-missing
```

## Development

### Project Structure

```
paraprof/
├── src/paraprof/          # Source code
│   ├── sampler.py         # Main sampler class
│   ├── master.py          # Master orchestration
│   ├── worker.py          # Worker event loop
│   ├── logger.py          # Logging utilities
│   ├── exceptions.py      # Custom exceptions
│   ├── jobs/              # Job classes
│   ├── visualization.py   # Plotting utilities
│   ├── interpolation.py   # Grid refinement
│   └── test_functions.py  # Benchmark functions
├── tests/                 # Test suite
├── examples/              # Example scripts
└── docs/                  # Documentation
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ParaProf in your research, please cite:

```bibtex
@software{paraprof2025,
  title = {ParaProf: Parallel Profile Likelihood Computation},
  author = {Kvellestad, Anders},
  year = {2025},
  url = {https://github.com/anderkve/paraprof}
}
```

## Acknowledgments

ParaProf implements grid-anchored differential evolution with adaptive sampling strategies inspired by modern global optimization research.

---

**Maintainer**: Anders Kvellestad
**Python Version**: 3.10+
**Status**: Active Development
