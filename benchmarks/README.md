# ParaProf Performance Benchmarks

This directory contains the performance benchmark suite for ParaProf.

## Overview

The benchmark suite measures and tracks performance across various dimensions:

- **Grid size scaling**: How performance scales with grid resolution
- **Mutation strategies**: Comparison of different DE mutation strategies
- **Convergence efficiency**: Function evaluations needed to reach target accuracy
- **Refinement efficiency**: Speedup from using grid refinement vs direct fine grid
- **Test function comparison**: Performance across different benchmark functions

## Requirements

**IMPORTANT**: Benchmarks require MPI and must be run with `mpiexec`:

```bash
# Install paraprof with dependencies
pip install -e ".[all]"

# Requires mpi4py
pip install mpi4py
```

## Running Benchmarks

### Run All Benchmarks

```bash
# Run with 4 MPI processes (1 master + 3 workers)
mpiexec -n 4 python benchmarks/benchmark_suite.py --all
```

### Run Specific Benchmarks

```bash
# Grid size scaling
mpiexec -n 4 python benchmark_suite.py --benchmark grid_sizes

# Mutation strategy comparison
mpiexec -n 4 python benchmark_suite.py --benchmark mutation_strategies

# Convergence efficiency
mpiexec -n 4 python benchmark_suite.py --benchmark convergence

# Refinement efficiency
mpiexec -n 4 python benchmark_suite.py --benchmark refinement

# Test function comparison
mpiexec -n 4 python benchmark_suite.py --benchmark test_functions
```

### Options

```bash
# Custom output directory
mpiexec -n 4 python benchmark_suite.py --all --output my_results

# Custom grid size for single benchmark
mpiexec -n 4 python benchmark_suite.py --benchmark convergence --grid-size 25
```

## Output

Results are saved to JSON files in the `benchmark_results/` directory (or custom output directory):

```
benchmark_results/
└── benchmark_results_20251112_143022.json
```

Each benchmark result includes:
- Timestamp
- Description
- Timing metrics (seconds)
- Function evaluation counts
- Best likelihood values (where applicable)
- Derived metrics (evaluations per second, speedup factors, etc.)

## Example Output

```
================================================================================
Benchmark: grid_size_scaling
Description: Performance scaling with grid size on himmelblau_4d
Timestamp: 2025-11-12T14:30:22.123456
--------------------------------------------------------------------------------
  grid_sizes: [10, 15, 20, 25, 30]
  timings_seconds: [12.34, 28.56, 51.23, 82.45, 121.67]
  evaluation_counts: [1234, 2856, 5123, 8245, 12167]
  evaluations_per_second: [100.0, 100.0, 100.0, 100.0, 100.0]
================================================================================
```

## Interpreting Results

### Grid Size Scaling
- **Linear scaling**: Evaluation count grows as O(N²) for 2D grids
- **Time per evaluation**: Should remain roughly constant
- **Parallel efficiency**: Check evaluations_per_second consistency

### Mutation Strategies
- **best_strategy**: The mutation strategy that found the highest likelihood
- Compare `evaluation_counts` to see efficiency differences
- Compare `best_likelihoods` to see solution quality differences

### Convergence Efficiency
- **accuracy_delta**: Should be close to 0 (found true maximum)
- **evaluations_per_grid_point**: Lower is better (more efficient)
- **total_evaluations**: Total cost to convergence

### Refinement Efficiency
- **time_speedup**: How much faster refinement is vs direct fine grid
- **evaluation_speedup**: How many fewer evaluations are needed
- Values > 1.0 indicate refinement is beneficial

## Customizing Benchmarks

To add custom benchmarks, edit `benchmark_suite.py`:

1. Add a new method to the `BenchmarkSuite` class
2. Create a `BenchmarkResult` object
3. Run experiments with `_run_mpi_benchmark()`
4. Collect metrics and add to result
5. Add to the `main()` function's argument parser

Example:

```python
def benchmark_my_custom_test(self, ...):
    result = BenchmarkResult(
        name="my_custom_benchmark",
        description="Description of what this measures"
    )

    # Run experiments...

    result.add_metric('my_metric', value)
    self.results.append(result)
    result.print_summary()
    return result
```

## Performance Tips

1. **Use more MPI processes** for larger grids (e.g., `-n 8` for 30x30 grids)
2. **Reduce grid sizes** for quicker testing (use `--grid-size 10`)
3. **Run specific benchmarks** instead of `--all` during development
4. **Check log files** if benchmarks hang or fail

## Troubleshooting

### "MPI is required to run benchmarks"
- Make sure to run with `mpiexec -n <cores> python ...`
- Requires at least 2 processes (1 master + 1 worker)

### Benchmarks hang indefinitely
- Check MPI is working: `mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"`
- Ensure you have at least 2 MPI processes

### Poor performance
- Use more workers: `-n 8` or `-n 16`
- Reduce grid sizes for testing
- Check system load (benchmarks use 100% CPU)

## Citation

If you use these benchmarks in published work, please cite ParaProf:

```bibtex
@software{paraprof2025,
  title = {ParaProf: Parallel Profile Likelihood Computation},
  author = {Kvellestad, Anders},
  year = {2025},
  url = {https://github.com/anderkve/paraprof}
}
```
