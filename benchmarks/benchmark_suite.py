"""
Performance Benchmark Suite for ParaProf

This module provides benchmarks to measure and track performance across:
- Different grid sizes
- Different mutation strategies
- Function evaluation efficiency
- Serial vs parallel scaling

Usage:
    # Run all benchmarks
    python benchmark_suite.py --all

    # Run specific benchmark
    python benchmark_suite.py --benchmark grid_sizes

    # Compare mutation strategies
    python benchmark_suite.py --benchmark mutation_strategies

    # Run with specific number of cores
    mpiexec -n 4 python benchmark_suite.py --benchmark parallel_scaling
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import numpy as np

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Parallel benchmarks will be skipped.")

# Import paraprof components
try:
    from paraprof import (
        ProfileProjector,
        run_projection,
        terminate_workers,
        worker_main,
        get_test_function,
        set_log_level
    )
except ImportError as e:
    print(f"Error: Could not import paraprof. Make sure it's installed.")
    print(f"Run: pip install -e .")
    print(f"Error details: {e}")
    sys.exit(1)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metrics: Dict[str, Any] = {}
        self.timestamp = datetime.now().isoformat()

    def add_metric(self, key: str, value: Any):
        """Add a metric to the results."""
        self.metrics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'timestamp': self.timestamp,
            'metrics': self.metrics
        }

    def print_summary(self):
        """Print a formatted summary of the results."""
        print(f"\n{'='*80}")
        print(f"Benchmark: {self.name}")
        print(f"Description: {self.description}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'-'*80}")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*80}\n")


class BenchmarkSuite:
    """Main benchmark suite for ParaProf."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

        # Set logging to WARNING to reduce noise during benchmarks
        set_log_level('WARNING')

    def save_results(self, filename: Optional[str] = None):
        """Save all results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename
        data = {
            'benchmark_suite_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    def benchmark_grid_sizes(self, test_function: str = "himmelblau_4d",
                            grid_sizes: List[int] = None,
                            num_cores: int = 1) -> BenchmarkResult:
        """
        Benchmark performance across different grid sizes.

        Parameters
        ----------
        test_function : str
            Name of test function to use
        grid_sizes : list of int
            Grid sizes to test (e.g., [10, 20, 30, 40, 50])
        num_cores : int
            Number of MPI cores to use

        Returns
        -------
        BenchmarkResult
            Results containing timing and evaluation counts for each grid size
        """
        if grid_sizes is None:
            grid_sizes = [10, 20, 30, 40, 50]

        result = BenchmarkResult(
            name="grid_size_scaling",
            description=f"Performance scaling with grid size on {test_function}"
        )

        print(f"\nBenchmarking grid sizes: {grid_sizes}")
        print(f"Test function: {test_function}")
        print(f"Cores: {num_cores}\n")

        log_likelihood, param_bounds, _ = get_test_function(test_function)

        timings = []
        eval_counts = []

        for grid_size in grid_sizes:
            print(f"Testing grid size: {grid_size}x{grid_size}...")

            # 2D projection on first two dimensions
            projection = {
                'dims': [0, 1],
                'grid_points': [grid_size, grid_size],
                'lbfgsb': False,  # Disable for consistent comparison
                'patching_coarse': False,
                'enable_refinement': False
            }

            sampler = ProfileProjector(
                target_func=log_likelihood,
                bounds=param_bounds,
                projections=[projection],
                pop_per_grid_point=2,
                n_initial_optimizations=20,
                roi_threshold=4.0,
            )

            start_time = time.time()

            # Run with MPI (broadcast function only on first iteration)
            broadcast = (grid_size == grid_sizes[0])
            self._run_mpi_benchmark(sampler, projection, broadcast_func=broadcast)

            elapsed = time.time() - start_time
            eval_count = sampler.target_calls

            timings.append(elapsed)
            eval_counts.append(eval_count)

            print(f"  Completed in {elapsed:.2f}s with {eval_count} evaluations")

        # Store results
        result.add_metric('grid_sizes', grid_sizes)
        result.add_metric('timings_seconds', timings)
        result.add_metric('evaluation_counts', eval_counts)
        result.add_metric('evaluations_per_second', [e/t for e, t in zip(eval_counts, timings)])

        self.results.append(result)
        result.print_summary()

        return result

    def benchmark_mutation_strategies(self, test_function: str = "himmelblau_4d",
                                     grid_size: int = 20,
                                     num_cores: int = 1) -> BenchmarkResult:
        """
        Compare different DE mutation strategies.

        Parameters
        ----------
        test_function : str
            Name of test function to use
        grid_size : int
            Grid size for benchmark
        num_cores : int
            Number of MPI cores to use

        Returns
        -------
        BenchmarkResult
            Results comparing mutation strategies
        """
        strategies = ['current-to-rand/1', 'rand/1', 'current-to-pbest/1']

        result = BenchmarkResult(
            name="mutation_strategy_comparison",
            description=f"Comparing DE mutation strategies on {test_function}"
        )

        print(f"\nBenchmarking mutation strategies: {strategies}")
        print(f"Test function: {test_function}")
        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Cores: {num_cores}\n")

        log_likelihood, param_bounds, _ = get_test_function(test_function)

        timings = []
        eval_counts = []
        best_likelihoods = []

        for strategy in strategies:
            print(f"Testing strategy: {strategy}...")

            projection = {
                'dims': [0, 1],
                'grid_points': [grid_size, grid_size],
                'lbfgsb': True,
                'patching_coarse': True,
                'enable_refinement': False
            }

            sampler = ProfileProjector(
                target_func=log_likelihood,
                bounds=param_bounds,
                projections=[projection],
                pop_per_grid_point=3,
                mutation_strategy=strategy,
                n_initial_optimizations=50,
                roi_threshold=4.0,
            )

            start_time = time.time()
            # Broadcast function only on first strategy
            broadcast = (strategy == strategies[0])
            self._run_mpi_benchmark(sampler, projection, broadcast_func=broadcast)
            elapsed = time.time() - start_time
            eval_count = sampler.target_calls
            best_likelihood = sampler.global_max_target_val

            timings.append(elapsed)
            eval_counts.append(eval_count)
            best_likelihoods.append(best_likelihood)

            print(f"  Completed in {elapsed:.2f}s with {eval_count} evaluations")
            print(f"  Best likelihood: {best_likelihood:.6f}")

        # Store results
        result.add_metric('strategies', strategies)
        result.add_metric('timings_seconds', timings)
        result.add_metric('evaluation_counts', eval_counts)
        result.add_metric('best_likelihoods', best_likelihoods)

        # Find best strategy
        best_idx = np.argmax(best_likelihoods)
        result.add_metric('best_strategy', strategies[best_idx])
        result.add_metric('best_likelihood', best_likelihoods[best_idx])

        self.results.append(result)
        result.print_summary()

        return result

    def benchmark_convergence_efficiency(self, test_function: str = "himmelblau_4d",
                                        grid_size: int = 20) -> BenchmarkResult:
        """
        Measure convergence efficiency (evaluations to reach target accuracy).

        Parameters
        ----------
        test_function : str
            Name of test function to use
        grid_size : int
            Grid size for benchmark

        Returns
        -------
        BenchmarkResult
            Results showing convergence characteristics
        """
        result = BenchmarkResult(
            name="convergence_efficiency",
            description=f"Convergence efficiency on {test_function}"
        )

        print(f"\nBenchmarking convergence efficiency")
        print(f"Test function: {test_function}")
        print(f"Grid size: {grid_size}x{grid_size}\n")

        log_likelihood, param_bounds, true_peaks = get_test_function(test_function)

        projection = {
            'dims': [0, 1],
            'grid_points': [grid_size, grid_size],
            'lbfgsb': True,
            'patching_coarse': True,
            'enable_refinement': False
        }

        sampler = ProfileProjector(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=[projection],
            pop_per_grid_point=3,
            mutation_strategy='current-to-pbest/1',
            n_initial_optimizations=100,
            roi_threshold=4.0,
        )

        start_time = time.time()
        self._run_mpi_benchmark(sampler, projection, num_generations=10000)
        elapsed = time.time() - start_time

        # Calculate metrics
        eval_count = sampler.target_calls
        best_likelihood = sampler.global_max_target_val

        # Evaluate function at all known peaks to find true maximum
        if len(true_peaks) > 0:
            peak_values = [log_likelihood(peak) for peak in true_peaks]
            true_max = max(peak_values)
        else:
            true_max = 0.0  # Many test functions have optimum at 0

        accuracy = best_likelihood - true_max  # Should be close to 0 or negative

        result.add_metric('total_time_seconds', elapsed)
        result.add_metric('total_evaluations', eval_count)
        result.add_metric('best_likelihood_found', best_likelihood)
        result.add_metric('true_maximum', true_max)
        result.add_metric('accuracy_delta', accuracy)
        result.add_metric('evaluations_per_grid_point', eval_count / (grid_size ** 2))

        self.results.append(result)
        result.print_summary()

        return result

    def benchmark_refinement_efficiency(self, test_function: str = "himmelblau_4d",
                                       coarse_size: int = 20,
                                       refinement_factor: int = 2) -> BenchmarkResult:
        """
        Measure efficiency of grid refinement.

        Parameters
        ----------
        test_function : str
            Name of test function to use
        coarse_size : int
            Coarse grid size
        refinement_factor : int
            Refinement factor

        Returns
        -------
        BenchmarkResult
            Results comparing coarse vs refined performance
        """
        result = BenchmarkResult(
            name="refinement_efficiency",
            description=f"Grid refinement efficiency on {test_function}"
        )

        print(f"\nBenchmarking refinement efficiency")
        print(f"Test function: {test_function}")
        print(f"Coarse grid: {coarse_size}x{coarse_size}")
        print(f"Refinement factor: {refinement_factor}x\n")

        log_likelihood, param_bounds, _ = get_test_function(test_function)

        # Test with refinement
        projection_refined = {
            'dims': [0, 1],
            'grid_points': [coarse_size, coarse_size],
            'lbfgsb': True,
            'patching_coarse': True,
            'enable_refinement': True,
            'refinement_factor': refinement_factor,
            'patching_refined': False
        }

        sampler_refined = ProfileProjector(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=[projection_refined],
            pop_per_grid_point=3,
            n_initial_optimizations=50,
        )

        print("Running with refinement...")
        start_refined = time.time()
        self._run_mpi_benchmark(sampler_refined, projection_refined, num_generations=5000, broadcast_func=True)
        time_refined = time.time() - start_refined
        evals_refined = sampler_refined.target_calls

        # Test without refinement (direct fine grid)
        fine_size = coarse_size * refinement_factor
        projection_direct = {
            'dims': [0, 1],
            'grid_points': [fine_size, fine_size],
            'lbfgsb': True,
            'patching_coarse': True,
            'enable_refinement': False
        }

        sampler_direct = ProfileProjector(
            target_func=log_likelihood,
            bounds=param_bounds,
            projections=[projection_direct],
            pop_per_grid_point=3,
            n_initial_optimizations=50,
        )

        print("Running direct on fine grid...")
        start_direct = time.time()
        # Don't broadcast again - workers already have the function
        self._run_mpi_benchmark(sampler_direct, projection_direct, num_generations=5000, broadcast_func=False)
        time_direct = time.time() - start_direct
        evals_direct = sampler_direct.target_calls

        # Calculate speedup
        time_speedup = time_direct / time_refined
        eval_speedup = evals_direct / evals_refined

        result.add_metric('coarse_grid_size', coarse_size)
        result.add_metric('fine_grid_size', fine_size)
        result.add_metric('refinement_factor', refinement_factor)
        result.add_metric('time_with_refinement_seconds', time_refined)
        result.add_metric('time_direct_fine_seconds', time_direct)
        result.add_metric('evals_with_refinement', evals_refined)
        result.add_metric('evals_direct_fine', evals_direct)
        result.add_metric('time_speedup', time_speedup)
        result.add_metric('evaluation_speedup', eval_speedup)

        self.results.append(result)
        result.print_summary()

        return result

    def _run_mpi_benchmark(self, sampler, projection_config, num_generations=5000, broadcast_func=True):
        """
        Helper to run a benchmark using MPI (must be called with mpiexec).

        This requires running the script with mpiexec -n <cores>.
        Workers remain active after this call - terminate them separately.
        """
        if not MPI_AVAILABLE:
            print("Warning: MPI not available, skipping benchmark")
            return

        from paraprof.master import run_projection

        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()

        # Only master runs the benchmark, workers handled externally
        if myrank == 0:
            # Broadcast target function if requested
            if broadcast_func:
                comm.bcast(sampler.target_func, root=0)

            run_projection(
                comm=comm,
                sampler=sampler,
                projection_config=projection_config,
                num_generations=num_generations,
                max_num_to_evolve=None,
                save_plots=False,
                plot_settings=None,
                skip_init_opt_on_warm_start=False,
                myrank=myrank
            )

    def benchmark_test_functions(self, grid_size: int = 20) -> BenchmarkResult:
        """
        Compare performance across different test functions.

        Parameters
        ----------
        grid_size : int
            Grid size to use for all tests

        Returns
        -------
        BenchmarkResult
            Results comparing different test functions
        """
        test_functions = ["himmelblau_4d", "rosenbrock_4d", "rastrigin_4d"]

        result = BenchmarkResult(
            name="test_function_comparison",
            description="Performance on different test functions"
        )

        print(f"\nBenchmarking test functions: {test_functions}")
        print(f"Grid size: {grid_size}x{grid_size}\n")

        timings = []
        eval_counts = []

        for func_name in test_functions:
            print(f"Testing function: {func_name}...")

            log_likelihood, param_bounds, _ = get_test_function(func_name)

            projection = {
                'dims': [0, 1],
                'grid_points': [grid_size, grid_size],
                'lbfgsb': True,
                'patching_coarse': True,
                'enable_refinement': False
            }

            sampler = ProfileProjector(
                target_func=log_likelihood,
                bounds=param_bounds,
                projections=[projection],
                pop_per_grid_point=3,
                n_initial_optimizations=50,
            )

            start_time = time.time()
            # Broadcast function only on first test function
            broadcast = (func_name == test_functions[0])
            self._run_mpi_benchmark(sampler, projection, num_generations=5000, broadcast_func=broadcast)
            elapsed = time.time() - start_time
            eval_count = sampler.target_calls

            timings.append(elapsed)
            eval_counts.append(eval_count)

            print(f"  Completed in {elapsed:.2f}s with {eval_count} evaluations")

        result.add_metric('test_functions', test_functions)
        result.add_metric('timings_seconds', timings)
        result.add_metric('evaluation_counts', eval_counts)

        self.results.append(result)
        result.print_summary()

        return result


def run_all_benchmarks(output_dir: str = "benchmark_results"):
    """Run all benchmarks and save results."""
    suite = BenchmarkSuite(output_dir=output_dir)

    print("="*80)
    print("ParaProf Performance Benchmark Suite")
    print("="*80)

    # Run benchmarks
    suite.benchmark_grid_sizes(grid_sizes=[10, 15, 20, 25])
    suite.benchmark_mutation_strategies(grid_size=15)
    suite.benchmark_convergence_efficiency(grid_size=20)
    suite.benchmark_refinement_efficiency(coarse_size=15, refinement_factor=2)
    suite.benchmark_test_functions(grid_size=15)

    # Save results
    suite.save_results()

    print("\n" + "="*80)
    print("All benchmarks complete!")
    print("="*80)


def main():
    """Main entry point for benchmark suite."""
    # Check if running with MPI
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        myrank = comm.Get_rank()
    else:
        print("Error: MPI is required to run benchmarks.")
        print("Please run with: mpiexec -n <cores> python benchmark_suite.py ...")
        sys.exit(1)

    # Only master process handles arguments and orchestrates benchmarks
    if myrank == 0:
        from paraprof.master import terminate_workers

        parser = argparse.ArgumentParser(
            description="ParaProf Performance Benchmark Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run all benchmarks (requires at least 2 MPI processes)
  mpiexec -n 4 python benchmark_suite.py --all

  # Run specific benchmark
  mpiexec -n 4 python benchmark_suite.py --benchmark grid_sizes

  # Run with custom output directory
  mpiexec -n 4 python benchmark_suite.py --all --output results_2025

Note: Benchmarks must be run with mpiexec and at least 2 processes.
        """
        )

        parser.add_argument(
            '--all',
            action='store_true',
            help='Run all benchmarks'
        )

        parser.add_argument(
            '--benchmark',
            choices=['grid_sizes', 'mutation_strategies', 'convergence',
                    'refinement', 'test_functions'],
            help='Run a specific benchmark'
        )

        parser.add_argument(
            '--output',
            default='benchmark_results',
            help='Output directory for results (default: benchmark_results)'
        )

        parser.add_argument(
            '--grid-size',
            type=int,
            default=15,
            help='Grid size for benchmarks (default: 15)'
        )

        args = parser.parse_args()

        if not args.all and not args.benchmark:
            parser.print_help()
            sys.exit(1)

        suite = BenchmarkSuite(output_dir=args.output)

        num_cores = comm.Get_size()
        print(f"\nRunning benchmarks with {num_cores} MPI processes ({num_cores-1} workers)\n")

        try:
            if args.all:
                run_all_benchmarks(output_dir=args.output)
            else:
                if args.benchmark == 'grid_sizes':
                    suite.benchmark_grid_sizes(grid_sizes=[10, 20, 40, 80], num_cores=num_cores)
                elif args.benchmark == 'mutation_strategies':
                    suite.benchmark_mutation_strategies(grid_size=args.grid_size, num_cores=num_cores)
                elif args.benchmark == 'convergence':
                    suite.benchmark_convergence_efficiency(grid_size=args.grid_size)
                elif args.benchmark == 'refinement':
                    suite.benchmark_refinement_efficiency(coarse_size=args.grid_size)
                elif args.benchmark == 'test_functions':
                    suite.benchmark_test_functions(grid_size=args.grid_size)

                suite.save_results()
        finally:
            # Always terminate workers when done
            print("\nTerminating workers...")
            terminate_workers(comm, myrank)
    else:
        # Worker processes run worker_main and wait for tasks
        from paraprof.worker import worker_main
        worker_main(comm, myrank)


if __name__ == '__main__':
    main()
