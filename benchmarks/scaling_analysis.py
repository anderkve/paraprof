#!/usr/bin/env python
"""
Scalability Analysis Tool for ParaProf

Automatically runs benchmarks with different numbers of processes and network
simulations to identify bottlenecks and scaling efficiency.

This helps answer questions like:
- Does my code scale well to 1000+ processes?
- Where are communication bottlenecks?
- What's the overhead of MPI communication vs computation?
- Strong scaling: How does runtime change with more processes (fixed problem)?
- Weak scaling: Does efficiency stay constant with proportional work increase?

Usage:
    # Run full scaling analysis (strong + weak scaling)
    python scaling_analysis.py --mode all

    # Test strong scaling with different process counts
    python scaling_analysis.py --mode strong --max-procs 256

    # Test with different network simulations
    python scaling_analysis.py --mode strong --networks infiniband,10gbe,cloud

    # Quick test (fewer data points)
    python scaling_analysis.py --mode quick
"""
import argparse
import subprocess
import sys
import json
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ScalingAnalyzer:
    """Automated scaling analysis for MPI applications."""

    def __init__(self, script_path, max_procs=256, networks=None):
        self.script_path = script_path
        self.max_procs = max_procs
        self.networks = networks or ['infiniband', '10gbe', 'cloud']
        self.results = []

    def get_process_counts(self, mode='strong'):
        """Generate list of process counts to test."""
        physical_cores = os.cpu_count() or 8

        if mode == 'quick':
            # Quick test: just a few points
            return [4, physical_cores, physical_cores * 2]
        elif mode == 'strong':
            # Strong scaling: double processes each time
            counts = []
            n = 4
            while n <= self.max_procs:
                counts.append(n)
                n *= 2
            return counts
        elif mode == 'weak':
            # Weak scaling: increase both processes and problem size
            return self.get_process_counts('strong')
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_benchmark(self, num_procs, network='none', grid_size=20,
                     max_iterations=50):
        """
        Run a single benchmark with specified configuration.

        Returns dict with timing and performance metrics.
        """
        print(f"\n{'='*70}")
        print(f"Running: {num_procs} processes, {network} network, "
              f"grid_size={grid_size}")
        print(f"{'='*70}")

        # Set environment variables
        env = os.environ.copy()
        env['PARAPROF_NETWORK'] = network
        env['PARAPROF_GRID_SIZE'] = str(grid_size)
        env['PARAPROF_MAX_ITERATIONS'] = str(max_iterations)

        # Build command
        cmd = [
            'mpiexec',
            '-n', str(num_procs),
            '--oversubscribe',  # Allow oversubscription
            sys.executable,
            self.script_path
        ]

        try:
            # Run benchmark
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse output for metrics
            metrics = self._parse_output(result.stdout)
            metrics['num_procs'] = num_procs
            metrics['network'] = network
            metrics['grid_size'] = grid_size
            metrics['success'] = result.returncode == 0

            if not metrics['success']:
                print(f"ERROR: Benchmark failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")

            return metrics

        except subprocess.TimeoutExpired:
            print(f"ERROR: Benchmark timed out after 300 seconds")
            return {
                'num_procs': num_procs,
                'network': network,
                'grid_size': grid_size,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"ERROR: {e}")
            return {
                'num_procs': num_procs,
                'network': network,
                'grid_size': grid_size,
                'success': False,
                'error': str(e)
            }

    def _parse_output(self, output):
        """Extract metrics from benchmark output."""
        metrics = {
            'wall_time': None,
            'n_evals': None,
            'throughput': None,
            'simulated_delay': None,
            'best_likelihood': None
        }

        for line in output.split('\n'):
            if 'Wall time:' in line:
                try:
                    metrics['wall_time'] = float(line.split(':')[1].split()[0])
                except:
                    pass
            elif 'Function evaluations:' in line:
                try:
                    metrics['n_evals'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'Throughput:' in line:
                try:
                    metrics['throughput'] = float(line.split(':')[1].split()[0])
                except:
                    pass
            elif 'Total simulated delay:' in line:
                try:
                    metrics['simulated_delay'] = float(line.split(':')[1].split()[0])
                except:
                    pass
            elif 'Best likelihood:' in line:
                try:
                    metrics['best_likelihood'] = float(line.split(':')[1].strip())
                except:
                    pass

        return metrics

    def strong_scaling_test(self):
        """
        Strong scaling: Fixed problem size, varying number of processes.

        Ideal: Runtime halves when doubling processes (efficiency = 1.0)
        """
        print("\n" + "="*70)
        print("STRONG SCALING TEST")
        print("="*70)

        process_counts = self.get_process_counts('strong')
        grid_size = 30  # Fixed problem size

        for network in self.networks:
            print(f"\n--- Testing network: {network} ---")
            for num_procs in process_counts:
                result = self.run_benchmark(
                    num_procs=num_procs,
                    network=network,
                    grid_size=grid_size,
                    max_iterations=50
                )
                result['test_type'] = 'strong_scaling'
                self.results.append(result)

    def weak_scaling_test(self):
        """
        Weak scaling: Increase problem size proportionally with processes.

        Ideal: Runtime stays constant when both scale together (efficiency = 1.0)
        """
        print("\n" + "="*70)
        print("WEAK SCALING TEST")
        print("="*70)

        process_counts = self.get_process_counts('weak')
        base_grid_size = 15  # Grid size for 4 processes

        for network in self.networks:
            print(f"\n--- Testing network: {network} ---")
            for num_procs in process_counts:
                # Scale grid size with sqrt(num_procs) to keep work/process constant
                grid_size = int(base_grid_size * np.sqrt(num_procs / 4))

                result = self.run_benchmark(
                    num_procs=num_procs,
                    network=network,
                    grid_size=grid_size,
                    max_iterations=50
                )
                result['test_type'] = 'weak_scaling'
                result['work_per_process'] = grid_size**2 / num_procs
                self.results.append(result)

    def communication_overhead_test(self):
        """
        Test communication overhead by comparing with/without network delays.

        High overhead indicates communication bottleneck.
        """
        print("\n" + "="*70)
        print("COMMUNICATION OVERHEAD TEST")
        print("="*70)

        process_counts = [16, 64, 256]
        networks = ['none', 'infiniband', '10gbe']

        for num_procs in process_counts:
            for network in networks:
                result = self.run_benchmark(
                    num_procs=num_procs,
                    network=network,
                    grid_size=25,
                    max_iterations=30
                )
                result['test_type'] = 'communication_overhead'
                self.results.append(result)

    def analyze_results(self):
        """Compute scaling efficiency and identify bottlenecks."""
        print("\n" + "="*70)
        print("SCALING ANALYSIS RESULTS")
        print("="*70)

        # Group by test type and network
        by_test = {}
        for result in self.results:
            if not result.get('success', False):
                continue

            test_type = result.get('test_type', 'unknown')
            network = result.get('network', 'unknown')
            key = (test_type, network)

            if key not in by_test:
                by_test[key] = []
            by_test[key].append(result)

        # Analyze each group
        for (test_type, network), results_group in by_test.items():
            print(f"\n--- {test_type.upper()} ({network}) ---")

            # Sort by number of processes
            results_group.sort(key=lambda r: r['num_procs'])

            # Calculate efficiencies
            baseline = results_group[0]
            baseline_time = baseline.get('wall_time')
            baseline_procs = baseline['num_procs']

            print(f"{'Procs':<8} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<12} "
                  f"{'Throughput':<15} {'Overhead':<10}")
            print("-" * 75)

            for r in results_group:
                procs = r['num_procs']
                time = r.get('wall_time')
                delay = r.get('simulated_delay', 0) or 0

                if time and baseline_time:
                    speedup = baseline_time / time
                    ideal_speedup = procs / baseline_procs
                    efficiency = speedup / ideal_speedup
                    throughput = r.get('throughput', 0) or 0
                    overhead_pct = (delay / time * 100) if time > 0 else 0

                    print(f"{procs:<8} {time:<10.2f} {speedup:<10.2f} "
                          f"{efficiency:<12.2%} {throughput:<15.1f} {overhead_pct:<10.1f}%")
                else:
                    print(f"{procs:<8} {'FAILED':<10}")

    def plot_results(self, output_dir='scaling_plots'):
        """Generate visualization plots."""
        os.makedirs(output_dir, exist_ok=True)

        # Strong scaling plot
        self._plot_strong_scaling(output_dir)

        # Weak scaling plot
        self._plot_weak_scaling(output_dir)

        # Communication overhead plot
        self._plot_communication_overhead(output_dir)

        print(f"\nPlots saved to {output_dir}/")

    def _plot_strong_scaling(self, output_dir):
        """Plot strong scaling efficiency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for network in self.networks:
            data = [r for r in self.results
                   if r.get('test_type') == 'strong_scaling'
                   and r.get('network') == network
                   and r.get('success', False)]

            if not data:
                continue

            data.sort(key=lambda r: r['num_procs'])
            procs = [r['num_procs'] for r in data]
            times = [r.get('wall_time', 0) for r in data]

            # Speedup plot
            if times and times[0] > 0:
                speedups = [times[0] / t if t > 0 else 0 for t in times]
                ax1.plot(procs, speedups, 'o-', label=network)

            # Efficiency plot
            baseline_procs = procs[0]
            if times and times[0] > 0:
                efficiencies = [(times[0] / t) / (p / baseline_procs) if t > 0 else 0
                               for p, t in zip(procs, times)]
                ax2.plot(procs, efficiencies, 'o-', label=network)

        # Ideal scaling line
        if procs:
            baseline = procs[0]
            ideal_speedup = [p / baseline for p in procs]
            ax1.plot(procs, ideal_speedup, 'k--', label='Ideal', alpha=0.5)
            ax2.axhline(y=1.0, color='k', linestyle='--', label='Ideal', alpha=0.5)

        ax1.set_xlabel('Number of Processes')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Strong Scaling: Speedup')
        ax1.legend()
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=2)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Parallel Efficiency')
        ax2.set_title('Strong Scaling: Efficiency')
        ax2.legend()
        ax2.set_xscale('log', base=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/strong_scaling.png', dpi=150)
        plt.close()

    def _plot_weak_scaling(self, output_dir):
        """Plot weak scaling efficiency."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for network in self.networks:
            data = [r for r in self.results
                   if r.get('test_type') == 'weak_scaling'
                   and r.get('network') == network
                   and r.get('success', False)]

            if not data:
                continue

            data.sort(key=lambda r: r['num_procs'])
            procs = [r['num_procs'] for r in data]
            times = [r.get('wall_time', 0) for r in data]

            if times and times[0] > 0:
                efficiency = [times[0] / t if t > 0 else 0 for t in times]
                ax.plot(procs, efficiency, 'o-', label=network)

        ax.axhline(y=1.0, color='k', linestyle='--', label='Ideal', alpha=0.5)
        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Parallel Efficiency')
        ax.set_title('Weak Scaling Efficiency')
        ax.legend()
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/weak_scaling.png', dpi=150)
        plt.close()

    def _plot_communication_overhead(self, output_dir):
        """Plot communication overhead vs computation."""
        fig, ax = plt.subplots(figsize=(10, 6))

        networks = ['none', 'infiniband', '10gbe', 'cloud']
        for network in networks:
            data = [r for r in self.results
                   if r.get('test_type') == 'communication_overhead'
                   and r.get('network') == network
                   and r.get('success', False)]

            if not data:
                continue

            data.sort(key=lambda r: r['num_procs'])
            procs = [r['num_procs'] for r in data]
            overhead = []
            for r in data:
                delay = r.get('simulated_delay', 0) or 0
                time = r.get('wall_time', 1)
                overhead.append((delay / time * 100) if time > 0 else 0)

            ax.plot(procs, overhead, 'o-', label=network)

        ax.set_xlabel('Number of Processes')
        ax.set_ylabel('Communication Overhead (%)')
        ax.set_title('Communication Overhead vs Process Count')
        ax.legend()
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/communication_overhead.png', dpi=150)
        plt.close()

    def save_results(self, filename='scaling_results.json'):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Scalability analysis for ParaProf',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', choices=['quick', 'strong', 'weak', 'overhead', 'all'],
                       default='quick',
                       help='Type of scaling test to run')
    parser.add_argument('--max-procs', type=int, default=256,
                       help='Maximum number of processes to test')
    parser.add_argument('--networks', type=str, default='infiniband,10gbe',
                       help='Comma-separated list of networks to simulate')
    parser.add_argument('--script', type=str,
                       default='examples/run_with_simulation.py',
                       help='Path to benchmark script')
    parser.add_argument('--output-dir', type=str, default='scaling_plots',
                       help='Directory for output plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')

    args = parser.parse_args()

    # Parse networks
    networks = [n.strip() for n in args.networks.split(',')]

    # Create analyzer
    analyzer = ScalingAnalyzer(
        script_path=args.script,
        max_procs=args.max_procs,
        networks=networks
    )

    # Run tests
    if args.mode in ['strong', 'all']:
        analyzer.strong_scaling_test()

    if args.mode in ['weak', 'all']:
        analyzer.weak_scaling_test()

    if args.mode in ['overhead', 'all']:
        analyzer.communication_overhead_test()

    if args.mode == 'quick':
        # Quick test: just strong scaling with few points
        print("\n--- Quick scaling test ---")
        for num_procs in [4, 16, 64]:
            result = analyzer.run_benchmark(
                num_procs=num_procs,
                network='infiniband',
                grid_size=20,
                max_iterations=30
            )
            result['test_type'] = 'strong_scaling'
            analyzer.results.append(result)

    # Analyze and plot
    analyzer.analyze_results()

    if not args.no_plots:
        analyzer.plot_results(output_dir=args.output_dir)

    analyzer.save_results()


if __name__ == '__main__':
    main()
