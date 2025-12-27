#!/bin/bash
# Quick Scaling Check - Compare performance with different process counts
#
# Usage: ./quick_scaling_check.sh [max_processes]
#
# This script runs your code with increasing numbers of processes to quickly
# identify if there are obvious scaling issues.

set -e  # Exit on error

MAX_PROCS=${1:-128}  # Default to 128 if not specified
SCRIPT="examples/run_test_with_simulation.py"

echo "========================================================================"
echo "ParaProf Quick Scaling Check"
echo "========================================================================"
echo "Testing with processes: 4, 8, 16, 32, 64, $MAX_PROCS"
echo "Network simulation: InfiniBand (1μs latency, 100 Gbps)"
echo ""

# Create results directory
mkdir -p quick_scaling_results

# Test function
run_test() {
    local nprocs=$1
    local network=$2
    local label=$3

    echo "------------------------------------------------------------------------"
    echo "Running with $nprocs processes ($label)..."
    echo "------------------------------------------------------------------------"

    # Set environment
    export PARAPROF_NETWORK=$network

    # Run and capture output
    output_file="quick_scaling_results/${label}_${nprocs}procs.log"

    if timeout 180 mpiexec -n $nprocs --oversubscribe python $SCRIPT > $output_file 2>&1; then
        # Extract key metrics
        wall_time=$(grep "Wall time:" $output_file | awk '{print $3}')
        throughput=$(grep "Throughput:" $output_file | awk '{print $2}')
        overhead=$(grep "Total simulated delay:" $output_file | awk '{print $4}')

        if [ -n "$wall_time" ] && [ -n "$throughput" ]; then
            echo "  ✓ Success: ${wall_time}s, ${throughput} evals/sec"

            # Calculate overhead percentage if available
            if [ -n "$overhead" ]; then
                overhead_pct=$(echo "scale=1; ($overhead / $wall_time) * 100" | bc)
                echo "    Communication overhead: ${overhead_pct}%"
            fi
        else
            echo "  ⚠ Completed but couldn't extract metrics"
        fi
    else
        echo "  ✗ Failed or timed out (>180s)"
    fi
    echo ""
}

# Test 1: Quick baseline without simulation
echo "========================================================================"
echo "PHASE 1: Baseline (no network simulation)"
echo "========================================================================"
echo ""

for nprocs in 4 8 16 32; do
    run_test $nprocs "none" "baseline"
done

# Test 2: With InfiniBand simulation
echo "========================================================================"
echo "PHASE 2: With InfiniBand simulation (realistic HPC)"
echo "========================================================================"
echo ""

for nprocs in 4 8 16 32 64 $MAX_PROCS; do
    run_test $nprocs "infiniband" "infiniband"
done

# Test 3: With 10GbE simulation (higher latency)
echo "========================================================================"
echo "PHASE 3: With 10GbE simulation (higher latency network)"
echo "========================================================================"
echo ""

for nprocs in 16 64 $MAX_PROCS; do
    run_test $nprocs "10gbe" "10gbe"
done

# Generate summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "Results saved to: quick_scaling_results/"
echo ""
echo "Quick analysis:"
echo ""

# Extract and compare runtimes
echo "Process Count | Baseline | InfiniBand | 10GbE"
echo "--------------|----------|------------|-------"

for nprocs in 16 64 $MAX_PROCS; do
    baseline_time=$(grep "Wall time:" quick_scaling_results/baseline_${nprocs}procs.log 2>/dev/null | awk '{print $3}' || echo "N/A")
    ib_time=$(grep "Wall time:" quick_scaling_results/infiniband_${nprocs}procs.log 2>/dev/null | awk '{print $3}' || echo "N/A")
    gbe_time=$(grep "Wall time:" quick_scaling_results/10gbe_${nprocs}procs.log 2>/dev/null | awk '{print $3}' || echo "N/A")

    printf "%-13s | %-8s | %-10s | %-8s\n" "$nprocs" "$baseline_time" "$ib_time" "$gbe_time"
done

echo ""
echo "========================================================================"
echo "INTERPRETATION GUIDE"
echo "========================================================================"
echo ""
echo "1. Baseline vs InfiniBand:"
echo "   - If times are similar: Good! Computation dominates (ideal)."
echo "   - If InfiniBand >>10% slower: Communication bottleneck exists."
echo ""
echo "2. InfiniBand vs 10GbE:"
echo "   - Shows sensitivity to network latency."
echo "   - Large difference means many small messages (consider batching)."
echo ""
echo "3. Scaling with process count:"
echo "   - Runtime should decrease as processes increase (strong scaling)."
echo "   - If runtime plateaus or increases: severe bottleneck."
echo ""
echo "4. Communication overhead (from logs):"
echo "   - <10%: Excellent"
echo "   - 10-25%: Good"
echo "   - 25-50%: Could be improved"
echo "   - >50%: Serious issue, redesign needed"
echo ""
echo "For detailed analysis, run:"
echo "  python benchmarks/scaling_analysis.py --mode all"
echo ""
