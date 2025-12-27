# Scalability Testing Guide for ParaProf

This guide explains how to test ParaProf's performance with hundreds or thousands of MPI processes on your laptop.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Approaches Overview](#approaches-overview)
3. [MPI Simulator](#mpi-simulator)
4. [Scaling Analysis](#scaling-analysis)
5. [Profiling Tools](#profiling-tools)
6. [Interpreting Results](#interpreting-results)
7. [Common Bottlenecks](#common-bottlenecks)

---

## Quick Start

**Test with 128 processes and simulated InfiniBand network:**
```bash
mpiexec -n 128 --oversubscribe python examples/run_with_simulation.py
```

**Run automated scaling analysis:**
```bash
python benchmarks/scaling_analysis.py --mode all
```

**Compare different network types:**
```bash
# InfiniBand (1μs latency, 100 Gbps) - typical HPC
PARAPROF_NETWORK=infiniband mpiexec -n 256 --oversubscribe python examples/run_with_simulation.py

# 10 Gigabit Ethernet (10μs latency, 10 Gbps)
PARAPROF_NETWORK=10gbe mpiexec -n 256 --oversubscribe python examples/run_with_simulation.py

# Cloud network (100μs latency, 25 Gbps)
PARAPROF_NETWORK=cloud mpiexec -n 128 --oversubscribe python examples/run_with_simulation.py
```

---

## Approaches Overview

### 1. **Oversubscription (Simplest)**
Run more MPI ranks than physical cores:
```bash
mpiexec -n 256 --oversubscribe python run_test.py
```

**Pros:**
- No code changes required
- Immediately reveals communication patterns and synchronization issues
- Tests master-worker load balancing

**Cons:**
- Performance too optimistic (no network latency)
- Won't catch network-specific bottlenecks
- Context switching overhead differs from real HPC

**When to use:** Quick check for algorithmic scalability issues

---

### 2. **MPI Simulator (Recommended for Development)**
Add realistic network delays to emulate HPC performance:

```python
from paraprof.mpi_simulator import get_simulated_communicator

# Instead of: comm = MPI.COMM_WORLD
comm = get_simulated_communicator('infiniband')
```

**Pros:**
- Realistic network latency and bandwidth limits
- Test different network types (InfiniBand, 10GbE, cloud)
- Identifies communication bottlenecks
- Measures communication overhead percentage

**Cons:**
- Requires minor code changes (or use provided examples)
- Sleep-based delays (not exact hardware simulation)

**When to use:** Primary development and testing tool

---

### 3. **Automated Scaling Analysis (Comprehensive)**
Systematically test strong/weak scaling:

```bash
# Full analysis: strong + weak scaling + overhead tests
python benchmarks/scaling_analysis.py --mode all --max-procs 512

# Quick test (fewer data points)
python benchmarks/scaling_analysis.py --mode quick

# Only strong scaling
python benchmarks/scaling_analysis.py --mode strong --networks infiniband,10gbe
```

**Outputs:**
- Speedup and efficiency curves
- Communication overhead graphs
- JSON results for further analysis
- PNG plots in `scaling_plots/`

**When to use:** Before major releases, after optimization work

---

### 4. **MPI Profiling Tools (Production Testing)**

#### mpiP (Lightweight Profiling)
```bash
# Install: sudo apt-get install libmpi-dev
export LD_PRELOAD=/usr/lib/libmpiP.so
mpiexec -n 128 --oversubscribe python run_test.py
# Generates mpiP report with time spent in MPI calls
```

#### Score-P + Scalasca (Advanced)
```bash
# Trace MPI calls
scorep mpiexec -n 256 --oversubscribe python run_test.py
# Analyze traces
scalasca -analyze scorep_*
```

**When to use:** Final validation before deployment to real HPC

---

## MPI Simulator Details

### Network Profiles

| Profile | Latency | Bandwidth | Typical Use Case |
|---------|---------|-----------|------------------|
| `infiniband` | 1 μs | 100 Gbps | Modern HPC (InfiniBand HDR) |
| `10gbe` | 10 μs | 10 Gbps | Older clusters, departmental systems |
| `gigabit` | 50 μs | 1 Gbps | Very old systems, testing worst-case |
| `cloud` | 100 μs | 25 Gbps | AWS, Azure, GCP compute clusters |
| `none` | 0 μs | ∞ | Baseline (no simulation) |

### Usage in Your Code

**Simple replacement:**
```python
# Old code
from mpi4py import MPI
comm = MPI.COMM_WORLD

# New code
from paraprof.mpi_simulator import get_simulated_communicator
comm = get_simulated_communicator('infiniband')
```

**Environment variable control:**
```bash
# Enable simulation (default)
PARAPROF_SIMULATE_MPI=1 mpiexec -n 128 python your_script.py

# Disable simulation (normal MPI)
PARAPROF_SIMULATE_MPI=0 mpiexec -n 8 python your_script.py
```

**Custom network parameters:**
```python
from paraprof.mpi_simulator import MPISimulator

comm = MPISimulator(
    MPI.COMM_WORLD,
    latency_us=5.0,      # 5 microsecond latency
    bandwidth_gbps=50,   # 50 Gbps bandwidth
    jitter_factor=0.1    # ±10% random jitter
)
```

### Statistics Reporting

The simulator tracks communication statistics:
```python
comm.print_stats()
```

Output:
```
============================================================
MPI Simulation Statistics
============================================================
Network: 1.0 μs latency, 100 Gbps bandwidth
Total simulated delay: 12.456 seconds
Send operations: 45,231 (234.56 MB)
Recv operations: 45,231 (234.56 MB)
Average message size: 5440 bytes
============================================================
```

Use this to calculate **communication overhead**:
```
Overhead % = (Simulated Delay / Total Runtime) × 100
```

**Guideline:**
- < 10%: Communication well-optimized
- 10-25%: Acceptable for most applications
- 25-50%: Consider reducing communication
- > 50%: Serious bottleneck, redesign needed

---

## Scaling Analysis

### Strong Scaling
**Definition:** Fixed problem size, increase processes
**Ideal:** Runtime halves when doubling processes
**Measures:** How well your code parallelizes a single problem

```bash
python benchmarks/scaling_analysis.py --mode strong
```

**Interpreting Results:**
- **Efficiency > 0.8** (80%): Excellent scaling
- **Efficiency 0.5-0.8**: Good scaling, minor bottlenecks
- **Efficiency < 0.5**: Poor scaling, investigate bottlenecks

Example output:
```
Procs    Time(s)    Speedup    Efficiency   Throughput      Overhead
------------------------------------------------------------------------
4        120.00     1.00       100.00%      85.3            2.5%
8        62.00      1.94       96.77%       165.2           3.1%
16       33.00      3.64       90.91%       310.3           4.8%
32       19.00      6.32       79.00%       538.9           7.2%
64       12.00      10.00      62.50%       853.3           12.5%
128      9.00       13.33      52.08%       1137.8          18.3%
```

**Analysis:** Efficiency drops from 97% → 52% as we scale from 8 → 128 processes.
Overhead increases from 3% → 18%, indicating communication bottleneck.

### Weak Scaling
**Definition:** Increase both problem size and processes proportionally
**Ideal:** Runtime stays constant
**Measures:** Can your code handle larger problems efficiently?

```bash
python benchmarks/scaling_analysis.py --mode weak
```

**Interpreting Results:**
- **Efficiency > 0.9**: Excellent weak scaling
- **Efficiency 0.7-0.9**: Good weak scaling
- **Efficiency < 0.7**: Poor weak scaling, algorithmic issues

---

## Interpreting Results

### Key Metrics

1. **Speedup** = T₁ / Tₙ
   Where T₁ = runtime with baseline processes, Tₙ = runtime with n processes

2. **Parallel Efficiency** = Speedup / (n / baseline)
   Values close to 1.0 (100%) are ideal

3. **Communication Overhead** = (MPI time) / (Total time)
   Lower is better; < 10% is excellent

4. **Throughput** = Evaluations / Second
   Should increase linearly with processes (strong scaling)

### Diagnostic Questions

**Efficiency drops significantly at high process counts?**
- Check communication overhead → likely network bottleneck
- Consider batch communication (send multiple items together)
- Review master's task distribution logic

**Overhead increases superlinearly with processes?**
- Master may be overwhelmed (single-process bottleneck)
- Consider hierarchical master-worker (master delegates to sub-masters)

**Good strong scaling but poor weak scaling?**
- Algorithmic complexity issue (e.g., O(n²) operations)
- Memory bandwidth limitation
- Review grid refinement and search algorithms

**Similar performance between 'none' and 'infiniband' networks?**
- Good! Means computation dominates communication
- Code is well-balanced for HPC

---

## Common Bottlenecks and Solutions

### 1. Master Overwhelmed (Single-Process Bottleneck)
**Symptoms:**
- Master CPU usage at 100%
- Workers idle waiting for tasks
- Efficiency plateaus at ~10-20 processes

**Solutions:**
```python
# Use non-blocking sends in batches
requests = []
for worker_id, task in zip(free_workers, tasks):
    req = comm.isend(task, dest=worker_id)
    requests.append(req)
MPI.Request.Waitall(requests)  # Wait once at end

# Reduce result processing overhead
# Instead of processing each result immediately:
results_buffer = []
while comm.Iprobe(source=MPI.ANY_SOURCE):
    results_buffer.append(comm.recv(source=MPI.ANY_SOURCE))
# Process in batch
for result in results_buffer:
    process_result(result)
```

### 2. Excessive Communication
**Symptoms:**
- High overhead % (>25%)
- Large performance difference between 'none' and 'infiniband'
- Efficiency drops sharply with process count

**Solutions:**
```python
# Pack multiple parameters into single message
task = {
    'batch': [params1, params2, params3, ...],  # Send batch
    'job_ids': [id1, id2, id3, ...]
}

# Use broadcast for common data instead of individual sends
shared_config = comm.bcast(config, root=0)

# Reduce message frequency - let workers request tasks
# Instead of: master pushes every task
# Use: worker pulls when ready (request-based)
```

### 3. Load Imbalance
**Symptoms:**
- Some workers finish early, sit idle
- Total runtime dominated by slowest worker
- Good speedup initially, then plateaus

**Solutions:**
```python
# Use smaller task granularity
# Instead of: 1 grid point per task
# Use: Multiple points per task, but not too many

# Implement work stealing
# Fast workers can steal tasks from slow workers' queues

# Dynamic task sizing based on worker performance
worker_speed = track_worker_completion_times()
assign_more_tasks_to_faster_workers(worker_speed)
```

### 4. Synchronization Points
**Symptoms:**
- Periodic stalls where all processes wait
- Barrier-like patterns in trace visualization
- Sawtooth pattern in throughput

**Solutions:**
```python
# Avoid unnecessary barriers
# comm.barrier()  # Remove unless truly necessary

# Use asynchronous collectives
# Instead of: result = comm.bcast(data, root=0)  # Blocking
# Use: req = comm.Ibcast(data, root=0); ...; req.Wait()

# Overlap communication and computation
req = comm.isend(result, dest=0)
# Do other work here
req.wait()
```

---

## Example Workflow

**Step 1: Quick sanity check**
```bash
# Test with moderate oversubscription
mpiexec -n 32 --oversubscribe python run_test.py
```

**Step 2: Add network simulation**
```bash
# Test realistic HPC conditions
PARAPROF_NETWORK=infiniband mpiexec -n 128 --oversubscribe python examples/run_with_simulation.py
```

**Step 3: Comprehensive scaling analysis**
```bash
# Generate scaling plots and identify bottlenecks
python benchmarks/scaling_analysis.py --mode all --max-procs 256
# Review plots in scaling_plots/
```

**Step 4: Profile with mpiP**
```bash
# Detailed MPI profiling
export LD_PRELOAD=/usr/lib/libmpiP.so
mpiexec -n 256 --oversubscribe python run_test.py
# Review mpiP report for hotspots
```

**Step 5: Optimize based on findings**
- Address bottlenecks identified in step 3-4
- Re-run scaling analysis to verify improvements

**Step 6: Test on real HPC (if available)**
```bash
# Final validation on target system
sbatch --nodes=16 --ntasks-per-node=64 run_job.sh
```

---

## References

- **MPI Best Practices:** https://www.mpi-forum.org/docs/
- **Parallel Performance Analysis:** "Performance Analysis of Parallel Applications" (Jost et al.)
- **Amdahl's Law:** Theoretical speedup limits due to serial portions
- **Gustafson's Law:** Scaled speedup with problem size growth

---

## Getting Help

If you encounter unexpected scaling behavior:

1. Check `scaling_plots/` for visual patterns
2. Review `scaling_results.json` for detailed metrics
3. Run with `PARAPROF_NETWORK=none` to isolate computation vs. communication
4. Enable verbose output in your scripts for debugging
5. Compare against baseline benchmarks in `benchmarks/`

For further assistance, see the main ParaProf documentation or file an issue.
