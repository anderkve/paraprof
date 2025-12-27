# ParaProf Scalability Testing - Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
# Test with 128 processes on your laptop
mpiexec -n 128 --oversubscribe python examples/run_test_with_simulation.py

# See communication statistics and identify bottlenecks immediately!
```

## 📊 Three Ways to Test Scalability

### 1. Quick Check (5 minutes)
```bash
./benchmarks/quick_scaling_check.sh 128
```
**Output:** Side-by-side comparison of different process counts and networks

### 2. Detailed Analysis (30 minutes)
```bash
python benchmarks/scaling_analysis.py --mode all --max-procs 256
```
**Output:** Scaling plots, efficiency curves, bottleneck analysis

### 3. Manual Testing
```bash
# Test specific configurations
PARAPROF_NETWORK=infiniband mpiexec -n 256 --oversubscribe python examples/run_with_simulation.py
PARAPROF_NETWORK=10gbe mpiexec -n 128 --oversubscribe python examples/run_with_simulation.py
PARAPROF_NETWORK=cloud mpiexec -n 64 --oversubscribe python examples/run_with_simulation.py
```

## 🔧 How to Add to Your Own Scripts

**Before:**
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
```

**After:**
```python
from paraprof.mpi_simulator import get_simulated_communicator
comm = get_simulated_communicator('infiniband')  # That's it!
```

## 📈 Interpreting Results

### Parallel Efficiency
- **>80%**: Excellent scaling ✅
- **50-80%**: Good scaling, minor bottlenecks ⚠️
- **<50%**: Poor scaling, investigate 🔴

### Communication Overhead
- **<10%**: Excellent ✅
- **10-25%**: Acceptable ⚠️
- **>25%**: Bottleneck, consider optimization 🔴

## 🎯 Common Issues and Fixes

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| Efficiency drops at 32+ processes | Master overwhelmed | Use batch sends (`isend` + `Waitall`) |
| Overhead >25% | Too many small messages | Batch multiple tasks per message |
| Workers idle | Load imbalance | Smaller task granularity |
| Similar performance across all networks | Good! | Computation dominates (ideal) |

## 📖 Full Documentation

See `docs/SCALABILITY_TESTING.md` for comprehensive guide.

## 🔬 Network Types Available

| Type | Latency | Bandwidth | Use For |
|------|---------|-----------|---------|
| `infiniband` | 1 μs | 100 Gbps | Modern HPC clusters |
| `10gbe` | 10 μs | 10 Gbps | Older clusters |
| `cloud` | 100 μs | 25 Gbps | AWS/Azure/GCP |
| `none` | 0 μs | ∞ | Baseline comparison |

## ⚡ Pro Tips

1. **Start small:** Test 4 → 8 → 16 → 32 processes first
2. **Compare networks:** Run with `none` and `infiniband` to isolate communication
3. **Watch overhead:** If >10%, you have room for optimization
4. **Check logs:** Look for idle workers or master bottlenecks
5. **Iterate:** Optimize → measure → repeat

## 🎓 Example Output

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

Procs    Time(s)    Speedup    Efficiency   Overhead
----------------------------------------------------------------
4        120.00     1.00       100.00%      2.5%
8        62.00      1.94       96.77%       3.1%
16       33.00      3.64       90.91%       4.8%
32       19.00      6.32       79.00%       7.2%
64       12.00      10.00      62.50%       12.5%
128      9.00       13.33      52.08%       18.3%
```

**Analysis:** Strong scaling efficiency degrades from 97% → 52% as we scale to 128 processes. Communication overhead increases from 3% → 18%, indicating the bottleneck is network communication. Consider batching messages or reducing communication frequency.
