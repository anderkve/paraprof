# BOBYQA Integration Documentation

This directory contains complete design and implementation documentation for integrating BOBYQA optimization into ParaProf.

## What is BOBYQA?

**BOBYQA** (Bound Optimization BY Quadratic Approximation) is a derivative-free optimization algorithm that builds quadratic models of the objective function and optimizes within trust regions. It typically requires fewer function evaluations than gradient-based methods like L-BFGS-B.

## Why Add BOBYQA to ParaProf?

**Goal:** Reduce likelihood evaluations by 20-40% while maintaining robustness

**Benefits:**
- Fewer evaluations per grid point (15-35 vs 20-50 for L-BFGS-B)
- No numerical gradient calculations needed
- Better for noisy or expensive likelihood functions
- Parallelizes similarly to L-BFGS-B

**Use cases:**
- Expensive likelihood evaluations (e.g., running simulations)
- Smooth parameter spaces
- When every evaluation counts

## Document Overview

### 📖 Start Here

1. **[BOBYQA_QUICKSTART.md](BOBYQA_QUICKSTART.md)** ⭐ **READ THIS FIRST**
   - 5-minute overview
   - Implementation checklist
   - Quick reference guide
   - Usage examples

2. **[BOBYQA_PHASE1_SUMMARY.md](BOBYQA_PHASE1_SUMMARY.md)** 📊 **Decision Document**
   - Executive summary and recommendation
   - Expected performance gains
   - Success criteria
   - Risk assessment
   - Timeline estimate

### 🔧 Implementation Guides

3. **[BOBYQA_JOB_SKELETON.py](BOBYQA_JOB_SKELETON.py)** 💻 **Code Template**
   - Complete BOBYQAJob class skeleton (~500 lines)
   - Fully commented implementation
   - Trust region solver
   - Model building logic
   - Ready to copy and flesh out

4. **[BOBYQA_INTEGRATION_GUIDE.md](BOBYQA_INTEGRATION_GUIDE.md)** 📝 **Step-by-Step**
   - 7-step implementation plan
   - Code snippets for each integration point
   - Test examples
   - Benchmark scripts
   - Configuration examples
   - Troubleshooting guide

### 📚 Deep Dive

5. **[BOBYQA_INTEGRATION_DESIGN.md](BOBYQA_INTEGRATION_DESIGN.md)** 🏗️ **Design Document**
   - Algorithm overview
   - Parallelization strategy
   - Architecture decisions
   - Performance analysis
   - Testing strategy
   - Phase 2 roadmap (model transfer)

## Reading Path by Role

### If you're deciding whether to implement this:
```
1. BOBYQA_QUICKSTART.md (5 min)
2. BOBYQA_PHASE1_SUMMARY.md (15 min)
   → Make go/no-go decision
```

### If you're implementing:
```
1. BOBYQA_QUICKSTART.md (review checklist)
2. BOBYQA_JOB_SKELETON.py (study code)
3. BOBYQA_INTEGRATION_GUIDE.md (follow steps 1-7)
4. BOBYQA_INTEGRATION_DESIGN.md (reference as needed)
```

### If you're reviewing the implementation:
```
1. BOBYQA_PHASE1_SUMMARY.md (understand goals)
2. BOBYQA_JOB_SKELETON.py (review code structure)
3. Check success criteria (in PHASE1_SUMMARY.md)
```

### If you're using BOBYQA in ParaProf:
```
1. BOBYQA_QUICKSTART.md (usage examples)
2. Examples in BOBYQA_INTEGRATION_GUIDE.md (configuration)
```

## Quick Implementation Timeline

```
Week 1: Core Implementation
  ├─ BOBYQAJob class
  ├─ Trust region solver
  ├─ Model building
  └─ Basic unit tests

Week 2: Integration
  ├─ Sampler factory methods
  ├─ Master workflow updates
  ├─ Configuration
  └─ Integration tests

Week 3: Testing & Debugging
  ├─ Full test suite
  ├─ MPI testing
  ├─ Edge cases
  └─ Benchmarks

Week 4: Polish
  ├─ Documentation
  ├─ Examples
  ├─ Performance tuning
  └─ Code review

Total: 3-4 weeks
```

## Files You'll Create/Modify

### New Files (1000 lines total)
```
src/paraprof/jobs/bobyqa_job.py              [500 lines]
examples/run_himmelblau_4d_bobyqa.py         [100 lines]
tests/test_bobyqa_job.py                     [150 lines]
tests/test_bobyqa_integration.py             [50 lines]
benchmarks/benchmark_bobyqa_vs_lbfgsb.py     [80 lines]
```

### Modified Files (100 lines total)
```
src/paraprof/sampler.py                      [+80 lines]
src/paraprof/master.py                       [+60 lines]
src/paraprof/jobs/__init__.py                [+2 lines]
```

## Key Concepts

### Trust Region
A "safe zone" around the current point where the quadratic model is trusted:
- Start with radius = 0.1
- Expand if model is good (×2)
- Shrink if model is bad (×0.5)
- Stop when radius < 1e-6

### Quadratic Model
Approximate likelihood as: `f(x) ≈ c + g^T*x + 0.5*x^T*H*x`
- `g`: gradient (n-vector)
- `H`: Hessian (n×n matrix)
- Fit from 2n+1 evaluation points

### Interpolation Set
Collection of points where likelihood has been evaluated:
- Used to build/update quadratic model
- Keep ~2n+1 points around current location
- Replace old points as optimization progresses

## Expected Performance

### Himmelblau 4D (smooth landscape)
```
Method      Grid Points    Evaluations    Reduction
L-BFGS-B    50×50         ~45,000        baseline
BOBYQA      50×50         ~30,000        33% fewer
```

### Rosenbrock 4D (ill-conditioned)
```
Method      Grid Points    Evaluations    Reduction
L-BFGS-B    50×50         ~55,000        baseline
BOBYQA      50×50         ~42,000        24% fewer
```

## Success Criteria Checklist

Phase 1 succeeds if:
- [ ] ≥20% fewer evaluations on Himmelblau 4D
- [ ] ≥15% fewer evaluations on Rosenbrock 4D
- [ ] Finds correct global optimum (within 1%)
- [ ] No MPI deadlocks or race conditions
- [ ] BOBYQAJob <600 lines of code
- [ ] >95% test coverage
- [ ] Passes code review

## Phase 2 Preview (Future)

If Phase 1 succeeds, consider:

**Phase 2A: Model Transfer** (2 weeks)
- Seed neighbors with Hessian from adjacent grid points
- Skip some interpolation point evaluations
- Expected: Additional 10-30% reduction

**Phase 2B: Enhanced Parallelization** (2 weeks)
- Generate multiple trust region candidates
- Evaluate in parallel, pick best
- Expected: Better parallel efficiency

**Phase 2C: Adaptive Selection** (1 week)
- Auto-choose BOBYQA vs L-BFGS-B based on smoothness
- Hybrid strategy for robustness

## Troubleshooting

Common issues and solutions:

| Problem | Solution |
|---------|----------|
| Trust radius → 0 | Increase initial_trust_radius |
| More evals than L-BFGS-B | Check n_dims (overhead for n<5) |
| Wrong optimum | Increase n_initial_optimizations |
| MPI deadlock | Check state machine transitions |
| Model building fails | Add regularization to Hessian |

## Testing Strategy

```bash
# Quick check
pytest tests/test_bobyqa_job.py::test_bobyqa_job_initialization

# Full unit tests
pytest tests/test_bobyqa_job.py -v

# Integration tests (requires MPI)
mpiexec -n 4 python -m pytest tests/test_bobyqa_integration.py

# Benchmark vs L-BFGS-B
mpiexec -n 8 python benchmarks/benchmark_bobyqa_vs_lbfgsb.py

# Run example
mpiexec -n 4 python examples/run_himmelblau_4d_bobyqa.py
```

## Usage Example

```python
from paraprof import GridAnchoredDESampler, run_all_projections

# Define projections with BOBYQA
projections = [
    {
        'dims': [0, 1],
        'grid_points': [50, 50],
        'optimization_method': 'bobyqa',  # Use BOBYQA
        'patching_coarse': True,
        'enable_refinement': True,
        'refinement_factor': 2
    }
]

# Create sampler with BOBYQA settings
sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=projections,
    # BOBYQA-specific parameters
    bobyqa_initial_trust_radius=0.1,
    bobyqa_max_iterations=50,
    bobyqa_min_trust_radius=1e-6,
    # Standard settings
    n_initial_optimizations=100,
    roi_threshold=4.0
)

# Run (same as before!)
results = run_all_projections(
    comm=comm,
    sampler=sampler,
    projections=projections,
    save_plots=True
)
```

## References

### Papers
- Powell, M. J. D. (2009). "The BOBYQA algorithm for bound constrained optimization without derivatives"
- Conn, A. R., Gould, N. I., & Toint, P. L. (2000). "Trust-Region Methods"

### Software
- **PDFO**: https://www.pdfo.net/ (Powell's Derivative-Free Optimization solvers)
- **Py-BOBYQA**: https://github.com/numericalalgorithmsgroup/pybobyqa

### ParaProf Code
- `src/paraprof/jobs/lbfgsb_job.py` - Follow these patterns
- `src/paraprof/jobs/cd_job.py` - Similar state machine structure

## Contributing

When implementing:
1. Follow existing code style (LBFGSBJob patterns)
2. Add comprehensive docstrings
3. Write tests as you go
4. Benchmark early and often
5. Document any deviations from the design

## Questions?

Check the troubleshooting sections in:
- BOBYQA_INTEGRATION_GUIDE.md (implementation issues)
- BOBYQA_QUICKSTART.md (usage questions)
- BOBYQA_INTEGRATION_DESIGN.md (algorithm questions)

---

## TL;DR for Busy People

**What:** Add BOBYQA optimization to ParaProf
**Why:** 20-40% fewer likelihood evaluations
**How:** Copy skeleton, follow guide, test thoroughly
**When:** 3-4 weeks implementation time
**Decision:** ✅ Recommended - good ROI, low risk

**Start here:** [BOBYQA_QUICKSTART.md](BOBYQA_QUICKSTART.md)

Good luck! 🚀
