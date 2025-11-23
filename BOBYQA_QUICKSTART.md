# BOBYQA Integration - Quick Start

## TL;DR

**Goal:** Add BOBYQA as an optimization method to reduce evaluations by 20-40%

**Effort:** ~3-4 weeks

**Files to read:**
1. `BOBYQA_PHASE1_SUMMARY.md` - Read this first (decision summary)
2. `BOBYQA_JOB_SKELETON.py` - Working code template
3. `BOBYQA_INTEGRATION_GUIDE.md` - Step-by-step instructions
4. `BOBYQA_INTEGRATION_DESIGN.md` - Detailed design (optional)

## 30-Second Overview

BOBYQA (Bound Optimization BY Quadratic Approximation) is a derivative-free optimizer that:
- Builds quadratic models of your likelihood function
- Optimizes within trust regions (safe zones)
- Requires fewer evaluations than gradient-based methods
- Parallelizes similarly to L-BFGS-B (model points vs gradient points)

## How It Fits ParaProf

```
Current: Grid Point → L-BFGS-B → Optimized Parameters
         (20-50 evaluations per grid point)

New:     Grid Point → BOBYQA → Optimized Parameters
         (15-35 evaluations per grid point)

User chooses: optimization_method='bobyqa' or 'lbfgsb'
```

## Implementation Checklist

```
□ Week 1: Core BOBYQAJob class
  □ Copy skeleton to src/paraprof/jobs/bobyqa_job.py
  □ Implement trust region solver
  □ Implement model building
  □ Write basic unit tests

□ Week 2: Integration
  □ Add factory methods to sampler.py
  □ Update master.py workflow
  □ Update jobs/__init__.py
  □ Write integration tests

□ Week 3: Testing
  □ Run full test suite
  □ Test MPI parallelization
  □ Debug edge cases
  □ Benchmark vs L-BFGS-B

□ Week 4: Polish
  □ Documentation
  □ Example scripts
  □ Performance tuning
  □ Code review
```

## Key Code Locations

```
src/paraprof/jobs/bobyqa_job.py          [NEW: 500 lines]
  └─ BOBYQAJob class (main implementation)

src/paraprof/sampler.py                  [EDIT: +80 lines]
  ├─ __init__: Add bobyqa_* parameters
  ├─ create_post_activation_bobyqa_jobs()
  └─ create_bobyqa_loop_jobs()

src/paraprof/master.py                   [EDIT: +60 lines]
  ├─ POST_ACTIVATION_BOBYQA stage
  ├─ PROCESSING_POST_ACTIVATION_BOBYQA stage
  ├─ BOBYQA_LOOP stage
  └─ PROCESSING_BOBYQA_LOOP stage

src/paraprof/jobs/__init__.py            [EDIT: +2 lines]
  └─ Import BOBYQAJob

tests/test_bobyqa_job.py                 [NEW: 150 lines]
tests/test_bobyqa_integration.py         [NEW: 50 lines]
benchmarks/benchmark_bobyqa_vs_lbfgsb.py [NEW: 80 lines]
examples/run_himmelblau_4d_bobyqa.py     [NEW: 100 lines]
```

## Usage Example

```python
# Before (L-BFGS-B)
projection = {
    'dims': [0, 1],
    'grid_points': [50, 50],
    'optimization_method': 'lbfgsb'
}

# After (BOBYQA)
projection = {
    'dims': [0, 1],
    'grid_points': [50, 50],
    'optimization_method': 'bobyqa',  # That's it!
}

# Advanced tuning
sampler = GridAnchoredDESampler(
    # ... other params ...
    bobyqa_initial_trust_radius=0.1,
    bobyqa_max_iterations=50,
    bobyqa_min_trust_radius=1e-6
)
```

## Testing Strategy

```bash
# Unit tests
pytest tests/test_bobyqa_job.py -v

# Integration tests
mpiexec -n 4 python -m pytest tests/test_bobyqa_integration.py

# Benchmark
mpiexec -n 8 python benchmarks/benchmark_bobyqa_vs_lbfgsb.py

# Example
mpiexec -n 4 python examples/run_himmelblau_4d_bobyqa.py
```

## Expected Benchmark Results

On Himmelblau 4D (2D projection, 50×50 grid):

```
Method      Evaluations    Time    Quality
L-BFGS-B    ~45,000       100%    -0.0000 (baseline)
BOBYQA      ~30,000       ~70%    -0.0000 (same)

Reduction: 33% fewer evaluations
```

## Troubleshooting

**Problem:** Trust radius shrinks to zero immediately
**Fix:** Increase initial_trust_radius or improve model building

**Problem:** More evaluations than L-BFGS-B
**Fix:** Check n_opt_dims - BOBYQA has overhead for small problems (n<5)

**Problem:** Wrong optimum found
**Fix:** Increase n_initial_optimizations or use hybrid approach

## Key Algorithms

### Trust Region Subproblem (Cauchy Point)
```python
# Minimize: g^T*s + 0.5*s^T*H*s
# Subject to: ||s|| <= trust_radius, l <= x+s <= u

direction = -gradient / ||gradient||  # Steepest descent
step = min(trust_radius, bounds_limit)
s = step * direction
```

### Accept/Reject Step
```python
actual_reduction = f_old - f_new
predicted_reduction = model_prediction(s)
ratio = actual_reduction / predicted_reduction

if ratio > 0.75:
    trust_radius *= 2  # Expand
elif ratio < 0.1:
    trust_radius *= 0.5  # Shrink
```

### Model Building
```python
# Collect 2n+1 points: center + coordinate perturbations
# Fit: f(x) ≈ c + g^T*(x-x0) + 0.5*(x-x0)^T*H*(x-x0)

gradient = least_squares_fit(differences, values)
hessian = finite_difference_pairs(coordinate_points)
```

## Success Criteria

Phase 1 succeeds if:
- ✅ 20%+ fewer evaluations on Himmelblau 4D
- ✅ Finds correct global optimum
- ✅ No MPI deadlocks
- ✅ Code <600 lines, well-tested

## Next Steps After Success

1. **Write paper:** "Sample-Efficient Profile Likelihood with BOBYQA"
2. **Phase 2A:** Hessian transfer between neighbors (10-30% more savings)
3. **Phase 2B:** Parallel trust region candidates
4. **Real application:** Apply to your actual physics problem!

## Getting Help

If stuck:
1. Check `BOBYQA_JOB_SKELETON.py` for implementation details
2. Compare with `LBFGSBJob` (same patterns)
3. Read Powell (2009) BOBYQA paper for algorithm details
4. Test on simple quadratic first (easy to debug)

## Resources

- **BOBYQA paper:** Powell (2009) "The BOBYQA algorithm for bound constrained optimization without derivatives"
- **PDFO library:** https://www.pdfo.net/ (reference implementation)
- **ParaProf patterns:** `src/paraprof/jobs/lbfgsb_job.py` (follow this closely)
- **Trust regions:** Conn, Gould, Toint (2000) book

## One-Line Summary

**BOBYQA = Quadratic model + Trust region + Parallel evaluation → Fewer function calls**

---

Ready to start? Begin with Week 1 of the checklist above! 🚀
