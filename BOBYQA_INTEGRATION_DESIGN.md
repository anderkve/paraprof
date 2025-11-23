# BOBYQA Integration Design - Phase 1

## Overview

This document outlines the design for integrating BOBYQA (Bound Optimization BY Quadratic Approximation) into ParaProf as an alternative optimization method for grid-anchored continuous parameter optimization.

## Goals

- **Sample Efficiency**: Reduce number of likelihood evaluations per grid point by 20-40% compared to L-BFGS-B
- **Parallelizability**: Maintain full MPI parallelization through job-based task distribution
- **Robustness**: Provide reliable convergence across diverse problem landscapes
- **Drop-in Replacement**: Use same interface as LBFGSBJob for easy integration

## Architecture

### BOBYQAJob Class

Similar to `LBFGSBJob`, the `BOBYQAJob` will:
1. Break BOBYQA algorithm into asynchronous evaluation tasks
2. Maintain internal state (trust region, interpolation model)
3. Support warm-starting from neighbors (Phase 2 feature)
4. Handle both global and grid-anchored optimization

### Key State Machine States

```
NEEDS_INITIAL_MODEL     → Build initial interpolation set (2*n+1 points)
NEEDS_TRUST_STEP        → Evaluate candidate trust region step
NEEDS_MODEL_IMPROVEMENT → Evaluate points to improve model geometry
FINISHED                → Optimization complete
```

### Implementation Strategy

**Option A: Wrapper Approach** (Recommended for Phase 1)
- Use existing BOBYQA library (scipy.optimize.minimize with method='COBYLA' or pdfo)
- Intercept function calls via custom callback/wrapper
- Convert synchronous calls → asynchronous tasks

**Option B: Custom Implementation** (Phase 2 consideration)
- Full stateful BOBYQA reimplementation
- Better control over parallelization
- More complex, but enables advanced features (model transfer)

## Phase 1 Implementation Plan

### 1. Core BOBYQA Job (`src/paraprof/jobs/bobyqa_job.py`)

**Key Components:**

```python
class BOBYQAJob(Job):
    """
    BOBYQA optimization job using function interception for parallelization.

    This job wraps an existing BOBYQA implementation and intercepts its
    function evaluations, converting them into asynchronous MPI tasks.
    """

    # State management
    - current_params: Current optimization parameters
    - current_fitness: Current best fitness
    - trust_radius: Current trust region radius
    - interpolation_points: Model interpolation point set
    - pending_evals: Queue of evaluations in flight

    # Optimization control
    - max_iterations: Maximum BOBYQA iterations
    - initial_trust_radius: Starting trust region size
    - convergence_tolerance: Stopping criterion
```

**State Machine Flow:**

1. **Initialization** (`start()`):
   - Test neighbor parameters (like LBFGSBJob)
   - Initialize trust region around best starting point
   - Queue initial interpolation set evaluations

2. **Model Building** (`NEEDS_INITIAL_MODEL`):
   - Generate 2*n+1 interpolation points
   - Return all as parallel tasks
   - Wait for all results before proceeding

3. **Trust Region Step** (`NEEDS_TRUST_STEP`):
   - Solve trust region subproblem (quadratic model + bounds)
   - Generate candidate point
   - Evaluate in parallel

4. **Model Update** (`process_result()`):
   - Accept/reject trust region step
   - Update trust radius
   - Optionally queue model improvement points

5. **Convergence Check**:
   - Trust radius < tolerance
   - OR max iterations reached
   - OR no improvement over N iterations

### 2. Function Interception Strategy

**Synchronous → Asynchronous Conversion:**

```python
class AsyncBOBYQA:
    """
    Wrapper that converts BOBYQA's synchronous function calls
    into asynchronous task generation.
    """

    def __init__(self, job):
        self.job = job
        self.pending_tasks = []
        self.completed_evals = {}

    def objective_function(self, x):
        """
        This gets called by BOBYQA library.
        Instead of evaluating, we:
        1. Store the evaluation request
        2. Raise a special exception to pause BOBYQA
        3. Let job.start() return these as tasks
        """
        task_id = self.job.next_task_id()
        self.pending_tasks.append({
            'task_id': task_id,
            'params': x,
            'status': 'pending'
        })
        raise PendingEvaluationException(task_id)

    def resume_with_result(self, task_id, value):
        """Called by job.process_result() to provide evaluation."""
        self.completed_evals[task_id] = value
```

**Alternative: Batch Evaluation Mode:**

If the library supports batch evaluations (like some BOBYQA implementations):

```python
def start(self):
    # Generate next batch of points BOBYQA needs
    next_points = self.bobyqa_state.get_next_evaluation_batch()

    tasks = []
    for i, point in enumerate(next_points):
        full_params = self._construct_full_params_for_task(point)
        context = {
            'type': self.type,
            'job_id': self.id,
            'sub_type': 'BOBYQA_EVAL',
            'point_idx': i
        }
        tasks.append({'params': full_params, 'context': context})

    return tasks
```

### 3. Trust Region Subproblem Solver

BOBYQA needs to solve: `min_s  m(s)  s.t. ||s|| ≤ Δ, l ≤ x+s ≤ u`

Where `m(s) = c + g^T s + (1/2) s^T H s` (quadratic model)

**Implementation Options:**

1. **Use scipy's existing solver** (easiest):
   ```python
   from scipy.optimize import minimize

   result = minimize(
       quadratic_model,
       x0=np.zeros(n),
       method='trust-constr',
       bounds=adjusted_bounds,
       options={'maxiter': 100}
   )
   ```

2. **Simple Cauchy point** (fallback):
   ```python
   # Steepest descent direction
   s = -gradient / ||gradient||
   # Step to trust region boundary or bounds
   alpha = min(trust_radius, bounds_step_limit)
   s_cauchy = alpha * s
   ```

### 4. Integration Points

**In `sampler.py`:**

```python
def create_post_activation_bobyqa_jobs(self, next_job_id):
    """
    Create BOBYQA jobs for all activated grid points.

    Similar to create_post_activation_lbfgsb_jobs() but uses BOBYQA.
    """
    from .jobs.bobyqa_job import BOBYQAJob

    jobs = []
    for grid_idx, state in self.population.items():
        if state['status'] == 'active':
            # ... (similar to LBFGSB version)

            job = BOBYQAJob(
                job_id=next_job_id,
                job_type='POST_ACTIVATION_BOBYQA',
                sampler=self,
                opt_dims=tuple(self.continuous_dims),
                start_params=start_params_partial,
                grid_idx=grid_idx,
                start_params_full=start_params_full,
                start_fitness=start_fitness,
                initial_trust_radius=0.1,  # Configurable
                max_iterations=50
            )
            jobs.append(job)
            next_job_id += 1

    return jobs, next_job_id
```

**In `master.py`:**

Add new workflow stage:

```python
# In master_main state machine
elif stage == 'BOBYQA_LOOP':
    # Similar to LBFGSB_LOOP but for BOBYQA
    jobs, next_job_id = sampler.create_bobyqa_loop_jobs(next_job_id)
    # ...
```

**Configuration Parameter:**

```python
# In example scripts
PROJECTIONS_TO_RUN = [
    {
        'dims': [0, 1],
        'grid_points': [50, 50],
        'optimization_method': 'bobyqa',  # NEW OPTION
        'bobyqa_initial_trust_radius': 0.1,
        'bobyqa_max_iterations': 50,
        # ...
    }
]
```

### 5. Detailed BOBYQAJob Implementation

See `BOBYQA_JOB_SKELETON.py` for full implementation sketch.

## Expected Performance

### Sample Complexity per Grid Point

**L-BFGS-B (current):**
- ~20-50 evaluations for convergence
- Central differences: `iterations × (2*n_dims + 1)` evaluations
- Forward differences: `iterations × (n_dims + 1)` evaluations

**BOBYQA (expected):**
- Initial model: `2*n_dims + 1` evaluations
- Iterations: `iterations × 1-2` evaluations (mostly single trust region steps)
- Total: ~15-35 evaluations for convergence
- **Expected reduction: 25-40%**

### Parallelization Efficiency

Both L-BFGS-B and BOBYQA can saturate workers:

**L-BFGS-B:**
- Gradient calc: `n_dims` parallel tasks (central: `2*n_dims`)
- Line search: 1 task at a time (sequential backtracking)

**BOBYQA:**
- Initial model: `2*n_dims + 1` parallel tasks
- Trust region steps: 1 task at a time
- Model improvement: 1-5 parallel tasks (opportunistic)

**Verdict:** Similar parallelization potential, BOBYQA may have slight edge during model building.

## Testing Strategy

### Unit Tests

1. **BOBYQAJob state machine:**
   - Test each state transition
   - Verify task generation
   - Check convergence detection

2. **Trust region subproblem:**
   - Test with known quadratic models
   - Verify bound handling
   - Check trust radius updates

3. **Integration points:**
   - Test job factory methods
   - Verify configuration parsing

### Integration Tests

1. **Simple test functions:**
   - Himmelblau 4D (well-conditioned)
   - Rosenbrock 4D (ill-conditioned)
   - Compare evaluations vs L-BFGS-B

2. **Grid-anchored optimization:**
   - 2D projection on 4D function
   - Verify profile likelihood accuracy
   - Check dynamic activation behavior

3. **MPI parallelization:**
   - Run with 4-8 workers
   - Monitor task distribution
   - Check for deadlocks/race conditions

### Benchmarks

Create `benchmark_bobyqa_vs_lbfgsb.py`:

```python
# Compare on standard test suite:
# - Total evaluations
# - Convergence quality
# - Parallel efficiency
# - Wall-clock time

functions = ['himmelblau_4d', 'rosenbrock_4d', 'rastrigin_4d']
methods = ['lbfgsb', 'bobyqa']

for func_name in functions:
    for method in methods:
        results = run_projection(
            optimization_method=method,
            # ...
        )
        print(f"{func_name} + {method}: {results['total_calls']} evals")
```

## Dependencies

### Required Libraries

1. **PDFO** (Recommended):
   ```bash
   pip install pdfo
   ```
   - Pure Python BOBYQA implementation
   - Good parallelization control
   - BSD license

2. **OR scipy** (Alternative):
   ```bash
   # Already a dependency
   ```
   - Use `method='trust-constr'` with quadratic model
   - More limited control over algorithm internals

### Optional (Phase 2)

- **Py-BOBYQA**: More modern implementation with better parallelization hooks

## Implementation Checklist

- [ ] Create `src/paraprof/jobs/bobyqa_job.py`
- [ ] Implement BOBYQAJob class with basic state machine
- [ ] Add trust region subproblem solver
- [ ] Implement function interception mechanism
- [ ] Add factory methods in `sampler.py`
- [ ] Update workflow stages in `master.py`
- [ ] Add configuration validation in `sampler._reset_for_new_projection()`
- [ ] Create unit tests in `tests/test_bobyqa_job.py`
- [ ] Create integration test script
- [ ] Run benchmarks vs L-BFGS-B
- [ ] Update documentation and examples

## Risk Mitigation

### Potential Issues

1. **Function interception complexity:**
   - Mitigation: Start with simple batch evaluation mode
   - Fallback: Use synchronous wrapper with periodic task generation

2. **Trust region subproblem failures:**
   - Mitigation: Implement robust Cauchy point fallback
   - Fallback: Fall back to L-BFGS-B for that grid point

3. **Convergence issues on rough landscapes:**
   - Mitigation: Tune trust radius parameters
   - Fallback: Hybrid approach (BOBYQA for smooth regions, L-BFGS-B for rough)

## Success Criteria

Phase 1 is successful if:

1. ✅ BOBYQAJob integrates cleanly into existing job architecture
2. ✅ Reduces evaluations by ≥20% on smooth test functions (Himmelblau, Rosenbrock)
3. ✅ Maintains comparable convergence quality to L-BFGS-B
4. ✅ No degradation in parallel efficiency
5. ✅ Passes all unit and integration tests
6. ✅ Code complexity remains manageable (<500 lines for BOBYQAJob)

## Timeline Estimate

- Week 1: Implement BOBYQAJob skeleton + trust region solver
- Week 2: Function interception + integration points
- Week 3: Testing + debugging + benchmarks
- Week 4: Documentation + refinement

**Total: ~4 weeks for complete Phase 1**

## Next Steps (Phase 2)

If Phase 1 succeeds:

1. **Model Transfer Between Neighbors:**
   - Save Hessian approximation in `optimizer_state`
   - Seed neighbor jobs with transferred Hessian
   - Measure speedup on grid sweeps

2. **Adaptive Method Selection:**
   - Use BOBYQA for smooth regions (high neighbor correlation)
   - Use L-BFGS-B for rough regions (low correlation)
   - Hybrid strategy based on local landscape analysis

3. **Enhanced Parallelization:**
   - Generate multiple trust region candidates
   - Evaluate in parallel, pick best
   - Opportunistic model improvement points

## References

- Powell, M. J. D. (2009). "The BOBYQA algorithm for bound constrained optimization without derivatives"
- Zhang et al. (2020). "PDFO: A cross-platform package for Powell's derivative-free optimization solvers"
- ParaProf existing jobs: `LBFGSBJob`, `CoordinateDescentJob` (good design patterns)
