# BOBYQA Integration Guide - Implementation Steps

## Quick Summary

This guide walks through implementing Phase 1 BOBYQA integration into ParaProf. The complete implementation includes:

1. `BOBYQAJob` class (~400-500 lines)
2. Sampler integration (~50 lines)
3. Master workflow updates (~30 lines)
4. Configuration validation (~20 lines)
5. Tests (~200 lines)

**Total estimated effort:** 3-4 weeks for complete, tested implementation.

## Step-by-Step Implementation

### Step 1: Create BOBYQAJob Class

**File:** `src/paraprof/jobs/bobyqa_job.py`

Use the skeleton from `BOBYQA_JOB_SKELETON.py` as your starting point.

**Key Implementation Details:**

1. **Trust Region Subproblem Solver** (Lines 150-200):
   - Start with simple Cauchy point (steepest descent)
   - Later enhance with dogleg or exact solver

2. **Interpolation Point Generation** (Lines 220-260):
   - Use coordinate perturbations: ±step in each dimension
   - Step size = trust_radius / 2
   - Respect parameter bounds

3. **Model Building** (Lines 270-330):
   - Fit gradient via least squares on differences
   - Estimate Hessian from coordinate pairs
   - Add regularization for positive definiteness

4. **State Machine** (Lines 350-500):
   - Follow LBFGSBJob patterns
   - Handle neighbor testing, initial model, trust steps
   - Update trust radius based on ratio test

**Testing as you go:**

```python
# Unit test for trust region solver
def test_trust_region_solver():
    # Quadratic: f(x) = x^2 + y^2
    job = BOBYQAJob(...)
    job.model_gradient = np.array([2.0, 2.0])
    job.model_hessian = 2 * np.eye(2)
    job.trust_radius = 1.0

    s, pred_red = job._solve_trust_region_subproblem()

    # Should step in -gradient direction to TR boundary
    assert np.linalg.norm(s) <= 1.0 + 1e-6
    assert np.dot(s, job.model_gradient) < 0  # Descent direction
```

### Step 2: Add Sampler Configuration

**File:** `src/paraprof/sampler.py`

**Add parameters to `__init__`:**

```python
def __init__(self,
             # ... existing parameters ...
             bobyqa_initial_trust_radius=0.1,
             bobyqa_max_iterations=50,
             bobyqa_min_trust_radius=1e-6):
    """
    Parameters
    ----------
    # ... existing docs ...
    bobyqa_initial_trust_radius : float, optional
        Initial trust region radius for BOBYQA (default: 0.1)
    bobyqa_max_iterations : int, optional
        Maximum BOBYQA iterations per grid point (default: 50)
    bobyqa_min_trust_radius : float, optional
        Minimum trust radius before convergence (default: 1e-6)
    """
    # ... existing code ...

    # BOBYQA configuration
    self.bobyqa_initial_trust_radius = bobyqa_initial_trust_radius
    self.bobyqa_max_iterations = bobyqa_max_iterations
    self.bobyqa_min_trust_radius = bobyqa_min_trust_radius
```

**Add validation in `_reset_for_new_projection`:**

```python
# In _reset_for_new_projection, around line 350:

# Validate optimization method
valid_methods = ['de', 'lbfgsb', 'bobyqa']  # ADD 'bobyqa'
if self.optimization_method not in valid_methods:
    raise ConfigurationError(
        f"Invalid optimization_method: '{self.optimization_method}'. "
        f"Must be one of {valid_methods}",
        parameter="optimization_method",
        value=self.optimization_method
    )
```

**Add job factory methods:**

```python
# Around line 1000, after create_post_activation_lbfgsb_jobs:

def create_post_activation_bobyqa_jobs(self, next_job_id):
    """
    Create BOBYQA jobs for all activated grid points.

    Similar to create_post_activation_lbfgsb_jobs() but uses BOBYQA
    optimization instead of L-BFGS-B.

    Parameters
    ----------
    next_job_id : int
        The next available job ID

    Returns
    -------
    jobs : list
        List of BOBYQAJob instances
    next_job_id : int
        Updated job ID counter
    """
    from .jobs.bobyqa_job import BOBYQAJob

    jobs = []

    for grid_idx, state in self.population.items():
        if state['status'] == 'active':
            # Direct evaluation mode: no continuous dimensions
            if self.direct_eval_mode or self.n_cont_dims == 0:
                state['status'] = 'bobyqa_optimized'
                continue

            # Mark as claimed
            state['status'] = 'BOBYQA_queued'

            # Find best individual to start from
            best_ind_idx = np.argmax(state['fitnesses'])
            start_params_partial = state['continuous_params'][best_ind_idx]
            start_fitness = state['fitnesses'][best_ind_idx]
            start_params_full = self._construct_params(grid_idx, start_params_partial)

            # Create BOBYQA job
            job = BOBYQAJob(
                job_id=next_job_id,
                job_type='POST_ACTIVATION_BOBYQA',
                sampler=self,
                opt_dims=tuple(self.continuous_dims),
                start_params=start_params_partial,
                grid_idx=grid_idx,
                start_params_full=start_params_full,
                start_fitness=start_fitness,
                initial_trust_radius=self.bobyqa_initial_trust_radius,
                max_iterations=self.bobyqa_max_iterations
            )

            jobs.append(job)
            next_job_id += 1

    self.logger.info(f"Created {len(jobs)} post-activation BOBYQA jobs")
    return jobs, next_job_id


def create_bobyqa_loop_jobs(self, next_job_id):
    """
    Create BOBYQA jobs for active grid points in BOBYQA_LOOP stage.

    Similar to create_lbfgsb_loop_jobs() but for iterative BOBYQA workflow.

    Parameters
    ----------
    next_job_id : int
        The next available job ID

    Returns
    -------
    jobs : list
        List of BOBYQAJob instances
    next_job_id : int
        Updated job ID counter
    """
    from .jobs.bobyqa_job import BOBYQAJob

    jobs = []

    for grid_idx, state in self.population.items():
        if state['status'] == 'active':
            # Direct evaluation mode
            if self.direct_eval_mode or self.n_cont_dims == 0:
                state['status'] = 'converged'
                continue

            # Mark as claimed
            state['status'] = 'BOBYQA_queued'

            # Find best individual to start from
            best_ind_idx = np.argmax(state['fitnesses'])
            start_params_partial = state['continuous_params'][best_ind_idx]
            start_fitness = state['fitnesses'][best_ind_idx]
            start_params_full = self._construct_params(grid_idx, start_params_partial)

            # Create BOBYQA job
            job = BOBYQAJob(
                job_id=next_job_id,
                job_type='BOBYQA_LOOP',
                sampler=self,
                opt_dims=tuple(self.continuous_dims),
                start_params=start_params_partial,
                grid_idx=grid_idx,
                start_params_full=start_params_full,
                start_fitness=start_fitness,
                initial_trust_radius=self.bobyqa_initial_trust_radius,
                max_iterations=self.bobyqa_max_iterations
            )

            jobs.append(job)
            next_job_id += 1

    return jobs, next_job_id
```

### Step 3: Update Master Workflow

**File:** `src/paraprof/master.py`

Find the `master_main` function. Locate the workflow stage logic (around line 400-600).

**Add BOBYQA stages after DE stages:**

```python
# Around line 500, after POST_ACTIVATION_LBFGSB stage:

elif stage == 'POST_ACTIVATION_BOBYQA':
    logger.info("=" * 80)
    logger.info(f"--- Stage: POST_ACTIVATION_BOBYQA (Generation {sampler.current_generation}) ---")
    logger.info("=" * 80)

    jobs, next_job_id = sampler.create_post_activation_bobyqa_jobs(next_job_id)

    if not jobs:
        logger.info("No grid points need BOBYQA optimization. Moving to next stage.")
        if sampler.patching_coarse:
            stage = 'PATCHING'
        else:
            stage = 'DONE'
        continue

    # Queue all jobs
    for job in jobs:
        active_jobs[job.id] = job
        new_tasks = job.start()
        for task in new_tasks:
            task_queue.append(task)

    stage = 'PROCESSING_POST_ACTIVATION_BOBYQA'

elif stage == 'PROCESSING_POST_ACTIVATION_BOBYQA':
    # Same logic as PROCESSING_POST_ACTIVATION_LBFGSB
    if not active_jobs and not task_queue and pending_results == 0:
        logger.info("All BOBYQA jobs complete.")

        if sampler.patching_coarse:
            stage = 'PATCHING'
        else:
            stage = 'DONE'

# Similarly for BOBYQA_LOOP:

elif stage == 'BOBYQA_LOOP':
    logger.info("=" * 80)
    logger.info(f"--- Stage: BOBYQA_LOOP (Generation {sampler.current_generation}) ---")
    logger.info("=" * 80)

    jobs, next_job_id = sampler.create_bobyqa_loop_jobs(next_job_id)

    if not jobs:
        logger.info("No active grid points remain. BOBYQA loop complete.")
        if sampler.patching_coarse:
            stage = 'PATCHING'
        else:
            stage = 'DONE'
        continue

    # Queue jobs and process
    for job in jobs:
        active_jobs[job.id] = job
        new_tasks = job.start()
        for task in new_tasks:
            task_queue.append(task)

    stage = 'PROCESSING_BOBYQA_LOOP'

elif stage == 'PROCESSING_BOBYQA_LOOP':
    if not active_jobs and not task_queue and pending_results == 0:
        sampler.current_generation += 1

        # Dynamic activation
        new_jobs, next_job_id = sampler.create_dynamic_activation_jobs(next_job_id)
        if new_jobs:
            # ... (queue activation jobs, stay in BOBYQA_LOOP)
            stage = 'BOBYQA_LOOP'
        else:
            # No new activations
            if sampler.patching_coarse:
                stage = 'PATCHING'
            else:
                stage = 'DONE'
```

**Update stage selection logic:**

```python
# Around line 350, in initial stage selection:

if sampler.optimization_method == 'lbfgsb':
    stage = 'POST_ACTIVATION_LBFGSB'
elif sampler.optimization_method == 'bobyqa':
    stage = 'POST_ACTIVATION_BOBYQA'  # NEW
elif sampler.optimization_method == 'de':
    stage = 'DE_LOOP'
```

### Step 4: Update Jobs Module

**File:** `src/paraprof/jobs/__init__.py`

```python
from .bobyqa_job import BOBYQAJob

__all__ = [
    'Job',
    'LBFGSBJob',
    'ActivationJob',
    'DEGridPointJob',
    'PatchingTestJob',
    'CoordinateDescentJob',
    'BOBYQAJob'  # ADD THIS
]
```

### Step 5: Create Example Script

**File:** `examples/run_himmelblau_4d_bobyqa.py`

```python
"""
Example: BOBYQA optimization on Himmelblau 4D.

Compares BOBYQA vs L-BFGS-B vs DE on profile likelihood computation.
"""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed.")
    sys.exit(1)

from paraprof import (
    GridAnchoredDESampler,
    run_all_projections,
    terminate_workers,
    worker_main,
    get_test_function,
    set_log_level
)

set_log_level('INFO')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

np.random.seed(750123)

TEST_FUNCTION = "himmelblau_4d"

# Test different optimization methods
PROJECTIONS_TO_RUN = [
    # 1D projection with BOBYQA
    {
        'dims': [0],
        'grid_points': [100],
        'optimization_method': 'bobyqa',
        'patching_coarse': True,
        'enable_refinement': True,
        'refinement_factor': 2
    },

    # 2D projection with BOBYQA
    {
        'dims': [0, 1],
        'grid_points': [50, 50],
        'optimization_method': 'bobyqa',
        'patching_coarse': True,
        'patching_refined': True,
        'enable_refinement': True,
        'refinement_factor': 2
    },
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # Master process
    output_file = f"samples_bobyqa_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        n_initial_optimizations=100,
        roi_threshold=4.0,
        convergence_threshold=1e-7,
        convergence_window=3,
        # BOBYQA-specific parameters
        bobyqa_initial_trust_radius=0.1,
        bobyqa_max_iterations=50,
        bobyqa_min_trust_radius=1e-6,
        # Other settings
        max_patching_waves=50,
        memory_size=25,
        samples_output_file=output_file,
    )

    # Broadcast target function
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # Run all projections
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=100000,
        save_plots=True,
        plot_settings={'dpi': 300, 'filetype': 'png'},
        myrank=myrank
    )

    # Print summary
    print("\n" + "="*80)
    print("=== BOBYQA Benchmark Results ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
    print("="*80 + "\n")

    # Terminate workers
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # Worker process
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
```

### Step 6: Create Tests

**File:** `tests/test_bobyqa_job.py`

```python
"""
Unit tests for BOBYQAJob.
"""
import pytest
import numpy as np
from paraprof.jobs.bobyqa_job import BOBYQAJob
from paraprof import GridAnchoredDESampler, get_test_function


@pytest.fixture
def simple_sampler():
    """Create a simple sampler for testing."""
    log_likelihood, bounds, _ = get_test_function("himmelblau_4d")
    projections = [{'dims': [0, 1], 'grid_points': [10, 10]}]

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=bounds,
        projections=projections,
        bobyqa_initial_trust_radius=0.1,
        bobyqa_max_iterations=20
    )
    return sampler


def test_bobyqa_job_initialization(simple_sampler):
    """Test BOBYQAJob initialization."""
    start_params = np.array([0.5, 0.5])
    grid_idx = (0, 0)
    start_params_full = simple_sampler._construct_params(grid_idx, start_params)

    job = BOBYQAJob(
        job_id=1,
        job_type='POST_ACTIVATION_BOBYQA',
        sampler=simple_sampler,
        opt_dims=tuple(simple_sampler.continuous_dims),
        start_params=start_params,
        grid_idx=grid_idx,
        start_params_full=start_params_full,
        initial_trust_radius=0.1,
        max_iterations=20
    )

    assert job.id == 1
    assert job.n_opt_dims == 2
    assert job.trust_radius == 0.1
    assert job.status == 'NEEDS_INITIAL_F'


def test_trust_region_subproblem_solver(simple_sampler):
    """Test trust region subproblem solver."""
    job = BOBYQAJob(
        job_id=1,
        job_type='POST_ACTIVATION_BOBYQA',
        sampler=simple_sampler,
        opt_dims=(2, 3),
        start_params=np.array([0.0, 0.0]),
        grid_idx=(0, 0),
        start_params_full=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_trust_radius=1.0,
        max_iterations=20
    )

    # Set up a simple quadratic model
    job.model_gradient = np.array([1.0, 1.0])  # Gradient points in (1,1) direction
    job.model_hessian = np.eye(2)
    job.model_center = np.array([0.0, 0.0])
    job.trust_radius = 1.0

    s, pred_red = job._solve_trust_region_subproblem()

    # Step should be in descent direction
    assert np.dot(s, job.model_gradient) < 0

    # Step should respect trust radius
    assert np.linalg.norm(s) <= job.trust_radius + 1e-6

    # Predicted reduction should be positive
    assert pred_red > 0


def test_interpolation_point_generation(simple_sampler):
    """Test initial interpolation point generation."""
    job = BOBYQAJob(
        job_id=1,
        job_type='POST_ACTIVATION_BOBYQA',
        sampler=simple_sampler,
        opt_dims=(2, 3),
        start_params=np.array([0.0, 0.0]),
        grid_idx=(0, 0),
        start_params_full=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_trust_radius=0.1,
        max_iterations=20
    )

    points = job._generate_initial_interpolation_points()

    # Should have 2*n points (coordinate +/- perturbations)
    assert len(points) == 2 * job.n_opt_dims

    # All points should be within bounds
    cont_dims = simple_sampler.continuous_dims
    for point in points:
        for i, dim_idx in enumerate(cont_dims):
            assert simple_sampler.bounds[dim_idx, 0] <= point[i] <= simple_sampler.bounds[dim_idx, 1]


def test_bobyqa_convergence_detection(simple_sampler):
    """Test convergence detection logic."""
    job = BOBYQAJob(
        job_id=1,
        job_type='POST_ACTIVATION_BOBYQA',
        sampler=simple_sampler,
        opt_dims=(2, 3),
        start_params=np.array([0.0, 0.0]),
        grid_idx=(0, 0),
        start_params_full=np.array([0.0, 0.0, 0.0, 0.0]),
        initial_trust_radius=0.1,
        max_iterations=50
    )

    # Test: Trust radius too small
    job.trust_radius = 1e-8
    assert job._check_convergence() == True

    # Reset
    job.trust_radius = 0.1

    # Test: Max iterations
    job.iteration = 51
    assert job._check_convergence() == True

    # Reset
    job.iteration = 10

    # Test: No improvement
    job.no_improvement_count = 15
    assert job._check_convergence() == True
```

**File:** `tests/test_bobyqa_integration.py`

```python
"""
Integration tests for BOBYQA in full workflow.
"""
import pytest
import numpy as np
from mpi4py import MPI
from paraprof import GridAnchoredDESampler, get_test_function


def test_bobyqa_simple_optimization():
    """Test BOBYQA on a simple 2D quadratic."""
    # Simple quadratic: f(x,y) = -(x-1)^2 - (y-2)^2
    def quadratic(params):
        x, y = params
        return -((x - 1.0)**2 + (y - 2.0)**2)

    bounds = np.array([[-5, 5], [-5, 5]])
    projections = [{'dims': [0, 1], 'grid_points': [5, 5], 'optimization_method': 'bobyqa'}]

    sampler = GridAnchoredDESampler(
        target_func=quadratic,
        bounds=bounds,
        projections=projections,
        n_initial_optimizations=10,
        bobyqa_max_iterations=30
    )

    # This would normally be run with MPI, but for testing we can check job creation
    sampler._reset_for_new_projection(projections[0])

    # Run initial optimizations (synchronous for testing)
    # ... (simplified, full test would use MPI)

    assert sampler.optimization_method == 'bobyqa'
```

### Step 7: Run Benchmarks

**File:** `benchmarks/benchmark_bobyqa_vs_lbfgsb.py`

```python
"""
Benchmark BOBYQA vs L-BFGS-B on standard test functions.

Usage:
    mpiexec -n 8 python benchmark_bobyqa_vs_lbfgsb.py
"""
import sys
import numpy as np
from mpi4py import MPI
from paraprof import (
    GridAnchoredDESampler,
    run_projection,
    terminate_workers,
    worker_main,
    get_test_function,
    set_log_level
)

set_log_level('WARNING')  # Reduce noise

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

TEST_FUNCTIONS = ['himmelblau_4d', 'rosenbrock_4d']
METHODS = ['lbfgsb', 'bobyqa']

if myrank == 0:
    print("="*80)
    print("BOBYQA vs L-BFGS-B Benchmark")
    print("="*80)

    results = {}

    for func_name in TEST_FUNCTIONS:
        log_likelihood, bounds, _ = get_test_function(func_name)
        results[func_name] = {}

        for method in METHODS:
            projection_config = {
                'dims': [0, 1],
                'grid_points': [20, 20],
                'optimization_method': method
            }

            sampler = GridAnchoredDESampler(
                target_func=log_likelihood,
                bounds=bounds,
                projections=[projection_config],
                n_initial_optimizations=50,
                bobyqa_max_iterations=50,
                LBFGSB_max_iter=50
            )

            comm.bcast(sampler.target_func, root=0)

            result = run_projection(
                comm=comm,
                sampler=sampler,
                projection_config=projection_config,
                num_generations=10000,
                myrank=myrank
            )

            results[func_name][method] = {
                'total_calls': result['metrics']['total_target_calls'],
                'global_max': result['metrics']['global_max']
            }

    # Print comparison
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    for func_name in TEST_FUNCTIONS:
        print(f"\n{func_name}:")
        for method in METHODS:
            calls = results[func_name][method]['total_calls']
            max_val = results[func_name][method]['global_max']
            print(f"  {method:10s}: {calls:6d} calls, max = {max_val:.6e}")

        # Compute reduction
        lbfgsb_calls = results[func_name]['lbfgsb']['total_calls']
        bobyqa_calls = results[func_name]['bobyqa']['total_calls']
        reduction = 100 * (lbfgsb_calls - bobyqa_calls) / lbfgsb_calls
        print(f"  {'reduction':10s}: {reduction:6.1f}%")

    terminate_workers(comm, myrank)

else:
    worker_main(comm, myrank)
```

## Configuration Examples

### Minimal BOBYQA Usage

```python
sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=[{
        'dims': [0, 1],
        'grid_points': [50, 50],
        'optimization_method': 'bobyqa'  # That's it!
    }]
)
```

### Advanced Configuration

```python
sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=PROJECTIONS,
    # BOBYQA-specific tuning
    bobyqa_initial_trust_radius=0.1,  # Smaller = more conservative
    bobyqa_max_iterations=50,         # More iterations for complex landscapes
    bobyqa_min_trust_radius=1e-6,     # Convergence threshold
    # Other settings
    n_initial_optimizations=100,
    roi_threshold=4.0,
    max_patching_waves=50
)
```

### Hybrid Strategy (Phase 2 Idea)

```python
# Use BOBYQA for smooth projections, L-BFGS-B for rough ones
PROJECTIONS = [
    # Smooth low-D projection
    {'dims': [0, 1], 'grid_points': [50, 50], 'optimization_method': 'bobyqa'},

    # Higher-D or rough projection
    {'dims': [0, 1, 2], 'grid_points': [20, 20, 20], 'optimization_method': 'lbfgsb'},
]
```

## Troubleshooting

### Issue: Trust region gets stuck at minimum radius

**Solution:** Increase `bobyqa_min_trust_radius` or add better model improvement logic

### Issue: BOBYQA uses more evaluations than L-BFGS-B

**Possible causes:**
- Initial model building overhead dominates (small n_dims)
- Trust region solver not finding good steps
- Model quality is poor

**Debug:** Add logging in `_solve_trust_region_subproblem()` to check predicted vs actual reduction

### Issue: Convergence to wrong optimum

**Solution:** Increase `n_initial_optimizations` or tune `bobyqa_initial_trust_radius`

## Success Metrics

Track these metrics to evaluate Phase 1 success:

```python
# In benchmarks
metrics = {
    'total_evaluations': sampler.target_calls,
    'evaluations_per_grid_point': sampler.target_calls / len(sampler.active_grid_indices),
    'convergence_quality': sampler.global_max_target_val,
    'parallel_efficiency': wall_time / (total_cpu_time / n_workers)
}
```

**Target:** ≥20% reduction in evaluations_per_grid_point vs L-BFGS-B

## Next Steps After Phase 1

If benchmarks show promise:

1. **Phase 2A: Model Transfer**
   - Implement Hessian seeding from neighbors
   - Measure speedup on grid sweeps

2. **Phase 2B: Enhanced Parallelization**
   - Generate multiple trust region candidates
   - Opportunistic model improvement

3. **Phase 2C: Adaptive Selection**
   - Auto-choose method based on local landscape
   - Hybrid BOBYQA + L-BFGS-B

## References

- BOBYQA paper: Powell (2009) "The BOBYQA algorithm for bound constrained optimization without derivatives"
- Trust region methods: Conn, Gould, Toint (2000) "Trust-Region Methods"
- PDFO library: https://www.pdfo.net/
