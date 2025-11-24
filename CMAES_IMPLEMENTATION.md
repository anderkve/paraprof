# CMA-ES Optimization Method Implementation

## Overview

This document describes the implementation of CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as a third optimization method for ParaProf, joining the existing 'de' (Differential Evolution) and 'lbfgsb' (L-BFGS-B) options.

The implementation focuses on **Opportunity 3: Neighbor-Informed Initialization**, leveraging the grid structure to warm-start CMA-ES runs from solutions found at neighboring grid points.

## Key Features

### 1. Neighbor-Informed Mean Initialization

The CMA-ES mean vector is initialized from the best neighbor solution rather than randomly:

```python
def _get_neighbor_informed_mean(self):
    """Initialize mean from best neighbor solution."""
    for neighbor_idx in self.sampler._get_valid_neighbors(self.grid_idx):
        if neighbor_idx in self.sampler.population:
            # Use best neighbor's continuous parameters
            best_neighbor_params = ...
            return best_neighbor_params.copy()
    # Fallback to global pool or bounds center
```

**Benefit**: Reduces wasted evaluations exploring low-likelihood regions.

### 2. Adaptive Step Size Initialization

Initial step size σ is estimated from neighbor fitness variance:

```python
def _estimate_initial_sigma(self):
    """Estimate sigma based on neighbor fitness variance."""
    if fitness_range > roi_threshold / 2:
        sigma = 0.3 * typical_scale  # Steep region - explore
    else:
        sigma = 0.1 * typical_scale  # Flat region - exploit
```

**Benefit**: Avoids overly large or small initial step sizes.

### 3. Population Seeding from Neighbors

In the first generation, up to 30% of the population is seeded from neighbor solutions:

```python
def _sample_offspring(self):
    if self.generation == 0:
        # Mix neighbor solutions with CMA-ES samples
        offspring.extend(neighbor_solutions[:lambda//3])
    # Remaining offspring from N(m, σ²C)
```

**Benefit**: Provides diverse high-quality starting points.

## Architecture Integration

### Configuration Parameters

Three new parameters in `GridAnchoredDESampler.__init__()`:

- `cmaes_lambda`: Population size (default: 4 + 3*log(n_cont_dims))
- `cmaes_mu`: Number of parents (default: lambda/2)
- `cmaes_max_generations`: Max iterations per grid point (default: 100)

### Workflow Integration

The CMA-ES method fits naturally into the existing workflow:

```
INITIAL_OPTIMIZATION → ACTIVATION → CMAES_LOOP → [PATCHING_WAVES]
```

The `CMAES_LOOP` stage:
1. Creates `CMAESGridPointJob` for each active grid point
2. Handles dynamic activation of neighboring points
3. Checks convergence (improvement history, sigma, condition number)
4. Optionally spawns L-BFGS-B refinement jobs

### Job-Based Parallelization

`CMAESGridPointJob` follows the same pattern as `DEGridPointJob`:

- `start()`: Sample λ offspring, return evaluation tasks
- `process_result()`: Update fitness, check if generation complete
- When generation complete: Update CMA-ES state (m, C, σ, evolution paths)
- `on_finish()`: Check convergence, spawn refinement if configured

## Usage

### Basic Usage

```python
sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=[{'dims': [0, 1], 'grid_points': [50, 50]}],
    cmaes_lambda=None,  # Auto-set based on dimensionality
    cmaes_mu=None,      # Auto-set as lambda/2
    cmaes_max_generations=100
)

projections = [
    {
        'dims': [0, 1],
        'grid_points': [50, 50],
        'optimization_method': 'cmaes',  # Use CMA-ES
        'lbfgsb_refinement': True,       # Optional L-BFGS-B polish
        'patching_coarse': True          # Optional patching
    }
]
```

### Test Script

Run `test_cmaes.py` to verify the implementation:

```bash
mpiexec -n 2 python test_cmaes.py
```

This runs a 1D projection on the Himmelblau 4D test function using CMA-ES.

## Performance Characteristics

### Test Results (Himmelblau 4D, 1D projection, 20 grid points)

- **Function calls**: ~3400
- **Grid points explored**: 6 (ROI-focused activation)
- **Status**: Successfully converges to global optimum

### Expected Benefits

1. **Fewer wasted evaluations**: Neighbor-informed initialization avoids exploring poor regions
2. **Faster convergence**: Better starting points reduce iterations needed
3. **Robust to rugged landscapes**: CMA-ES handles correlations and ill-conditioning
4. **Complementary to DE**: Different search characteristics may suit different problems

## Future Opportunities

The current implementation realizes **Opportunity 3**. Additional enhancements could include:

### Opportunity 1: Shared Covariance Matrix Pool

Maintain a global pool of learned covariance matrices:
- Donate converged covariance matrices to pool
- Initialize new points from weighted blend of neighbor covariances
- **Expected savings**: 30-50% fewer evaluations

### Opportunity 2: Enhanced Step Size Inheritance

More sophisticated σ initialization using:
- Neighbor fitness gradients
- Local curvature estimates
- **Expected savings**: 10-20% fewer evaluations

### Opportunity 4: Meta-Learning

Learn optimal CMA-ES hyperparameters (learning rates, damping) across the grid:
- Track successful hyperparameter values
- Use median/best values for new grid points
- **Expected savings**: 5-15% fewer evaluations

## Implementation Details

### Files Modified

- `src/paraprof/jobs/cmaes_job.py` (new): CMA-ES job class
- `src/paraprof/sampler.py`: Configuration and job creation
- `src/paraprof/master.py`: CMAES_LOOP workflow stage
- `src/paraprof/jobs/__init__.py`: Export CMAESGridPointJob
- `test_cmaes.py` (new): Test script

### CMA-ES Algorithm Implementation

The implementation follows standard CMA-ES (Hansen & Ostermeier 2001):

- **Recombination**: Weighted mean of μ best offspring
- **Evolution paths**: Momentum for C and σ adaptation
- **Covariance update**: Rank-μ + rank-1 update
- **Step size adaptation**: Cumulative step size adaptation (CSA)
- **Convergence checks**: Improvement history, σ threshold, condition number

### Convergence Criteria

CMA-ES jobs converge when:
1. Average improvement < `convergence_threshold` (same as DE)
2. Step size σ < 1e-10 (stagnation)
3. Condition number of C > 1e14 (ill-conditioning)
4. Max generations reached

## Comparison with DE and L-BFGS-B

| Feature | DE | L-BFGS-B | CMA-ES |
|---------|----|----|--------|
| **Type** | Population-based | Gradient-based | Population-based |
| **Parallelization** | λ evals/gen | Sequential | λ evals/gen |
| **Grid sharing** | Parent pool, neighbor-pull | Warm starts | Neighbor init, pop seeding |
| **Strengths** | Robust, simple | Fast convergence | Handles correlations |
| **Weaknesses** | Many evaluations | Needs good gradient | More complex |
| **Best for** | Multimodal | Smooth landscapes | Correlated parameters |

## Testing

### Unit Tests

The implementation has been tested with:
- 1D projections (test_cmaes.py)
- Various population sizes
- Convergence verification
- Grid activation dynamics

### Integration Tests

Verified compatibility with:
- Dynamic activation
- L-BFGS-B refinement
- Patching (optional)
- Emulator pre-screening (optional)

## References

- Hansen, N., & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies". *Evolutionary Computation*, 9(2), 159-195.
- Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial". arXiv:1604.00772.

---

**Implementation completed**: 2025-01-24
**Branch**: `feature/cmaes-optimization`
**Status**: ✅ Working and tested
