# Nuisance Parameter Framework

This document describes the nuisance parameter framework for ParaProf, designed to test profile likelihood computation in realistic physics scenarios.

## Overview

In many physics analyses, likelihood functions contain:
- **Parameters of Interest (POI)**: Few dimensions, wide ranges, complex surfaces
- **Nuisance Parameters**: Many dimensions, tightly constrained by Gaussian penalties
- **Coupling**: Nuisance parameters affect the main likelihood (not just additive)

The nuisance parameter framework allows you to augment any test function with constrained nuisance parameters that couple to the POI in physically motivated ways.

## Key Features

### Parameter Structure

Full parameter vector: `[poi_0, ..., poi_n, nuis_0, ..., nuis_m]`

The augmented likelihood has the form:
```
log L_total = log L_base(transformed_POI) + Σ log L_constraint(nuisance_i)
```

### Coupling Modes

The framework supports multiple ways nuisance parameters can affect POI:

1. **Shift** (`coupling_mode='shift'`): Additive systematic shifts
   - `x_poi → x_poi + Σ M_ij * d_j`
   - Models: calibration uncertainties, zero-point offsets
   - Example: detector calibration shifts in measurements

2. **Scale** (`coupling_mode='scale'`): Multiplicative scaling
   - `x_poi → x_poi * (1 + Σ M_ij * s_j)`
   - Models: normalization uncertainties, efficiency factors
   - Example: luminosity scaling in particle physics

3. **Rotation** (`coupling_mode='rotation'`): Coordinate transformations
   - Applies small rotations in POI space
   - Models: basis choice uncertainties, alignment
   - Example: detector coordinate system uncertainties

4. **Additive** (`coupling_mode='additive'`): No coupling
   - Baseline case for comparison
   - Only constraint penalties, no POI transformation

5. **Mixed** (`coupling_mode='mixed'`): Custom linear combinations
   - Use `coupling_matrix` for arbitrary linear coupling
   - Most general case

### Constraint Types

1. **Gaussian** (`constraint_mode='gaussian'`): Standard Gaussian penalty
   - `-0.5 * ((x - μ) / σ)²`
   - Most common in physics analyses
   - Default choice

2. **Uniform** (`constraint_mode='uniform'`): Flat within bounds
   - 0 within ±σ, -∞ outside
   - Hard constraints

3. **Soft Uniform** (`constraint_mode='soft_uniform'`): Hybrid
   - Flat within ±σ, quadratic penalty outside
   - Smoother than hard uniform

## Usage Examples

### Basic Example

```python
from paraprof import get_test_function
from paraprof.nuisance_wrapper import create_nuisance_wrapped_function

# Get base test function
base_func, base_bounds, peaks = get_test_function("himmelblau_4d")

# Wrap with 8 shift-type nuisance parameters
wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=4,
    n_nuisance=8,
    coupling_mode='shift',
    constraint_sigma=0.5,
    nuisance_mean=0.0
)

# Now wrapped_func accepts 12D input: [4 POI + 8 nuisance]
import numpy as np
params = np.zeros(12)
log_likelihood = wrapped_func(params)
```

### Custom Coupling Matrix

```python
import numpy as np

# Define how each nuisance parameter affects each POI
# Shape: (n_poi, n_nuisance)
coupling_matrix = np.array([
    [1.0, 0.5, 0.0, 0.0],  # POI 0 gets nuisance 0 fully, nuisance 1 at 0.5
    [0.0, 0.5, 1.0, 0.0],  # POI 1 gets nuisance 1 at 0.5, nuisance 2 fully
    [0.0, 0.0, 1.0, 0.5],  # etc.
    [0.0, 0.0, 0.0, 1.0]
])

wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=4,
    n_nuisance=4,
    coupling_mode='shift',
    coupling_matrix=coupling_matrix,
    constraint_sigma=0.3
)
```

### Per-Parameter Constraint Strengths

```python
# Different constraint widths for each nuisance parameter
constraint_sigmas = np.array([0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 0.3, 0.3])
# Some parameters tightly constrained (0.1), others loose (1.0)

wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=4,
    n_nuisance=8,
    coupling_mode='shift',
    constraint_sigma=constraint_sigmas
)
```

### Profile Likelihood Computation

```python
# Compute profile likelihood by optimizing over nuisance parameters
poi_values = np.array([3.0, 2.0, 3.0, 2.0])  # Fix POI values

profile_ll, optimal_nuis = wrapper.profile_over_nuisance(
    poi_values, method='analytical'
)

print(f"Profile log-likelihood: {profile_ll}")
print(f"Optimal nuisance parameters: {optimal_nuis}")
```

### Using Pre-Registered Test Functions

```python
from paraprof.nuisance_wrapper import register_nuisance_wrapped_test_functions

# Get registry of pre-configured test cases
registry = register_nuisance_wrapped_test_functions()

# Available configurations include:
# - himmelblau_4d_shift_8nuis_sigma0.5
# - himmelblau_4d_shift_16nuis_sigma0.5
# - rosenbrock_4d_shift_8nuis_sigma0.5
# - rastrigin_4d_shift_8nuis_sigma0.5
# etc.

func, bounds, wrapper, base_peaks = registry['himmelblau_4d_shift_8nuis_sigma0.5']
```

## Running Profile Likelihood Analysis

### Complete MPI Example

See `examples/run_nuisance_example.py` for a complete working example.

```bash
mpiexec -n 8 python examples/run_nuisance_example.py
```

Key points:
1. Only project over POI dimensions (not nuisance)
2. The full parameter space is explored during optimization
3. Nuisance parameters are automatically handled by the wrapper

### Configuration Recommendations

For testing with nuisance parameters:

**POI Grid Configuration:**
- Project only over POI dimensions: `dims=[0, 1]` for first 2 POI
- Use same grid resolutions as without nuisance parameters
- Enable refinement for detailed structure

**Sampler Configuration:**
- Increase `n_initial_optimizations` (e.g., 100-200)
  - More dimensions = more starting points needed
- Enable emulator pre-screening: `use_de_prescreening=True`
  - Reduces evaluations by 30-50%
- Adjust `roi_threshold` based on nuisance count
  - More nuisance params → larger likelihood variations

**Example Configuration:**
```python
sampler = GridAnchoredDESampler(
    target_func=wrapped_func,
    bounds=wrapped_bounds,
    projections=PROJECTIONS_TO_RUN,
    pop_per_grid_point=3,
    n_initial_optimizations=100,
    use_de_prescreening=True,
    emulator_min_neighbors=10,
    emulator_max_neighbors=200,
    # ... other parameters
)
```

## Physics Interpretation

### Calibration Uncertainty Example

```python
# 4 observables, 8 calibration parameters (2 per observable)
# Models detector response: measured = true + calibration_shift
base_func, base_bounds, _ = get_test_function("himmelblau_4d")

wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=4,
    n_nuisance=8,
    coupling_mode='shift',
    constraint_sigma=0.2,  # 20% calibration uncertainty
)
```

**Physical Meaning:**
- POI: True physical parameters being measured
- Nuisance: Systematic shifts in detector calibration
- Constraint: Prior knowledge from calibration studies (σ = 0.2)
- Coupling: Measurements are biased by calibration errors

### Normalization Uncertainty Example

```python
# Cross-section measurements with luminosity uncertainty
wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=4,
    n_nuisance=4,
    coupling_mode='scale',
    constraint_sigma=0.05,  # 5% luminosity uncertainty
)
```

**Physical Meaning:**
- POI: Physics cross-sections
- Nuisance: Luminosity scale factors
- Constraint: Luminosity measurement precision
- Coupling: Cross-sections scale with luminosity

## Testing Strategy

### Difficulty Scaling

Test difficulty can be systematically varied:

1. **Number of nuisance parameters:**
   - Few (4-8): Manageable, quick tests
   - Medium (16-32): Realistic physics scenarios
   - Many (50+): Stress test for high-dimensional spaces

2. **Constraint strength (σ):**
   - Tight (σ = 0.1-0.2): Strongly constrained, easier
   - Medium (σ = 0.5-1.0): Realistic constraints
   - Loose (σ = 2.0+): Weakly constrained, harder

3. **Base function complexity:**
   - Unimodal (sphere, rosenbrock): Test basic functionality
   - Few peaks (himmelblau): Realistic multimodal
   - Many peaks (rastrigin): Challenge for global search

### Validation

Check algorithm performance by:

1. **Recovery of known optima:**
   ```python
   # Base function has known peak at peak_poi
   params_optimal = np.concatenate([peak_poi, np.zeros(n_nuisance)])
   ll_optimal = wrapped_func(params_optimal)
   # Should be at base function optimum
   ```

2. **Nuisance parameter behavior:**
   ```python
   # Nuisance parameters should stay near constraint mean
   best_nuis = best_params[n_poi:]
   deviations = (best_nuis - nuisance_mean) / constraint_sigma
   # Most deviations should be < 1-2σ
   ```

3. **Profile structure:**
   - 1D profiles should show expected shape from base function
   - Nuisance parameters should profile out smoothly
   - Constraint penalties should be visible at boundaries

## Implementation Details

### Default Coupling Matrix

When `coupling_matrix=None`, the framework creates sensible defaults:

**Shift/Scale modes:**
- Distributes nuisance parameters evenly across POI dimensions
- Each POI gets approximately `n_nuisance / n_poi` nuisance params
- Example (4 POI, 8 nuisance):
  ```
  POI 0: nuisance 0, 1
  POI 1: nuisance 2, 3
  POI 2: nuisance 4, 5
  POI 3: nuisance 6, 7
  ```

**Rotation mode:**
- Creates rotation angles for all POI pairs
- Needs at least `n_poi * (n_poi - 1) / 2` nuisance parameters
- Each nuisance controls rotation in a specific plane

**Mixed mode:**
- Random sparse coupling (30% connectivity)
- Fixed seed for reproducibility
- Good for general testing

### Bounds Selection

Nuisance parameter bounds are set to:
```
[mean - k*sigma, mean + k*sigma]
```
where `k = nuisance_bounds_sigma_multiple` (default: 5.0)

This covers:
- k=3: 99.7% probability (3σ)
- k=5: 99.9999% probability (5σ, default)

### Computational Considerations

**Evaluation Cost:**
- Wrapper adds negligible overhead (<1% typically)
- Main cost is increased dimensionality
- Emulator pre-screening helps significantly

**Memory:**
- No significant increase in memory usage
- Evaluation history scales with total dimensions

**Convergence:**
- More dimensions → more generations needed
- Tight constraints help (nuisance space is easier)
- Initial optimization phase is critical

## Future Extensions

Potential enhancements:

1. **Correlated nuisance parameters:**
   - Full covariance matrix for constraints
   - Models correlated systematic uncertainties

2. **Asymmetric constraints:**
   - Different σ for positive/negative deviations
   - Common in physics (e.g., theory uncertainties)

3. **Non-Gaussian constraints:**
   - Log-normal for positive-definite parameters
   - Custom distributions from experimental data

4. **Adaptive coupling:**
   - Coupling strength depends on POI values
   - Models state-dependent systematics

## References

For physics context on nuisance parameters:

- Cowan et al., "Asymptotic formulae for likelihood-based tests of new physics", Eur. Phys. J. C (2011)
- Read, "Presentation of search results: CLs technique", J. Phys. G (2002)
- Cranmer et al., "HistFactory: A tool for creating statistical models", CERN-OPEN-2012-016

## Summary

The nuisance parameter framework provides:
- **Realistic test scenarios** mimicking physics analyses
- **Flexible coupling modes** representing different systematics
- **Scalable difficulty** for comprehensive testing
- **Physics interpretation** for meaningful validation

Use this framework to test ParaProf's performance on high-dimensional, constrained optimization problems typical of modern physics analyses.
