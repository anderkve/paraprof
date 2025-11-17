# Nuisance Parameters - Quick Start Guide

## TL;DR

Add constrained nuisance parameters to any test function in 3 lines:

```python
from paraprof import get_test_function
from paraprof.nuisance_wrapper import create_nuisance_wrapped_function

base_func, base_bounds, peaks = get_test_function("himmelblau_4d")
wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func, base_bounds, n_poi=4, n_nuisance=8,
    coupling_mode='shift', constraint_sigma=0.5
)
# Now use wrapped_func with ParaProf as usual!
```

## What This Does

Transforms a simple test function into a realistic physics-like scenario:
- **Before**: 4D Himmelblau function
- **After**: 12D function (4 POI + 8 nuisance) where:
  - The 4 POI have the Himmelblau structure (wide range, multimodal)
  - The 8 nuisance params are Gaussian-constrained (tight, near zero)
  - Nuisance params shift the POI values before evaluation

## Key Parameters

```python
create_nuisance_wrapped_function(
    base_func,           # Your test function
    base_bounds,         # POI bounds
    n_poi,              # Number of parameters of interest
    n_nuisance,         # Number of nuisance parameters (can be >> n_poi)
    coupling_mode,      # How nuisance affects POI: 'shift', 'scale', 'rotation', 'additive'
    constraint_sigma,   # Constraint width (0.2=tight, 1.0=loose)
)
```

## Coupling Modes Explained

| Mode | Transform | Physics Example |
|------|-----------|----------------|
| `'shift'` | `x → x + d` | Detector calibration offsets |
| `'scale'` | `x → x * (1+s)` | Luminosity normalization |
| `'rotation'` | `x → R(θ)·x` | Coordinate system alignment |
| `'additive'` | No coupling | Baseline (constraint only) |

## Usage with ParaProf

**Important**: Only project over POI dimensions!

```python
# Define projections using POI indices only (not nuisance)
PROJECTIONS = [
    {'dims': [0, 1], 'grid_points': [50, 50], ...},  # First 2 POI
]

sampler = GridAnchoredDESampler(
    target_func=wrapped_func,      # Use wrapped function
    bounds=wrapped_bounds,         # Use wrapped bounds (POI + nuisance)
    projections=PROJECTIONS,       # Only POI dimensions
    # ... other settings
)
```

The algorithm automatically optimizes over all nuisance parameters while projecting the POI.

## Running the Example

```bash
mpiexec -n 8 python examples/run_nuisance_example.py
```

This runs a complete profile likelihood analysis with:
- 4 POI (Himmelblau structure)
- 8 nuisance parameters (shift coupling, σ=0.5)
- 1D and 2D projections over POI
- Automatic nuisance parameter profiling

## Common Use Cases

### Many Weakly-Coupled Nuisance Parameters
```python
# Typical physics scenario: 4 POI, 20+ nuisance
wrapped_func, bounds, wrapper = create_nuisance_wrapped_function(
    base_func, base_bounds, n_poi=4, n_nuisance=24,
    coupling_mode='shift', constraint_sigma=0.3
)
```

### Tight vs Loose Constraints
```python
# Test algorithm performance with different constraint strengths
tight = create_nuisance_wrapped_function(..., constraint_sigma=0.1)    # 10σ = 5% prob
medium = create_nuisance_wrapped_function(..., constraint_sigma=0.5)   # More realistic
loose = create_nuisance_wrapped_function(..., constraint_sigma=2.0)    # Harder problem
```

### Different Coupling Mechanisms
```python
# Calibration shifts (additive systematics)
shift_func, _, _ = create_nuisance_wrapped_function(
    base_func, base_bounds, n_poi=4, n_nuisance=8,
    coupling_mode='shift', constraint_sigma=0.3
)

# Efficiency uncertainties (multiplicative systematics)
scale_func, _, _ = create_nuisance_wrapped_function(
    base_func, base_bounds, n_poi=4, n_nuisance=4,
    coupling_mode='scale', constraint_sigma=0.1
)
```

## Pre-Registered Test Functions

For quick testing, use pre-configured scenarios:

```python
from paraprof.nuisance_wrapper import register_nuisance_wrapped_test_functions

registry = register_nuisance_wrapped_test_functions()
func, bounds, wrapper, peaks = registry['himmelblau_4d_shift_8nuis_sigma0.5']
```

Available configurations:
- `himmelblau_4d_shift_4nuis_sigma0.5`
- `himmelblau_4d_shift_8nuis_sigma0.5`
- `himmelblau_4d_shift_16nuis_sigma0.5`
- `himmelblau_4d_shift_8nuis_sigma0.2` (tight)
- `himmelblau_4d_shift_8nuis_sigma1.0` (loose)
- `rosenbrock_4d_shift_8nuis_sigma0.5`
- `rastrigin_4d_shift_8nuis_sigma0.5`
- And more...

## Validation Tips

After running ParaProf, check nuisance parameter behavior:

```python
# Get best-fit parameters
best_params = sampler.global_best_params
best_poi = best_params[:n_poi]
best_nuis = best_params[n_poi:]

# Check nuisance parameter pulls (should be << 1σ for well-behaved)
deviations = (best_nuis - wrapper.nuisance_mean) / wrapper.constraint_sigma
print(f"Nuisance pulls (σ): {deviations}")

# Compare to analytical optimum
optimal_nuis = wrapper.get_optimal_nuisance(best_poi)
difference = np.linalg.norm(best_nuis - optimal_nuis)
print(f"Distance from analytical optimum: {difference}")
```

## Performance Tips

With nuisance parameters:
1. **Use emulator pre-screening**: `use_de_prescreening=True` (saves 30-50% evaluations)
2. **Increase initial optimizations**: `n_initial_optimizations=100-200`
3. **Consider tighter convergence**: High-dimensional spaces need more generations
4. **Monitor nuisance behavior**: If pulls are large, constraint_sigma may be too tight

## Summary

The nuisance parameter framework lets you:
- ✅ Test ParaProf on realistic high-dimensional problems
- ✅ Model physics-like constrained parameters
- ✅ Control problem difficulty systematically
- ✅ Validate algorithm behavior on known structures

For complete details, see `NUISANCE_PARAMETERS.md`.
