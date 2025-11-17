# Nuisance Parameter Example - Bug Fix

## Issue

The `run_nuisance_example.py` script had an `AttributeError` at the end when trying to access `sampler.global_best_params`, which doesn't exist.

```python
AttributeError: 'GridAnchoredDESampler' object has no attribute 'global_best_params'
```

## Root Cause

The `GridAnchoredDESampler` class stores best solutions in the `global_solution_pool` list, not in single `global_best_params` and `global_max` attributes.

## Fix

Changed the code to correctly access the best solution from the sampler:

**Before:**
```python
# Get best point from sampler
best_params = sampler.global_best_params  # ❌ Doesn't exist
best_ll = sampler.global_max              # ❌ Doesn't exist
```

**After:**
```python
# Get best point from sampler's global solution pool
# Pool is sorted by fitness (best first)
best_solution = sampler.global_solution_pool[0]  # ✅ First element is best
best_params = best_solution['full_params']       # ✅ Full parameter vector
best_ll = sampler.global_max_target_val          # ✅ Correct attribute name
```

## Sampler's Global Solution Storage

The `GridAnchoredDESampler` class maintains:

1. **`global_solution_pool`** (list of dicts):
   - Sorted by fitness (best first)
   - Each entry contains:
     - `'full_params'`: Complete parameter vector
     - `'fitness'`: Fitness value (same as log-likelihood)
     - `'grid_idx'`: Grid point index (or None for global optimizations)
   - Maximum size controlled by `global_pool_size` parameter
   - Automatically pruned to keep top solutions

2. **`global_max_target_val`** (float):
   - Single best fitness value found across all evaluations
   - Updated whenever a better solution is found
   - Used for ROI threshold calculations

## Usage Pattern

To access the best solution found by ParaProf:

```python
# Check that solutions were found
if sampler.global_solution_pool:
    # Get best solution (pool is sorted, best first)
    best_solution = sampler.global_solution_pool[0]

    # Extract parameters and fitness
    best_params = best_solution['full_params']
    best_fitness = best_solution['fitness']
    best_grid_idx = best_solution['grid_idx']

    # Alternative: use the global max value directly
    global_max = sampler.global_max_target_val
```

For nuisance parameter analysis:

```python
# Split parameters into POI and nuisance
n_poi = 4
n_nuisance = 8

best_params = sampler.global_solution_pool[0]['full_params']
best_poi = best_params[:n_poi]
best_nuis = best_params[n_poi:]

# Analyze nuisance parameter behavior
deviations = (best_nuis - nuisance_mean) / constraint_sigma
```

## File Updated

- `examples/run_nuisance_example.py` (lines 190-198)

## Status

✅ **Fixed** - The script now correctly accesses the sampler's best solution.

## Related

The results dictionary from `run_all_projections()` includes the global maximum:

```python
results = run_all_projections(comm, sampler, projections, ...)

# Access via results
global_max = results[0]['metrics']['global_max']  # From first projection

# Or directly from sampler
global_max = sampler.global_max_target_val
```

Both approaches are valid. The example script now uses direct sampler access for consistency.
