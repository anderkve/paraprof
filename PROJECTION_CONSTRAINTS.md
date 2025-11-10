# Projection Modes and Configurations for ParaProf

## Two Operating Modes

**ParaProf now supports TWO modes depending on your projection configuration:**

### 1. Normal Mode (Profile Likelihood with Optimization)
**When:** `len(projection_dims) < total_dimensions` (has continuous dimensions)
- Projects onto fewer dimensions than the function has
- Optimizes continuous dimensions at each grid point using DE + L-BFGS-B
- **Use for:** Profile likelihood computations, parameter inference

### 2. Direct Evaluation Mode (NEW!)
**When:** `len(projection_dims) == total_dimensions` (no continuous dimensions)
- Projects onto ALL dimensions of the function
- Evaluates directly at grid points (no optimization)
- Uses intelligent sparse grid activation for efficiency
- **Use for:** 2D likelihood surface visualization, efficient 2D scanning

## Mode Selection

**The mode is automatically detected based on your projection configuration:**

```python
if len(projection_dims) < total_dimensions:
    # Normal Mode: Has continuous dimensions to optimize
    mode = "Profile Likelihood with Optimization"
elif len(projection_dims) == total_dimensions:
    # Direct Evaluation Mode: No continuous dimensions
    mode = "Direct Evaluation at Grid Points"
```

When direct evaluation mode activates, you'll see:
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  DIRECT EVALUATION MODE
  ----------------------------------------------------------------------------
  Grid dimensionality equals function dimensionality (2D)
  No continuous dimensions to optimize - will evaluate at grid points directly
  Workflow: Initial optimization → Sparse grid activation → Direct evaluation
  Skipping: DE optimization and L-BFGS-B refinement stages
  Benefit: Efficient sparse likelihood map vs. dense grid scanning
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Valid Projection Configurations

### Rule
```
len(projection_dims) < total_dimensions
```

### Examples by Dimensionality

#### 2D Functions (beale_2d, eggholder_2d, sphere_2d, etc.)

| Configuration | Mode | Explanation |
|---------------|------|-------------|
| `dims=[0]` | ✓ Normal | 1D projection, optimizes dimension 1 |
| `dims=[1]` | ✓ Normal | 1D projection, optimizes dimension 0 |
| `dims=[0, 1]` | ✓ **Direct Eval** | 2D projection, direct evaluation (NEW!) |

**Example Configurations:**

**Option 1: Direct Evaluation Mode (NEW!)**
```python
# Projects onto both dimensions - evaluates at grid points directly
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [75, 75]},  # Direct evaluation mode
]
```

**Option 2: Normal Mode (with optimization)**
```python
# 1D projection - optimizes the other dimension at each point
PROJECTIONS_TO_RUN = [
    {'dims': [0], 'grid_points': [75]},  # Normal mode
]
```

#### 4D Functions (rosenbrock_4d, himmelblau_4d, rastrigin_4d, etc.)

| Configuration | Status | Explanation |
|---------------|--------|-------------|
| `dims=[0]` | ✓ Valid | 1D projection, optimizes [1, 2, 3] |
| `dims=[0, 1]` | ✓ Valid | 2D projection, optimizes [2, 3] |
| `dims=[0, 1, 2]` | ✓ Valid | 3D projection, optimizes [3] |
| `dims=[0, 1, 2, 3]` | ✗ **INVALID** | No continuous dimensions left! |

**Example Configuration:**
```python
# CORRECT: 2D projection for 4D function
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [75, 75]},
]
```

#### 6D Functions (rosenbrock_6d, rastrigin_6d, etc.)

| Configuration | Status | Explanation |
|---------------|--------|-------------|
| `dims=[0, 1]` | ✓ Valid | 2D projection, optimizes [2, 3, 4, 5] |
| `dims=[0, 1, 2]` | ✓ Valid | 3D projection, optimizes [3, 4, 5] |
| `dims=[0, 1, 2, 3, 4]` | ✓ Valid | 5D projection, optimizes [5] |
| `dims=[0, 1, 2, 3, 4, 5]` | ✗ **INVALID** | No continuous dimensions left! |

**Example Configuration:**
```python
# CORRECT: 2D projection for 6D function
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [50, 50]},
]
```

## Error Messages

### Invalid Configuration Error

If you try to project onto all dimensions, you'll get:

```
ValueError: Invalid projection configuration: projection_dims=[0, 1]
uses all 2 dimensions. ParaProf requires at least 1 continuous
dimension to optimize. For a 2D function, use at most 1 projection
dimensions. Example: for 2D functions use dims=[0] or dims=[1],
not dims=[0,1].
```

## Recommended Projection Configurations

### For Visualization and Testing

| Function Dimensionality | Recommended Projection | Grid Size | Purpose |
|------------------------|----------------------|-----------|---------|
| 2D | `dims=[0]` or `dims=[1]` | 75-150 | 1D profile plot |
| 4D | `dims=[0, 1]` | 50-75 per dim | 2D contour plot |
| 6D | `dims=[0, 1]` | 40-60 per dim | 2D contour plot |
| 10D | `dims=[0, 1]` | 30-50 per dim | 2D contour plot |

### Multiple Projections

You can run multiple projections for different dimension pairs:

```python
# For 4D function: explore multiple 2D projections
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [50, 50]},  # Project on dims 0-1
    {'dims': [0, 2], 'grid_points': [50, 50]},  # Project on dims 0-2
    {'dims': [1, 2], 'grid_points': [50, 50]},  # Project on dims 1-2
]
```

## Common Use Cases

### Use Case 1: 2D Likelihood Surface Visualization

✓ **Direct Evaluation Mode (NEW!):**
```python
TEST_FUNCTION = "beale_2d"
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [75, 75]},  # Direct eval - efficient!
]
```
- Evaluates at sparse grid points (intelligently activated)
- Much faster than dense grid scanning
- Produces full 2D likelihood surface

### Use Case 2: Profile Likelihood with Optimization

✓ **Normal Mode:**
```python
TEST_FUNCTION = "beale_2d"
PROJECTIONS_TO_RUN = [
    {'dims': [0], 'grid_points': [75]},  # Profile likelihood of dim 0
]
```
- Optimizes dimension 1 at each point on dimension 0 grid
- True profile likelihood computation

### Mistake 2: Projecting onto all dimensions

❌ **WRONG:**
```python
TEST_FUNCTION = "sphere_4d"  # 4D function
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1, 2, 3], 'grid_points': [20, 20, 20, 20]},  # ERROR!
]
```

✓ **CORRECT:**
```python
TEST_FUNCTION = "sphere_4d"  # 4D function
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1, 2], 'grid_points': [20, 20, 20]},  # 3D projection
]
```

## Quick Reference Table

| Function Dims | Max Projection Dims | Example Valid Config |
|---------------|--------------------|--------------------|
| 2 | 1 | `dims=[0]` |
| 3 | 2 | `dims=[0, 1]` |
| 4 | 3 | `dims=[0, 1, 2]` |
| 5 | 4 | `dims=[0, 1, 2, 3]` |
| 6 | 5 | `dims=[0, 1, 2, 3, 4]` |
| 10 | 9 | `dims=[0, 1, 2, 3, 4, 5, 6, 7, 8]` |

## Testing Your Configuration

Use this quick test to verify your projection is valid:

```python
from test_functions import get_test_function
from sampler import GridAnchoredDESampler

# Load your test function
func, bounds, peaks = get_test_function("your_function_name")

# Try to create sampler - will raise ValueError if invalid
try:
    sampler = GridAnchoredDESampler(
        target_func=func,
        bounds=bounds,
        projections=[
            {'dims': [0], 'grid_points': [10]}  # Your config
        ]
    )
    print("✓ Configuration is valid!")
    print(f"  Projection dims: {sampler.projection_dims}")
    print(f"  Continuous dims: {sampler.continuous_dims}")
except ValueError as e:
    print("✗ Invalid configuration!")
    print(f"  Error: {e}")
```

## Understanding Profile Likelihoods

Profile likelihood for parameters θ = (θ_p, θ_c):
- θ_p: projection parameters (fixed on grid)
- θ_c: continuous parameters (optimized)

For each grid point, compute:
```
PL(θ_p) = max_{θ_c} L(θ_p, θ_c)
```

**This requires continuous parameters θ_c to optimize!**

## Summary

### Two Modes, More Flexibility

**Normal Mode** (`len(projection_dims) < total_dimensions`):
- ✓ Profile likelihood computation with optimization
- ✓ For 2D functions: Use 1D projections (`dims=[0]` or `dims=[1]`)
- ✓ For 4D+ functions: Use 2D projections (`dims=[0, 1]`)

**Direct Evaluation Mode** (`len(projection_dims) == total_dimensions`) - NEW!:
- ✓ Efficient 2D surface visualization
- ✓ For 2D functions: Use 2D projections (`dims=[0, 1]`)
- ✓ Sparse grid activation reduces evaluations vs. dense scanning
- ✓ No optimization overhead - perfect for visualization

### Key Benefits

1. **Flexibility**: Choose the mode that fits your needs
2. **Efficiency**: Direct eval mode uses sparse grids intelligently
3. **Simplicity**: Mode is automatically detected
4. **Backward Compatible**: All existing code still works

---

**Updated:** 2025 (with Direct Evaluation Mode and extended test function suite)
