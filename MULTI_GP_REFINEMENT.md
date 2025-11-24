# Multi-GP Refinement Method

## Overview

The multi-GP refinement method provides an improved interpolation approach for grid refinement that uses Gaussian Process regression to predict optimal continuous parameters at fine grid points.

## Key Concept

Instead of modeling the full high-dimensional likelihood function, we train **one low-dimensional GP per continuous parameter** to predict optimal values as a function of projection coordinates only:

- `θ_cont[0] = f₀(θ_projection)`
- `θ_cont[1] = f₁(θ_projection)`
- ...

For a 2D projection with 2 continuous dimensions, this means training 2 separate 2D GPs instead of one 4D GP.

## Advantages Over Linear Interpolation

1. **Better predictions**: Captures non-linear variation in optimal parameters
2. **Uncertainty quantification**: GP provides prediction uncertainties for adaptive refinement
3. **Robust to noise**: Handles sparse/noisy coarse grid data better
4. **Zero extra cost**: Uses existing coarse grid optima as training data

## Usage

### Basic Example

```python
from paraprof import GridAnchoredDESampler, run_projection

projection_config = {
    'dims': [0, 1],
    'grid_points': [50, 50],
    'enable_refinement': True,
    'refinement_factor': 2,
    'refinement_method': 'multi_gp',  # Use multi-GP instead of 'linear'
}

sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=[projection_config],
    # ... other parameters
)

results = run_projection(
    comm=comm,
    sampler=sampler,
    projection_config=projection_config,
    save_plots=True
)
```

### Configuration Options

The `refinement_method` can be specified per projection:

- `'linear'` (default): Fast linear interpolation
- `'multi_gp'`: Gaussian Process interpolation (requires scikit-learn)

### Requirements

The multi-GP method requires scikit-learn:

```bash
pip install scikit-learn
```

Or install paraprof with the `emulator` extra:

```bash
pip install -e ".[emulator]"
```

## How It Works

### Training Phase (After Coarse Grid Convergence)

1. Extract optimal continuous parameters from all coarse ROI grid points
2. Train one GP per continuous dimension using projection coordinates as inputs
3. Each GP learns: `optimal_θ_cont[i] = f_i(θ_projection)`

### Refinement Phase (For Each Fine Grid Point)

1. Query all GPs with fine grid projection coordinates
2. Get predicted optimal continuous parameters + uncertainties
3. Evaluate true likelihood at predicted point
4. Optionally run L-BFGS-B refinement (especially if GP uncertainty is high)

## Performance Characteristics

### When Multi-GP Excels

- **Non-linear likelihood surfaces** where optimal parameters vary smoothly
- **Moderate dimensionality** (2-6 continuous dimensions)
- **Higher refinement factors** (3x, 4x) with many fine points
- **Complex correlation structure** between parameters

### When Linear May Be Sufficient

- Nearly linear likelihood surfaces
- Very simple parameter relationships
- Small refinement factor (2x)
- Speed is critical over accuracy

## Implementation Details

### GP Configuration

The `MultiGPInterpolator` class supports:

- **Kernel types**: RBF (smooth) or Matérn (robust, default)
- **Length scale**: Auto-estimated from grid spacing or manually specified
- **Noise level**: Regularization for numerical stability
- **Input normalization**: Automatic scaling to [0, 1] for stability

### Training Statistics

The interpolator logs training metrics:

```
MultiGPInterpolator: Trained 2 GPs on 185 coarse grid points
  GP 0: log_marginal_likelihood=-45.32, y_range=[-1.234e+00, 2.345e+00]
  GP 1: log_marginal_likelihood=-38.91, y_range=[-5.678e-01, 4.567e-01]
```

### Uncertainty-Based Refinement

The GP provides uncertainty estimates that could be used for adaptive refinement:

```python
# Example (not yet implemented in refinement workflow)
cont_params, uncertainties = interpolator.interpolate(grid_coords)

if max(uncertainties) > threshold:
    # High uncertainty → run full L-BFGS-B refinement
    run_lbfgsb = True
else:
    # Low uncertainty → trust GP prediction
    run_lbfgsb = False
```

## Testing

### Unit Test

```bash
python test_multigp_simple.py
```

Tests the `MultiGPInterpolator` class with synthetic data.

### Comparison Test

```bash
OMP_NUM_THREADS=1 mpiexec -n 2 python test_multi_gp_refinement.py
```

Runs two identical projections with different refinement methods and compares:
- Total evaluations
- Refinement stage efficiency
- Final likelihood accuracy

## Limitations

1. **Multi-modal surfaces**: If optimal continuous parameters "jump" between modes at nearby grid points, GPs will interpolate between modes and may give poor predictions. Always run L-BFGS-B refinement as a safety net.

2. **Discontinuities**: GPs assume smooth functions. Discontinuous optimal parameters will have high prediction uncertainty near boundaries.

3. **Dimensionality**: While better than full-dimensional GPs, training still scales as O(n³) where n is the number of coarse grid points. For very large coarse grids (>1000 points), consider using sparse GP approximations.

## Technical Details

### Class: `MultiGPInterpolator`

Location: `src/paraprof/interpolation.py`

**Constructor parameters:**
- `coarse_grid_solution`: Dictionary from `sampler.export_grid_solution()`
- `kernel_type`: `'rbf'` or `'matern'` (default)
- `length_scale`: Initial kernel length scale (auto-estimated if None)
- `noise_level`: Regularization parameter (default: 1e-5)
- `normalize_inputs`: Scale inputs to [0,1] (default: True)

**Key methods:**
- `interpolate(projection_coords, return_std=True)`: Predict continuous parameters
- `get_max_uncertainty(projection_coords)`: Get maximum prediction uncertainty
- `get_coverage_fraction()`: Fraction of coarse grid with valid solutions

## Future Enhancements

Potential improvements:

1. **Adaptive L-BFGS-B**: Use GP uncertainty to decide whether to run refinement
2. **Parallel GP training**: Distribute GP training across MPI workers
3. **Sparse GPs**: Use inducing points for very large coarse grids
4. **ARD kernels**: Automatic Relevance Determination to identify important dimensions
5. **Multi-fidelity**: Combine coarse and fine grid evaluations in hierarchical GP

## References

The multi-GP approach is inspired by:
- Surrogate modeling in expensive black-box optimization
- Multi-output GP regression with independent outputs
- Profile likelihood computation in statistical inference

## See Also

- `src/paraprof/interpolation.py`: Implementation
- `GridInterpolator`: Linear interpolation baseline
- Grid refinement documentation in README.md
