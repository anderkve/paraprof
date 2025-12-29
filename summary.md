# ParaProf: Parallel Profile Likelihood Computation

## Overview

**ParaProf** is a high-performance Python library for computing multi-dimensional profile likelihood projections using parallelized grid-based optimization. The library implements a novel grid-anchored optimization approach that combines differential evolution, gradient-based methods, and adaptive sampling strategies to efficiently explore complex likelihood surfaces in high-dimensional parameter spaces.

**Author:** Anders Kvellestad
**License:** MIT
**Language:** Python 3.10+
**Core Dependencies:** NumPy, SciPy, MPI4py
**Repository:** https://github.com/anderkve/paraprof

## Scientific Context and Applications

Profile likelihood is a fundamental tool in statistical inference, particularly in physics, statistics, and machine learning for:

- **Parameter estimation** with nuisance parameters
- **Confidence interval construction** via likelihood ratio tests
- **Model comparison** and hypothesis testing
- **High-dimensional optimization** with constraints

The profile likelihood for parameters of interest (POI) $\theta$ given nuisance parameters $\nu$ is:

$$
\mathcal{L}_{\text{prof}}(\theta) = \max_{\nu} \mathcal{L}(\theta, \nu)
$$

Computing this requires optimization over nuisance parameters at each point in the POI space, making it computationally expensive for high-dimensional problems.

## Novel Algorithmic Contributions

### 1. Grid-Anchored Differential Evolution

ParaProf introduces a **grid-anchored population strategy** that differs from standard differential evolution:

- **Spatial decomposition:** Parameters are split into projection dimensions (gridded) and continuous dimensions (optimized)
- **Population anchoring:** Each grid point maintains its own DE population, preventing premature convergence to single modes
- **Cross-grid mutation:** Populations can exchange information through neighbor-based mutation operators with probability $p_{\text{neighbor}}$ (default 0.5)
- **Adaptive activation:** Grid points are dynamically activated when neighbors enter the region of interest (ROI), defined by $\Delta \chi^2 < \text{threshold}$ (default 3.0)

This architecture enables exploration of multi-modal likelihood surfaces without populations collapsing to a single mode, a common problem in global optimization.

### 2. Emulator-Enhanced Sampling (30-50% Reduction in Evaluations)

A key innovation is the integration of **Gaussian Process (GP) emulators** for trial pre-screening in differential evolution:

**Algorithm:**
1. Build local GP emulator from nearby evaluations using Matérn kernel with $\nu = 1.5$
2. For each DE trial point $x_{\text{trial}}$, predict fitness $\mu(x)$ and uncertainty $\sigma(x)$
3. Compute Upper Confidence Bound: $\text{UCB}(x) = \mu(x) + \beta \sigma(x)$ (default $\beta = 2.0$)
4. Screen out trial if $\text{UCB}(x_{\text{trial}}) < f(x_{\text{current}})$
5. Only evaluate surviving trials

**Key features:**
- Input standardization (z-score normalization) for each local emulator
- Local cache gathering from grid point and neighbors for efficiency
- Dynamic emulator building on workers (parallel GP fitting)
- Maintains solution quality (max likelihood difference $< 10^{-6}$)

**Performance:** Reduces likelihood evaluations by 30-50% on standard benchmarks while preserving solution quality.

### 3. Wave-Based Patching Algorithm

The **patching algorithm** propagates gradient information across the grid to escape local optima:

**Algorithm (Wave $k$):**
1. For each grid point $i$ with neighbors $\mathcal{N}(i)$
2. Compute gradient estimate: $g_i = \nabla_{\nu} \mathcal{L}(\theta_i, \nu_i^*)$
3. Test improved point: $\nu_{\text{test}} = \nu_i^* + \alpha \cdot \text{neighbor\_gradient}$
4. If improvement found, spawn L-BFGS-B refinement job
5. Track updated points $U_k$ in wave $k$
6. Use $U_k$ as seeds for wave $k+1$
7. Terminate when $U_k = \emptyset$ or $k > k_{\max}$ (default $k_{\max} = 10$)

This creates wave-like propagation of improvements across the grid, effectively performing gradient-based refinement in a spatially-aware manner.

### 4. Cluster-Aware Grid Refinement

For grid refinement (increasing resolution), ParaProf employs **mode clustering** to handle multi-modal likelihood surfaces:

**Algorithm:**
1. Cluster coarse grid points by optimal continuous parameters using DBSCAN
2. Features: continuous parameters + projection coordinates (weighted)
3. Auto-estimate DBSCAN $\epsilon$ from k-nearest-neighbor distances
4. Identify boundary points where clusters meet
5. For fine grid points near boundaries:
   - Generate multiple candidate initializations (one per nearby cluster)
   - Evaluate all candidates, select best
6. For interior points: use standard linear interpolation

**DBSCAN configuration:**
- $\epsilon = \text{percentile}_{90}(d_{k\text{-nn}}) \times \alpha$ where $\alpha = 3.0$ (default)
- $\text{min\_samples} = \max(2, n_{\text{continuous}})$
- Includes projection coordinates with weight 1.0 for spatial context

This prevents incorrect interpolation across mode boundaries, which would otherwise seed optimization in poor regions.

### 5. Adaptive Memory Mechanisms

**Differential Evolution Memory:**
- Success-History based Adaptive DE (SHADE) inspired approach
- Memory pools of size $25 \times \max(\text{grid\_sizes})$ store successful $(F, CR)$ values
- Sampling: $F \sim \text{Cauchy}(\mu_F, 0.1)$, $CR \sim \mathcal{N}(\mu_{CR}, 0.1)$
- Memory updated with successful mutations each generation

**Global Solution Pool:**
- Maintains top 10,000 (configurable) evaluated points across all projections
- Enables warm-starting subsequent projections
- Used for activation initialization: 50% neighbors + 25% global + 25% random (default)

## Architecture and Implementation

### Master-Worker Parallelization

**MPI-based asynchronous task distribution:**
- Master (rank 0): State machine managing workflow, job queues, convergence tracking
- Workers (rank 1+): Stateless function evaluators with optional GP emulator building
- Two-priority task queues: high (L-BFGS-B, patching) and low (DE, CMA-ES)
- Non-blocking sends with `MPI.isend()` for communication overlap
- `MPI.Iprobe()` polling loop for result collection

**Job System:**
Object-oriented job framework with inheritance:
- `Job` (base): Lifecycle management, state tracking
- `DEGridPointJob`: Single DE generation for one grid point
- `LBFGSBJob`: Gradient-based local optimization
- `ActivationJob`: Initialize new grid point populations
- `PatchingTestJob`: Test gradient-based improvements
- `CMAESJob`: CMA-ES evolution for grid points
- `CoordinateDescentJob`: Axis-aligned local search

Jobs can spawn new jobs (e.g., DE convergence → L-BFGS-B refinement).

### Workflow State Machine

**Standard Run:**
```
INITIAL_OPTIMIZATION → ACTIVATION → {DE_LOOP | LBFGSB_LOOP | CMAES_LOOP} → PATCHING_WAVES
```

**Refinement Run:**
```
REFINEMENT_LBFGSB → [PATCHING_WAVES]
```

Each stage completes before transitioning. Looping stages (DE_LOOP, LBFGSB_LOOP, CMAES_LOOP) iterate until convergence.

### Data Structures

**Population State (per grid point):**
```python
{
    'continuous_params': np.ndarray,  # (pop_size, n_cont_dims)
    'fitnesses': np.ndarray,          # (pop_size,)
    'best_fitness': float,
    'status': str,                     # 'active' | 'converged' | 'optimized'
    'improvement_history': deque,      # Rolling window for convergence
    'last_update_gen': int
}
```

**Evaluation Caches:**
- **Local caches:** Per-grid-point evaluation history for local GP emulators
- **Global cache:** Pruned best + recent evaluations (LRU-like with quality retention)
- **Samples buffer:** Batched CSV writing to reduce I/O overhead

## Optimization Methods

ParaProf supports three primary optimization engines:

### 1. Differential Evolution (DE)
- **Mutation strategies:** `current-to-pbest/1` (default), `rand/1`, `current-to-rand/1`
- **Neighbor-aware mutation:** Incorporates best neighbor solutions with probability 0.5
- **Convergence:** Rolling window (default 3 generations) with improvement threshold $< 10^{-3} \times \text{ROI\_threshold}$
- **Generations:** Default 100,000 (typically converges much earlier)

### 2. L-BFGS-B
- **Gradient approximation:** Forward differences (configurable)
- **Tolerance:** $f_{\text{tol}} = 10^{-9}$ (default)
- **Max iterations:** 50 (default)
- **Use cases:** Initial optimization, refinement, polishing after DE/CMA-ES

### 3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- **Population size:** $\lambda = 4 + \lfloor 3 \log(n) \rfloor$ where $n = n_{\text{continuous}}$
- **Parent size:** $\mu = \lfloor \lambda / 2 \rfloor$
- **Max generations per run:** 100 (default)
- **Iterative approach:** Similar to DE_LOOP, with dynamic activation

### 4. Coordinate Descent (for refinement)
- **Fast alternative** to L-BFGS-B for grid refinement
- **Cycles:** Max 3 (default)
- **Step size:** 1% of parameter bounds (default)
- **Use case:** Quick refinement when gradient computation is expensive

## Interpolation and Refinement

**Linear Interpolation:**
- Uses `scipy.interpolate.RegularGridInterpolator`
- Separate interpolator per continuous parameter
- Nearest-neighbor extrapolation beyond grid bounds
- Likelihood interpolation for visualization

**Refinement Strategy:**
1. Export coarse grid solution (grid axes, continuous params, likelihoods)
2. Build `GridInterpolator` from coarse solution
3. Optionally detect mode clusters and boundaries
4. For each fine grid point:
   - Standard region: interpolate continuous parameters
   - Boundary region: generate multiple cluster-based candidates
5. Optimize from interpolated/candidate starting points
6. Optional: Apply patching on refined grid

**Speedup:** Refinement by factor $r$ typically requires $\mathcal{O}(r)$ evaluations vs $\mathcal{O}(r^2)$ for direct fine grid in 2D.

## Nuisance Parameter Framework

ParaProf includes a sophisticated **nuisance parameter wrapper** for testing:

**Augmented Likelihood:**
$$
\log \mathcal{L}_{\text{total}}(\theta, \nu) = \log \mathcal{L}_{\text{base}}(T(\theta, \nu)) + \sum_i \log \mathcal{L}_{\text{constraint}}(\nu_i)
$$

**Coupling Modes:**
1. **Shift:** $\theta'_i = \theta_i + \sum_j M_{ij} \nu_j$ (systematic shifts)
2. **Scale:** $\theta'_i = \theta_i (1 + \sum_j M_{ij} \nu_j)$ (normalization uncertainties)
3. **Rotation:** Small rotations in parameter space (detector misalignment)
4. **Additive:** No coupling, pure penalty term
5. **Mixed:** Custom linear combinations via matrix $M$

**Constraint Types:**
- **Gaussian:** $\log \mathcal{L} = -\frac{1}{2}\left(\frac{\nu - \mu}{\sigma}\right)^2$
- **Uniform:** Flat within $\pm \sigma$, $-\infty$ outside
- **Soft uniform:** Flat within $\pm \sigma$, quadratic penalty outside

**Example:** Himmelblau 4D + 8 shift nuisance parameters creates a 12D problem where the first 4 parameters are profiled and the last 8 are optimized at each grid point with Gaussian constraints.

## Performance Characteristics

**Benchmark Results (Himmelblau 4D, 2D projection):**
- Grid sizes: 10×10 to 30×30
- Scaling: Approximately $\mathcal{O}(N^2)$ evaluations for $N \times N$ grid
- Parallel efficiency: Linear with number of workers (up to ~16 workers)
- Emulator speedup: 1.3-2.0× (30-50% fewer evaluations)

**Convergence:**
- DE typically converges in 10-100 generations per grid point
- L-BFGS-B refinement: 10-50 iterations
- Patching waves: 1-5 waves typically sufficient

**Memory:**
- Scales with grid size and population size
- Typical: $\mathcal{O}(\text{grid\_points} \times \text{pop\_size} \times n_{\text{continuous}})$
- Global pool: Fixed size (10,000 points default)

## Test Functions and Benchmarking

ParaProf includes 20+ test functions across categories:

**Unimodal:**
- Sphere: Simplest benchmark, convex
- Rosenbrock: Narrow valley, tests exploration in corridors

**Multimodal (few peaks):**
- Himmelblau 4D: 4 equal global optima (combined 2×2D Himmelblau)
- Beale: Steep valleys
- Eggholder: Highly rugged, asymmetric

**Multimodal (many regular peaks):**
- Rastrigin: Regularly spaced local optima, tests basin-hopping
- Ackley: Flat outer region with narrow global basin
- Griewank: Product of cosines creates regularity

**Multimodal (steep/rugged):**
- Michalewicz: Steep valleys, $m=10$ steepness parameter
- Styblinski-Tang: $2^n$ local minima
- Levy: Complex multi-modal structure
- Schwefel: Deceptive surface, global optimum far from center

**Dimensionalities:** 2D, 4D, 6D, 10D variants for scaling studies.

All functions are:
1. Negated for maximization
2. Shifted so global optimum = 0.0 (when known)
3. Scaled to appropriate dynamic range

## Key Configuration Parameters

**Core Parameters:**
- `roi_threshold`: ROI threshold in $\chi^2$ units (default: 3.0)
- `pop_per_grid_point`: DE population size (default: 1-5)
- `max_patching_waves`: Maximum patching iterations (default: 10)
- `n_initial_optimizations`: Global L-BFGS-B runs (default: $\min(100, 20n)$)

**Emulator Parameters:**
- `use_emulator`: Enable GP pre-screening (default: False)
- `emulator_confidence_threshold`: UCB $\beta$ parameter (default: 2.0)
- `emulator_min_neighbors`: Minimum training points (default: 10)
- `emulator_length_scale`: Matérn kernel length scale (default: 1.0)

**DE Parameters:**
- `mutation_strategy`: DE mutation type (default: `current-to-pbest/1`)
- `neighbor_pull_probability`: Use neighbor mutation (default: 0.5)
- `convergence_threshold`: Improvement threshold (default: `roi_threshold/1000`)

**Activation Mix Ratios:**
- Neighbor samples: 50% (default)
- Global pool samples: 25% (default)
- Random (LHS) samples: 25% (default)

## Visualization

**Automatic Plotting:**
- **1D profiles:** Line plots with 68%, 95% confidence bands
- **2D profiles:** Heatmaps with contours, active point markers
- **3D+ profiles:** Pairwise 2D slices through maximum or marginalized

**Continuous Parameter Plots:**
- Shows optimal continuous parameter values across projection space
- 1D: Multi-panel line plots per continuous parameter
- 2D: Heatmap per continuous parameter
- Reveals parameter correlations and likelihood structure

**Output formats:** PNG (default), PDF, SVG (via matplotlib)

## Research Extensions and Future Directions

**Potential Research Directions:**

1. **Hybrid surrogate modeling:** Combine GP emulators with physics-informed neural networks for multi-fidelity optimization

2. **Uncertainty quantification:** Leverage GP predictive variance for adaptive sampling strategies beyond UCB

3. **Multi-objective profiling:** Extend to Pareto-front computation for competing objectives

4. **Adaptive grid coarsening:** Dynamically adjust grid resolution based on local likelihood curvature

5. **Hierarchical Bayesian integration:** Use profile likelihoods as priors in hierarchical models

6. **Quantum optimization backends:** Replace DE/CMA-ES with quantum annealing for discrete parameter spaces

7. **Transfer learning:** Use global pool as pre-training for subsequent likelihood functions

8. **Automatic differentiation:** Replace finite differences with JAX/PyTorch for exact gradients

## Implementation Quality

**Software Engineering:**
- Modern Python packaging (`pyproject.toml`, src layout)
- Type hints and docstrings throughout
- Comprehensive test suite with pytest
- GitHub Actions CI/CD
- Code formatting: Black, linting: Ruff
- Exception hierarchy for error handling

**Testing:**
- Unit tests for core algorithms
- Integration tests for full workflows
- Regression tests for performance
- MPI-aware test fixtures

**Documentation:**
- Extensive inline documentation
- API documentation with examples
- Benchmark suite with performance tracking
- Multiple example scripts

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Parameters of interest (projection dimensions) |
| $\nu$ | Nuisance parameters (continuous dimensions) |
| $\mathcal{L}(\theta, \nu)$ | Joint likelihood function |
| $\mathcal{L}_{\text{prof}}(\theta)$ | Profile likelihood |
| $\Delta \chi^2$ | Likelihood ratio test statistic: $-2\log(\mathcal{L}/\mathcal{L}_{\max})$ |
| $F, CR$ | DE mutation factor and crossover rate |
| $\beta$ | UCB exploration parameter |
| $\epsilon$ | DBSCAN distance parameter |
| $\alpha$ | Learning rate / step size parameter |

## Dependencies and Ecosystem

**Required:**
- `numpy >= 1.21`: Numerical computing
- `scipy >= 1.7`: Optimization, interpolation, statistics
- `mpi4py >= 3.1`: MPI bindings for parallelization

**Optional:**
- `matplotlib >= 3.5`: Visualization
- `scikit-learn >= 1.3.0`: GP emulators, clustering
- `pytest >= 7.0`: Testing framework

**Compatibility:** Linux, macOS, Windows (with MPI installation)

## Conclusion

ParaProf represents a significant contribution to computational statistics and optimization, introducing several novel techniques:

1. **Grid-anchored population management** preventing mode collapse in multi-modal optimization
2. **Emulator-enhanced DE** reducing evaluations by 30-50% with minimal quality loss
3. **Wave-based gradient propagation** for spatially-aware refinement
4. **Cluster-aware interpolation** handling mode boundaries during grid refinement
5. **Comprehensive nuisance parameter framework** for realistic statistical inference scenarios

The codebase demonstrates production-quality software engineering with a clean API, extensive testing, and strong documentation. The modular job system and MPI parallelization make it suitable for both research exploration and large-scale computational campaigns.

**Primary innovation:** The integration of spatial structure (grid anchoring), machine learning (GP emulators), and classical optimization (DE, L-BFGS-B) into a unified framework that outperforms any single technique in isolation.
