# paraprof

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**paraprof** is a robust and MPI-parallelised algorithm for computing profile likelihood projections – or more generally for mapping out low-dimensional projections of a target function by optimizing over the remaining dimensions.

First an initial global optimization identifies promising starting points. These starting points are used to activate initial grid points in the user-selected projection space. At each active grid point the remaining parameters are profiled (optimized) out using differential evolution (DE) and/or L-BFGS-B. If the optimized target value for a grid point meets the user-set threshold, the neighboring grid points are activated. This way the set of optimized grid points dynamically expands to cover the full region of interest, e.g. the high-likelihood region. Information from already converged grid points is communicated to neighboring grid points for faster convergence.


<p align="center">
  <img src="examples/example_plots/animation/paraprof_rosenbrock_himmelblau_4D.gif" alt="ParaProf scanning the 4D Rosenbrock and 4D Himmelblau log-likelihoods" width="600"/>
</p>

## Installation

```bash
pip install git+https://github.com/anderkve/paraprof.git
```

Optional extras: `pip install -e ".[viz]"`, `".[dev]"`, or `".[all]"`.

Requires Python 3.10+, NumPy, SciPy, and mpi4py (with an MPI implementation like OpenMPI or MPICH). Matplotlib and scikit-learn are optional.


## Example output

Below are 1D and 2D projections of three 4D test functions used as example log-likelihood functions. Dimensions not shown on the axes are profiled out. The white star marks the best-fit point; the contours are 68% and 95% confidence regions (assuming Wilks' theorem). Plots come from `examples/run_showcase_scan.py` followed by `examples/make_showcase_plots.py`.

<p align="center">
  <img src="examples/example_plots/showcase/himmelblau_4d_1d.png" alt="Himmelblau 4D 1D profile for x0" width="300"/>
  <img src="examples/example_plots/showcase/himmelblau_4d_2d_a.png" alt="Himmelblau 4D 2D profile for (x0, x1)" width="240"/>
  <img src="examples/example_plots/showcase/himmelblau_4d_2d_b.png" alt="Himmelblau 4D 2D profile for (x0, x2)" width="240"/>
</p>

Himmelblau 4D: 264,617 target-function evaluations across all three projections.

<p align="center">
  <img src="examples/example_plots/showcase/rosenbrock_4d_1d.png" alt="Rosenbrock 4D 1D profile for x0" width="300"/>
  <img src="examples/example_plots/showcase/rosenbrock_4d_2d_a.png" alt="Rosenbrock 4D 2D profile for (x0, x1)" width="240"/>
  <img src="examples/example_plots/showcase/rosenbrock_4d_2d_b.png" alt="Rosenbrock 4D 2D profile for (x1, x3)" width="240"/>
</p>

Rosenbrock 4D: 63,904 evaluations.

<p align="center">
  <img src="examples/example_plots/showcase/levy_4d_1d.png" alt="Levy 4D 1D profile for x0" width="300"/>
  <img src="examples/example_plots/showcase/levy_4d_2d_a.png" alt="Levy 4D 2D profile for (x0, x1)" width="240"/>
  <img src="examples/example_plots/showcase/levy_4d_2d_b.png" alt="Levy 4D 2D profile for (x0, x3)" width="240"/>
</p>

Levy 4D: 117,757 evaluations. The last dimension uses `sin(2π·w₃)` rather than `sin(π·w + 1)`, so the (x₀, x₃) projection has denser horizontal ridges than (x₀, x₁).


## Quick start

```python
from mpi4py import MPI
from paraprof import (
    ProfileProjector, run_all_projections, terminate_workers, worker_main,
    get_test_function,
)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

log_likelihood, bounds, _ = get_test_function("himmelblau_4d")
projections = [
    {'dims': [0, 1], 'grid_points': [50, 50]},
]

if myrank == 0:
    with ProfileProjector(
        target_func=log_likelihood,
        bounds=bounds,
        projections=projections,
        roi_threshold=4.0,
        pop_per_grid_point=3,
    ) as sampler:
        comm.bcast(sampler.target_func, root=0)
        results = run_all_projections(
            comm=comm, sampler=sampler, projections=projections,
            save_plots=True,
        )
    terminate_workers(comm, myrank)
else:
    worker_main(comm, myrank)
```

Run with `mpiexec -n 4 python your_script.py`.

More examples live in `examples/`. The `*_advanced.py` scripts show the full `advanced_config` layout.

## How it works

Rank 0 is the master: it owns the grid, hands out optimization jobs, and tracks which cells are active. Ranks 1+ are stateless workers that evaluate `target_func` on demand.

A scan proceeds roughly as follows:

1. Lay a regular grid over the projection dimensions; the rest are profiled at each grid point.
2. Run global L-BFGS-B starts to find initial maxima. Rather than firing a fixed number of Latin-hypercube starts and hoping it was enough, paraprof runs a *rolling* multistart with **basin detection**: each converged optimum is clustered online into a registry of distinct optima, and a Bayesian stopping rule (Boender–Rinnooy Kan, restricted to region-of-interest optima) halts the stage once the expected number of undiscovered ROI optima drops below a threshold — aborting any still-running optimizations at that point so their remaining evaluations aren't wasted. `n_initial_optimizations` becomes the upper bound. On later projections these are seeded from an in-memory pool built up by earlier projections, often skipping the global step entirely.
3. Anchor a DE population (or an initial point for L-BFGS-B optimization) at each promising cell. One population slot is filled with the highest-fitness past evaluation nearest the cell (proximity warm-start), so later projections inherit useful starting points.
4. Optimize the profiled parameters at each active cell. L-BFGS-B cells also reuse the best already-converged neighbour's quasi-Newton history and its best profiled parameters as an alternative start, propagating local curvature outward.
5. Activate the neighbours of high-likelihood cells, expanding the active set into the region of interest.
6. Patching: re-test each cell with its neighbours' best profiled parameters and polish any improvement.
7. Suspect recheck: cells whose profiled parameters look discontinuous compared to their neighbourhood are re-optimized from diverse seed points. This breaks contiguous wrong-optimum strips that the fitness-only filter of the patching step cannot fix.
8. Optionally refine the grid: interpolate the coarse grid to warm-start a finer one.

Key code paths: `ProfileProjector` (`sampler.py`) holds state and configuration, `master_main()` (`master.py`) is the state machine, `worker_main()` (`worker.py`) is the evaluation loop, and `jobs/` contains the asynchronous optimization jobs.

## Configuration

Common constructor arguments:

- `roi_threshold` — region-of-interest cutoff in log-likelihood; cells with `logL > global_max - roi_threshold` are inside the ROI. Default 3.0.
- `pop_per_grid_point` — DE population size per cell. Default 3.
- `n_initial_optimizations` — cap on global L-BFGS-B starts before grid optimization. Default `min(400, 50 * n_dims)`: a safe ceiling, since the Bayesian stopping rule controls the actual spend, so set it generously. (If you disable early stopping with `basin_detection.undiscovered_threshold = 0`, this becomes a fixed count, so set it explicitly.)
- `max_patching_waves` — cap on patching iterations. Default 10.
- `lbfgsb_max_iter`, `lbfgsb_polish` — L-BFGS-B iteration cap and whether to polish DE results. Defaults 50 and `True`.
- `initial_points` — explicit starting points to activate; useful when you already know where the good regions are.
- `use_clustering` — detect multiple modes during refinement. Default `True`.
- `refinement_direct_eval` — skip optimization in the refinement run and just evaluate the interpolated point. Default `False`.
- `samples_output_file` — path to log every evaluation. The format is chosen from the extension: `.csv` for plain text or `.h5`/`.hdf5` for HDF5 binary (needs the optional `h5py` dependency; see [Sample file formats](#sample-file-formats)).
- `warm_start_file` — sample file read at the start of each projection to pre-populate `initial_maxima`. Any supported format is accepted (the extension selects the reader). Set this equal to `samples_output_file` to feed a run's samples into the next one.
- `grad_func` — analytic gradient (see below).

### User-supplied gradients

By default L-BFGS-B uses finite differences, which costs 2N (central difference) or N (forward difference) extra calls per gradient. If you have an analytic gradient, pass it via `grad_func`:

```python
def target(p):
    return -float(np.sum(p**2))         # log-likelihood, maximized

def grad(p):
    return -2.0 * np.asarray(p)         # ∇target_func

sampler = ProfileProjector(target_func=target, grad_func=grad, ...)
```

`grad_func` returns the gradient of the function being *maximized*; ParaProf negates internally. Getting the sign wrong sends L-BFGS-B uphill.

You can return either a length-`n_dims` array or a `{dim_index: value}` dict for partial gradients. Entries that are `NaN`/`±inf` (or dims missing from the dict) fall back to finite differences using `lbfgsb.gradient_method`.

Only the L-BFGS-B paths use the gradient; DE is gradient-free. `sampler.target_calls_saved_by_user_gradient` and `sampler.user_gradient_errors` track usage and fallbacks.

### Sample file formats

Each evaluated point — `n_dims` parameter values plus its target value — is one row of the sample file. The format follows the `samples_output_file` extension:

| Extension        | Format      | Notes                                                                 |
|------------------|-------------|-----------------------------------------------------------------------|
| `.csv` (default) | plain text  | Headerless, `%.10e` columns. No extra dependencies.                   |
| `.h5` / `.hdf5`  | HDF5 binary | ~Half the size and faster I/O. Requires `h5py` (`pip install paraprof[hdf5]`). |

Both formats round-trip through `warm_start_file`, so a run can append to and re-read its own file. They differ in crash safety: CSV re-opens per flush, so a crash loses at most the un-flushed buffer, while HDF5 keeps the file open and can truncate the final chunk on a hard kill — prefer CSV if that matters more than file size.

To pool several independent runs (or convert between formats), use `combine_samples`. It streams chunk by chunk and may mix formats:

```python
import glob
from paraprof import combine_samples, read_samples, write_samples

combine_samples(glob.glob("run_*/samples.*"), "all_samples.h5")

samples = read_samples("all_samples.h5")   # (n_samples, n_dims + 1) array
params, target = samples[:, :-1], samples[:, -1]

write_samples(samples[target > target.max() - 4.0], "roi.csv")  # one-shot save
```

`read_samples`/`write_samples` are the one-shot load/save pair (`write_samples` replaces the target file). For very large files, iterate with `paraprof.sample_io.iter_sample_batches(path)` instead of loading the whole array.

### Advanced configuration

Pass an `advanced_config` dict for the knobs that actually move solution quality or are real iteration budgets:

| Key                                | Default                        | What it does                                                     |
|------------------------------------|--------------------------------|------------------------------------------------------------------|
| `memory_size`                      | `max(grid_sizes) * 25`         | DE F/CR adaptation memory size                                   |
| `convergence_threshold`            | `1e-6`                         | DE per-cell convergence cutoff                                   |
| `de.convergence_window`            | `3`                            | Generations of no-improvement before DE declares convergence     |
| `de.num_generations`               | `100000`                       | Hard cap on DE generations                                       |
| `de.max_num_to_evolve`             | `None`                         | Cap on cells evolved per generation                              |
| `lbfgsb.ftol`                      | `1e-9`                         | L-BFGS-B function tolerance                                      |
| `lbfgsb.gradient_method`           | `'forward'`                    | `'forward'` or `'central'` (~50% more calls)                     |
| `clustering.*`                     | auto-DBSCAN                    | Mode detection during refinement                                 |
| `cross_projection.proximity_warm_start`       | `True`             | Swap one LHS seed for the best nearby past evaluation. |
| `cross_projection.pool_seeded_initial_maxima` | `True`             | Seed `initial_maxima` from the pool on later projections and skip the global L-BFGS-B starts. |
| `suspect_recheck.enabled`                     | `True`             | Run the suspect-cell recheck pass after patching. |
| `suspect_recheck.max_waves`                   | `3`                | Cap on suspect-recheck waves. |
| `suspect_recheck.param_k`                     | `3.0`              | MAD multiplier for the discontinuity threshold. Lower flags more cells. |
| `suspect_recheck.max_fraction`                | `0.25`             | Hard cap on the fraction of ROI cells flagged per wave. |
| `suspect_recheck.seeds_k_ring`                | `3`                | Max Chebyshev radius for extended-neighbour seeds. |
| `suspect_recheck.seeds_from_pool`             | `3`                | Cross-projection pool seeds tested per suspect cell. |
| `suspect_recheck.polish_threshold`            | `1e-4`             | Min logL improvement to trigger the L-BFGS-B polish. |
| `basin_detection.batch_size`                  | `None`             | Optimizations kept in flight at once in the rolling multistart. `None` = FD-aware auto (≈ `n_workers` / per-gradient finite-difference fan-out, floored at 2). |
| `basin_detection.undiscovered_threshold`      | `0.5`              | Stop once the expected number of undiscovered ROI optima falls below this. Higher = stops sooner; `0` disables early stopping (the stage then runs the full `n_initial_optimizations`). |
| `basin_detection.min_starts`                  | `None`             | Minimum starts before the stopping rule may fire. `None` = `max(10, 3·n_dims)` (capped at `n_initial_optimizations`). |

**Usage:** with basin detection on, set `n_initial_optimizations` generously — it caps the worst case, while the stopping rule keeps the actual spend proportional to how multimodal the target turns out to be. Easy targets stop early; hard ones use the budget.

See the `ProfileProjector` docstring for the full structure. Several DE knobs that did not change ROI quality in benchmarking (`mutation_strategy`, `pbest_fraction`, `neighbor_pull_probability`, `global_pool_size`, `patching.n_neighbors`, `activation.mix_ratios`) are module-level constants in `sampler.py` and are intentionally not user-tunable.

### Projection options

Each projection is a dict. Required: `dims` (parameter indices) and `grid_points` (resolution per dimension). Optional: `optimization_method` (`'de'` or `'lbfgsb'`, default `'de'`), `patch_coarse_grid` (default `True`), `patch_refined_grid` (default `False`), `grid_refinement_factor` (integer > 1 to enable refinement), and `refinement_method` (default `'linear'`).

## Visualization

With `save_plots=True`, paraprof writes 1D line plots, 2D heatmaps with contours, and pairwise 2D slices for higher-dimensional projections (either the max slice or marginalized). It also writes plots of the optimal profiled-parameter values across the projection grid, which is useful for seeing what the profiling actually did.

Override the defaults with a `plot_settings` dict:

```python
plot_settings = {
    'dpi': 300,
    'filetype': 'png',
    'slice_mode': 'max',  # or 'all' for marginalization (3D+)
    'vmin': -4.0,
    'vmax': 0.0,
    'plot_profiled_params': True,
    'output_dir': '.',  # created automatically if missing
}
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/paraprof --cov-report=term-missing
```

## Project structure

```
paraprof/
├── src/paraprof/
│   ├── sampler.py            # ProfileProjector (state + config)
│   ├── master.py             # Master event loop
│   ├── worker.py             # Worker event loop
│   ├── jobs/                 # Optimization jobs (LBFGSB, DE, activation, patching)
│   ├── interpolation.py      # Grid interpolation + clustering for refinement
│   ├── visualization.py      # Plot helpers
│   ├── nuisance_wrapper.py   # Wrap test functions with nuisance parameters
│   ├── test_functions.py     # Himmelblau, Rosenbrock, Rastrigin, etc.
│   ├── exceptions.py
│   └── logger.py
├── tests/
└── examples/                 # `*_advanced.py` exercises advanced_config
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{paraprof2025,
  title = {paraprof},
  author = {Kvellestad, Anders},
  year = {2025},
  url = {https://github.com/anderkve/paraprof}
}
```

Maintainer: Anders Kvellestad.
