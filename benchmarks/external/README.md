# External-tool comparison

Harness used to benchmark **paraprof** against external profile-likelihood
algorithms for the comparison section of the paraprof paper.

See `plan.md` and the surrounding section of the paper for the full
methodological rationale. The short version:

| What | How |
|---|---|
| Primary metric | Target-function evaluations |
| Secondary metric | Wall-clock time (informational only) |
| Test problems | `rosenbrock_4d`, `himmelblau_4d`, `rastrigin_4d`, `levy_4d` |
| Projections | Two 2-D projections per problem |
| Grid | 50 × 50 |
| Seeds | 3 per (method, problem, projection) |
| Oracle | `paraprof_oracle` (high-budget paraprof with refinement, run with two seeds and sanity-checked) |

## Methods

| Adapter name              | Where it appears |
|---------------------------|------------------|
| `paraprof_default`        | Body figures |
| `paraprof_kernel`         | Appendix figures (patching / suspect-recheck / refinement off) |
| `paraprof_oracle`         | Reference grid only |
| `iminuit_grid`            | Body figures (per-cell MIGRAD with multistart) |
| `iminuit_mncontour`       | Contour-overlay figure + table (global MIGRAD + MNCONTOUR) |
| `scipy_de`                | Body figures (`scipy.optimize.differential_evolution`) |
| `scipy_lbfgsb`            | Appendix figures (multistart L-BFGS-B with Latin-hypercube starts) |
| `nlopt_crs2_bobyqa`       | Appendix figures (NLopt CRS2-LM global + BOBYQA local) |

## Wall-clock fairness caveat

Only the `paraprof_*` adapters use paraprof's native MPI master/worker pool.
The per-cell competitor adapters run their inner optimisation on the master
rank only. This is intentional and acceptable because:

1. The paper's primary metric is **target-function evaluations**, which is
   parallelisation-invariant.
2. Each per-cell competitor optimisation is independent and would be trivially
   parallelisable across grid cells — the wall-clock numbers reported here
   are upper bounds on what a parallelised port could achieve.

If you need fair wall-clock numbers, extend each non-MPI adapter to run its
per-cell loop through an MPI worker pool.

## Files

```
benchmarks/external/
├── adapters/                # One adapter per method
├── oracle.py                # Build/cache the reference grid per (problem, projection)
├── metrics.py               # Solution quality, coverage, evals-to-ε
├── run_one.py               # MPI entry point: one (method × problem × proj × seed)
├── run_comparison.py        # Top-level orchestrator
├── plot_comparison.py       # Renders every figure from the runs/ JSONs
├── style.mplstyle           # Paper-quality matplotlib style
├── results/oracle/          # Cached oracle grids (gitignored runtime artefacts)
└── results/runs/            # Per-task result JSONs (gitignored)
```

## Usage

Install the bench extras and confirm MPI works:

```bash
pip install -e ".[bench-extras]"
mpiexec --allow-run-as-root -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
```

Build the oracles first (this is the only step that is *required* before any
metric or figure can be computed):

```bash
python -m benchmarks.external.run_comparison --build-oracles-only --mpi-ranks 4
```

Run the full sweep:

```bash
python -m benchmarks.external.run_comparison --mpi-ranks 4
```

Render the figures:

```bash
python -m benchmarks.external.plot_comparison
```

Outputs land in `benchmarks/external/results/figures/` as PDFs (vector) and
PNGs (raster).

## Smoke test

To check the pipeline end-to-end on a tiny grid without spending the full
budget:

```bash
python -m benchmarks.external.run_comparison \
    --problems himmelblau_4d \
    --methods paraprof_oracle paraprof_default iminuit_grid scipy_de \
              scipy_lbfgsb nlopt_crs2_bobyqa iminuit_mncontour \
    --grid 10 10 \
    --seeds 1 \
    --max-evals-per-cell 400 \
    --mpi-ranks 4
python -m benchmarks.external.plot_comparison
```
