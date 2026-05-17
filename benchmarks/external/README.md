# External-tool comparison

Harness used to benchmark **paraprof** against external profile-likelihood
algorithms for the comparison section of the paraprof paper.

See `plan.md` and the surrounding section of the paper for the full
methodological rationale. The short version:

| What | How |
|---|---|
| Primary metric | Target-function evaluations |
| Wall-clock | Not part of the comparison (see below) |
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

## Why no wall-clock comparison

The paper compares methods on **target-function evaluation count**, which is
parallelisation-invariant: it doesn't matter whether one method runs in MPI
and another runs serially. The eval count for `paraprof_default` on a 50×50
grid is the same number whether the run takes seconds (cheap analytic
target) or hours (expensive likelihood).

The harness reflects this: only the `paraprof_*` adapters use paraprof's
native MPI master/worker pool; the per-cell competitor adapters run their
inner optimisation on the master rank only. Adding MPI to every competitor
would change nothing about the eval-count comparison and is not done.

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
budget. Runs all production methods (everything except `paraprof_oracle`,
which is a high-budget reference grid and intentionally expensive — the
smoke synthesizes a stand-in below). Completes in well under a minute.

```bash
# 1. Run all seven production methods on a single 5x5 projection
python -m benchmarks.external.run_comparison \
    --problems himmelblau_4d \
    --methods paraprof_default paraprof_kernel \
              scipy_de scipy_lbfgsb nlopt_crs2_bobyqa \
              iminuit_grid iminuit_mncontour \
    --dims-override 0 1 \
    --grid 5 5 \
    --seeds 1 \
    --max-evals-per-cell 600 \
    --mpi-ranks 4

# 2. Synthesize a stand-in oracle (cell-wise max across the grid-producing
#    methods) so the metrics + plot pipeline can run without paraprof_oracle.
python - <<'PY'
import json, numpy as np
from pathlib import Path
runs = Path("benchmarks/external/results/runs")
arrs, template = [], None
for p in sorted(runs.glob("*.json")):
    d = json.load(open(p))
    if d["method"] == "iminuit_mncontour":
        continue
    arrs.append(np.array(d["logL_grid"]))
    template = d
template["method"] = "paraprof_oracle"
template["logL_grid"] = np.nanmax(np.stack(arrs), axis=0).tolist()
template["extra"] = {"smoke_pseudo_oracle": True}
out = Path("benchmarks/external/results/oracle")
out.mkdir(parents=True, exist_ok=True)
(out / "himmelblau_4d__dims-0_1__grid-5x5.json").write_text(json.dumps(template))
PY

# 3. Render all figures from the smoke run
python -m benchmarks.external.plot_comparison
```

Generated PDFs and PNGs land in `benchmarks/external/results/figures/`.

For the real paper sweep, run step 1 with the full method list (including
`paraprof_oracle`) and the full grid/seed settings from the table at the
top. Replace step 2 with a real oracle build:

```bash
python -m benchmarks.external.run_comparison --build-oracles-only --mpi-ranks 4
```

A laptop is fine for analytic targets; only the oracle takes meaningful
wall-clock time, and only because of how many evaluations it is
deliberately configured to spend.
