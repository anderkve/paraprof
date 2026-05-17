"""Build in-container pseudo-oracle grids by taking the cell-wise max over
all grid-producing runs.

Used when the real ``paraprof_oracle`` build (high pop, refinement,
many patching waves) is too expensive for the runtime environment. The
real paper sweep should use ``oracle.py`` / ``--build-oracles-only``
on hardware with enough budget.

Reads every result JSON in ``results/runs/`` that has a finite ``logL_grid``
(i.e. all the grid methods; ``iminuit_mncontour`` is skipped because it
emits contour polygons rather than a grid). Groups by (problem, projection,
grid shape). For each group, the oracle is the per-cell maximum across
all contributing (method, seed) grids; cells that are NaN in every
contributor stay NaN.

Limitations the user should know about:

* Solution-quality Δ for methods that contributed to the per-cell max
  will be biased toward zero in those cells. Coverage and evals-to-ε
  remain meaningful.
* Cells that no contributing method reaches stay NaN in the oracle and
  are dropped from every metric.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .adapters.base import ProjectionResult
from .oracle import ORACLE_DIR, oracle_key, save_oracle

RUNS_DIR = Path(__file__).parent / "results" / "runs"

# Methods whose ``logL_grid`` is a real grid (not a contour polygon).
GRID_PRODUCERS = {
    "paraprof_default",
    "paraprof_kernel",
    "iminuit_grid",
    "scipy_de",
    "scipy_lbfgsb",
    "nlopt_crs2_bobyqa",
}


def _group_runs(runs_dir: Path) -> dict[tuple, list[ProjectionResult]]:
    """Group run JSONs by (problem, dims, grid_shape)."""
    groups: dict[tuple, list[ProjectionResult]] = defaultdict(list)
    for path in sorted(runs_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        if data["method"] not in GRID_PRODUCERS:
            continue
        result = ProjectionResult.from_dict(data)
        key = (result.problem, tuple(result.dims), tuple(result.grid_points))
        groups[key].append(result)
    return groups


def build_pseudo_oracle(group: list[ProjectionResult]) -> ProjectionResult:
    stack = np.stack([r.logL_grid for r in group])
    # nanmax: cells NaN in every contributor stay NaN, cells finite in
    # any contributor take that maximum.
    merged = np.nanmax(stack, axis=0)
    template = group[0]
    return ProjectionResult(
        method="paraprof_oracle",
        problem=template.problem,
        dims=list(template.dims),
        grid_points=list(template.grid_points),
        seed=0,
        grid_axes=template.grid_axes,
        logL_grid=merged,
        profiled_params_grid=template.profiled_params_grid,
        cell_evals=np.zeros_like(template.cell_evals),
        total_evals=int(sum(r.total_evals for r in group)),
        n_cells_capped=0,
        wall_time=0.0,
        extra={
            "pseudo_oracle": True,
            "contributors": sorted({r.method for r in group}),
            "seeds": sorted({r.seed for r in group}),
            "n_contributing_runs": len(group),
        },
    )


def main(argv=None) -> int:
    groups = _group_runs(RUNS_DIR)
    if not groups:
        raise SystemExit(f"no grid-method runs found in {RUNS_DIR}")
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, group in sorted(groups.items()):
        oracle = build_pseudo_oracle(group)
        path = save_oracle(oracle)
        problem, dims, grid = key
        finite = np.isfinite(oracle.logL_grid)
        print(
            f"  {problem:>14}  dims={list(dims)}  grid={list(grid)}  "
            f"contributors={oracle.extra['n_contributing_runs']:>2}  "
            f"coverage={finite.mean()*100:5.1f}%  -> {path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
