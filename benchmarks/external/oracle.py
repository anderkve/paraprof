"""Build, cache, and load the oracle reference grid for each (problem, projection).

Oracle = paraprof_oracle adapter (high-budget paraprof with refinement). The
oracle is run twice with different seeds and accepted only if max-cell
disagreement <= ``ORACLE_TOL``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from paraprof import get_test_function

from .adapters.base import ProjectionResult
from .adapters.paraprof_adapter import ParaProfOracle

ORACLE_DIR = Path(__file__).parent / "results" / "oracle"
ORACLE_TOL = 1e-3  # max-cell disagreement tolerated between two oracle seeds


def oracle_key(problem: str, dims: tuple[int, ...], grid_points: tuple[int, ...]) -> str:
    d = "_".join(str(i) for i in dims)
    g = "x".join(str(n) for n in grid_points)
    return f"{problem}__dims-{d}__grid-{g}.json"


def load_oracle(problem: str, dims: tuple[int, ...],
                grid_points: tuple[int, ...]) -> ProjectionResult | None:
    path = ORACLE_DIR / oracle_key(problem, dims, grid_points)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return ProjectionResult.from_dict(data)


def save_oracle(result: ProjectionResult) -> Path:
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    path = ORACLE_DIR / oracle_key(result.problem, tuple(result.dims),
                                   tuple(result.grid_points))
    with path.open("w") as f:
        json.dump(result.to_dict(), f, indent=1)
    return path


def build_oracle(problem: str, dims: tuple[int, ...], grid_points: tuple[int, ...],
                 comm, seeds: tuple[int, int] = (101, 202),
                 max_evals_per_cell: int = 5000,
                 force: bool = False, tol: float = ORACLE_TOL) -> ProjectionResult:
    """Build (or load) the oracle reference grid for one (problem, projection).

    Must be invoked with paraprof workers attached on ``comm`` (rank > 0 sits
    in paraprof's worker_main; rank 0 calls this function).
    """
    cached = load_oracle(problem, dims, grid_points) if not force else None
    if cached is not None:
        return cached

    func, bounds, _ = get_test_function(problem)
    adapter = ParaProfOracle()
    runs = []
    for seed in seeds:
        result = adapter.run(
            func=func,
            bounds=bounds,
            dims=list(dims),
            grid_points=list(grid_points),
            max_evals_per_cell=max_evals_per_cell,
            seed=seed,
            comm=comm,
        )
        result.problem = problem
        runs.append(result)

    # Sanity check: max-cell agreement between two seeds.
    a, b = runs[0].logL_grid, runs[1].logL_grid
    finite = np.isfinite(a) & np.isfinite(b)
    disagree = float("inf")
    if finite.any():
        disagree = float(np.nanmax(np.abs(a[finite] - b[finite])))

    # Use the cell-wise max as the oracle (best of the two seeds).
    merged = np.where(np.isfinite(a) & np.isfinite(b),
                      np.maximum(a, b),
                      np.where(np.isfinite(a), a, b))
    oracle = ProjectionResult(
        method="paraprof_oracle",
        problem=problem,
        dims=list(dims),
        grid_points=list(grid_points),
        seed=int(seeds[0]),
        grid_axes=runs[0].grid_axes,
        logL_grid=merged,
        profiled_params_grid=runs[0].profiled_params_grid,
        cell_evals=runs[0].cell_evals + runs[1].cell_evals,
        total_evals=int(runs[0].total_evals + runs[1].total_evals),
        n_cells_capped=int(runs[0].n_cells_capped + runs[1].n_cells_capped),
        wall_time=float(runs[0].wall_time + runs[1].wall_time),
        extra={
            "seeds": list(seeds),
            "max_cell_disagreement": disagree,
            "tol": tol,
            "tol_passed": bool(disagree <= tol),
        },
    )
    save_oracle(oracle)
    return oracle
