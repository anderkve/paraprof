"""iminuit per-cell MIGRAD adapter.

At every grid cell we fix the projection-dim coordinates and run MIGRAD over
the profiled dims. To keep multimodal problems tractable we do ``n_restarts``
random restarts per cell and keep the best.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from iminuit import Minuit

from .base import (
    BaseAdapter,
    CountingFunction,
    ProjectionResult,
    cell_centres,
    register_adapter,
)


class _CellBudgetExceeded(Exception):
    pass


class _CellBudget:
    """Counter-aware wrapper that raises when a per-cell budget is hit
    and tracks the best logL seen so far inside this cell."""

    def __init__(self, counting: CountingFunction, max_evals: int):
        self.counting = counting
        self.start = counting.count
        self.max_evals = max_evals
        self.hit = False
        self.best_logL = -np.inf
        self.best_full = None

    def __call__(self, params):
        if self.counting.count - self.start >= self.max_evals:
            self.hit = True
            raise _CellBudgetExceeded()
        val = self.counting(params)
        if val > self.best_logL:
            self.best_logL = val
            self.best_full = np.asarray(params, dtype=float).copy()
        return val


def _make_objective(budget: _CellBudget, cell_coords: np.ndarray,
                    projection_dims: list[int], profiled_dims: list[int],
                    n_dims: int):
    """Return a negative-logL objective taking positional profiled-param args."""

    def fcn(*args):
        full = np.empty(n_dims)
        full[projection_dims] = cell_coords
        full[profiled_dims] = args
        return -budget(full)

    return fcn


@register_adapter
class IMinuitGrid(BaseAdapter):
    name = "iminuit_grid"
    n_restarts = 3  # restarts per cell; identical for every method that has restarts

    def run(self, func, bounds, dims, grid_points, max_evals_per_cell, seed, comm=None):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        counting = CountingFunction(func)
        n_dims = len(bounds)
        projection_dims = list(dims)
        profiled_dims = [i for i in range(n_dims) if i not in projection_dims]
        n_profiled = len(profiled_dims)
        profiled_bounds = [tuple(bounds[d]) for d in profiled_dims]
        axes = cell_centres(bounds, projection_dims, grid_points)
        shape = tuple(grid_points)

        logL_grid = np.full(shape, np.nan, dtype=float)
        profiled_grid = np.full(shape + (n_profiled,), np.nan, dtype=float)
        cell_evals = np.zeros(shape, dtype=np.int64)
        n_capped = 0
        param_names = tuple(f"p{i}" for i in range(n_profiled))

        for cell_idx in np.ndindex(*shape):
            cell_coords = np.array([axes[i][cell_idx[i]] for i in range(len(axes))])
            budget = _CellBudget(counting, max_evals_per_cell)
            objective = _make_objective(
                budget, cell_coords, projection_dims, profiled_dims, n_dims
            )

            for r in range(self.n_restarts):
                if budget.hit:
                    break
                # Restart 0 = centre of profiled bounds, rest = uniform-random.
                if r == 0:
                    start = np.array([0.5 * (lo + hi) for lo, hi in profiled_bounds])
                else:
                    start = np.array(
                        [rng.uniform(lo, hi) for lo, hi in profiled_bounds]
                    )
                try:
                    m = Minuit(objective, *start.tolist(), name=param_names)
                    m.errordef = 0.5
                    m.limits = profiled_bounds
                    m.print_level = 0
                    m.migrad()
                except _CellBudgetExceeded:
                    n_capped += 1
                    break
                except Exception:
                    # MIGRAD instability at this start; skip.
                    continue

            cell_evals[cell_idx] = counting.count - budget.start
            if np.isfinite(budget.best_logL):
                logL_grid[cell_idx] = budget.best_logL
                if n_profiled > 0 and budget.best_full is not None:
                    # ``best_full`` is the FULL params vector (n_dims); slice it.
                    profiled_grid[cell_idx] = budget.best_full[profiled_dims]

        return ProjectionResult(
            method=self.name,
            problem="",
            dims=projection_dims,
            grid_points=list(grid_points),
            seed=int(seed),
            grid_axes=axes,
            logL_grid=logL_grid,
            profiled_params_grid=profiled_grid,
            cell_evals=cell_evals,
            total_evals=counting.count,
            n_cells_capped=n_capped,
            wall_time=time.perf_counter() - t0,
            extra={"n_restarts": self.n_restarts},
        )
