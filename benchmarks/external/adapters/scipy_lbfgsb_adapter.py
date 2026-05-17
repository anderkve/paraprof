"""SciPy L-BFGS-B multistart per-cell adapter."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from .base import (
    BaseAdapter,
    CountingFunction,
    ProjectionResult,
    cell_centres,
    register_adapter,
)


class _CellBudgetExceeded(Exception):
    pass


@register_adapter
class ScipyLBFGSB(BaseAdapter):
    name = "scipy_lbfgsb"
    n_starts = 5  # Latin-hypercube random starts per cell

    def run(self, func, bounds, dims, grid_points, max_evals_per_cell, seed, comm=None):
        t0 = time.perf_counter()
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

        for cell_idx in np.ndindex(*shape):
            cell_coords = np.array([axes[i][cell_idx[i]] for i in range(len(axes))])
            cell_start = counting.count
            best = {"logL": -np.inf, "x": np.full(n_profiled, np.nan)}

            def fcn(profiled, _best=best, _cell_coords=cell_coords):
                if counting.count - cell_start >= max_evals_per_cell:
                    raise _CellBudgetExceeded()
                full = np.empty(n_dims)
                full[projection_dims] = _cell_coords
                full[profiled_dims] = profiled
                val = counting(full)
                if val > _best["logL"]:
                    _best["logL"] = val
                    _best["x"] = np.asarray(profiled, dtype=float).copy()
                return -val

            if n_profiled == 0:
                full = np.empty(n_dims)
                full[projection_dims] = cell_coords
                best["logL"] = counting(full)
            else:
                cell_seed = (seed + int(np.ravel_multi_index(cell_idx, shape))) & 0xFFFFFFFF
                lhs = LatinHypercube(d=n_profiled, seed=cell_seed)
                starts = lhs.random(n=self.n_starts)
                for s in starts:
                    start = np.array([lo + s_i * (hi - lo)
                                      for s_i, (lo, hi) in zip(s, profiled_bounds)])
                    try:
                        minimize(
                            fcn, start, method="L-BFGS-B",
                            bounds=profiled_bounds,
                            options={"maxiter": 200, "ftol": 1e-9},
                        )
                    except _CellBudgetExceeded:
                        n_capped += 1
                        break
                    except Exception:
                        continue

            if np.isfinite(best["logL"]):
                logL_grid[cell_idx] = best["logL"]
                if n_profiled > 0:
                    profiled_grid[cell_idx] = best["x"]
            cell_evals[cell_idx] = counting.count - cell_start

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
            extra={"n_starts": self.n_starts},
        )
