"""NLopt CRS2 + BOBYQA polish per-cell adapter."""
from __future__ import annotations

import time

import numpy as np
import nlopt

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
class NLoptCRS2BOBYQA(BaseAdapter):
    name = "nlopt_crs2_bobyqa"

    def run(self, func, bounds, dims, grid_points, max_evals_per_cell, seed, comm=None):
        t0 = time.perf_counter()
        counting = CountingFunction(func)
        n_dims = len(bounds)
        projection_dims = list(dims)
        profiled_dims = [i for i in range(n_dims) if i not in projection_dims]
        n_profiled = len(profiled_dims)
        profiled_bounds = [tuple(bounds[d]) for d in profiled_dims]
        lower = np.array([b[0] for b in profiled_bounds])
        upper = np.array([b[1] for b in profiled_bounds])
        axes = cell_centres(bounds, projection_dims, grid_points)
        shape = tuple(grid_points)

        logL_grid = np.full(shape, np.nan, dtype=float)
        profiled_grid = np.full(shape + (n_profiled,), np.nan, dtype=float)
        cell_evals = np.zeros(shape, dtype=np.int64)
        n_capped = 0

        # Per-cell evaluation budget split between global and local stage.
        global_budget = max(50, int(0.7 * max_evals_per_cell))
        local_budget = max(20, int(0.3 * max_evals_per_cell))

        for cell_idx in np.ndindex(*shape):
            cell_coords = np.array([axes[i][cell_idx[i]] for i in range(len(axes))])
            cell_start = counting.count
            best = {"logL": -np.inf, "x": np.full(n_profiled, np.nan)}

            def fcn_array(profiled, grad=None,
                          _best=best, _cell_coords=cell_coords):
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
                # Global stage: CRS2-LM.
                start = 0.5 * (lower + upper)
                global_x = start.copy()
                try:
                    opt = nlopt.opt(nlopt.GN_CRS2_LM, n_profiled)
                    opt.set_lower_bounds(lower)
                    opt.set_upper_bounds(upper)
                    opt.set_min_objective(fcn_array)
                    opt.set_maxeval(global_budget)
                    opt.set_ftol_rel(1e-7)
                    cell_seed = (seed + int(np.ravel_multi_index(cell_idx, shape))) & 0xFFFFFFFF
                    nlopt.srand(cell_seed)
                    global_x = opt.optimize(start)
                except _CellBudgetExceeded:
                    n_capped += 1
                    if np.isfinite(best["logL"]):
                        logL_grid[cell_idx] = best["logL"]
                        profiled_grid[cell_idx] = best["x"]
                    cell_evals[cell_idx] = counting.count - cell_start
                    continue
                except Exception:
                    pass

                # Local polish: BOBYQA from the best global point.
                try:
                    opt = nlopt.opt(nlopt.LN_BOBYQA, n_profiled)
                    opt.set_lower_bounds(lower)
                    opt.set_upper_bounds(upper)
                    opt.set_min_objective(fcn_array)
                    opt.set_maxeval(local_budget)
                    opt.set_ftol_rel(1e-9)
                    opt.optimize(global_x)
                except _CellBudgetExceeded:
                    n_capped += 1
                except Exception:
                    pass

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
            extra={
                "global_budget": global_budget,
                "local_budget": local_budget,
            },
        )
