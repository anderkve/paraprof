"""iminuit MIGRAD + MNCONTOUR adapter.

Runs ONE global MIGRAD over all parameters, then calls ``mncontour`` for the
two projection dims at the 68% and 95% Wilks levels. Produces contours but
no grid. Used for the contour-overlay figure and the IoU/Hausdorff metrics.
"""
from __future__ import annotations

import time

import numpy as np
from iminuit import Minuit

from .base import (
    BaseAdapter,
    CountingFunction,
    ProjectionResult,
    cell_centres,
    register_adapter,
)


# 2-D Wilks ΔlogL for 68% and 95%: Δχ² ≈ 2.30 and 6.18, halved for logL form.
_CL_LEVELS_2D = {
    "68": 2.30 / 2.0,
    "95": 6.18 / 2.0,
}


@register_adapter
class IMinuitMNContour(BaseAdapter):
    name = "iminuit_mncontour"
    n_restarts = 10  # global multistart before contour scan
    contour_points = 80  # per CL contour

    def run(self, func, bounds, dims, grid_points, max_evals_per_cell, seed, comm=None):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        counting = CountingFunction(func)
        n_dims = len(bounds)
        projection_dims = list(dims)
        if len(projection_dims) != 2:
            raise ValueError("iminuit_mncontour adapter expects exactly two projection dims")

        param_names = tuple(f"p{i}" for i in range(n_dims))
        full_bounds = [tuple(b) for b in bounds]

        def objective(*args):
            return -counting(np.asarray(args, dtype=float))

        best_state = None
        for r in range(self.n_restarts):
            if r == 0:
                start = np.array([0.5 * (lo + hi) for lo, hi in full_bounds])
            else:
                start = np.array([rng.uniform(lo, hi) for lo, hi in full_bounds])
            try:
                m = Minuit(objective, *start.tolist(), name=param_names)
                m.errordef = 0.5
                m.limits = full_bounds
                m.print_level = 0
                m.migrad()
                if not np.isfinite(m.fval):
                    continue
                if best_state is None or m.fval < best_state["fval"]:
                    best_state = {
                        "fval": float(m.fval),
                        "values": list(m.values),
                    }
            except Exception:
                continue

        contours = {}
        if best_state is not None:
            # Restart at the best vector to make sure HESSE/MNCONTOUR run from there.
            m = Minuit(objective, *best_state["values"], name=param_names)
            m.errordef = 0.5
            m.limits = full_bounds
            m.print_level = 0
            m.migrad()
            try:
                m.hesse()
            except Exception:
                pass

            pname_a, pname_b = param_names[projection_dims[0]], param_names[projection_dims[1]]
            for cl, dlogL in _CL_LEVELS_2D.items():
                try:
                    # iminuit takes cl in (0, 1); use the chi-squared mapping it expects.
                    # For Δχ² = 2.30 (68% 2D) and 6.18 (95% 2D) we set ``cl`` explicitly.
                    cl_arg = float(0.68 if cl == "68" else 0.95)
                    pts = m.mncontour(pname_a, pname_b, cl=cl_arg,
                                      size=self.contour_points)
                    if pts is not None and len(pts) > 0:
                        contours[cl] = np.asarray(pts, dtype=float).tolist()
                except Exception:
                    continue

        # Build empty grid placeholders so the result fits the standard schema.
        shape = tuple(grid_points)
        logL_grid = np.full(shape, np.nan, dtype=float)
        n_profiled = n_dims - 2
        profiled_grid = np.full(shape + (n_profiled,), np.nan, dtype=float)
        cell_evals = np.zeros(shape, dtype=np.int64)
        axes = cell_centres(bounds, projection_dims, grid_points)

        extra = {
            "best_logL": float(-best_state["fval"]) if best_state else float("nan"),
            "best_params": best_state["values"] if best_state else [],
            "contours": contours,  # {"68": [[x, y], ...], "95": [[x, y], ...]}
        }

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
            n_cells_capped=0,
            wall_time=time.perf_counter() - t0,
            extra=extra,
        )
