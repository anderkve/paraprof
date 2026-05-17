"""Solution-quality, coverage, and evals-to-threshold metrics."""
from __future__ import annotations

import numpy as np

from .adapters.base import ProjectionResult

# 2-D Wilks ΔlogL halved (we work in logL not chi²): 68% -> 1.15, 95% -> 3.09.
DELTA_LOGL_68_2D = 2.30 / 2.0
DELTA_LOGL_95_2D = 6.18 / 2.0
ROI_THRESHOLD_2D = DELTA_LOGL_95_2D  # ROI = 95% Wilks region for the 2-D case


def _aligned_grids(oracle: ProjectionResult, method: ProjectionResult) -> tuple[np.ndarray, np.ndarray]:
    """Return the oracle and method logL grids aligned to a common cell mask.

    Cells where the oracle is NaN are dropped on both sides; cells where the
    method is NaN are penalised — they are treated as -inf logL so the
    per-cell Δ becomes oracle - (-inf) = +inf, marking complete failure.
    """
    if oracle.logL_grid.shape != method.logL_grid.shape:
        raise ValueError(
            f"grid-shape mismatch: oracle {oracle.logL_grid.shape} vs "
            f"method {method.logL_grid.shape}"
        )
    oracle_grid = oracle.logL_grid.astype(float).copy()
    method_grid = method.logL_grid.astype(float).copy()
    # Method NaN -> -inf (worst-case suboptimal).
    method_grid[np.isnan(method_grid)] = -np.inf
    return oracle_grid, method_grid


def solution_quality(oracle: ProjectionResult, method: ProjectionResult) -> dict:
    """Per-cell ΔlogL stats over the full grid and over the ROI."""
    oracle_grid, method_grid = _aligned_grids(oracle, method)
    mask_oracle = np.isfinite(oracle_grid)
    if not mask_oracle.any():
        return {
            "max_delta": float("nan"),
            "rms_delta_full": float("nan"),
            "rms_delta_roi": float("nan"),
            "n_cells_failed": int(np.isinf(method_grid).sum()),
        }

    delta = np.full_like(oracle_grid, np.nan)
    delta[mask_oracle] = oracle_grid[mask_oracle] - method_grid[mask_oracle]
    # Clamp Δ to be >= 0 to handle tiny numerical overshoots above the oracle.
    delta_clamped = np.where(mask_oracle, np.maximum(delta, 0.0), np.nan)

    full = delta_clamped[mask_oracle]
    full_finite = full[np.isfinite(full)]

    oracle_max = float(np.nanmax(oracle_grid))
    roi_mask = mask_oracle & (oracle_grid >= oracle_max - ROI_THRESHOLD_2D)
    roi = delta_clamped[roi_mask]
    roi_finite = roi[np.isfinite(roi)]

    return {
        "max_delta": float(np.nanmax(full)) if full.size > 0 else float("nan"),
        "rms_delta_full": (
            float(np.sqrt(np.nanmean(full_finite**2))) if full_finite.size > 0 else float("inf")
        ),
        "rms_delta_roi": (
            float(np.sqrt(np.nanmean(roi_finite**2))) if roi_finite.size > 0 else float("inf")
        ),
        "n_cells_failed": int(np.isinf(method_grid).sum()),
        "n_roi_cells": int(roi_mask.sum()),
    }


def _region_mask(grid: np.ndarray, delta_logL: float) -> np.ndarray:
    """Boolean grid mask for cells inside the Δχ² confidence region of ``grid``."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return np.zeros_like(grid, dtype=bool)
    g_max = float(np.nanmax(grid))
    out = np.zeros_like(grid, dtype=bool)
    finite_mask = np.isfinite(grid)
    out[finite_mask] = grid[finite_mask] >= g_max - delta_logL
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return float("nan")
    return inter / union


def _fpfn(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    truth = truth.astype(bool)
    pred = pred.astype(bool)
    n_true_positive = int(np.logical_and(truth, pred).sum())
    n_false_positive = int(np.logical_and(~truth, pred).sum())
    n_false_negative = int(np.logical_and(truth, ~pred).sum())
    fp = n_false_positive / max(1, int((~truth).sum()))
    fn = n_false_negative / max(1, int(truth.sum()))
    return fp, fn


def _hausdorff(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """Two-set Hausdorff distance between two sets of 2-D points (Nx2 each).

    Uses an O(N*M) explicit pairwise distance — N is bounded by the grid cell
    count, so this is fine for 50x50 grids.
    """
    if set_a.size == 0 or set_b.size == 0:
        return float("nan")
    diffs = set_a[:, None, :] - set_b[None, :, :]
    d2 = np.sum(diffs**2, axis=-1)
    a_to_b = np.sqrt(d2.min(axis=1)).max()
    b_to_a = np.sqrt(d2.min(axis=0)).max()
    return float(max(a_to_b, b_to_a))


def _boundary_cells(mask: np.ndarray) -> np.ndarray:
    """Return the (i, j) -> (x, y) cell-index coordinates that lie on a region's boundary."""
    if not mask.any():
        return np.zeros((0, 2), dtype=float)
    pad = np.pad(mask, 1, mode="constant", constant_values=False)
    # A cell is "boundary" if it's True and any 4-neighbour is False (or off-grid).
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    boundary = mask & (~up | ~down | ~left | ~right)
    idx = np.argwhere(boundary)
    return idx.astype(float)


def coverage(oracle: ProjectionResult, method: ProjectionResult) -> dict:
    """Region-mask IoU + FP/FN + Hausdorff at 68% and 95%."""
    oracle_grid, method_grid = _aligned_grids(oracle, method)
    axes = oracle.grid_axes

    results: dict = {}
    for cl_name, dlogL in (("68", DELTA_LOGL_68_2D), ("95", DELTA_LOGL_95_2D)):
        truth = _region_mask(oracle_grid, dlogL)
        pred = _region_mask(method_grid, dlogL)
        # Convert grid-index boundary cells to physical coordinates for Hausdorff.
        truth_idx = _boundary_cells(truth)
        pred_idx = _boundary_cells(pred)
        if axes is not None and len(axes) == 2 and truth_idx.size and pred_idx.size:
            ax0, ax1 = axes
            truth_coords = np.column_stack([ax0[truth_idx[:, 0].astype(int)],
                                            ax1[truth_idx[:, 1].astype(int)]])
            pred_coords = np.column_stack([ax0[pred_idx[:, 0].astype(int)],
                                           ax1[pred_idx[:, 1].astype(int)]])
            haus = _hausdorff(truth_coords, pred_coords)
        else:
            haus = float("nan")

        fp, fn = _fpfn(truth, pred)
        results[f"iou_{cl_name}"] = _iou(truth, pred)
        results[f"fp_{cl_name}"] = fp
        results[f"fn_{cl_name}"] = fn
        results[f"hausdorff_{cl_name}"] = haus
    return results


def coverage_from_polygons(oracle: ProjectionResult, method: ProjectionResult) -> dict:
    """Coverage metrics for a contour-only method (e.g. iminuit_mncontour).

    Rasterises the method's contour polygon at the oracle grid and reuses the
    standard grid-mask coverage calculation.
    """
    contours = method.extra.get("contours", {}) or {}
    oracle_grid = oracle.logL_grid.astype(float)
    axes = oracle.grid_axes
    if len(axes) != 2:
        raise ValueError("polygon coverage only defined for 2-D projections")

    # Build raster mask per CL.
    raster_grids = {}
    for cl_name in ("68", "95"):
        poly = contours.get(cl_name)
        if not poly:
            raster_grids[cl_name] = np.zeros_like(oracle_grid, dtype=bool)
            continue
        poly_arr = np.asarray(poly, dtype=float)
        raster_grids[cl_name] = _polygon_to_grid_mask(poly_arr, axes)

    # Wrap into a synthetic ProjectionResult so coverage() does the math.
    synthetic = ProjectionResult(
        method=method.method,
        problem=method.problem,
        dims=method.dims,
        grid_points=method.grid_points,
        seed=method.seed,
        grid_axes=axes,
        logL_grid=_mask_to_logL(raster_grids),
        profiled_params_grid=method.profiled_params_grid,
        cell_evals=method.cell_evals,
        total_evals=method.total_evals,
        n_cells_capped=0,
        wall_time=method.wall_time,
        extra=method.extra,
    )
    return coverage(oracle, synthetic)


def _mask_to_logL(raster: dict) -> np.ndarray:
    """Encode two nested masks (68 inside 95) as a synthetic logL grid.

    Inside-68 -> 0 (peak), 95-but-not-68 -> -DELTA_LOGL_68_2D - 1e-6
    (just below 68 threshold), outside-95 -> -DELTA_LOGL_95_2D - 1.
    """
    m68 = raster["68"]
    m95 = raster["95"]
    out = np.full(m68.shape, -DELTA_LOGL_95_2D - 1.0, dtype=float)
    out[m95] = -DELTA_LOGL_68_2D - 1e-6
    out[m68] = 0.0
    return out


def _polygon_to_grid_mask(poly: np.ndarray, axes) -> np.ndarray:
    """Point-in-polygon for every grid cell. Polygon is a closed Nx2 array."""
    from matplotlib.path import Path

    if poly.shape[0] < 3:
        return np.zeros((axes[0].size, axes[1].size), dtype=bool)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    path = Path(poly)
    xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return path.contains_points(pts).reshape(xx.shape)


def evals_to_threshold(oracle: ProjectionResult, method: ProjectionResult,
                       epsilons=(1.0, 0.1, 0.01, 0.001)) -> dict[str, float]:
    """For each ε, return total target-function evaluations if the method's
    max-cell ΔlogL is below ε, else NaN (the method "failed" at that ε)."""
    quality = solution_quality(oracle, method)
    max_delta = quality["max_delta"]
    out: dict[str, float] = {}
    for eps in epsilons:
        key = f"eps_{eps:g}"
        if not np.isfinite(max_delta):
            out[key] = float("nan")
        elif max_delta <= eps:
            out[key] = float(method.total_evals)
        else:
            out[key] = float("nan")
    out["max_delta_achieved"] = float(max_delta)
    return out


def summarise(oracle: ProjectionResult, method: ProjectionResult) -> dict:
    """Convenience: bundle all metrics for one (oracle, method) pair."""
    if method.method.startswith("iminuit_mncontour"):
        cov = coverage_from_polygons(oracle, method)
        qual = {
            "max_delta": float("nan"),
            "rms_delta_full": float("nan"),
            "rms_delta_roi": float("nan"),
            "n_cells_failed": 0,
            "n_roi_cells": 0,
        }
    else:
        qual = solution_quality(oracle, method)
        cov = coverage(oracle, method)
    return {
        "method": method.method,
        "seed": method.seed,
        "total_evals": int(method.total_evals),
        "wall_time": float(method.wall_time),
        "n_cells_capped": int(method.n_cells_capped),
        **qual,
        **cov,
        **evals_to_threshold(oracle, method),
    }
