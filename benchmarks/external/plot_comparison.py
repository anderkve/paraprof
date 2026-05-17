"""Generate every paper figure from a directory of result JSONs.

Deterministic given the JSONs — no live re-runs of any optimiser. The plot
scripts here own the visual styling; data shaping is delegated to
``metrics.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .adapters.base import ProjectionResult
from .metrics import (
    DELTA_LOGL_68_2D,
    DELTA_LOGL_95_2D,
    solution_quality,
    coverage,
    coverage_from_polygons,
    summarise,
)
from .oracle import load_oracle

# --- Layout & visual conventions ---------------------------------------------

BODY_METHODS = [
    ("paraprof_default", "paraprof"),
    ("iminuit_grid",     "iminuit"),
    ("scipy_de",         "scipy DE"),
]
APPENDIX_METHODS = [
    ("paraprof_kernel",     "paraprof (kernel)"),
    ("scipy_lbfgsb",        "L-BFGS-B"),
    ("nlopt_crs2_bobyqa",   "nlopt"),
]
EVALS_METHODS = [
    ("paraprof_default",    "paraprof"),
    ("paraprof_kernel",     "paraprof (kernel)"),
    ("iminuit_grid",        "iminuit"),
    ("scipy_de",            "scipy DE"),
    ("scipy_lbfgsb",        "scipy L-BFGS-B"),
    ("nlopt_crs2_bobyqa",   "nlopt CRS2+BOBYQA"),
]
CONTOUR_METHODS = [
    ("paraprof_default",    "paraprof"),
    ("iminuit_grid",        "iminuit (per-cell)"),
    ("scipy_de",            "scipy DE"),
    ("iminuit_mncontour",   "MIGRAD+MNCONTOUR"),
]

METHOD_COLOURS = {
    "paraprof_default":   "#1f77b4",
    "paraprof_kernel":    "#aec7e8",
    "iminuit_grid":       "#d62728",
    "iminuit_mncontour":  "#ff7f0e",
    "scipy_de":           "#2ca02c",
    "scipy_lbfgsb":       "#9467bd",
    "nlopt_crs2_bobyqa":  "#8c564b",
}
METHOD_LINESTYLES = {
    "paraprof_default":   "-",
    "paraprof_kernel":    "-",
    "iminuit_grid":       "--",
    "iminuit_mncontour":  "--",
    "scipy_de":           ":",
    "scipy_lbfgsb":       (0, (5, 1, 1, 1)),
    "nlopt_crs2_bobyqa":  "-.",
}
METHOD_MARKERS = {
    "paraprof_default":   "o",
    "paraprof_kernel":    "s",
    "iminuit_grid":       "^",
    "iminuit_mncontour":  "v",
    "scipy_de":           "D",
    "scipy_lbfgsb":       "P",
    "nlopt_crs2_bobyqa":  "X",
}

# Match the README plots: white star + dashed white CL contours on heatmaps.
CONTOUR_LINE_KWARGS = {
    "colors": "white",
    "linewidths": 1.0,
}


# --- Result loading & grouping -----------------------------------------------

RESULT_NAME_RE = re.compile(
    r"^(?P<problem>[^_]+_\d+d)__dims-(?P<dims>[\d_]+)__"
    r"(?P<method>[a-z0-9_]+?)__seed-(?P<seed>\d+)\.json$"
)


def _load_one(path: Path) -> ProjectionResult:
    with path.open() as f:
        return ProjectionResult.from_dict(json.load(f))


def load_results(runs_dir: Path) -> dict[tuple, list[ProjectionResult]]:
    """Group result JSONs by (problem, dims_tuple, method)."""
    out: dict[tuple, list[ProjectionResult]] = defaultdict(list)
    for p in sorted(runs_dir.glob("*.json")):
        try:
            r = _load_one(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not load {p.name}: {exc}")
            continue
        key = (r.problem, tuple(r.dims), r.method)
        out[key].append(r)
    return out


def pick_seed(rs: list[ProjectionResult], oracle: ProjectionResult) -> ProjectionResult:
    """Pick the median-quality seed for visualisations."""
    if len(rs) == 1:
        return rs[0]
    qualities = []
    for r in rs:
        try:
            q = solution_quality(oracle, r).get("rms_delta_full", float("inf"))
        except Exception:
            q = float("inf")
        qualities.append(q)
    order = np.argsort(qualities)
    return rs[order[len(rs) // 2]]


# --- Figure 1 / Appendix 1: per-test-function panel --------------------------

def _heatmap_panel(ax, axes, grid, *, vmin, vmax, cmap="viridis",
                   contour_levels=None, title=""):
    extent = [axes[1].min(), axes[1].max(), axes[0].min(), axes[0].max()]
    im = ax.imshow(grid, extent=extent, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if contour_levels is not None:
        try:
            ax.contour(axes[1], axes[0], grid, levels=contour_levels,
                       **CONTOUR_LINE_KWARGS,
                       linestyles=["dashed", "solid"])
        except Exception:
            pass
    # Best-fit star
    if np.isfinite(grid).any():
        flat = np.argmax(np.where(np.isfinite(grid), grid, -np.inf))
        i, j = np.unravel_index(flat, grid.shape)
        ax.plot(axes[1][j], axes[0][i], marker="*", color="white",
                markersize=8, markeredgecolor="black", markeredgewidth=0.5)
    ax.set_title(title)
    return im


def plot_per_function_panel(problem: str, results: dict, oracles: dict,
                            method_list, out_path: Path) -> None:
    projections = sorted({k[1] for k in results.keys() if k[0] == problem})
    if not projections:
        print(f"[skip] no projections for {problem}")
        return
    n_methods = 1 + len(method_list)  # +1 for oracle column
    panel_w = 2.0
    panel_h = 2.4
    fig, axes_arr = plt.subplots(
        len(projections), n_methods,
        figsize=(panel_w * n_methods + 0.8, panel_h * len(projections) + 0.5),
        squeeze=False, sharex="col",
    )

    last_im = None
    last_vmin = last_vmax = None
    for row, dims in enumerate(projections):
        oracle = oracles.get((problem, dims))
        if oracle is None:
            continue
        finite = oracle.logL_grid[np.isfinite(oracle.logL_grid)]
        if finite.size == 0:
            continue
        vmax = float(finite.max())
        vmin = float(max(vmax - 10.0, finite.min()))
        last_vmin, last_vmax = vmin, vmax
        cl_levels = [vmax - DELTA_LOGL_95_2D, vmax - DELTA_LOGL_68_2D]

        im = _heatmap_panel(
            axes_arr[row, 0], oracle.grid_axes, oracle.logL_grid,
            vmin=vmin, vmax=vmax, contour_levels=cl_levels,
            title=("oracle" if row == 0 else ""),
        )
        last_im = im
        axes_arr[row, 0].set_ylabel(f"$x_{{{dims[0]}}}$")

        for col, (method_id, method_label) in enumerate(method_list, start=1):
            ax = axes_arr[row, col]
            seeds = results.get((problem, dims, method_id), [])
            if not seeds:
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, color="0.4", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(method_label)
                continue
            r = pick_seed(seeds, oracle)
            grid = r.logL_grid.copy()
            grid[~np.isfinite(grid)] = vmin
            _heatmap_panel(ax, r.grid_axes, grid, vmin=vmin, vmax=vmax,
                           contour_levels=cl_levels,
                           title=(method_label if row == 0 else ""))
            ax.tick_params(labelleft=False)

        for col in range(n_methods):
            axes_arr[row, col].set_xlabel(f"$x_{{{dims[1]}}}$")
            axes_arr[row, col].set_aspect("equal", adjustable="box")

    # Single shared colorbar to the right of all panels.
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.ravel().tolist(),
                            location="right", fraction=0.018, pad=0.02,
                            shrink=0.9)
        cbar.set_label(r"$\log\mathcal{L}$")

    fig.suptitle(problem, y=1.0)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


# --- Figure 2: summary table -------------------------------------------------

def _collect_metric_rows(results, oracles) -> list[dict]:
    rows: list[dict] = []
    for (problem, dims, method), seeds in results.items():
        oracle = oracles.get((problem, dims))
        if oracle is None:
            continue
        for r in seeds:
            rows.append({
                "problem": problem,
                "dims": dims,
                **summarise(oracle, r),
            })
    return rows


def plot_summary_table(rows: list[dict], out_path: Path) -> None:
    """Median per (method, problem) over seeds. One block of columns per problem."""
    if not rows:
        return
    methods = sorted({r["method"] for r in rows})
    problems = sorted({r["problem"] for r in rows})
    cell_cols = ["max_delta", "rms_delta_roi", "iou_68", "iou_95",
                 "hausdorff_95", "eps_0.01"]
    short = {"max_delta": "max Δ", "rms_delta_roi": "RMS Δ (ROI)",
             "iou_68": "IoU₆₈", "iou_95": "IoU₉₅",
             "hausdorff_95": "Hausdorff₉₅", "eps_0.01": "evals @ ε=0.01"}

    fig_w = 1.6 + 1.05 * len(cell_cols) * len(problems)
    fig_h = 0.5 + 0.28 * (len(methods) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    header_top = [""] + [p for p in problems for _ in cell_cols]
    header_bottom = [""] + [short[c] for _ in problems for c in cell_cols]
    table_data = []
    for m in methods:
        row = [m]
        for p in problems:
            for c in cell_cols:
                vals = [r[c] for r in rows
                        if r["method"] == m and r["problem"] == p
                        and r.get(c) is not None and np.isfinite(r.get(c, float("nan")))]
                if vals:
                    val = float(np.median(vals))
                    row.append(_fmt(val, c))
                else:
                    row.append("—")
        table_data.append(row)

    tbl = ax.table(cellText=[header_bottom] + table_data,
                   cellLoc="center",
                   colLabels=header_top,
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


def _fmt(v: float, key: str) -> str:
    if key in ("iou_68", "iou_95"):
        return f"{v:.3f}"
    if key == "eps_0.01":
        return f"{int(v):d}"
    return f"{v:.2g}"


# --- Figure 3: evals-to-ε ----------------------------------------------------

def plot_evals_to_eps(rows: list[dict], problems: list[str], out_path: Path) -> None:
    epsilons = [1.0, 0.1, 0.01, 0.001]
    n_panels = len(problems)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(7.5, 5.5),
                                 squeeze=False, sharex=True)
    for ax_flat, problem in zip(axes_arr.ravel(), problems):
        method_ids = [m for m, _ in EVALS_METHODS]
        for method, label in EVALS_METHODS:
            xs = []
            ys = []
            yerr_lo = []
            yerr_hi = []
            for eps in epsilons:
                vals = [r[f"eps_{eps:g}"] for r in rows
                        if r["method"] == method and r["problem"] == problem
                        and np.isfinite(r.get(f"eps_{eps:g}", float("nan")))]
                if not vals:
                    continue
                med = float(np.median(vals))
                lo = float(np.quantile(vals, 0.25)) if len(vals) > 1 else med
                hi = float(np.quantile(vals, 0.75)) if len(vals) > 1 else med
                xs.append(eps)
                ys.append(med)
                yerr_lo.append(lo)
                yerr_hi.append(hi)
            if not xs:
                continue
            ax_flat.plot(xs, ys, label=label,
                         color=METHOD_COLOURS.get(method, "0.4"),
                         linestyle=METHOD_LINESTYLES.get(method, "-"),
                         marker=METHOD_MARKERS.get(method, "o"))
            ax_flat.fill_between(xs, yerr_lo, yerr_hi, alpha=0.15,
                                 color=METHOD_COLOURS.get(method, "0.4"),
                                 linewidth=0)
        ax_flat.set_xscale("log")
        ax_flat.set_yscale("log")
        ax_flat.invert_xaxis()
        ax_flat.set_title(problem)
        ax_flat.set_xlabel(r"$\varepsilon$ (max-cell $\Delta\log\mathcal{L}$)")
        ax_flat.set_ylabel("target evaluations")
    # Turn off any unused panels.
    for ax in axes_arr.ravel()[len(problems):]:
        ax.set_visible(False)
    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.18, 0.5),
               frameon=False)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


# --- Figure 4: contour overlay -----------------------------------------------

def _extract_polygon(method_result: ProjectionResult, cl: str):
    contours = method_result.extra.get("contours", {}) or {}
    poly = contours.get(cl)
    if poly is None or len(poly) < 3:
        return None
    arr = np.asarray(poly, dtype=float)
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def _level_set_polygon(grid_result: ProjectionResult, level: float):
    """Use matplotlib's contour engine to extract a polygon at ``level`` from a grid."""
    axes = grid_result.grid_axes
    grid = grid_result.logL_grid
    g = grid.copy()
    g[~np.isfinite(g)] = -np.inf
    try:
        cs = plt.contour(axes[1], axes[0], g, levels=[level])
    except Exception:
        plt.close("all")
        return []
    paths = []
    for collection in cs.collections:
        for path in collection.get_paths():
            paths.append(path.vertices)
    plt.close("all")
    return paths


def plot_contour_overlay(results: dict, oracles: dict, problems: list[str],
                         out_path: Path) -> None:
    n_cols = 2
    n_rows = (len(problems) + n_cols - 1) // n_cols
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(7.5, 6.0), squeeze=False)
    for ax, problem in zip(axes_arr.ravel(), problems):
        # Use the FIRST projection per problem for the overlay.
        proj_dims = sorted({k[1] for k in results.keys() if k[0] == problem})
        if not proj_dims:
            continue
        dims = proj_dims[0]
        oracle = oracles.get((problem, dims))
        if oracle is None:
            ax.set_visible(False)
            continue
        # Truth contour from oracle.
        g_max = float(np.nanmax(oracle.logL_grid))
        for cl_label, level_offset, style in (
            ("68%", -DELTA_LOGL_68_2D, "-"),
            ("95%", -DELTA_LOGL_95_2D, "--"),
        ):
            cs = ax.contour(oracle.grid_axes[1], oracle.grid_axes[0],
                             oracle.logL_grid, levels=[g_max + level_offset],
                             colors="black", linewidths=1.4, linestyles=style)
        for method, label in CONTOUR_METHODS:
            rs = results.get((problem, dims, method), [])
            if not rs:
                continue
            r = pick_seed(rs, oracle)
            colour = METHOD_COLOURS.get(method, "0.4")
            style = METHOD_LINESTYLES.get(method, "-")
            if method == "iminuit_mncontour":
                for cl, ls in (("68", "-"), ("95", "--")):
                    poly = _extract_polygon(r, cl)
                    if poly is not None:
                        ax.plot(poly[:, 0], poly[:, 1], color=colour, linestyle=ls,
                                linewidth=1.0, alpha=0.85,
                                label=label if cl == "68" else None)
            else:
                g_max_r = float(np.nanmax(r.logL_grid))
                for level_offset, ls in ((-DELTA_LOGL_68_2D, "-"),
                                         (-DELTA_LOGL_95_2D, "--")):
                    g = r.logL_grid.copy()
                    g[~np.isfinite(g)] = g_max_r - DELTA_LOGL_95_2D - 5
                    try:
                        ax.contour(r.grid_axes[1], r.grid_axes[0], g,
                                   levels=[g_max_r + level_offset],
                                   colors=[colour], linewidths=1.0, linestyles=ls,
                                   alpha=0.85)
                    except Exception:
                        pass
                # Manual legend entry per method.
                ax.plot([], [], color=colour, linestyle=style, label=label)

        ax.set_title(f"{problem}  (dims {list(dims)})")
        ax.set_xlabel(f"$x_{{{dims[1]}}}$")
        ax.set_ylabel(f"$x_{{{dims[0]}}}$")
        ax.legend(fontsize=7, loc="best")
    for ax in axes_arr.ravel()[len(problems):]:
        ax.set_visible(False)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved {out_path}")


# --- Driver ------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path,
                        default=Path("benchmarks/external/results"))
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/external/results/figures"))
    parser.add_argument("--style", type=Path,
                        default=Path(__file__).parent / "style.mplstyle")
    args = parser.parse_args(argv)

    plt.style.use(str(args.style))
    args.out.mkdir(parents=True, exist_ok=True)

    runs_dir = args.results_root / "runs"
    results = load_results(runs_dir)
    if not results:
        print(f"No results found under {runs_dir}. Run benchmarks.external.run_comparison first.")
        return 1

    # Load oracles for every (problem, dims) pair encountered.
    oracles: dict[tuple, ProjectionResult] = {}
    for (problem, dims, _method) in results.keys():
        key = (problem, dims)
        if key in oracles:
            continue
        oracle = load_oracle(problem, dims, tuple(results[(problem, dims, _method)][0].grid_points))
        if oracle is not None:
            oracles[key] = oracle
    if not oracles:
        print("No oracle JSONs found; build oracles first.")
        return 1

    problems = sorted({k[0] for k in results.keys()})

    # 1. Per-test-function body figure.
    for problem in problems:
        plot_per_function_panel(problem, results, oracles, BODY_METHODS,
                                args.out / f"body_{problem}.pdf")
    # 2. Per-test-function appendix figure.
    for problem in problems:
        plot_per_function_panel(problem, results, oracles, APPENDIX_METHODS,
                                args.out / f"appendix_{problem}.pdf")

    rows = _collect_metric_rows(results, oracles)
    # 3. Summary table.
    plot_summary_table(rows, args.out / "summary_table.pdf")
    # 4. Evals-to-eps.
    plot_evals_to_eps(rows, problems, args.out / "evals_to_eps.pdf")
    # 5. Contour overlay.
    plot_contour_overlay(results, oracles, problems, args.out / "contour_overlay.pdf")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
