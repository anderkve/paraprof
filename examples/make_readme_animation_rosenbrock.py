"""
Render a README GIF for the 4D Rosenbrock function with all six 2D
projections scanned in sequence and shown in a 3 x 2 panel grid:

    row 0:  (x0, x1)   (x0, x2)
    row 1:  (x0, x3)   (x1, x2)
    row 2:  (x1, x3)   (x2, x3)

The first projection runs the full ParaProf pipeline (initial global
L-BFGS-B sweep -> dynamic grid activation -> differential-evolution
refinement -> patching waves). Each subsequent projection is warm-started
from the accumulated global solution pool, so its initial L-BFGS-B sweep
is skipped and dynamic activation can start immediately. The animation
shows, in every frame, the live grid for the projection currently being
scanned, the frozen final grid for projections that have already
finished, and an inert grey placeholder for projections still queued.

Style, timing and frame budget mirror ``make_readme_animation.py``.

Run with MPI:

    mpiexec -n 4 python examples/make_readme_animation_rosenbrock.py

Requires the optional ``viz`` dependencies (matplotlib) plus ``imageio``.
``gifsicle`` on PATH automatically halves the final GIF size.
"""

from __future__ import annotations

import collections
import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpi4py import MPI

import imageio.v2 as imageio

from paraprof import (
    ProfileProjector,
    run_projection,
    terminate_workers,
    worker_main,
    get_test_function,
    set_log_level,
)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

FUNC_NAME = "rosenbrock_4d"
GRID_PER_DIM = 50
N_INITIAL_OPT = 20
ROI_THRESHOLD = 10.0
POP_PER_CELL = 3
LBFGSB_ITER = 15
MAX_PATCHING_WAVES = 10

SNAPSHOT_INTERVAL_FIRST = 25  # interval (in target calls) for projection 1
SNAPSHOT_INTERVAL_OTHER = 4   # finer sampling for the warm-started later projs
SCATTER_HISTORY = 300

# Six 2D projections, in the order asked for in the README animation:
#     (x0, x1), (x0, x2), (x0, x3), (x1, x2), (x1, x3), (x2, x3)
PROJECTIONS = [
    {"dims": [0, 1], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [0, 2], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [0, 3], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [1, 2], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [1, 3], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [2, 3], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
]
N_PROJ = len(PROJECTIONS)
N_ROWS = (N_PROJ + 1) // 2

OUT_DIR = os.path.join(os.path.dirname(__file__), "example_plots", "animation")
GIF_PATH = os.path.join(OUT_DIR, "paraprof_rosenbrock_4D.gif")

VMIN = -float(ROI_THRESHOLD)
VMAX = 0.0
CONTOUR_LEVELS = [-3.0, -1.0]
CONTOUR_LINESTYLES = ["--", "-"]
CONTOUR_LINEWIDTHS = [1.0, 1.6]

# Visible axis range applied to every panel (the data extent is still the
# full parameter-bounds rectangle; we just crop the view to where the
# Rosenbrock valley actually lives).
PANEL_XLIM = (-4.0, 4.0)
PANEL_YLIM = (-2.0, 6.0)


# ---------------------------------------------------------------------------
# Snapshot capture (identical to make_readme_animation.py)
# ---------------------------------------------------------------------------


class SnapshotCapturer:
    """Hooks the ProfileProjector to record periodic state snapshots."""

    def __init__(self, sampler, interval: int, scatter_history: int):
        self.sampler = sampler
        self.interval = interval
        self.scatter_history = scatter_history
        self.scatter_buf: collections.deque = collections.deque(maxlen=scatter_history)
        self.frames: list[dict] = []
        self.current_proj_idx = 0
        self._last_snapshot_calls = -1
        self._max_active_seen = 0
        self._wrap_register()

    def _wrap_register(self):
        orig = self.sampler._register_target_call

        def wrapped(params, target_val):
            orig(params, target_val)
            self.scatter_buf.append(
                (np.asarray(params, dtype=float).copy(), float(target_val))
            )
            if (self.sampler.target_calls % self.interval == 0
                    and self.sampler.target_calls != self._last_snapshot_calls):
                self._last_snapshot_calls = self.sampler.target_calls
                self.capture()

        self.sampler._register_target_call = wrapped

    def reset_for_new_projection(self):
        self._max_active_seen = 0
        self.scatter_buf.clear()

    def _classify_phase(self, forced: Optional[str] = None) -> str:
        if forced is not None:
            return forced
        s = self.sampler
        pop_size = len(s.population)
        active_count = sum(
            1 for st in s.population.values() if st.get("status") == "active"
        )
        gen = int(getattr(s, "current_generation", 0) or 0)
        self._max_active_seen = max(self._max_active_seen, active_count)
        if pop_size == 0:
            return "initial_global_search"
        if gen == 0:
            return "dynamic_activation"
        if active_count == 0:
            return "patching"
        active_ratio = active_count / max(pop_size, 1)
        if active_ratio < 0.35 and pop_size >= self._max_active_seen * 0.9:
            return "de_refinement"
        return "dynamic_activation"

    def capture(self, forced_phase: Optional[str] = None):
        s = self.sampler
        best_idx = None
        best_v = -np.inf
        for idx, v in s.profile_likelihood_grid.items():
            if v > best_v:
                best_v = v
                best_idx = idx
        recent = (np.array([p for p, _ in self.scatter_buf])
                  if self.scatter_buf else np.empty((0, s.dims)))
        self.frames.append({
            "target_calls": int(s.target_calls),
            "global_max": float(s.global_max_target_val),
            "proj_idx": self.current_proj_idx,
            "proj_dims": tuple(s.projection_dims),
            "grid_values": dict(s.profile_likelihood_grid),
            "active_cells": {idx for idx, state in s.population.items()
                             if state.get("status") == "active"},
            "best_fit_idx": best_idx,
            "best_fit_value": best_v if np.isfinite(best_v) else None,
            "recent_samples": recent,
            "phase": self._classify_phase(forced=forced_phase),
        })


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _build_grid_image(grid_values: dict, grid_shape: tuple,
                      global_max: float) -> tuple[np.ndarray, np.ndarray]:
    img = np.full(grid_shape, -np.inf)
    for idx, v in grid_values.items():
        img[idx] = v
    finite = np.isfinite(img)
    if not finite.any():
        return img, finite
    ref = max(global_max, img[finite].max())
    img = np.where(finite, img - ref, -np.inf)
    return img, finite


def _select_recent_samples(samples: np.ndarray, n_recent: int):
    if samples is None or len(samples) == 0:
        return None, None
    n = len(samples)
    if n_recent >= n:
        return samples, None
    return samples[-n_recent:], samples[:-n_recent]


def _draw_panel(ax, axes_x, axes_y, grid_img, mask,
                active_cells, best_fit_idx,
                scatter_xy_recent, scatter_xy_old,
                title, cmap, show_scatter):
    extent = [axes_x[0], axes_x[-1], axes_y[0], axes_y[-1]]
    masked = np.ma.masked_where(~mask, grid_img)
    im = ax.imshow(masked.T, extent=extent, origin="lower",
                   aspect="equal", cmap=cmap, vmin=VMIN, vmax=VMAX,
                   interpolation="nearest")

    if mask.any():
        X, Y = np.meshgrid(axes_x, axes_y)
        try:
            ax.contour(X, Y, np.where(mask, grid_img, np.nan).T,
                       levels=sorted(CONTOUR_LEVELS),
                       colors="white",
                       linestyles=CONTOUR_LINESTYLES,
                       linewidths=CONTOUR_LINEWIDTHS,
                       alpha=0.95)
        except Exception:
            pass

    if show_scatter:
        if scatter_xy_old is not None and len(scatter_xy_old):
            ax.scatter(scatter_xy_old[:, 0], scatter_xy_old[:, 1],
                       s=4, c="#ff6420", alpha=0.32, linewidths=0, zorder=4)
        if scatter_xy_recent is not None and len(scatter_xy_recent):
            ax.scatter(scatter_xy_recent[:, 0], scatter_xy_recent[:, 1],
                       s=10, c="#ff6420", alpha=0.90, linewidths=0.3,
                       edgecolors="#8a3010", zorder=5)

    if best_fit_idx is not None:
        bx = axes_x[best_fit_idx[0]]
        by = axes_y[best_fit_idx[1]]
        ax.scatter([bx], [by], s=110, marker="*", c="white",
                   edgecolors="black", linewidths=0.9, zorder=10)

    # Crop the visible view to where the Rosenbrock valley lives (data
    # extent stays at the full bounds rectangle).
    ax.set_xlim(*PANEL_XLIM)
    ax.set_ylim(*PANEL_YLIM)
    ax.set_facecolor("#dadada")
    ax.tick_params(axis="both", length=3.0, pad=3)
    ax.set_xlabel(title["xlabel"])
    ax.set_ylabel(title["ylabel"])
    ax.grid(True, linestyle=":", linewidth=0.4, color="white", alpha=0.5)
    return im


def render_animation(frames, final_states, bounds, gif_path):
    if not frames:
        raise RuntimeError("No snapshots captured -- nothing to render.")

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dadada")

    # 3 rows x 2 cols of square heat-map panels, plus a header strip on top.
    # We use the same per-row colour-bar pattern as the 2-panel Himmelblau
    # animation: right panel gets a real colour-bar appended via
    # make_axes_locatable, left panel gets an identically-sized invisible
    # appendage to keep both panels in a row exactly the same width (so
    # aspect='equal' renders them at the same size).
    #
    # The layout uses an OUTER 2-row gridspec (info bar + panel block) and
    # a 3 x 2 INNER gridspec for the panels themselves. This lets us set
    # the info -> first-row gap independently of the panel-to-panel gap.
    fig = plt.figure(figsize=(7.6, 10.4), dpi=110, facecolor="white")
    outer_gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[0.06, float(N_ROWS)],
        left=0.08, right=0.91, top=0.97, bottom=0.06,
        # Halve the gap between the info text and the first row of panels.
        hspace=0.06,
    )
    inner_gs = outer_gs[1].subgridspec(
        nrows=N_ROWS, ncols=2,
        width_ratios=[1.0, 1.0],
        # Panel-to-panel gaps stay roughly the same as before; horizontal
        # gap is reduced by 20% (0.22 -> 0.176).
        hspace=0.23, wspace=0.176,
    )

    info_ax = fig.add_subplot(outer_gs[0])
    info_ax.axis("off")

    panel_axes: list = []
    caxes: list = []
    for r in range(N_ROWS):
        ax_l = fig.add_subplot(inner_gs[r, 0])
        ax_r = fig.add_subplot(inner_gs[r, 1])
        panel_axes.append(ax_l)
        panel_axes.append(ax_r)
        # Phantom on the LEFT panel, real colour-bar on the RIGHT panel.
        ghost = make_axes_locatable(ax_l).append_axes("right", size="4.6%", pad=0.10)
        ghost.set_visible(False)
        cax = make_axes_locatable(ax_r).append_axes("right", size="4.6%", pad=0.10)
        caxes.append(cax)

    sm = ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=cmap)
    sm.set_array([])
    for cax in caxes:
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label(r"$\Delta \log L = \log L - \log L_{\max}$", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    bounds = np.asarray(bounds)
    grid_shape = (GRID_PER_DIM + 1, GRID_PER_DIM + 1)
    # Per-projection 1D axis arrays in the order PROJECTIONS lists them.
    proj_axes = []
    for proj in PROJECTIONS:
        d0, d1 = proj["dims"]
        proj_axes.append([
            np.linspace(bounds[d0, 0], bounds[d0, 1], GRID_PER_DIM + 1),
            np.linspace(bounds[d1, 0], bounds[d1, 1], GRID_PER_DIM + 1),
        ])

    # -------------------------------------------------------------------
    # Frame budget. The first projection runs the full pipeline and gets
    # the largest share; the warm-started later projections each get a
    # smaller cap because the run finishes quickly.
    # -------------------------------------------------------------------
    per_proj_frames = [[f for f in frames if f["proj_idx"] == k]
                       for k in range(N_PROJ)]

    max_first = 95
    max_other = 35
    transition_hold = 8     # frames held at the end of each projection
    final_hold = 56         # clean trailing hold (~2 s at 28 FPS)

    chosen: list = []
    for k, projection_frames in enumerate(per_proj_frames):
        if not projection_frames:
            continue
        max_cap = max_first if k == 0 else max_other
        stride = max(1, len(projection_frames) // max_cap)
        sub = projection_frames[::stride]
        if sub[-1] is not projection_frames[-1]:
            sub.append(projection_frames[-1])
        chosen.extend(sub)
        if k < N_PROJ - 1:
            chosen.extend([sub[-1]] * transition_hold)

    if not chosen:
        chosen = list(frames)

    # Build a marker-free copy of the very last frame so the animation
    # closes on a clean view with all six projections complete.
    last_snap = chosen[-1]
    last_recent = last_snap["recent_samples"]
    empty_recent = np.empty((0, last_recent.shape[1] if last_recent.size else 4))
    clean_last = dict(last_snap)
    clean_last["recent_samples"] = empty_recent
    clean_last["active_cells"] = set()
    chosen = chosen + [clean_last] * final_hold

    images = []
    final_global_max = max(
        f["global_max"] for f in frames if np.isfinite(f["global_max"])
    )

    for i, snap in enumerate(chosen):
        info_ax.cla()
        info_ax.axis("off")
        for ax in panel_axes:
            ax.cla()

        cur = snap["proj_idx"]
        snap_recent, snap_old = _select_recent_samples(
            snap["recent_samples"], n_recent=90,
        )

        for k, proj in enumerate(PROJECTIONS):
            ax = panel_axes[k]
            d0, d1 = proj["dims"]
            xlabel = rf"$x_{{{d0}}}$"
            ylabel = rf"$x_{{{d1}}}$"
            axis_x, axis_y = proj_axes[k]

            if k < cur:
                # Projection k already finished: show its frozen final state,
                # no scatter, no active markers.
                state = final_states[k]
                if state is None:
                    continue
                grid_img, mask = _build_grid_image(
                    state["grid_values"], grid_shape, final_global_max,
                )
                _draw_panel(
                    ax, axis_x, axis_y,
                    grid_img, mask, set(),
                    state["best_fit_idx"],
                    None, None,
                    title={"xlabel": xlabel, "ylabel": ylabel},
                    cmap=cmap, show_scatter=False,
                )
            elif k == cur:
                # Live projection: scatter shown, active heat-map drawn.
                grid_img, mask = _build_grid_image(
                    snap["grid_values"], grid_shape,
                    max(snap["global_max"], final_global_max),
                )
                recent_xy = (snap_recent[:, [d0, d1]]
                             if snap_recent is not None else None)
                old_xy = (snap_old[:, [d0, d1]]
                          if snap_old is not None else None)
                _draw_panel(
                    ax, axis_x, axis_y,
                    grid_img, mask, snap["active_cells"],
                    snap["best_fit_idx"],
                    recent_xy, old_xy,
                    title={"xlabel": xlabel, "ylabel": ylabel},
                    cmap=cmap, show_scatter=True,
                )
            else:
                # Queued projection: empty placeholder, axes only.
                empty_img = np.full(grid_shape, -np.inf)
                empty_mask = np.zeros(grid_shape, dtype=bool)
                _draw_panel(
                    ax, axis_x, axis_y,
                    empty_img, empty_mask, set(),
                    None, None, None,
                    title={"xlabel": xlabel, "ylabel": ylabel},
                    cmap=cmap, show_scatter=False,
                )

        info_ax.text(
            0.0, 0.5,
            f"Target-function evaluations: {snap['target_calls']:,}",
            transform=info_ax.transAxes, ha="left", va="center",
            fontsize=12, color="#222",
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba[..., :3].copy())

        if i % 20 == 0 or i == len(chosen) - 1:
            print(f"  rendered frame {i + 1}/{len(chosen)} "
                  f"(stage={snap['phase']}, proj={snap['proj_idx']}, "
                  f"calls={snap['target_calls']:,})",
                  flush=True)

    plt.close(fig)

    print(f"Encoding GIF ({len(images)} frames) -> {gif_path}", flush=True)
    imageio.mimsave(gif_path, images, format="GIF", fps=28, loop=0)
    raw_mb = os.path.getsize(gif_path) / 1e6
    print(f"Wrote {gif_path} ({raw_mb:.2f} MB, pre-optimisation)")

    if shutil.which("gifsicle"):
        try:
            subprocess.run(
                ["gifsicle", "-O3", "--colors", "128", gif_path, "-o", gif_path],
                check=True,
            )
            opt_mb = os.path.getsize(gif_path) / 1e6
            print(f"gifsicle -O3 --colors 128: {raw_mb:.2f} MB -> {opt_mb:.2f} MB")
        except subprocess.CalledProcessError as exc:
            print(f"gifsicle optimisation skipped (exit {exc.returncode})")
    else:
        print("gifsicle not on PATH; skipping optimisation. "
              "Install it (e.g. `apt-get install gifsicle`) to shrink the GIF.")


# ---------------------------------------------------------------------------
# Master / worker entry points
# ---------------------------------------------------------------------------


def run_master(comm):
    np.random.seed(20250515)
    set_log_level("INFO")

    target_func, bounds, _ = get_test_function(FUNC_NAME)

    os.makedirs(OUT_DIR, exist_ok=True)

    with ProfileProjector(
        target_func=target_func,
        bounds=bounds,
        projections=PROJECTIONS,
        roi_threshold=ROI_THRESHOLD,
        pop_per_grid_point=POP_PER_CELL,
        n_initial_optimizations=N_INITIAL_OPT,
        lbfgsb_max_iter=LBFGSB_ITER,
        max_patching_waves=MAX_PATCHING_WAVES,
    ) as sampler:

        cap = SnapshotCapturer(sampler, SNAPSHOT_INTERVAL_FIRST, SCATTER_HISTORY)
        comm.bcast(sampler.target_func, root=0)

        final_states: list[Optional[dict]] = [None] * N_PROJ

        for proj_idx, proj in enumerate(PROJECTIONS):
            cap.current_proj_idx = proj_idx
            cap.reset_for_new_projection()
            cap.interval = (SNAPSHOT_INTERVAL_FIRST if proj_idx == 0
                            else SNAPSHOT_INTERVAL_OTHER)
            cap.scatter_buf = collections.deque(maxlen=SCATTER_HISTORY)

            # We drive run_projection() directly rather than going through
            # run_all_projections(), so we have to call
            # _reset_for_new_projection on the sampler ourselves between
            # projections (otherwise projection k > 0 would inherit
            # projection k-1's grid, population and initial_maxima).
            if proj_idx > 0:
                sampler._reset_for_new_projection(proj)

            # Snapshot the empty initial state at the very start of each
            # projection so the panel handover begins from a clean panel.
            cap.capture(forced_phase=(
                "initial_global_search" if proj_idx == 0
                else "dynamic_activation"
            ))

            run_projection(
                comm=comm,
                sampler=sampler,
                projection_config=proj,
                save_plots=False,
                # Enable warm-starting from the global solution pool for
                # every projection after the first (matches
                # run_all_projections).
                skip_init_opt_on_warm_start=(proj_idx > 0),
                myrank=0,
            )

            cap.capture(forced_phase="frozen")

            best_idx = None
            best_v = -np.inf
            for idx, v in sampler.profile_likelihood_grid.items():
                if v > best_v:
                    best_v = v
                    best_idx = idx
            final_states[proj_idx] = {
                "grid_values": dict(sampler.profile_likelihood_grid),
                "best_fit_idx": best_idx,
                "best_fit_value": best_v,
            }

        print(f"Captured {len(cap.frames)} snapshots across "
              f"{sampler.target_calls:,} target-function calls.")

    terminate_workers(comm, 0)

    render_animation(cap.frames, final_states, bounds, GIF_PATH)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        try:
            run_master(comm)
        except Exception:
            terminate_workers(comm, 0)
            raise
    else:
        worker_main(comm, rank)


if __name__ == "__main__":
    main()
