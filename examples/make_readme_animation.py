"""
Render a README GIF that visualizes how ParaProf explores the 4D Himmelblau
log-likelihood. Two 2D projections are scanned in sequence:

    1. (x0, x1), profiling over (x2, x3)
    2. (x2, x3), profiling over (x0, x1)

The animation is built from snapshots of the live ProfileProjector state,
captured every ``SNAPSHOT_INTERVAL`` target-function evaluations. Each frame
shows both panels side by side:

    - During projection 1, the left panel updates live while the right panel
      shows raw 4D samples projected onto (x2, x3) -- evidence that ParaProf
      is already mapping the second projection's landscape "for free".
    - During projection 2, the left panel is frozen at projection 1's final
      grid and the right panel updates live. Because projection 2 inherits
      the global solution pool from projection 1 (warm-start), the initial
      L-BFGS-B sweep is skipped and dynamic activation immediately starts
      growing outward from the four Himmelblau peaks.

Run with MPI:

    mpiexec -n 4 python examples/make_readme_animation.py

Requires the optional ``viz`` dependencies (matplotlib) plus ``imageio``.
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
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

FUNC_NAME = "himmelblau_4d"
GRID_PER_DIM = 40
N_INITIAL_OPT = 50
ROI_THRESHOLD = 6.0
POP_PER_CELL = 3
LBFGSB_ITER = 15
MAX_PATCHING_WAVES = 2

SNAPSHOT_INTERVAL_PROJ1 = 50  # target-function calls between snapshots in proj 1
SNAPSHOT_INTERVAL_PROJ2 = 3   # very fine sampling for the (fast) warm-started proj 2
SCATTER_HISTORY = 220         # most-recent raw samples kept for overlay scatter
SCATTER_HISTORY_INIT = 600    # larger buffer during initial L-BFGS-B phase

PROJECTIONS = [
    {"dims": [0, 1], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [2, 3], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "example_plots", "animation")
GIF_PATH = os.path.join(OUT_DIR, "paraprof_dynamic_scan.gif")

VMIN = -6.0
VMAX = 0.0
CONTOUR_LEVELS = [-6.18, -2.30]   # 95%, 68% Wilks-Δχ² for 2D

# Phase colour palette (for the phase-tag pill at the top of the frame).
PHASE_COLORS = {
    "initial_global_search": "#ff7849",
    "dynamic_activation":    "#5cb85c",
    "de_refinement":         "#3e8ed0",
    "patching":              "#f1b932",
    "frozen":                "#7a7a7a",
    "starting":              "#7a7a7a",
}

PHASE_LABELS = {
    "initial_global_search": "Initial global search",
    "dynamic_activation":    "Dynamic grid activation",
    "de_refinement":         "Differential evolution",
    "patching":              "Patching wave",
    "frozen":                "Projection complete",
    "starting":              "Initializing projection",
}


# ---------------------------------------------------------------------------
# Snapshot capture
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
        self.projection_started = False
        self._last_snapshot_calls = -1
        # Track the largest population size seen so far in the current projection.
        # This lets us detect the start of dynamic activation (population grows
        # past zero) versus the steady state of DE refinement.
        self._max_active_seen = 0
        self._wrap_register()

    # --- hooks ---------------------------------------------------------

    def _wrap_register(self):
        orig = self.sampler._register_target_call

        def wrapped(params, target_val):
            orig(params, target_val)
            self.scatter_buf.append((np.asarray(params, dtype=float).copy(), float(target_val)))
            if (self.sampler.target_calls % self.interval == 0
                    and self.sampler.target_calls != self._last_snapshot_calls):
                self._last_snapshot_calls = self.sampler.target_calls
                self.capture()

        self.sampler._register_target_call = wrapped

    def reset_for_new_projection(self):
        self.projection_started = False
        self._max_active_seen = 0
        self.scatter_buf.clear()

    # --- phase classification -----------------------------------------

    def _classify_phase(self, forced: Optional[str] = None) -> str:
        """Infer the algorithmic phase from observable sampler state.

        The sampler's *current_generation* counter is the cleanest tell:
            - == 0 and population empty -> initial L-BFGS-B global search
            - == 0 and population non-empty -> activation (warm-started or
              right after the initial sweep populated the grid)
            - > 0 with active cells -> DE/dynamic activation
            - > 0 with no active cells -> patching wave
        """
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

        # When the second projection warm-starts, the grid populates almost
        # instantly without an initial L-BFGS-B sweep -- skip straight to
        # dynamic activation in that case.
        if gen == 0:
            return "dynamic_activation"

        if active_count == 0:
            return "patching"

        # Differential evolution refinement: active cells exist but the active
        # frontier has stopped growing (population growth saturated).
        active_ratio = active_count / max(pop_size, 1)
        if active_ratio < 0.35 and pop_size >= self._max_active_seen * 0.9:
            return "de_refinement"
        return "dynamic_activation"

    # --- snapshot ------------------------------------------------------

    def capture(self, forced_phase: Optional[str] = None):
        s = self.sampler

        best_idx = None
        best_v = -np.inf
        for idx, v in s.profile_likelihood_grid.items():
            if v > best_v:
                best_v = v
                best_idx = idx

        recent = np.array([p for p, _ in self.scatter_buf]) if self.scatter_buf else np.empty((0, s.dims))

        self.frames.append({
            "target_calls": int(s.target_calls),
            "global_max": float(s.global_max_target_val),
            "proj_idx": self.current_proj_idx,
            "proj_dims": tuple(s.projection_dims),
            "grid_axes": [np.asarray(a).copy() for a in s.grid_axes],
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
    """Return (delta_image, mask) for a 2D sparse grid expressed as Δlog L."""
    img = np.full(grid_shape, -np.inf)
    for idx, v in grid_values.items():
        img[idx] = v
    finite = np.isfinite(img)
    if not finite.any():
        return img, finite
    # Use global_max for cross-projection consistency so the right panel keeps
    # the same colour scale during proj 2 once a meaningful global max exists.
    ref = max(global_max, img[finite].max())
    img = np.where(finite, img - ref, -np.inf)
    return img, finite


def _draw_panel(ax, axes_x, axes_y, grid_img, mask,
                active_cells, best_fit_idx,
                scatter_xy_recent, scatter_xy_old,
                title, cmap, show_scatter):
    """Render a single 2D panel."""
    extent = [axes_x[0], axes_x[-1], axes_y[0], axes_y[-1]]
    masked = np.ma.masked_where(~mask, grid_img)

    im = ax.imshow(masked.T, extent=extent, origin="lower",
                   aspect="equal", cmap=cmap, vmin=VMIN, vmax=VMAX,
                   interpolation="nearest")

    if mask.any():
        X, Y = np.meshgrid(axes_x, axes_y)
        try:
            ax.contour(X, Y, np.where(mask, grid_img, np.nan).T,
                       levels=CONTOUR_LEVELS,
                       colors="white", linewidths=0.9, alpha=0.95)
        except Exception:
            pass

    if show_scatter:
        if scatter_xy_old is not None and len(scatter_xy_old):
            ax.scatter(scatter_xy_old[:, 0], scatter_xy_old[:, 1],
                       s=4, c="#ff5d4f", alpha=0.18, linewidths=0,
                       zorder=4)
        if scatter_xy_recent is not None and len(scatter_xy_recent):
            ax.scatter(scatter_xy_recent[:, 0], scatter_xy_recent[:, 1],
                       s=10, c="#ff8a4f", alpha=0.65, linewidths=0.3,
                       edgecolors="#7a1d10", zorder=5)

    if active_cells:
        pts = np.array([(axes_x[i], axes_y[j]) for (i, j) in active_cells])
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#22d3ee",
                   edgecolors="black", linewidths=0.25, zorder=6)

    if best_fit_idx is not None:
        bx = axes_x[best_fit_idx[0]]
        by = axes_y[best_fit_idx[1]]
        ax.scatter([bx], [by], s=110, marker="*", c="white",
                   edgecolors="black", linewidths=0.9, zorder=10)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor("#dadada")
    ax.tick_params(axis="both", labelsize=8, length=2.5, pad=2)
    ax.set_xlabel(title["xlabel"], fontsize=10)
    ax.set_ylabel(title["ylabel"], fontsize=10)
    # Draw the per-panel title inside the panel (top-left) so it never
    # collides with the info bar above the panels.
    ax.text(0.02, 0.97, title["panel_title"],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, fontweight="bold",
            color=title.get("title_color", "#222"),
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none",
                      boxstyle="round,pad=0.25"))
    ax.grid(True, linestyle=":", linewidth=0.4, color="white", alpha=0.5)
    return im


def _make_phase_pill(ax, phase: str):
    """Draw the phase indicator pill in a dedicated axes."""
    color = PHASE_COLORS.get(phase, "#888")
    label = PHASE_LABELS.get(phase, phase)
    ax.axis("off")
    # Pill height in axes-fraction; keep it short so adjacent text has room.
    pill = FancyBboxPatch((0.0, 0.18), 1.0, 0.64,
                          boxstyle="round,pad=0.0,rounding_size=0.15",
                          transform=ax.transAxes,
                          linewidth=0, facecolor=color, alpha=0.95,
                          mutation_aspect=0.35)
    ax.add_patch(pill)
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color="white")


def _select_recent_samples(samples: np.ndarray, n_recent: int):
    """Split the scatter buffer into 'recent' (highlighted) and 'older' (faded)."""
    if samples is None or len(samples) == 0:
        return None, None
    n = len(samples)
    if n_recent >= n:
        return samples, None
    return samples[-n_recent:], samples[:-n_recent]


def render_animation(frames, frozen_p1, bounds, gif_path):
    if not frames:
        raise RuntimeError("No snapshots captured -- nothing to render.")

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dadada")

    # Pre-allocate the figure -- we reuse a single figure across frames to keep
    # rendering fast and consistent.
    fig = plt.figure(figsize=(7.8, 4.6), dpi=110, facecolor="white")
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[0.20, 0.11, 1.0],
        width_ratios=[1.0, 1.0, 0.04],
        left=0.07, right=0.94, top=0.95, bottom=0.09,
        hspace=0.20, wspace=0.22,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.72,
                  "ParaProf — profile-likelihood scan of the 4D Himmelblau function",
                  fontsize=13.5, fontweight="bold", color="#222",
                  transform=title_ax.transAxes, ha="left", va="center")
    title_ax.text(0.0, 0.20,
                  "Two 2D projections scanned sequentially; the second is warm-started from the first.",
                  fontsize=9.5, color="#666",
                  transform=title_ax.transAxes, ha="left", va="center")

    pill_ax = fig.add_subplot(gs[1, 0])
    info_ax = fig.add_subplot(gs[1, 1:])
    info_ax.axis("off")

    ax_left = fig.add_subplot(gs[2, 0])
    ax_right = fig.add_subplot(gs[2, 1])
    cax = fig.add_subplot(gs[2, 2])

    # Build the colour bar ONCE from a fixed mappable so we don't accumulate
    # locator references on every frame (which caused a recursion blow-up).
    sm = ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log L - \log L_{\mathrm{best}}$", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)

    bounds = np.asarray(bounds)
    proj_axes_p1 = [
        np.linspace(bounds[0, 0], bounds[0, 1], GRID_PER_DIM + 1),
        np.linspace(bounds[1, 0], bounds[1, 1], GRID_PER_DIM + 1),
    ]
    proj_axes_p2 = [
        np.linspace(bounds[2, 0], bounds[2, 1], GRID_PER_DIM + 1),
        np.linspace(bounds[3, 0], bounds[3, 1], GRID_PER_DIM + 1),
    ]
    grid_shape = (GRID_PER_DIM + 1, GRID_PER_DIM + 1)

    # Frame budget: target ~220 final frames so the GIF stays under ~6 MB and
    # plays in 12-15 seconds at 18 FPS.
    proj1_frames = [f for f in frames if f["proj_idx"] == 0]
    proj2_frames = [f for f in frames if f["proj_idx"] == 1]

    max_p1 = 130
    stride_p1 = max(1, len(proj1_frames) // max_p1)
    chosen_p1 = proj1_frames[::stride_p1]
    if chosen_p1 and chosen_p1[-1] is not proj1_frames[-1]:
        chosen_p1.append(proj1_frames[-1])

    # Hold on the final projection-1 state for ~0.7 s so the viewer
    # registers "first projection done" before the second one starts.
    transition_hold = 13
    chosen_p1 = list(chosen_p1) + [chosen_p1[-1]] * transition_hold

    # For projection 2 we want every captured snapshot in the output:
    # warm-starting collapses the entire run, so any sub-sampling loses the
    # short activation/DE phases entirely. Cap the total proj-2 frame count
    # to keep the GIF size sane.
    max_p2 = 80
    stride_p2 = max(1, len(proj2_frames) // max_p2)
    chosen_p2 = proj2_frames[::stride_p2]
    if chosen_p2 and chosen_p2[-1] is not proj2_frames[-1]:
        chosen_p2.append(proj2_frames[-1])

    chosen = chosen_p1 + chosen_p2
    if not chosen:
        chosen = list(frames)

    # Pad with trailing repeats so the final state holds for ~1.4 s.
    final_hold = 25
    chosen = chosen + [chosen[-1]] * final_hold

    images = []
    final_global_max = max(f["global_max"] for f in frames if np.isfinite(f["global_max"]))

    for i, snap in enumerate(chosen):
        ax_left.cla()
        ax_right.cla()
        pill_ax.cla()
        info_ax.cla()
        info_ax.axis("off")

        proj_idx = snap["proj_idx"]
        phase = snap["phase"]

        # ----- left panel: projection 1 (x0, x1) -----
        if proj_idx == 0:
            grid_img, mask = _build_grid_image(snap["grid_values"], grid_shape,
                                               max(snap["global_max"], -np.inf))
            active = snap["active_cells"]
            best_idx = snap["best_fit_idx"]
            samples_recent, samples_old = _select_recent_samples(
                snap["recent_samples"], n_recent=90,
            )
            recent_xy = samples_recent[:, [0, 1]] if samples_recent is not None else None
            old_xy = samples_old[:, [0, 1]] if samples_old is not None else None
            left_title_color = "#222"
            left_panel_title = "Projection 1 — (x₀, x₁) profile"
        else:
            # Projection 2 active: freeze left panel at proj 1's final state.
            grid_img, mask = _build_grid_image(
                frozen_p1["grid_values"], grid_shape, final_global_max,
            )
            active = set()
            best_idx = frozen_p1["best_fit_idx"]
            recent_xy = None
            old_xy = None
            left_title_color = "#7a7a7a"
            left_panel_title = "Projection 1 — (x₀, x₁) ✓ complete"

        _draw_panel(
            ax_left, proj_axes_p1[0], proj_axes_p1[1],
            grid_img, mask, active, best_idx,
            recent_xy, old_xy,
            title={
                "xlabel": "$x_0$", "ylabel": "$x_1$",
                "panel_title": left_panel_title,
                "title_color": left_title_color,
            },
            cmap=cmap,
            show_scatter=(proj_idx == 0),
        )

        # ----- right panel: projection 2 (x2, x3) -----
        if proj_idx == 0:
            # Show samples projected onto (x2, x3): no grid yet.
            empty_img = np.full(grid_shape, -np.inf)
            empty_mask = np.zeros(grid_shape, dtype=bool)
            samples_recent, samples_old = _select_recent_samples(
                snap["recent_samples"], n_recent=90,
            )
            recent_xy = samples_recent[:, [2, 3]] if samples_recent is not None else None
            old_xy = samples_old[:, [2, 3]] if samples_old is not None else None
            im_right = _draw_panel(
                ax_right, proj_axes_p2[0], proj_axes_p2[1],
                empty_img, empty_mask, set(), None,
                recent_xy, old_xy,
                title={
                    "xlabel": "$x_2$", "ylabel": "$x_3$",
                    "panel_title": "Projection 2 — (x₂, x₃) (queued)",
                    "title_color": "#a0a0a0",
                },
                cmap=cmap,
                show_scatter=True,
            )
        else:
            grid_img2, mask2 = _build_grid_image(
                snap["grid_values"], grid_shape, max(snap["global_max"], final_global_max),
            )
            active2 = snap["active_cells"]
            best_idx2 = snap["best_fit_idx"]
            samples_recent, samples_old = _select_recent_samples(
                snap["recent_samples"], n_recent=90,
            )
            recent_xy = samples_recent[:, [2, 3]] if samples_recent is not None else None
            old_xy = samples_old[:, [2, 3]] if samples_old is not None else None
            im_right = _draw_panel(
                ax_right, proj_axes_p2[0], proj_axes_p2[1],
                grid_img2, mask2, active2, best_idx2,
                recent_xy, old_xy,
                title={
                    "xlabel": "$x_2$", "ylabel": "$x_3$",
                    "panel_title": "Projection 2 — (x₂, x₃) warm-started",
                    "title_color": "#222",
                },
                cmap=cmap,
                show_scatter=False,
            )

        # ----- phase indicator pill + info text -----
        _make_phase_pill(pill_ax, phase)

        global_max = snap["global_max"]
        # Two-line info block, vertically centred in the info row.
        info_ax.text(
            0.04, 0.72,
            f"Target-function evaluations: {snap['target_calls']:,}",
            transform=info_ax.transAxes, ha="left", va="center",
            fontsize=10.5, color="#222", fontweight="bold",
        )
        if np.isfinite(global_max):
            info_ax.text(
                0.04, 0.24,
                f"Best log-likelihood found: {global_max:+.3e}",
                transform=info_ax.transAxes, ha="left", va="center",
                fontsize=9.5, color="#444",
            )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images.append(rgba[..., :3].copy())

        if i % 20 == 0 or i == len(chosen) - 1:
            print(f"  rendered frame {i + 1}/{len(chosen)} "
                  f"(stage={snap['phase']}, calls={snap['target_calls']:,})",
                  flush=True)

    plt.close(fig)

    print(f"Encoding GIF ({len(images)} frames) -> {gif_path}", flush=True)
    imageio.mimsave(gif_path, images, format="GIF", fps=18, loop=0)
    print(f"Wrote {gif_path} ({os.path.getsize(gif_path) / 1e6:.2f} MB)")


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

        cap = SnapshotCapturer(sampler, SNAPSHOT_INTERVAL_PROJ1, SCATTER_HISTORY_INIT)
        comm.bcast(sampler.target_func, root=0)

        frozen_p1: Optional[dict] = None

        for proj_idx, proj in enumerate(PROJECTIONS):
            cap.current_proj_idx = proj_idx
            cap.reset_for_new_projection()
            cap.interval = (SNAPSHOT_INTERVAL_PROJ1 if proj_idx == 0
                            else SNAPSHOT_INTERVAL_PROJ2)
            # Pre-size the scatter buffer: larger for projection 1 (initial
            # global L-BFGS-B sweep produces many scattered samples we want
            # to keep visible).
            cap.scatter_buf = collections.deque(
                maxlen=SCATTER_HISTORY_INIT if proj_idx == 0 else SCATTER_HISTORY,
            )
            if proj_idx == 0:
                # Only snapshot at the start of projection 1 -- the sampler
                # has not yet been reset for projection 2 at the equivalent
                # point in the loop, so an early forced capture there would
                # stamp projection 1's grid onto a projection-2 frame.
                cap.capture(forced_phase="initial_global_search")

            run_projection(
                comm=comm,
                sampler=sampler,
                projection_config=proj,
                save_plots=False,
                myrank=0,
            )

            # Final snapshot for this projection (phase: frozen).
            cap.capture(forced_phase="frozen")

            if proj_idx == 0:
                best_idx = None
                best_v = -np.inf
                for idx, v in sampler.profile_likelihood_grid.items():
                    if v > best_v:
                        best_v = v
                        best_idx = idx
                frozen_p1 = {
                    "grid_values": dict(sampler.profile_likelihood_grid),
                    "best_fit_idx": best_idx,
                    "best_fit_value": best_v,
                }

        print(f"Captured {len(cap.frames)} snapshots across "
              f"{sampler.target_calls:,} target-function calls.")

    terminate_workers(comm, 0)

    render_animation(cap.frames, frozen_p1, bounds, GIF_PATH)


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
