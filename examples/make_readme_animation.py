"""
Render a README GIF that visualizes how ParaProf explores the 4D Himmelblau
log-likelihood. Two 2D projections are scanned in sequence:

    1. (x0, x1), profiling over (x2, x3) -- yields the 2D Himmelblau shape
    2. (x0, x2), profiling over (x1, x3) -- yields a sum of two 1D
       Himmelblau profiles, with peaks on the cartesian product of the four
       1D-Himmelblau-peak coordinates

The animation is built from snapshots of the live ProfileProjector state,
captured every ``SNAPSHOT_INTERVAL`` target-function evaluations. Each frame
shows both panels side by side; only the panel for the projection currently
being scanned shows live activity.

Projection 2 is warm-started from projection 1 in the same way as
``run_all_projections`` does it: ``initial_maxima`` is seeded from the
accumulated global solution pool (skipping the initial global L-BFGS-B
sweep) and per-cell DE populations get one proximity warm-start from the
pool. The pool already contains points whose ``(x0, x2)`` projections
cover the full 4x4 cartesian product of 1D-Himmelblau peak coordinates,
so projection 2 quickly recovers all 16 peaks.

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

FUNC_NAME = "himmelblau_4d"
GRID_PER_DIM = 40
N_INITIAL_OPT = 20
ROI_THRESHOLD = 5.0
POP_PER_CELL = 3
LBFGSB_ITER = 15
MAX_PATCHING_WAVES = 2

SNAPSHOT_INTERVAL_PROJ1 = 25  # target-function calls between snapshots in proj 1
SNAPSHOT_INTERVAL_PROJ2 = 2   # finer sampling for the (fast) warm-started proj 2
SCATTER_HISTORY = 300         # rolling raw-sample buffer, identical for both projections

PROJECTIONS = [
    {"dims": [0, 1], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
    {"dims": [0, 2], "grid_points": [GRID_PER_DIM, GRID_PER_DIM]},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "example_plots", "animation")
GIF_PATH = os.path.join(OUT_DIR, "paraprof_dynamic_scan.gif")

VMIN = -5.0
VMAX = 0.0
# Δlog L contour levels (the maps below show log L − log L_best, so non-positive).
# Outer level (−3) is dashed/thin, inner level (−1) is solid/thicker -- matches
# the showcase plot styling in the "Example output" section of the README.
CONTOUR_LEVELS = [-3.0, -1.0]
CONTOUR_LINESTYLES = ['--', '-']
CONTOUR_LINEWIDTHS = [1.0, 1.6]

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
            # Sort levels ascending; matplotlib then applies linestyles and
            # linewidths in the same ascending order (most-negative first).
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
                       s=4, c="#ff6420", alpha=0.32, linewidths=0,
                       zorder=4)
        if scatter_xy_recent is not None and len(scatter_xy_recent):
            ax.scatter(scatter_xy_recent[:, 0], scatter_xy_recent[:, 1],
                       s=10, c="#ff6420", alpha=0.90, linewidths=0.3,
                       edgecolors="#8a3010", zorder=5)

    if best_fit_idx is not None:
        bx = axes_x[best_fit_idx[0]]
        by = axes_y[best_fit_idx[1]]
        ax.scatter([bx], [by], s=110, marker="*", c="white",
                   edgecolors="black", linewidths=0.9, zorder=10)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor("#dadada")
    ax.tick_params(axis="both", length=3.0, pad=3)
    ax.set_xlabel(title["xlabel"])
    ax.set_ylabel(title["ylabel"])
    ax.grid(True, linestyle=":", linewidth=0.4, color="white", alpha=0.5)
    return im


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

    # Match the showcase 2D plot styling: serif text body + STIX math, slightly
    # larger axis labels, ticks pointing inward on all four sides.
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dadada")

    # Pre-allocate the figure -- we reuse a single figure across frames to keep
    # rendering fast and consistent.
    # Use a 2x2 gridspec with just the info row + the two heat-map panels;
    # the colour bar is then attached to the right panel via
    # make_axes_locatable, which auto-sizes it to the same data-area height
    # as its parent panel (so it tracks the heat-map exactly under
    # aspect='equal'), and places it tight against the panel's right edge.
    # This matches the showcase 2D plots, which use the equivalent
    # ``fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)`` shortcut.
    fig = plt.figure(figsize=(8.2, 4.0), dpi=110, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.07, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.07, right=0.92, top=0.96, bottom=0.13,
        hspace=0.05, wspace=0.25,
    )

    info_ax = fig.add_subplot(gs[0, :])
    info_ax.axis("off")

    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # Attach a colour-bar axes to the right panel that matches its data-area
    # height (handled internally by axes_grid1), with a small fixed gap.
    divider_right = make_axes_locatable(ax_right)
    cax = divider_right.append_axes("right", size="4.6%", pad=0.10)

    # Append an identically-sized invisible axes to the LEFT panel so both
    # panels lose the same amount of width to their right-hand sibling.
    # Without this, axes_grid1 only steals width from ax_right, and under
    # aspect='equal' that makes the right panel visibly shorter than the
    # left one.
    divider_left = make_axes_locatable(ax_left)
    ghost_cax = divider_left.append_axes("right", size="4.6%", pad=0.10)
    ghost_cax.set_visible(False)

    # Build the colour bar ONCE from a fixed mappable so we don't accumulate
    # locator references on every frame (which caused a recursion blow-up).
    sm = ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(r"$\Delta \log L = \log L - \log L_{\max}$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    bounds = np.asarray(bounds)
    # Projection 1 covers (x0, x1); projection 2 covers (x0, x2). Sharing the
    # x0 axis emphasises that the two projections answer different questions
    # about the same parameter.
    proj_axes_p1 = [
        np.linspace(bounds[0, 0], bounds[0, 1], GRID_PER_DIM + 1),
        np.linspace(bounds[1, 0], bounds[1, 1], GRID_PER_DIM + 1),
    ]
    proj_axes_p2 = [
        np.linspace(bounds[0, 0], bounds[0, 1], GRID_PER_DIM + 1),
        np.linspace(bounds[2, 0], bounds[2, 1], GRID_PER_DIM + 1),
    ]
    grid_shape = (GRID_PER_DIM + 1, GRID_PER_DIM + 1)

    # Frame budget. Both projections now run the full pipeline, so we keep
    # the per-projection cap roughly balanced. Cap total around ~260 frames
    # at 9 FPS -> ~29 s.
    proj1_frames = [f for f in frames if f["proj_idx"] == 0]
    proj2_frames = [f for f in frames if f["proj_idx"] == 1]

    max_p1 = 190
    stride_p1 = max(1, len(proj1_frames) // max_p1)
    chosen_p1 = proj1_frames[::stride_p1]
    if chosen_p1 and chosen_p1[-1] is not proj1_frames[-1]:
        chosen_p1.append(proj1_frames[-1])

    # Hold on the final projection-1 state for ~0.5 s so the viewer
    # registers "first projection done" before the second one starts.
    transition_hold = 14
    chosen_p1 = list(chosen_p1) + [chosen_p1[-1]] * transition_hold

    # Projection 2 is warm-started so initial L-BFGS-B is skipped, but the
    # subsequent dynamic activation / DE / patching still spans a sizeable
    # number of evaluations; subsample to keep the GIF compact.
    max_p2 = 150
    stride_p2 = max(1, len(proj2_frames) // max_p2)
    chosen_p2 = proj2_frames[::stride_p2]
    if chosen_p2 and chosen_p2[-1] is not proj2_frames[-1]:
        chosen_p2.append(proj2_frames[-1])

    chosen = chosen_p1 + chosen_p2
    if not chosen:
        chosen = list(frames)

    # Build a "clean" copy of the last frame: same grids and best-fit
    # markers, but with the rolling scatter buffer and active-cell set
    # emptied. The animation closes on a complete, marker-free view of
    # both finished projections.
    last_snap = chosen[-1]
    last_recent = last_snap["recent_samples"]
    empty_recent = np.empty((0, last_recent.shape[1] if last_recent.size else 4))
    clean_last = dict(last_snap)
    clean_last["recent_samples"] = empty_recent
    clean_last["active_cells"] = set()

    # Pad with the clean version so the final state holds for ~2.5 s at
    # 28 FPS.
    final_hold = 70
    chosen = chosen + [clean_last] * final_hold

    images = []
    final_global_max = max(f["global_max"] for f in frames if np.isfinite(f["global_max"]))

    for i, snap in enumerate(chosen):
        ax_left.cla()
        ax_right.cla()
        info_ax.cla()
        info_ax.axis("off")

        proj_idx = snap["proj_idx"]

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
        else:
            # Projection 2 active: freeze left panel at proj 1's final state.
            grid_img, mask = _build_grid_image(
                frozen_p1["grid_values"], grid_shape, final_global_max,
            )
            active = set()
            best_idx = frozen_p1["best_fit_idx"]
            recent_xy = None
            old_xy = None

        _draw_panel(
            ax_left, proj_axes_p1[0], proj_axes_p1[1],
            grid_img, mask, active, best_idx,
            recent_xy, old_xy,
            title={"xlabel": "$x_0$", "ylabel": "$x_1$"},
            cmap=cmap,
            show_scatter=(proj_idx == 0),
        )

        # ----- right panel: projection 2 (x0, x2) -----
        if proj_idx == 0:
            # Projection 2 has not started yet -- show an inert "queued"
            # panel with no markers or sample overlay so the viewer can
            # focus on projection 1 on the left.
            empty_img = np.full(grid_shape, -np.inf)
            empty_mask = np.zeros(grid_shape, dtype=bool)
            im_right = _draw_panel(
                ax_right, proj_axes_p2[0], proj_axes_p2[1],
                empty_img, empty_mask, set(), None,
                None, None,
                title={"xlabel": "$x_0$", "ylabel": "$x_2$"},
                cmap=cmap,
                show_scatter=False,
            )
        else:
            grid_img2, mask2 = _build_grid_image(
                snap["grid_values"], grid_shape, max(snap["global_max"], final_global_max),
            )
            active2 = snap["active_cells"]
            best_idx2 = snap["best_fit_idx"]
            samples_recent_p2, samples_old_p2 = _select_recent_samples(
                snap["recent_samples"], n_recent=90,
            )
            recent_xy_p2 = (samples_recent_p2[:, [0, 2]]
                            if samples_recent_p2 is not None else None)
            old_xy_p2 = (samples_old_p2[:, [0, 2]]
                         if samples_old_p2 is not None else None)
            im_right = _draw_panel(
                ax_right, proj_axes_p2[0], proj_axes_p2[1],
                grid_img2, mask2, active2, best_idx2,
                recent_xy_p2, old_xy_p2,
                title={"xlabel": "$x_0$", "ylabel": "$x_2$"},
                cmap=cmap,
                show_scatter=True,
            )

        # Single-line counter, vertically centred in the info row.
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
                  f"(stage={snap['phase']}, calls={snap['target_calls']:,})",
                  flush=True)

    plt.close(fig)

    print(f"Encoding GIF ({len(images)} frames) -> {gif_path}", flush=True)
    imageio.mimsave(gif_path, images, format="GIF", fps=28, loop=0)
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

        cap = SnapshotCapturer(sampler, SNAPSHOT_INTERVAL_PROJ1, SCATTER_HISTORY)
        comm.bcast(sampler.target_func, root=0)

        frozen_p1: Optional[dict] = None

        for proj_idx, proj in enumerate(PROJECTIONS):
            cap.current_proj_idx = proj_idx
            cap.reset_for_new_projection()
            cap.interval = (SNAPSHOT_INTERVAL_PROJ1 if proj_idx == 0
                            else SNAPSHOT_INTERVAL_PROJ2)
            # Reset the rolling scatter buffer for this projection. The
            # buffer length is the same for both projections so the orange
            # marker density looks identical between panels.
            cap.scatter_buf = collections.deque(maxlen=SCATTER_HISTORY)

            # Reset the sampler's per-projection state for projections after
            # the first. ``run_all_projections`` does this internally, but
            # we drive ``run_projection`` directly so we have to do it here
            # -- without it, projection 2 would inherit projection 1's
            # ``initial_maxima``, ``population`` and grid, and would never
            # run its own initial L-BFGS-B sweep.
            if proj_idx > 0:
                sampler._reset_for_new_projection(proj)

            # Snapshot the empty initial state at the very start of each
            # projection so the GIF begins (and the proj-1 -> proj-2
            # handover begins) from a clean panel.
            cap.capture(forced_phase="initial_global_search")

            run_projection(
                comm=comm,
                sampler=sampler,
                projection_config=proj,
                save_plots=False,
                # Same setting that ``run_all_projections`` uses: enable
                # warm-starting (skip initial L-BFGS-B in favour of pool-
                # seeded initial_maxima) for every projection after the
                # first.
                skip_init_opt_on_warm_start=(proj_idx > 0),
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
