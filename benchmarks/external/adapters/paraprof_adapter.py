"""ParaProf adapter — full algorithm and kernel-only variants, plus oracle preset.

This is the only adapter that uses paraprof's native MPI master/worker
infrastructure. The other adapters run their per-cell optimisation on the
master rank only; see ``README.md`` for the wall-clock fairness caveat.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from paraprof import (
    ProfileProjector,
    run_projection,
    terminate_workers,
    worker_main,
)

from .base import (
    BaseAdapter,
    ProjectionResult,
    cell_centres,
    register_adapter,
)


def _extract_grid(solution: dict, grid_points: list[int],
                  n_profiled: int) -> tuple[np.ndarray, np.ndarray]:
    """Project a paraprof exported solution to dense logL and profiled-param grids."""
    shape = tuple(grid_points)
    logL = np.full(shape, np.nan, dtype=float)
    profiled = np.full(shape + (n_profiled,), np.nan, dtype=float)
    for grid_idx, entry in solution["solutions"].items():
        logL[tuple(grid_idx)] = float(entry["likelihood"])
        if n_profiled > 0:
            profiled[tuple(grid_idx)] = np.asarray(entry["profiled_params"], dtype=float)
    return logL, profiled


def _run_paraprof(
    func: Callable[[np.ndarray], float],
    bounds: list[list[float]],
    dims: list[int],
    grid_points: list[int],
    seed: int,
    comm,
    *,
    pop_per_grid_point: int,
    n_initial_optimizations: int,
    max_patching_waves: int,
    grid_refinement_factor: int | None,
    suspect_recheck_enabled: bool,
    initial_points: np.ndarray | None = None,
) -> tuple[dict, dict, int]:
    """Drive one paraprof projection on the master rank. Returns coarse solution,
    refined solution (or None), and paraprof's cluster-wide target_calls count.

    paraprof workers do the actual function evaluations, so wrapping ``func``
    in a CountingFunction on the master would always report zero. We rely on
    paraprof's own ``target_calls`` aggregate instead.
    """
    np.random.seed(seed)

    # paraprof internally adds 1 to each grid_points value (fencepost
    # convention). Pre-decrement so the resulting grid_shape matches what
    # the competitor adapters produce from np.linspace(lo, hi, n).
    paraprof_grid_points = [max(2, g - 1) for g in grid_points]
    projection = {
        "dims": list(dims),
        "grid_points": paraprof_grid_points,
    }
    if grid_refinement_factor is not None and grid_refinement_factor > 1:
        projection["grid_refinement_factor"] = grid_refinement_factor
        projection["patch_refined_grid"] = True

    advanced_config = {
        "suspect_recheck": {"enabled": suspect_recheck_enabled},
    }

    sampler = ProfileProjector(
        target_func=func,
        bounds=bounds,
        projections=[projection],
        roi_threshold=4.0,
        pop_per_grid_point=pop_per_grid_point,
        n_initial_optimizations=n_initial_optimizations,
        max_patching_waves=max_patching_waves,
        initial_points=initial_points,
        advanced_config=advanced_config,
    )

    comm.bcast(sampler.target_func, root=0)
    result = run_projection(
        comm=comm,
        sampler=sampler,
        projection_config=projection,
        save_plots=False,
        skip_init_opt_on_warm_start=False,
        myrank=0,
    )
    return (
        result["coarse_solution"],
        result.get("refined_solution"),
        int(result["metrics"]["total_target_calls"]),
    )


class _BaseParaProfAdapter(BaseAdapter):
    parallel_via_paraprof_mpi = True
    pop_per_grid_point = 3
    n_initial_optimizations = 80
    max_patching_waves = 20
    grid_refinement_factor: int | None = None
    suspect_recheck_enabled = True

    def run(self, func, bounds, dims, grid_points, max_evals_per_cell, seed, comm=None):
        if comm is None:
            raise RuntimeError(f"{self.name} requires an MPI communicator")
        t0 = time.perf_counter()
        coarse, refined, n_evals = _run_paraprof(
            func=func,
            bounds=bounds,
            dims=dims,
            grid_points=grid_points,
            seed=seed,
            comm=comm,
            pop_per_grid_point=self.pop_per_grid_point,
            n_initial_optimizations=self.n_initial_optimizations,
            max_patching_waves=self.max_patching_waves,
            grid_refinement_factor=self.grid_refinement_factor,
            suspect_recheck_enabled=self.suspect_recheck_enabled,
        )
        wall = time.perf_counter() - t0

        # Prefer refined grid when available, else coarse.
        solution = refined if refined is not None else coarse
        effective_grid_points = list(solution["grid_shape"])
        n_profiled = len(solution["profiled_dims"])
        logL, profiled = _extract_grid(solution, effective_grid_points, n_profiled)

        # If the run included refinement we downsample back to the requested
        # coarse grid so all methods are compared on the same grid. profiled
        # params take the values from whichever fine cell had the max logL.
        requested = [int(g) for g in grid_points]
        if [int(g) for g in effective_grid_points] != requested:
            logL, argmax_idx = _downsample_logL(logL, requested)
            profiled = _gather_profiled_at(profiled, argmax_idx, n_profiled)

        axes = cell_centres(bounds, dims, grid_points)
        cell_evals = np.zeros(tuple(grid_points), dtype=np.int64)
        # paraprof doesn't expose per-cell counts; record uniform attribution.
        n_active = int(np.sum(~np.isnan(logL)))
        if n_active > 0:
            cell_evals[~np.isnan(logL)] = max(1, n_evals // n_active)

        return ProjectionResult(
            method=self.name,
            problem="",  # filled in by the driver
            dims=list(dims),
            grid_points=list(grid_points),
            seed=int(seed),
            grid_axes=axes,
            logL_grid=logL,
            profiled_params_grid=profiled,
            cell_evals=cell_evals,
            total_evals=int(n_evals),
            n_cells_capped=0,
            wall_time=wall,
            extra={
                "had_refinement": refined is not None,
                "effective_grid_points": effective_grid_points,
            },
        )


def _downsample_logL(arr: np.ndarray, target_shape: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a refined logL grid to the coarse grid by per-block max.

    Returns the downsampled logL array and an array of per-coarse-cell fine
    indices (shape ``target_shape + (n_dims,)``) pointing at the fine-grid
    cell that supplied the max, so we can later gather the matching profiled
    params from the same fine cell.
    """
    target_shape = tuple(target_shape)
    if arr.shape == target_shape:
        idx = np.indices(target_shape).transpose(*range(1, len(target_shape) + 1), 0)
        return arr, idx

    src_shape = arr.shape
    factors = tuple(s // t for s, t in zip(src_shape, target_shape))
    assert all(s == t * f for s, t, f in zip(src_shape, target_shape, factors))

    # Reshape to (t0, f0, t1, f1, ...) then flatten the f-axes to one axis per coarse cell.
    reshape_dims = []
    for t, f in zip(target_shape, factors):
        reshape_dims.extend([t, f])
    reshaped = arr.reshape(reshape_dims)
    # Move all "factor" axes to the end so they form a single flattened axis.
    n = len(target_shape)
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
    moved = reshaped.transpose(perm)
    block_size = int(np.prod(factors))
    flat = moved.reshape(target_shape + (block_size,))
    flat_safe = np.where(np.isfinite(flat), flat, -np.inf)
    argmax_flat = np.argmax(flat_safe, axis=-1)
    out = np.take_along_axis(flat_safe, argmax_flat[..., None], axis=-1).squeeze(-1)
    out = np.where(np.isfinite(out), out, np.nan)
    # Unravel the flat block index back into a per-axis fine-cell offset.
    offsets = np.array(np.unravel_index(argmax_flat, factors))  # shape (n, *target)
    offsets = np.moveaxis(offsets, 0, -1)
    coarse_grid_idx = np.indices(target_shape).transpose(
        *range(1, n + 1), 0
    ) * np.array(factors)
    fine_idx = coarse_grid_idx + offsets
    return out, fine_idx


def _gather_profiled_at(profiled: np.ndarray, fine_idx: np.ndarray,
                        n_profiled: int) -> np.ndarray:
    """Gather profiled-param vectors from ``profiled`` at fine-grid indices."""
    target_shape = fine_idx.shape[:-1]
    out = np.full(target_shape + (n_profiled,), np.nan, dtype=float)
    if n_profiled == 0:
        return out
    flat_idx = fine_idx.reshape(-1, fine_idx.shape[-1])
    flat_out = out.reshape(-1, n_profiled)
    for k, idx in enumerate(flat_idx):
        flat_out[k] = profiled[tuple(idx)]
    return flat_out.reshape(target_shape + (n_profiled,))


@register_adapter
class ParaProfDefault(_BaseParaProfAdapter):
    name = "paraprof_default"
    pop_per_grid_point = 3
    n_initial_optimizations = 80
    max_patching_waves = 20
    grid_refinement_factor = None
    suspect_recheck_enabled = True


@register_adapter
class ParaProfKernel(_BaseParaProfAdapter):
    """Same kernel as default but with patching, suspect-recheck and refinement off."""

    name = "paraprof_kernel"
    pop_per_grid_point = 3
    n_initial_optimizations = 80
    max_patching_waves = 0
    grid_refinement_factor = None
    suspect_recheck_enabled = False


@register_adapter
class ParaProfOracle(_BaseParaProfAdapter):
    """High-budget paraprof used as the reference grid."""

    name = "paraprof_oracle"
    pop_per_grid_point = 6
    n_initial_optimizations = 400
    max_patching_waves = 30
    grid_refinement_factor = 2
    suspect_recheck_enabled = True


def paraprof_worker_loop(comm):
    """Worker-rank entry point for paraprof projections.

    Workers receive the wrapped CountingFunction via ``comm.bcast`` and then
    enter paraprof's worker event loop. Returns once the master sends the
    paraprof termination signal.
    """
    worker_main(comm, comm.Get_rank())


def terminate_paraprof_workers(comm) -> None:
    terminate_workers(comm, myrank=0)
