"""Shared types and helpers used by every adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Registry of method-name -> adapter instance.
ADAPTERS: dict[str, "BaseAdapter"] = {}


def register_adapter(cls):
    """Class decorator: instantiate and record the adapter in the registry."""
    instance = cls()
    if not getattr(instance, "name", None):
        raise ValueError(f"Adapter {cls!r} has no 'name' attribute.")
    if instance.name in ADAPTERS:
        raise ValueError(f"Adapter '{instance.name}' already registered.")
    ADAPTERS[instance.name] = instance
    return cls


class CountingFunction:
    """Wrapper that counts every call to ``func``.

    All single-process adapters call the target through this wrapper so the
    evaluation-count axis is the same currency for every method, including
    finite-difference gradient calls inside L-BFGS-B and MIGRAD. The paraprof
    adapter bypasses this and uses paraprof's own cluster-wide call counter.
    """

    __slots__ = ("_func", "_count")

    def __init__(self, func: Callable[[np.ndarray], float]):
        self._func = func
        self._count = 0

    def __call__(self, params: np.ndarray) -> float:
        self._count += 1
        return float(self._func(np.asarray(params, dtype=float)))

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0


@dataclass
class ProjectionResult:
    """Per-(method, problem, projection, seed) result, JSON-serialisable.

    ``logL_grid`` is the best logL the method found at each grid cell.
    Cells the method does not touch are stored as ``np.nan``.

    ``profiled_params_grid`` has shape ``grid_points + (n_profiled,)`` and
    holds the best profiled parameter vector at each cell.

    ``cell_evals`` has shape ``grid_points`` and holds the per-cell target-call
    count (zero for cells that were never touched).

    ``extra`` carries adapter-specific outputs (e.g. MNCONTOUR polygons).
    """

    method: str
    problem: str
    dims: list[int]
    grid_points: list[int]
    seed: int
    grid_axes: list[np.ndarray]
    logL_grid: np.ndarray
    profiled_params_grid: np.ndarray
    cell_evals: np.ndarray
    total_evals: int
    n_cells_capped: int
    wall_time: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "problem": self.problem,
            "dims": self.dims,
            "grid_points": self.grid_points,
            "seed": self.seed,
            "grid_axes": [a.tolist() for a in self.grid_axes],
            "logL_grid": self.logL_grid.tolist(),
            "profiled_params_grid": self.profiled_params_grid.tolist(),
            "cell_evals": self.cell_evals.tolist(),
            "total_evals": self.total_evals,
            "n_cells_capped": self.n_cells_capped,
            "wall_time": self.wall_time,
            "extra": _jsonify(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectionResult":
        return cls(
            method=d["method"],
            problem=d["problem"],
            dims=list(d["dims"]),
            grid_points=list(d["grid_points"]),
            seed=int(d["seed"]),
            grid_axes=[np.asarray(a, dtype=float) for a in d["grid_axes"]],
            logL_grid=np.asarray(d["logL_grid"], dtype=float),
            profiled_params_grid=np.asarray(d["profiled_params_grid"], dtype=float),
            cell_evals=np.asarray(d["cell_evals"], dtype=np.int64),
            total_evals=int(d["total_evals"]),
            n_cells_capped=int(d["n_cells_capped"]),
            wall_time=float(d["wall_time"]),
            extra=d.get("extra", {}) or {},
        )


def _jsonify(obj):
    """Coerce a value to JSON-friendly form, preserving numpy arrays as lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def cell_centres(bounds: list[list[float]], dims: list[int],
                 grid_points: list[int]) -> list[np.ndarray]:
    """Return the cell-centre coordinate axes for the projected dims.

    Mirrors paraprof's grid convention: ``grid_points[i]`` linearly spaced
    points between ``bounds[dims[i]][0]`` and ``bounds[dims[i]][1]`` inclusive.
    """
    axes = []
    for d, n in zip(dims, grid_points, strict=True):
        lo, hi = bounds[d]
        axes.append(np.linspace(lo, hi, n))
    return axes


class BaseAdapter(ABC):
    """Uniform interface every method must implement."""

    name: str = ""  # registered name, set on the subclass
    parallel_via_paraprof_mpi: bool = False  # True only for the paraprof adapter

    @abstractmethod
    def run(
        self,
        func: Callable[[np.ndarray], float],
        bounds: list[list[float]],
        dims: list[int],
        grid_points: list[int],
        max_evals_per_cell: int,
        seed: int,
        comm=None,
    ) -> ProjectionResult:
        """Run one projection. Master-rank only when ``comm`` is supplied."""
