"""Method adapters with a uniform projection-running interface.

Every adapter exposes::

    run(func, bounds, dims, grid_points, max_evals_per_cell, seed,
        comm=None) -> ProjectionResult
"""
from .base import (
    ADAPTERS,
    BaseAdapter,
    CountingFunction,
    ProjectionResult,
    cell_centres,
    register_adapter,
)
from . import (  # noqa: F401  (registers adapters via decorator side-effects)
    iminuit_grid_adapter,
    iminuit_mncontour_adapter,
    nlopt_adapter,
    paraprof_adapter,
    scipy_de_adapter,
    scipy_lbfgsb_adapter,
)

__all__ = [
    "ADAPTERS",
    "BaseAdapter",
    "CountingFunction",
    "ProjectionResult",
    "cell_centres",
    "register_adapter",
]
