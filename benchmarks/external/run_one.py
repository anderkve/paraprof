"""Run ONE adapter on ONE problem/projection/seed under MPI.

Usage (must be launched with mpiexec)::

    mpiexec --allow-run-as-root -n 4 python -m benchmarks.external.run_one \
        --method paraprof_default --problem himmelblau_4d \
        --dims 0 1 --grid 50 50 --seed 1 \
        --max-evals-per-cell 1500 \
        --output benchmarks/external/results/runs/result.json

Rank 0 drives the chosen adapter; ranks > 0 enter the worker loop appropriate
for the method (paraprof's worker_main for paraprof methods, an idle loop
for everything else).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mpi4py import MPI

from paraprof import get_test_function

from .adapters import ADAPTERS
from .adapters.paraprof_adapter import (
    paraprof_worker_loop,
    terminate_paraprof_workers,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run a single comparison task.")
    parser.add_argument("--method", required=True, choices=sorted(ADAPTERS.keys()))
    parser.add_argument("--problem", required=True)
    parser.add_argument("--dims", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-evals-per-cell", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_args(argv or sys.argv[1:])
    adapter = ADAPTERS[args.method]
    func, bounds, _ = get_test_function(args.problem)

    if adapter.parallel_via_paraprof_mpi:
        if rank == 0:
            result = adapter.run(
                func=func,
                bounds=bounds,
                dims=args.dims,
                grid_points=args.grid,
                max_evals_per_cell=args.max_evals_per_cell,
                seed=args.seed,
                comm=comm,
            )
            result.problem = args.problem
            terminate_paraprof_workers(comm)
            _save(result, args.output)
        else:
            paraprof_worker_loop(comm)
    else:
        # Non-MPI competitor adapters: master runs everything, workers idle.
        if rank == 0:
            result = adapter.run(
                func=func,
                bounds=bounds,
                dims=args.dims,
                grid_points=args.grid,
                max_evals_per_cell=args.max_evals_per_cell,
                seed=args.seed,
                comm=None,
            )
            result.problem = args.problem
            _save(result, args.output)
    return 0


def _save(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result.to_dict(), f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    sys.exit(main())
