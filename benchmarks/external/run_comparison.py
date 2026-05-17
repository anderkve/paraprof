"""Top-level (non-MPI) orchestrator for the comparison sweep.

Invokes ``run_one.py`` once per (method, problem, projection, seed) tuple.
The actual MPI driving lives in run_one.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Full sweep: 4 problems x 2 projections x (6 grid methods + 1 mncontour) x 3 seeds + 1 oracle/problem.
PROBLEMS = ["rosenbrock_4d", "himmelblau_4d", "rastrigin_4d", "levy_4d"]

PROJECTIONS: dict[str, list[list[int]]] = {
    "rosenbrock_4d": [[0, 1], [1, 3]],
    "himmelblau_4d": [[0, 1], [0, 2]],
    "rastrigin_4d":  [[0, 1], [0, 3]],
    "levy_4d":       [[0, 1], [0, 3]],
}

GRID_METHODS = [
    "paraprof_default", "paraprof_kernel",
    "iminuit_grid", "scipy_de", "scipy_lbfgsb", "nlopt_crs2_bobyqa",
]
CONTOUR_METHODS = ["iminuit_mncontour"]

DEFAULT_GRID = [50, 50]
DEFAULT_SEEDS = [1, 2, 3]
DEFAULT_MAX_EVALS_PER_CELL = 1500


@dataclasses.dataclass
class Task:
    method: str
    problem: str
    dims: list[int]
    grid: list[int]
    seed: int
    max_evals_per_cell: int
    output: Path
    mpi_ranks: int


def task_output(results_root: Path, method: str, problem: str,
                dims: list[int], seed: int) -> Path:
    d = "_".join(str(i) for i in dims)
    return results_root / "runs" / f"{problem}__dims-{d}__{method}__seed-{seed}.json"


def build_tasks(args) -> list[Task]:
    problems = args.problems or PROBLEMS
    methods = args.methods or (GRID_METHODS + CONTOUR_METHODS)
    seeds = args.seeds or DEFAULT_SEEDS
    grid = args.grid or DEFAULT_GRID

    tasks: list[Task] = []
    for problem in problems:
        projections = args.dims_override or PROJECTIONS[problem]
        for dims in projections:
            for method in methods:
                for seed in seeds:
                    out = task_output(args.results_root, method, problem, list(dims), seed)
                    if out.exists() and not args.force:
                        continue
                    tasks.append(Task(
                        method=method,
                        problem=problem,
                        dims=list(dims),
                        grid=list(grid),
                        seed=int(seed),
                        max_evals_per_cell=args.max_evals_per_cell,
                        output=out,
                        mpi_ranks=args.mpi_ranks,
                    ))
    return tasks


def run_task(task: Task, dry_run: bool = False) -> int:
    cmd = [
        "mpiexec",
        "--allow-run-as-root",
        "-n", str(task.mpi_ranks),
        sys.executable, "-m", "benchmarks.external.run_one",
        "--method", task.method,
        "--problem", task.problem,
        "--dims", *[str(d) for d in task.dims],
        "--grid", *[str(n) for n in task.grid],
        "--seed", str(task.seed),
        "--max-evals-per-cell", str(task.max_evals_per_cell),
        "--output", str(task.output),
    ]
    print(f"\n>>> {' '.join(shlex.quote(c) for c in cmd)}")
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full comparison sweep.")
    parser.add_argument("--results-root",
                        type=Path,
                        default=Path("benchmarks/external/results"))
    parser.add_argument("--problems", nargs="*", default=None,
                        help="Subset of problems to run; default is all four.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of adapters to run; default is all.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--grid", nargs="*", type=int, default=None)
    parser.add_argument("--dims-override", nargs="*", type=int, action="append",
                        default=None, metavar="DIM",
                        help="Override projection dims. May be specified multiple "
                             "times; each occurrence is one projection.")
    parser.add_argument("--max-evals-per-cell", type=int, default=DEFAULT_MAX_EVALS_PER_CELL)
    parser.add_argument("--mpi-ranks", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run tasks even when an output JSON already exists.")
    parser.add_argument("--build-oracles-only", action="store_true",
                        help="Run only the oracle builds (paraprof_oracle on every problem/projection).")
    args = parser.parse_args(argv)

    if args.build_oracles_only:
        args.methods = ["paraprof_oracle"]
        args.seeds = [101]  # oracle.py records two seeds internally

    tasks = build_tasks(args)
    print(f"{len(tasks)} task(s) queued.")

    fail = 0
    for i, t in enumerate(tasks, 1):
        t.output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rc = run_task(t, dry_run=args.dry_run)
        dt = time.perf_counter() - t0
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"[{i}/{len(tasks)}] {t.method:>22s}  {t.problem:>15s}  "
              f"dims={t.dims}  seed={t.seed}  {dt:6.1f}s  {status}")
        if rc != 0:
            fail += 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
