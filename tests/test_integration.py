"""
End-to-end integration test exercising master_main + worker_main via mpiexec.

This is the only test in the suite that covers the full event-loop / job
state machine. Most other tests are unit-level and stub out the master.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest


# Inline runner script. Written to a tempfile and invoked under mpiexec.
RUNNER = textwrap.dedent("""
    import json
    import sys
    import numpy as np
    from mpi4py import MPI

    from paraprof import (
        ProfileProjector, run_all_projections, terminate_workers, worker_main,
        get_test_function, set_log_level,
    )
    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    method = sys.argv[1]   # 'de' or 'lbfgsb'
    log_likelihood, bounds, _ = get_test_function('himmelblau_4d')
    projections = [
        {'dims': [0, 1], 'grid_points': [12, 12], 'optimization_method': method},
    ]

    np.random.seed(20260513)

    if rank == 0:
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=20,
            max_patching_waves=5,
            lbfgsb_max_iter=15,
        ) as sampler:
            comm.bcast(sampler.target_func, root=0)
            results = run_all_projections(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False, myrank=rank,
            )
            metrics = results[0]['metrics']
            grid = sampler.export_grid_solution()
            payload = {
                'method': method,
                'calls': metrics['total_target_calls'],
                'max_logL': float(metrics['global_max']),
                'n_activated_cells': len(grid['solutions']),
            }
            print('RESULT', json.dumps(payload), flush=True)
        terminate_workers(comm, rank)
    else:
        worker_main(comm, rank)
""")


def _have_mpiexec():
    return shutil.which('mpiexec') is not None


pytestmark = pytest.mark.skipif(
    not _have_mpiexec(),
    reason='mpiexec not on PATH; integration test requires an MPI runtime',
)


def _run(method):
    """Run the runner with `mpiexec -n 4` and parse the RESULT line."""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(RUNNER)
        runner_path = f.name
    # If pytest's parent process imported mpi4py (e.g. via earlier tests
    # importing paraprof), OMPI/PMIX env vars leak into the child mpiexec
    # and confuse it. Strip them before invoking the subprocess.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OMPI_', 'OPAL_', 'PMIX_', 'PMI_'))}
    try:
        proc = subprocess.run(
            ['mpiexec', '--allow-run-as-root', '--oversubscribe',
             '-n', '4', sys.executable, runner_path, method],
            capture_output=True, text=True, timeout=120, env=env,
        )
    finally:
        os.unlink(runner_path)
    if proc.returncode != 0:
        raise AssertionError(
            f"mpiexec runner failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[len('RESULT '):])
    raise AssertionError(f"Runner produced no RESULT line.\nSTDOUT:\n{proc.stdout}")


@pytest.mark.parametrize('method', ['de', 'lbfgsb'])
def test_end_to_end_scan(method):
    """A 12x12 Himmelblau scan should converge to the true peak."""
    result = _run(method)

    # Sanity: we ran something and produced a grid.
    assert result['calls'] > 0
    assert result['n_activated_cells'] > 0

    # The four Himmelblau peaks all sit at logL = 0; we should land on one
    # of them within numerical noise.
    assert result['max_logL'] > -1e-6, (
        f"max_logL = {result['max_logL']:.3e} for method={method!r}; "
        "expected close to 0"
    )

    # Sanity-check the call count is in the right ballpark (catches order-of-
    # magnitude regressions in budget). With grid 12x12, n_initial_optimizations=20,
    # pop_per_grid_point=3, both paths typically use 1k-15k target evaluations.
    assert 100 < result['calls'] < 50000, (
        f"calls = {result['calls']} for method={method!r} is out of expected "
        "range [100, 50000]"
    )
