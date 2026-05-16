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


# ---------------------------------------------------------------------------
# Suspect-cell recheck — end-to-end regression test
# ---------------------------------------------------------------------------
# Rosenbrock-4D on a coarse 80x80 grid reliably produces a small contiguous
# strip of grid cells stuck on a wrong optimum in the profiled dimensions:
# DE + L-BFGS-B polish + patching waves all leave the strip in place because
# its members agree with each other and the standard patching filter only
# tests neighbours with strictly higher fitness. The suspect-cell recheck
# pass is designed to catch exactly this. The runner below captures the grid
# both BEFORE the recheck stage starts and AFTER the full scan completes,
# then diffs the two so the feature's effect is isolated deterministically
# within one MPI run (sidestepping cross-run RNG variance from independent
# worker random states).

SUSPECT_RUNNER = textwrap.dedent("""
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

    log_likelihood, bounds, _ = get_test_function('rosenbrock_4d')
    projections = [{'dims': [0, 1], 'grid_points': [80, 80]}]
    np.random.seed(750123)

    if rank == 0:
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            roi_threshold=8.0,
            pop_per_grid_point=3,
            n_initial_optimizations=100,
            max_patching_waves=20,
            lbfgsb_max_iter=20,
            advanced_config={'convergence_threshold': 1e-7},
        ) as sampler:
            comm.bcast(sampler.target_func, root=0)

            # Snapshot the per-cell best_fitness right before the first
            # suspect-recheck wave is scheduled.
            before = {'snapshot': None}
            orig = sampler.create_suspect_recheck_jobs

            def patched(wave_number, updated_points_last_wave, next_job_id):
                if wave_number == 0 and before['snapshot'] is None:
                    snap = np.full(sampler.grid_shape, np.nan)
                    for idx, st in sampler.population.items():
                        snap[idx] = st['best_fitness']
                    before['snapshot'] = snap
                return orig(wave_number, updated_points_last_wave, next_job_id)
            sampler.create_suspect_recheck_jobs = patched

            run_all_projections(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False, myrank=rank,
            )

            after = np.full(sampler.grid_shape, np.nan)
            for idx, st in sampler.population.items():
                after[idx] = st['best_fitness']

            if before['snapshot'] is None:
                payload = {'fired': False}
            else:
                b = before['snapshot']
                mask = ~(np.isnan(b) | np.isnan(after))
                diff = np.where(mask, after - b, 0.0)
                payload = {
                    'fired': True,
                    'n_improved': int(np.sum(diff > 1e-6)),
                    'n_regressed': int(np.sum(diff < -1e-6)),
                    'max_improvement': float(diff.max()),
                    'min_change': float(diff.min()),
                    'total_delta': float(diff.sum()),
                }
            print('RESULT', json.dumps(payload), flush=True)
        terminate_workers(comm, rank)
    else:
        worker_main(comm, rank)
""")


def _run_suspect():
    """Run SUSPECT_RUNNER under mpiexec -n 4 and parse the RESULT line."""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(SUSPECT_RUNNER)
        runner_path = f.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OMPI_', 'OPAL_', 'PMIX_', 'PMI_'))}
    try:
        proc = subprocess.run(
            ['mpiexec', '--allow-run-as-root', '--oversubscribe',
             '-n', '4', sys.executable, runner_path],
            capture_output=True, text=True, timeout=180, env=env,
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


def test_suspect_recheck_fixes_rosenbrock_strip():
    """Rosenbrock-4D 80x80 should produce a wrong-optimum strip that the
    suspect-recheck pass detects and fixes — with zero regressions."""
    result = _run_suspect()

    assert result['fired'], "Suspect-recheck stage never ran"

    # The pass must improve at least one cell. Observed on the seed used
    # above: 4 cells improved by up to ~+8 logL.
    assert result['n_improved'] >= 1, (
        f"Suspect recheck did not improve any cells: {result}"
    )

    # Crucially: never regresses a cell. The LBFGSB polish only writes back
    # when current_fitness > state['best_fitness'], so this is a structural
    # guarantee — but the test makes the guarantee visible.
    assert result['n_regressed'] == 0, (
        f"Suspect recheck regressed {result['n_regressed']} cell(s): {result}"
    )
    assert result['min_change'] >= 0.0, (
        f"Suspect recheck produced a negative per-cell change "
        f"({result['min_change']}); the pass must never lower a cell's logL"
    )

    # The improvements should be meaningful (not just numerical noise from
    # an L-BFGS-B re-polish landing one ftol away). Observed ~+8 logL on the
    # worst strip cell; require at least +1 to leave headroom for variance.
    assert result['max_improvement'] > 1.0, (
        f"Largest improvement was only {result['max_improvement']:.3e}; "
        "expected at least one cell to gain >1 logL on this case."
    )
