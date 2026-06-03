"""
Tests for the host-framework integration API.

These cover the bits of the public surface added so paraprof can be embedded
inside an external master/worker MPI loop (e.g. as a ScannerBit plugin):

- ``ProfileProjector(parameter_names=...)`` and string ``dims`` resolution.
- ``worker_main(comm, myrank, target_func=...)`` accepting a pre-supplied
  callable instead of waiting on ``comm.bcast``.
- ``run_scan(..., broadcast_target_func=False)`` bundling the master-side
  setup/teardown into one call.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

from paraprof import ProfileProjector, run_scan, worker_main
from paraprof.exceptions import InvalidProjectionError


# ---------------------------------------------------------------------------
# Unit tests: parameter_names + string dims resolution
# ---------------------------------------------------------------------------

class TestParameterNames:

    def _f(self, x):
        return -float(np.sum(np.asarray(x) ** 2))

    def test_string_dims_resolve_to_indices(self, simple_bounds_2d):
        sampler = ProfileProjector(
            target_func=self._f,
            bounds=simple_bounds_2d,
            projections=[{'dims': ['x', 'y'], 'grid_points': [4, 4],
                          'patch_coarse_grid': False}],
            parameter_names=['x', 'y'],
        )
        # dims are rewritten in place to integer indices
        assert sampler.projections[0]['dims'] == [0, 1]

    def test_string_dims_rejected_without_names(self, simple_bounds_2d):
        with pytest.raises(InvalidProjectionError):
            ProfileProjector(
                target_func=self._f,
                bounds=simple_bounds_2d,
                projections=[{'dims': ['x', 'y'], 'grid_points': [3, 3]}],
            )

    def test_unknown_name_raises(self, simple_bounds_2d):
        with pytest.raises(InvalidProjectionError):
            ProfileProjector(
                target_func=self._f,
                bounds=simple_bounds_2d,
                projections=[{'dims': ['x', 'z'], 'grid_points': [3, 3]}],
                parameter_names=['x', 'y'],
            )


# ---------------------------------------------------------------------------
# End-to-end: run_scan + worker_main with a host-supplied target_func
# ---------------------------------------------------------------------------

# Mirrors test_integration.py but exercises the no-broadcast path: the worker
# is given target_func directly, and run_scan is called with
# broadcast_target_func=False. This is the path the GAMBIT/ScannerBit plugin
# uses (the loglike is a bound method that cannot be pickled).
RUNNER = textwrap.dedent("""
    import json
    import sys
    import numpy as np
    from mpi4py import MPI

    from paraprof import (
        ProfileProjector, run_scan, worker_main,
        get_test_function, set_log_level,
    )
    set_log_level('WARNING')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    log_likelihood, bounds, _ = get_test_function('himmelblau_4d')

    # NB: dims are passed as parameter NAMES, exercising the resolver.
    PARAM_NAMES = ['p0', 'p1', 'p2', 'p3']
    projections = [
        {'dims': ['p0', 'p1'], 'grid_points': [10, 10],
         'optimization_method': 'lbfgsb'},
    ]

    np.random.seed(20260514)

    if rank == 0:
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            parameter_names=PARAM_NAMES,
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=20,
            max_patching_waves=3,
            lbfgsb_max_iter=15,
        ) as sampler:
            results = run_scan(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False,
                broadcast_target_func=False,   # <-- no bcast: worker has func
                myrank=rank,
            )
            metrics = results[0]['metrics']
            payload = {
                'resolved_dims': list(projections[0]['dims']),
                'calls': metrics['total_target_calls'],
                'max_logL': float(metrics['global_max']),
            }
            print('RESULT', json.dumps(payload), flush=True)
    else:
        # Worker receives the target function directly (no comm.bcast).
        worker_main(comm, rank, target_func=log_likelihood)
""")


def _have_mpiexec():
    return shutil.which('mpiexec') is not None


pytestmark = pytest.mark.skipif(
    not _have_mpiexec(),
    reason='mpiexec not on PATH; integration test requires an MPI runtime',
)


def test_run_scan_no_broadcast_path():
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(RUNNER)
        runner_path = f.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OMPI_', 'OPAL_', 'PMIX_', 'PMI_'))}
    try:
        proc = subprocess.run(
            ['mpiexec', '--allow-run-as-root', '--oversubscribe',
             '-n', '4', sys.executable, runner_path],
            capture_output=True, text=True, timeout=120, env=env,
        )
    finally:
        os.unlink(runner_path)
    if proc.returncode != 0:
        raise AssertionError(
            f"mpiexec runner failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            result = json.loads(line[len('RESULT '):])
            break
    assert result is not None, f"No RESULT line.\nSTDOUT:\n{proc.stdout}"

    # String dims were resolved to integer indices.
    assert result['resolved_dims'] == [0, 1]

    # The scan ran and converged on a Himmelblau peak.
    assert result['calls'] > 0
    assert result['max_logL'] > -1e-6, (
        f"max_logL={result['max_logL']:.3e}, expected close to 0"
    )
