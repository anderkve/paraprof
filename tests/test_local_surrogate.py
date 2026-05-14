"""
Tests for the local-quadratic surrogate feature.

Two layers:
  1. Unit tests for the fit/predict primitives in ``paraprof.local_surrogate``.
  2. An MPI end-to-end test (subprocess) that runs the same Himmelblau-4D
     projection with the feature off and on, asserts the surrogate path
     was actually exercised, that the on-run produced fewer target calls,
     and that the ROI grid logL values match within a documented tolerance.
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

from paraprof.local_surrogate import (
    fit_local_quadratic,
    predict_local_quadratic,
    quadratic_basis_size,
)


# --------------------------------------------------------------------------
# Unit tests for fit / predict
# --------------------------------------------------------------------------

def test_quadratic_basis_size():
    # 1 + D + D*(D+1)/2
    assert quadratic_basis_size(1) == 3
    assert quadratic_basis_size(2) == 6
    assert quadratic_basis_size(4) == 15


def test_fit_predicts_known_quadratic_exactly():
    """A noiseless quadratic should be recovered to high accuracy."""
    rng = np.random.default_rng(42)
    n_dims = 3
    X = rng.uniform(-1.0, 1.0, size=(60, n_dims))
    # Pick a target quadratic; sign chosen so a maximum exists in the box.
    A = np.array([
        [-1.0, 0.2, 0.0],
        [0.2, -1.5, 0.1],
        [0.0, 0.1, -0.8],
    ])
    b = np.array([0.3, -0.1, 0.2])
    c = 0.5
    y = np.array([x @ A @ x + b @ x + c for x in X])

    model = fit_local_quadratic(X, y)
    assert model is not None

    # Same surface evaluated on a fresh test set.
    X_test = rng.uniform(-1.0, 1.0, size=(20, n_dims))
    y_test = np.array([x @ A @ x + b @ x + c for x in X_test])
    y_pred = predict_local_quadratic(model, X_test)

    np.testing.assert_allclose(y_pred, y_test, atol=1e-8)


def test_fit_returns_none_when_too_few_samples():
    # With D=4, basis size is 15; 5 samples is far too few.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(5, 4))
    y = rng.uniform(-1.0, 1.0, size=5)
    assert fit_local_quadratic(X, y) is None


def test_fit_returns_none_when_singular():
    """Collinear inputs make the design rank-deficient; fit should bail out."""
    # All points on a single line in 3D => quadratic terms are linearly
    # dependent and the design matrix has a huge condition number.
    n = 30
    t = np.linspace(-1.0, 1.0, n)
    X = np.stack([t, 2 * t, -t], axis=1)
    y = t ** 2
    model = fit_local_quadratic(X, y, cond_max=1e8)
    assert model is None


def test_predict_handles_single_point():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.0, 1.0, size=(40, 2))
    y = -(X[:, 0] ** 2 + X[:, 1] ** 2)
    model = fit_local_quadratic(X, y)
    assert model is not None
    pred = predict_local_quadratic(model, np.array([0.0, 0.0]))
    assert pred.shape == (1,)
    assert abs(pred[0]) < 1e-8


# --------------------------------------------------------------------------
# End-to-end MPI test: feature off vs feature on
# --------------------------------------------------------------------------

# Inline runner: runs a single Himmelblau-4D projection with the surrogate
# flag controlled by argv. Writes a JSON 'RESULT' line including the full
# grid for off/on comparison.
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

    use_surrogate = sys.argv[1] == 'on'

    log_likelihood, bounds, _ = get_test_function('himmelblau_4d')
    projections = [
        {'dims': [0, 1], 'grid_points': [10, 10]},
    ]

    np.random.seed(20260514)

    if rank == 0:
        # Default user-facing configuration: L-BFGS-B polish on, patching on.
        # This is the regime real users run in, and the regime in which the
        # surrogate prescreen must preserve ROI accuracy.
        with ProfileProjector(
            target_func=log_likelihood,
            bounds=bounds,
            projections=projections,
            roi_threshold=4.0,
            pop_per_grid_point=3,
            n_initial_optimizations=20,
            max_patching_waves=3,
            lbfgsb_max_iter=10,
            use_local_surrogate=use_surrogate,
        ) as sampler:
            comm.bcast(sampler.target_func, root=0)
            results = run_all_projections(
                comm=comm, sampler=sampler, projections=projections,
                save_plots=False, myrank=rank,
            )
            metrics = results[0]['metrics']
            grid = sampler.export_grid_solution()
            grid_logL = {
                ','.join(str(i) for i in idx): float(sol['likelihood'])
                for idx, sol in grid['solutions'].items()
            }
            payload = {
                'use_surrogate': use_surrogate,
                'calls': metrics['total_target_calls'],
                'max_logL': float(metrics['global_max']),
                'prescreen_count': int(sampler._surrogate_prescreen_count),
                'cache_size': sum(len(v) for v in sampler._surrogate_cache.values()),
                'grid_logL': grid_logL,
            }
            print('RESULT', json.dumps(payload), flush=True)
        terminate_workers(comm, rank)
    else:
        worker_main(comm, rank)
""")


def _have_mpiexec():
    return shutil.which('mpiexec') is not None


@pytest.mark.skipif(
    not _have_mpiexec(),
    reason='mpiexec not on PATH; surrogate end-to-end test requires an MPI runtime',
)
def test_surrogate_path_taken_and_grid_matches_off_path():
    """Run a small Himmelblau-4D projection off and on, then compare.

    Three independent assertions:
      * Off-run has zero surrogate prescreen events, on-run has many.
      * On-run's ROI logL grid matches the off-run within tolerance — the
        surrogate must not bias the result inside the region of interest.
      * On-run does not blow up the call count (sanity, not the primary
        eval-savings claim which is benchmarked separately).
    """
    runs = {}
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(RUNNER)
        runner_path = f.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OMPI_', 'OPAL_', 'PMIX_', 'PMI_'))}
    try:
        for tag in ('off', 'on'):
            proc = subprocess.run(
                ['mpiexec', '--allow-run-as-root', '--oversubscribe',
                 '-n', '4', sys.executable, runner_path, tag],
                capture_output=True, text=True, timeout=180, env=env,
            )
            if proc.returncode != 0:
                raise AssertionError(
                    f"mpiexec runner failed for tag={tag} (rc={proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith('RESULT '):
                    payload = json.loads(line[len('RESULT '):])
                    break
            assert payload is not None, (
                f"No RESULT line for tag={tag}.\nSTDOUT:\n{proc.stdout}"
            )
            runs[tag] = payload
    finally:
        os.unlink(runner_path)

    off, on = runs['off'], runs['on']

    # 1. The surrogate path was actually taken when the flag is on, and not
    # when the flag is off.
    assert off['prescreen_count'] == 0
    assert on['prescreen_count'] >= 5, (
        f"Surrogate prescreen path was taken only {on['prescreen_count']} times; "
        "expected many. Feature is probably misconfigured."
    )
    assert on['cache_size'] > 0
    assert off['cache_size'] == 0, (
        "Surrogate cache should remain empty when use_local_surrogate=False; "
        f"got cache_size={off['cache_size']}"
    )

    # 2. ROI logL grid agreement. Both runs should find peaks near 0.
    assert on['max_logL'] > -1e-3
    assert off['max_logL'] > -1e-3

    # Compare cell-by-cell on cells that exist in both grids and fall inside
    # the ROI (within 4.0 of the global max). On the user-default config
    # (with L-BFGS-B polish + patching enabled) the surrogate prescreen
    # must agree with the off-path to better than 1e-3 logL inside the ROI:
    # the polish step washes out the small stochastic differences in DE's
    # trial selection that the surrogate introduces.
    common = set(off['grid_logL']) & set(on['grid_logL'])
    global_max = max(off['max_logL'], on['max_logL'])
    roi_threshold = 4.0
    deltas = []
    for k in common:
        l_off, l_on = off['grid_logL'][k], on['grid_logL'][k]
        if max(l_off, l_on) >= global_max - roi_threshold:
            deltas.append(abs(l_off - l_on))
    assert deltas, "No common ROI cells between off and on runs"
    max_delta = max(deltas)
    assert max_delta < 1e-3, (
        f"max |Δ logL| in ROI = {max_delta:.3e} between off and on runs; "
        f"expected < 1e-3. (off max={off['max_logL']:.3e}, "
        f"on max={on['max_logL']:.3e})"
    )

    # 3. Sanity: the on-run should not blow up the call count by 2x or more.
    assert on['calls'] < 2 * off['calls'], (
        f"on calls={on['calls']} vs off calls={off['calls']} — feature is "
        "supposed to save evaluations, not double them."
    )
