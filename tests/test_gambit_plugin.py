"""
Standalone smoke test for the GAMBIT plugin.

The plugin is normally exercised by GAMBIT's ScannerBit machinery, which
provides `scanner_plugin`, `scannerbit`, and `utils` as ambient modules.
Here we stub those just well enough to run the plugin under mpiexec against
a known test function (4D Himmelblau), and check that:

  - The plugin instantiates against the stub base class.
  - Workers route through `loglike_hypercube` and call `print(1.0, "Posterior")`
    after each evaluation (i.e. the ScannerBit printer-flow contract holds).
  - The scan finishes with rc=0 and lands on a Himmelblau peak.
  - Parameter-name `dims` get resolved to integer indices end-to-end.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "GAMBIT_plugin"


RUNNER = textwrap.dedent("""
    import json
    import sys
    import types
    import numpy as np
    from mpi4py import MPI

    PLUGIN_DIR = sys.argv[1]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 4D Himmelblau (embedded: first two dims are the active Himmelblau pair,
    # last two are inert nuisance parameters minimised by adding a quadratic).
    BOUNDS = np.array([[-5., 5.], [-5., 5.], [-5., 5.], [-5., 5.]])

    def loglike_physical(x):
        a = x[0] * x[0] + x[1] - 11.0
        b = x[0] + x[1] * x[1] - 7.0
        return -(a * a + b * b) - (x[2] * x[2] + x[3] * x[3])


    # ---- Test-side config ----------------------------------------------------
    # The plugin reads every paraprof-native option from `self.run_args` (the
    # YAML `run:` block), so the stub must expose them there.
    TEST_RUN_ARGS = {
        'projections': [
            # dims supplied as parameter NAMES, exercising the resolver.
            {'dims': ['par_1', 'par_2'], 'grid_points': [10, 10],
             'optimization_method': 'lbfgsb'},
        ],
        'roi_threshold': 4.0,
        'pop_per_grid_point': 3,
        'n_initial_optimizations': 15,
        'max_patching_waves': 2,
        'lbfgsb_max_iter': 15,
        'save_plots': False,
    }


    # ---- Stubs for the ScannerBit ambient modules ---------------------------
    # Counters live in a module-level dict so subclasses can't accidentally
    # rebind a class attribute (a previous version used `type(self).x += 1`
    # which, when self is a ParaProf subclass instance, creates a fresh attr
    # on the subclass instead of incrementing the base-class one).
    STATE = {'point_id': 0, 'like_calls': 0, 'posterior_log': []}

    class _StubScanner:
        '''Mimics splug.scanner just enough for the paraprof plugin.'''

        def __init__(self, use_mpi=True, use_resume=False):
            self.mpi_rank = rank
            self.mpi_size = size
            self.dim = BOUNDS.shape[0]
            self.parameter_names = ('par_1', 'par_2', 'par_3', 'par_4')
            self.run_args = dict(TEST_RUN_ARGS)

        def loglike_hypercube(self, x):
            STATE['point_id'] += 1
            STATE['like_calls'] += 1
            x = np.asarray(x, dtype=float)
            x_phys = BOUNDS[:, 0] + x * (BOUNDS[:, 1] - BOUNDS[:, 0])
            return float(loglike_physical(x_phys))

        def print(self, weight, key):
            STATE['posterior_log'].append(
                (self.mpi_rank, STATE['point_id'], float(weight), str(key))
            )

        def transform_to_vec(self, x):
            x = np.asarray(x, dtype=float)
            return BOUNDS[:, 0] + x * (BOUNDS[:, 1] - BOUNDS[:, 0])

        def inverse_transform(self, x_phys):
            x_phys = np.asarray(x_phys, dtype=float)
            return (x_phys - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


    scanner_plugin = types.ModuleType('scanner_plugin')
    scanner_plugin.scanner = _StubScanner
    sys.modules['scanner_plugin'] = scanner_plugin

    scannerbit_mod = types.ModuleType('scannerbit')
    scannerbit_mod.with_mpi = True
    sys.modules['scannerbit'] = scannerbit_mod

    def _copydoc(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def _version(pkg):
        return getattr(pkg, '__version__', 'n/a') if pkg is not None else 'n/a'

    utils_mod = types.ModuleType('utils')
    utils_mod.copydoc = _copydoc
    utils_mod.version = _version
    utils_mod.with_mpi = True
    sys.modules['utils'] = utils_mod

    # ---- Load the plugin -----------------------------------------------------
    sys.path.insert(0, PLUGIN_DIR)
    import gambit_paraprof
    ParaProf = gambit_paraprof.__plugins__['paraprof']

    # Capture the run_scan results dict on rank 0 by wrapping run_scan.
    rs_results = []
    original_run_scan = gambit_paraprof.paraprof_run_scan
    def _capture_run_scan(*a, **kw):
        out = original_run_scan(*a, **kw)
        rs_results.append(out)
        return out
    gambit_paraprof.paraprof_run_scan = _capture_run_scan

    plugin = ParaProf()
    rc = plugin.run()

    # ---- Report from rank 0 --------------------------------------------------
    all_like_calls = comm.gather(STATE['like_calls'], root=0)
    all_post_counts = comm.gather(len(STATE['posterior_log']), root=0)

    if rank == 0:
        results = rs_results[0] if rs_results else []
        max_logL = float(results[0]['metrics']['global_max']) if results else None
        payload = {
            'rc': rc,
            'master_like_calls': all_like_calls[0],
            'worker_like_calls_total': sum(all_like_calls[1:]),
            'worker_post_count_total': sum(all_post_counts[1:]),
            # Read from the plugin's projection copy on rank 0: paraprof
            # rewrites string dims to integer indices in place at construction.
            'resolved_dims': list(plugin.projections[0]['dims']),
            'max_logL': max_logL,
        }
        print('RESULT', json.dumps(payload), flush=True)
""")


def _have_mpiexec():
    return shutil.which("mpiexec") is not None


pytestmark = pytest.mark.skipif(
    not _have_mpiexec(),
    reason="mpiexec not on PATH; standalone plugin test requires an MPI runtime",
)


def _run():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RUNNER)
        runner_path = f.name
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMPI_", "OPAL_", "PMIX_", "PMI_"))}
    try:
        proc = subprocess.run(
            ["mpiexec", "--allow-run-as-root", "--oversubscribe",
             "-n", "4", sys.executable, runner_path, str(PLUGIN_DIR)],
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
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):]), proc.stdout
    raise AssertionError(f"Runner produced no RESULT line.\nSTDOUT:\n{proc.stdout}")


def test_gambit_plugin_smoke():
    """End-to-end: build the plugin with stub ScannerBit modules and run a scan."""
    result, stdout = _run()

    # The plugin returned successfully.
    assert result["rc"] == 0

    # ScannerBit printer contract: the worker target wrapper must call
    # `self.print(1.0, "Posterior")` once per `loglike_hypercube` call.
    # We can't observe the count exactly (timing), but they must be equal.
    assert result["worker_post_count_total"] == result["worker_like_calls_total"], (
        f"posterior records ({result['worker_post_count_total']}) != "
        f"loglike calls ({result['worker_like_calls_total']}) on workers"
    )
    assert result["worker_like_calls_total"] > 0

    # Master rank does no target evaluations (paraprof master-worker model).
    assert result["master_like_calls"] == 0

    # String dims got resolved to integer indices.
    assert result["resolved_dims"] == [0, 1]

    # And the plugin's per-projection summary block should appear in stdout.
    assert "=== Scan summary ===" in stdout
    assert "best logL found:" in stdout

    # Sanity check the scan converged: this Himmelblau-with-quadratic-nuisance
    # has max logL = 0 (Himmelblau peaks at zero; nuisance dims optimised to
    # origin contribute zero penalty). Allow a generous tolerance — we only
    # ran a tiny 10x10 grid with few patching waves.
    assert result["max_logL"] is not None
    assert result["max_logL"] > -1e-4, (
        f"max_logL={result['max_logL']:.4e}, expected close to 0"
    )
