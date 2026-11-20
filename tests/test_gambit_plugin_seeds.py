"""The GAMBIT plugin's handling of `initial_points`, without MPI or GAMBIT.

The plugin module imports `scanner_plugin`, `scannerbit` and `utils` from
ScannerBit. They are stubbed here for the duration of one test.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PLUGIN_FILE = Path(__file__).resolve().parent.parent / "GAMBIT_plugin" / "gambit_paraprof.py"

NAMES = ("Model::alpha", "Model::beta", "Model::gamma")
LOWER = np.array([0.0, 1.0e-5, -2.0])
UPPER = np.array([4.0, 1.0e-4, 2.0])


@pytest.fixture
def plugin(monkeypatch):
    scanner_plugin = types.ModuleType("scanner_plugin")
    scanner_plugin.scanner = type("scanner", (), {})
    scannerbit = types.ModuleType("scannerbit")
    scannerbit.with_mpi = False
    utils = types.ModuleType("utils")
    utils.copydoc = lambda *a, **k: (lambda fn: fn)
    utils.version = lambda pkg: "n/a"
    utils.get_directory = lambda *a, **k: "."
    utils.with_mpi = False
    for name, mod in (("scanner_plugin", scanner_plugin), ("scannerbit", scannerbit),
                      ("utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)

    spec = importlib.util.spec_from_file_location("gambit_paraprof_under_test", PLUGIN_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.__plugins__["paraprof"]

    obj = object.__new__(cls)
    obj.print_prefix = "paraprof scanner plugin:"
    obj.parameter_names = NAMES
    calls = []

    def inverse_transform(physical):
        # ScannerBit's inverse prior transform takes {full name: value}.
        calls.append(dict(physical))
        x = np.array([physical[n] for n in NAMES])
        return (x - LOWER) / (UPPER - LOWER)

    obj.inverse_transform = inverse_transform
    obj.calls = calls
    return obj


def test_mapping_is_physical_and_goes_through_the_inverse_transform(plugin):
    out = plugin._initial_points_to_unit([{"alpha": 1.0, "Model::beta": 5.5e-5, "gamma": 2.0}])
    np.testing.assert_allclose(out, [[0.25, 0.5, 1.0]])
    assert plugin.calls == [{"Model::alpha": 1.0, "Model::beta": 5.5e-5, "Model::gamma": 2.0}]


def test_list_is_taken_as_unit_hypercube_coordinates(plugin):
    out = plugin._initial_points_to_unit([[0.1, 0.2, 0.3], (0.9, 0.8, 0.7)])
    assert out == [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
    assert plugin.calls == []


def test_unknown_and_missing_parameters_are_errors(plugin):
    with pytest.raises(RuntimeError, match="not a scanned parameter"):
        plugin._initial_points_to_unit([{"alpha": 1.0, "beta": 5e-5, "delta": 0.0}])
    with pytest.raises(RuntimeError, match="gives no value"):
        plugin._initial_points_to_unit([{"alpha": 1.0, "beta": 5e-5}])


def test_short_name_shared_by_two_models_is_ambiguous(plugin):
    plugin.parameter_names = ("A::x", "B::x", "A::y")
    with pytest.raises(RuntimeError, match="ambiguous"):
        plugin._initial_points_to_unit([{"x": 0.5, "y": 0.5}])
