"""The GAMBIT plugin's handling of `initial_points` and `branches`, without MPI or GAMBIT.

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
    obj.transform_to_vec = lambda unit: LOWER + np.asarray(unit) * (UPPER - LOWER)
    obj.calls = calls
    return obj


def test_mapping_is_physical_and_goes_through_the_inverse_transform(plugin):
    out = plugin._initial_points_to_unit([{"alpha": 1.0, "Model::beta": 5.5e-5, "gamma": 2.0}])
    np.testing.assert_allclose(out, [[0.25, 0.5, 1.0]])
    assert plugin.calls == [{"Model::alpha": 1.0, "Model::beta": 5.5e-5, "Model::gamma": 2.0}]


def test_a_point_that_is_not_a_mapping_is_rejected(plugin):
    point = {"alpha": 1.0, "beta": 5.5e-5, "gamma": 2.0}
    with pytest.raises(RuntimeError, match="entry 1 is not a mapping"):
        plugin._initial_points_to_unit([point, [0.1, 0.2, 0.3]])
    # A single mapping in place of the list of points.
    with pytest.raises(RuntimeError, match="list of mappings"):
        plugin._initial_points_to_unit(point)
    assert plugin.calls == [{"Model::alpha": 1.0, "Model::beta": 5.5e-5, "Model::gamma": 2.0}]


def test_unknown_and_missing_parameters_are_errors(plugin):
    with pytest.raises(RuntimeError, match="not a scanned parameter"):
        plugin._initial_points_to_unit([{"alpha": 1.0, "beta": 5e-5, "delta": 0.0}])
    with pytest.raises(RuntimeError, match="gives no value"):
        plugin._initial_points_to_unit([{"alpha": 1.0, "beta": 5e-5}])


def test_short_name_shared_by_two_models_is_ambiguous(plugin):
    plugin.parameter_names = ("A::x", "B::x", "A::y")
    with pytest.raises(RuntimeError, match="ambiguous"):
        plugin._initial_points_to_unit([{"x": 0.5, "y": 0.5}])


def test_split_points_are_mapped_from_physical_coordinates(plugin):
    out = plugin._branches_to_unit({"alpha": 1.0, "Model::gamma": [1.0, -1.0]}, "branches")
    assert list(out) == ["Model::alpha", "Model::gamma"]
    np.testing.assert_allclose(out["Model::alpha"], [0.25])
    np.testing.assert_allclose(out["Model::gamma"], [0.75, 0.25])
    # The other parameters sit at the centre of the hypercube in the transform.
    assert plugin.calls[0] == {"Model::alpha": 1.0, "Model::beta": 5.5e-5, "Model::gamma": 0.0}
    assert plugin._branches_to_unit({}, "branches") == {}
    assert plugin._branches_to_unit(None, "branches") == {}


def test_bad_branches_are_errors(plugin):
    with pytest.raises(RuntimeError, match="must be a mapping"):
        plugin._branches_to_unit([0.5], "branches")
    with pytest.raises(RuntimeError, match="not a scanned parameter"):
        plugin._branches_to_unit({"delta": 0.5}, "branches")
    with pytest.raises(RuntimeError, match="twice"):
        plugin._branches_to_unit({"alpha": 1.0, "Model::alpha": 2.0}, "branches")


def test_split_parameter_whose_prior_depends_on_another_is_rejected(plugin):
    plain = plugin.inverse_transform

    def alpha_depends_on_beta(physical):
        unit = plain(physical)
        unit[0] += 0.4 * (unit[1] - 0.5)
        return unit

    def beta_depends_on_alpha(physical):
        unit = plain(physical)
        unit[1] += 0.1 * (unit[0] - 0.5)
        return unit

    # The split point of alpha has no single unit coordinate.
    plugin.inverse_transform = alpha_depends_on_beta
    with pytest.raises(RuntimeError, match="depends on other parameters"):
        plugin._branches_to_unit({"alpha": 1.0}, "projections[0] branches")
    # Here it has one: only the position of beta moves with alpha.
    plugin.inverse_transform = beta_depends_on_alpha
    out = plugin._branches_to_unit({"alpha": 1.0}, "branches")
    np.testing.assert_allclose(out["Model::alpha"], [0.25])


def test_the_mapped_split_is_accepted_by_paraprof(plugin):
    from paraprof import ProfileProjector
    unit = plugin._branches_to_unit({"gamma": [0.5]}, "branches")
    with ProfileProjector(target_func=lambda x: 0.0, bounds=[(0.0, 1.0)] * 3,
                          projections=[{"dims": ["Model::alpha", "Model::beta"],
                                        "grid_points": [4, 4]}],
                          parameter_names=list(NAMES), branches=unit) as sampler:
        assert [b["ranges"] for b in sampler.projection_branches] == [
            {2: (0.0, 0.625)}, {2: (0.625, 1.0)}]
