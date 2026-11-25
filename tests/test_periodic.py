"""Tests for periodic parameters (`ProfileProjector(periodic=...)`).

The scans run on virtual workers (no MPI).
"""
import logging

import numpy as np
import pytest

from paraprof import ConfigurationError, ProfileProjector
from paraprof.interpolation import GridInterpolator
from virtual_pool import run_virtual_scan

TWO_PI = 2.0 * np.pi
ROI = 4.0
A = 6.0
DELTA0 = 5.9          # maximum, 0.38 below the upper bound


# --- target with the periodic parameter on the grid ---------------------------
# x0 = delta in [0, 2 pi] (periodic), x1 = theta, x2 and x3 profiled. The region
# of interest runs through delta = 2 pi and continues at delta = 0.

BOUNDS = [(0.0, TWO_PI), (0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
BEST = [DELTA0, 0.5 + 0.2 * np.sin(DELTA0), 0.5 * np.cos(DELTA0), 0.5 * np.sin(DELTA0)]


def target(x):
    d, t, n1, n2 = x
    return float(-A * (1.0 - np.cos(d - DELTA0))
                 - 0.5 * ((t - 0.5 - 0.2 * np.sin(d)) / 0.12) ** 2
                 - 0.5 * ((n1 - 0.5 * np.cos(d) - 0.3 * (t - 0.5)) / 0.08) ** 2
                 - 0.5 * ((n2 - 0.5 * np.sin(d) + 0.3 * (t - 0.5)) / 0.08) ** 2)


def profile(d, t):
    return -A * (1.0 - np.cos(d - DELTA0)) - 0.5 * ((t - 0.5 - 0.2 * np.sin(d)) / 0.12) ** 2


def roi_cells(out):
    """(cells of the exact region of interest, those of them the scan holds)."""
    D, T = np.meshgrid(*out['grid_axes'], indexing='ij')
    truth = {tuple(int(i) for i in idx) for idx in np.argwhere(profile(D, T) > -ROI)}
    held = {tuple(int(i) for i in idx) for idx, st in out['population'].items()
            if st['best_fitness'] > out['global_max'] - ROI}
    return truth, truth & held


# --- target with the periodic parameter profiled ------------------------------
# x0 = delta (periodic, profiled), x1 = t and x2 = u on the grid, x3 profiled.
# The best delta moves through the bound as t crosses 0.5.

BOUNDS_P = [(0.0, TWO_PI), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)]


def target_profiled(x):
    d, t, u, n = x
    best_d = 4.0 * (t - 0.5) + (u - 0.5)
    return float(-A * (1.0 - np.cos(d - best_d))
                 - 0.5 * ((u - 0.5) / 0.25) ** 2
                 - 0.5 * ((n - 0.4 * np.cos(d) - 0.2 * (t - 0.5)) / 0.08) ** 2)


def profile_deficits(out):
    """Exact profile minus stored value, over the exact region of interest."""
    ax_t, ax_u = out['grid_axes']
    deficits = []
    for (i, j), st in out['population'].items():
        exact = -0.5 * ((ax_u[j] - 0.5) / 0.25) ** 2
        if exact > -ROI:
            deficits.append(exact - st['best_fitness'])
    return np.array(deficits)


def make_sampler(periodic, dims=(0, 1), grid=(8, 4), **kwargs):
    return ProfileProjector(
        target_func=target, bounds=BOUNDS, periodic=periodic,
        projections=[{'dims': list(dims), 'grid_points': list(grid)}], **kwargs)


# --- configuration ------------------------------------------------------------

def test_periodic_accepts_indices_and_names():
    assert make_sampler(None).periodic_dims == []
    assert make_sampler([0]).periodic_dims == [0]
    names = ['delta', 'theta', 'n1', 'n2']
    sampler = make_sampler(['n1', 'delta'], parameter_names=names)
    assert sampler.periodic_dims == [0, 2]
    assert make_sampler('delta', parameter_names=names).periodic_dims == [0]


@pytest.mark.parametrize("periodic", [[4], [-1], ['delta'], [0.5], {0: True}, [True]])
def test_periodic_rejects_unknown_parameters(periodic):
    with pytest.raises(ConfigurationError, match="periodic"):
        make_sampler(periodic)


# --- grid neighbours ----------------------------------------------------------

def test_neighbours_wrap_along_the_periodic_dimension():
    sampler = make_sampler([0])            # grid 9 x 5; delta index 0 and 8 coincide
    assert sampler.grid_shape == (9, 5)

    # A point on the lower bound: its copy on the upper bound and the grid
    # points next to either.
    assert set(sampler._get_valid_neighbors((0, 2))) == {
        (8, 1), (8, 2), (8, 3), (7, 1), (7, 2), (7, 3),
        (0, 1), (0, 3), (1, 1), (1, 2), (1, 3)}
    assert set(sampler._get_valid_neighbors((8, 2))) == {
        (0, 1), (0, 2), (0, 3), (7, 1), (7, 2), (7, 3),
        (8, 1), (8, 3), (1, 1), (1, 2), (1, 3)}
    # Next to the bound: both copies of the point on the bound.
    assert {(0, 2), (8, 2)} <= set(sampler._get_valid_neighbors((1, 2)))
    assert {(0, 2), (8, 2)} <= set(sampler._get_valid_neighbors((7, 2)))
    # Away from the bound nothing changes, and theta is not periodic.
    assert set(sampler._get_valid_neighbors((4, 0))) == {
        (3, 0), (5, 0), (3, 1), (4, 1), (5, 1)}
    assert (4, 2) in set(sampler._get_valid_neighbors((4, 2), include_center=True))
    assert (4, 2) not in set(sampler._get_valid_neighbors((4, 2)))


def test_neighbours_without_periodic_parameters_stop_at_the_bound():
    sampler = make_sampler(None)
    assert set(sampler._get_valid_neighbors((0, 2))) == {
        (0, 1), (0, 3), (1, 1), (1, 2), (1, 3)}
    # A periodic parameter that is profiled does not change the grid either.
    sampler = make_sampler([2])
    assert set(sampler._get_valid_neighbors((0, 2))) == {
        (0, 1), (0, 3), (1, 1), (1, 2), (1, 3)}


def test_cells_further_out_wrap_too():
    sampler = make_sampler([0])
    within = sampler._cells_within((0, 2), 2)
    assert (6, 2) in within and (2, 2) in within and (0, 2) not in within
    assert len(within) == len(set(within))
    # Nearest first: the direct neighbours come before the cells two steps away.
    near = set(sampler._get_valid_neighbors((0, 2)))
    assert set(within[:len(near)]) == near


# --- wrapping and differences -------------------------------------------------

def test_wrap_and_short_way_difference():
    sampler = make_sampler([0], dims=(1, 2), grid=(4, 4))
    assert sampler.profiled_dims == [0, 3]

    wrapped = sampler._wrap_periodic(np.array([TWO_PI + 0.3, 1.7]), sampler.profiled_dims)
    np.testing.assert_allclose(wrapped, [0.3, 1.7])       # x3 is left alone
    wrapped = sampler._wrap_periodic(np.array([-0.3, -1.7]), sampler.profiled_dims)
    np.testing.assert_allclose(wrapped, [TWO_PI - 0.3, -1.7])

    # _ensure_bounds wraps the periodic parameter and clips the other.
    bounded = sampler._ensure_bounds(np.array([-0.3, 1.7]), sampler.profiled_dims)
    np.testing.assert_allclose(bounded, [TWO_PI - 0.3, 1.0])

    delta = sampler._param_delta(np.array([0.1, 0.5]), np.array([TWO_PI - 0.1, -0.5]),
                                 sampler.profiled_dims)
    np.testing.assert_allclose(delta, [0.2, 1.0])


def test_optima_on_either_side_of_the_bound_are_one_optimum():
    sampler = make_sampler([0])
    assert sampler.register_initial_optimum(np.array([0.01, 0.5, 0.0, 0.0]), -1.0)
    assert not sampler.register_initial_optimum(np.array([TWO_PI - 0.01, 0.5, 0.0, 0.0]), -1.0)

    sampler = make_sampler(None)
    assert sampler.register_initial_optimum(np.array([0.01, 0.5, 0.0, 0.0]), -1.0)
    assert sampler.register_initial_optimum(np.array([TWO_PI - 0.01, 0.5, 0.0, 0.0]), -1.0)


def test_refinement_interpolates_a_periodic_parameter_on_the_circle():
    solution = {
        'grid_axes': [np.array([0.0, 1.0])], 'projection_dims': [1], 'profiled_dims': [0, 2],
        'grid_shape': (2,),
        'solutions': {(0,): {'profiled_params': np.array([0.2, 0.0]), 'likelihood': 0.0},
                      (1,): {'profiled_params': np.array([TWO_PI - 0.4, 1.0]), 'likelihood': 0.0}},
    }
    plain = GridInterpolator(solution).interpolate(np.array([0.5]))
    np.testing.assert_allclose(plain, [np.pi - 0.1, 0.5])
    circular = GridInterpolator(solution, {0: (0.0, TWO_PI)}).interpolate(np.array([0.5]))
    np.testing.assert_allclose(circular, [TWO_PI - 0.1, 0.5])
    # At a grid point the value is the stored one.
    at_node = GridInterpolator(solution, {0: (0.0, TWO_PI)}).interpolate(np.array([1.0]))
    np.testing.assert_allclose(at_node, [TWO_PI - 0.4, 1.0])


# --- scans --------------------------------------------------------------------

def scan_plane(method, periodic, **kwargs):
    np.random.seed(0)
    return run_virtual_scan(
        target, BOUNDS,
        [{'dims': [0, 1], 'grid_points': [24, 12], 'optimization_method': method}],
        n_workers=16, roi_threshold=ROI, initial_points=[BEST], periodic=periodic, **kwargs)


@pytest.mark.parametrize("method", ['lbfgsb', 'de'])
def test_region_of_interest_is_followed_through_the_bound(method):
    truth, held = roi_cells(scan_plane(method, [0]))
    low = {c for c in truth if c[0] <= 6}
    assert len(low) > 10                    # the target does reach across the bound
    assert held == truth

    # Without the option the part below the bound is not reached from a
    # single start at the best-fit point.
    truth, held = roi_cells(scan_plane(method, None))
    assert not (held & low)


def test_both_grid_points_on_the_bound_hold_the_same_value():
    out = scan_plane('lbfgsb', [0])
    n = len(out['grid_axes'][0]) - 1
    pairs = [(out['population'][(0, j)], out['population'][(n, j)])
             for j in range(len(out['grid_axes'][1]))
             if (0, j) in out['population'] and (n, j) in out['population']]
    assert len(pairs) > 3
    for lower, upper in pairs:
        assert lower['best_fitness'] == pytest.approx(upper['best_fitness'], abs=1e-6)


def test_activation_ahead_wraps():
    truth, held = roi_cells(scan_plane(
        'lbfgsb', [0], advanced_config={'lbfgsb': {'activation_depth': 2}}))
    assert held == truth


@pytest.mark.parametrize("method", ['lbfgsb', 'de'])
def test_profiled_periodic_parameter_is_optimised_through_the_bound(method):
    def scan(periodic):
        np.random.seed(1)
        return run_virtual_scan(
            target_profiled, BOUNDS_P,
            [{'dims': [1, 2], 'grid_points': [16, 10], 'optimization_method': method,
              'patch_coarse_grid': False}],
            n_workers=16, roi_threshold=ROI, n_initial_optimizations=2, periodic=periodic)

    assert profile_deficits(scan([0])).max() < 1e-6
    # Without the option runs stop at the bound when the optimum lies beyond it.
    assert profile_deficits(scan(None)).max() > 1e-2


def test_grid_refinement_with_a_periodic_profiled_parameter():
    np.random.seed(2)
    out = run_virtual_scan(
        target_profiled, BOUNDS_P,
        [{'dims': [1, 2], 'grid_points': [10, 6], 'optimization_method': 'lbfgsb',
          'grid_refinement_factor': 2}],
        n_workers=16, roi_threshold=ROI, n_initial_optimizations=2, periodic=[0])
    assert out['grid_axes'][0].size == 21
    assert profile_deficits(out).max() < 1e-6


def test_volume_walkers_stay_inside_a_periodic_range(tmp_path):
    np.random.seed(3)
    out = run_virtual_scan(
        target, BOUNDS,
        [{'dims': [0, 1], 'grid_points': [16, 8], 'optimization_method': 'lbfgsb'}],
        n_workers=16, roi_threshold=ROI, initial_points=[BEST], periodic=[0],
        samples_output_file=str(tmp_path / 'samples.csv'),
        volume_sampling={'n_walkers': 60, 'n_steps': 10,
                         'output_file': str(tmp_path / 'volume.csv')})
    from paraprof import read_samples
    delta = read_samples(str(tmp_path / 'samples.csv'))[:, 0]
    assert delta.min() >= 0.0 and delta.max() <= TWO_PI + 1e-9
    volume = read_samples(str(tmp_path / 'volume.csv'))
    # Samples on both sides of the bound.
    assert np.any(volume[:, 0] < 1.0) and np.any(volume[:, 0] > TWO_PI - 1.0)
    assert out['target_calls'] > 0


# --- check of the bounds ------------------------------------------------------

class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append((record.levelno, record.getMessage()))


def run_periodic_check(bounds):
    sampler = ProfileProjector(
        target_func=target, bounds=bounds, periodic=['delta'],
        parameter_names=['delta', 'theta', 'n1', 'n2'],
        projections=[{'dims': [0, 1], 'grid_points': [8, 4]}])
    assert sampler.needs_periodic_check()
    (job,), _ = sampler.create_periodic_check_job(0)
    tasks = job.start()
    assert len(tasks) == 6                  # three base points, two bounds
    for task in tasks:
        assert task['params'][0] in (bounds[0][0], bounds[0][1])
        job.process_result({'target_val': target(task['params']), 'context': task['context']})
    assert job.is_finished()
    records = _Records()
    sampler.logger.addHandler(records)
    try:
        job.on_finish(1)
    finally:
        sampler.logger.removeHandler(records)
    assert not sampler.needs_periodic_check()
    return [m for level, m in records.messages if level >= logging.WARNING]


def test_periodic_check_passes_for_one_period():
    assert run_periodic_check(BOUNDS) == []


def test_periodic_check_warns_when_the_bounds_are_not_one_period():
    warnings = run_periodic_check([(-0.05, 6.3)] + BOUNDS[1:])
    assert len(warnings) == 1
    assert "delta is marked periodic" in warnings[0]
    assert "exactly one period" in warnings[0]


def test_periodic_check_runs_once_per_scan(capsys):
    np.random.seed(4)
    out = run_virtual_scan(
        target, [(-0.05, 6.3)] + BOUNDS[1:],
        [{'dims': [0, 1], 'grid_points': [12, 6], 'optimization_method': 'lbfgsb'},
         {'dims': [1, 2], 'grid_points': [6, 6], 'optimization_method': 'lbfgsb'}],
        n_workers=8, roi_threshold=ROI, initial_points=[BEST], periodic=[0])
    assert capsys.readouterr().err.count("is marked periodic") == 1
    assert out['target_calls'] > 0
