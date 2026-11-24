"""Labels of the plots written by paraprof.visualization."""
import re

import numpy as np
import pytest

pytest.importorskip('matplotlib')

from matplotlib.figure import Figure

from paraprof.visualization import plot_profiles, plot_volume_samples
from virtual_pool import run_virtual_scan

NAMES = ['alpha', 'beta', 'gamma', 'delta']
NOTE = 'relative to bounds'


def gauss4(x):
    return -50.0 * float(np.sum((np.asarray(x) - 0.4) ** 2))


def scanned_sampler(dims, named):
    """A sampler holding a finished projection over ``dims``."""
    kept = []
    np.random.seed(3)
    run_virtual_scan(
        gauss4, [(0.0, 1.0)] * 4,
        [{'dims': dims, 'grid_points': [6] * len(dims), 'optimization_method': 'lbfgsb'}],
        n_workers=20, seed=3, roi_threshold=4.0, n_initial_optimizations=6,
        max_patching_waves=1, lbfgsb_max_iter=20, sampler_hook=kept.append,
        parameter_names=NAMES if named else None)
    return kept[0]


@pytest.fixture
def figures(monkeypatch):
    """The text of every figure saved during the test, instead of the files."""
    saved = []

    def record(fig, *args, **kwargs):
        saved.append({
            'labels': [label for ax in fig.axes
                       for label in (ax.get_xlabel(), ax.get_ylabel()) if label],
            'titles': [ax.get_title() for ax in fig.axes if ax.get_title()]
                      + ([fig.get_suptitle()] if fig.get_suptitle() else []),
        })

    monkeypatch.setattr(Figure, 'savefig', record)
    return saved


def parameter_labels(figures):
    return [label for fig in figures for label in fig['labels']
            if 'log L' not in label and 'Log Likelihood' not in label]


@pytest.mark.parametrize('dims', [[0], [0, 1], [0, 1, 2]])
def test_named_parameters_and_coordinate_note(dims, figures, tmp_path):
    sampler = scanned_sampler(dims, named=True)
    plot_profiles(sampler, 'plot', {'output_dir': str(tmp_path), 'coordinate_note': NOTE})

    # The profile plot, and the plots of the profiled parameters.
    assert len(figures) >= 2
    labels = parameter_labels(figures)
    assert labels
    for label in labels:
        assert any(name in label for name in NAMES), label
        assert label.endswith(NOTE + ')'), label
        assert 'aram' not in label, label
    for d in dims:
        assert any(label.startswith(NAMES[d] + ' (') for label in labels)

    # The best-fit coordinates in the title carry the names and the note.
    best_fit = [t for t in figures[0]['titles'] if 'Best fit' in t]
    assert len(best_fit) == 1
    for d in dims:
        assert f'{NAMES[d]} = ' in best_fit[0]
    assert f'({NOTE})' in best_fit[0]
    assert not re.search(r'param \d', best_fit[0])


def test_labels_without_names_or_note(figures, tmp_path):
    sampler = scanned_sampler([0, 1], named=False)
    plot_profiles(sampler, 'plot', {'output_dir': str(tmp_path)})

    assert figures[0]['labels'][:2] == ['Parameter 0', 'Parameter 1']
    title = figures[0]['titles'][0]
    assert title.startswith('Profile likelihood for parameters [0, 1]\n'
                            'Best fit: param 0 = ')
    assert title.count('\n') == 1
    assert 'Parameter 2 Value' in figures[1]['labels']
    assert not any('(' in label for label in parameter_labels(figures))


def test_long_best_fit_line_is_broken(figures, tmp_path):
    sampler = scanned_sampler([0, 1], named=True)
    sampler.parameter_names = ['SomeLongModelName::' + n for n in NAMES]
    plot_profiles(sampler, 'plot', {'output_dir': str(tmp_path), 'coordinate_note': NOTE})

    lines = figures[0]['titles'][0].split('\n')[1:]
    assert len(lines) > 1
    assert all(len(line) <= 75 for line in lines)
    assert ' '.join(lines).count('SomeLongModelName::') == 2


def test_volume_plot_labels(figures, tmp_path):
    volume = {'samples': np.array([[0.1, 0.2, 0.3, 0.4, -1.0], [0.5, 0.6, 0.7, 0.8, -2.0]]),
              'band_lo_final': -4.0, 'global_max': 0.0}
    settings = {'output_dir': str(tmp_path)}
    plot_volume_samples(volume, (0, 2), 'volume', settings, parameter_names=NAMES)
    plot_volume_samples(volume, (0, 2), 'volume', dict(settings, coordinate_note=NOTE),
                        parameter_names=NAMES)

    assert figures[0]['labels'][:2] == ['alpha', 'gamma']
    assert figures[1]['labels'][:2] == [f'alpha ({NOTE})', f'gamma ({NOTE})']
