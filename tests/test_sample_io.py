"""Tests for the pluggable sample read/write layer (paraprof.sample_io)."""

import numpy as np
import pytest

from paraprof.sample_io import (
    create_sample_writer,
    read_samples,
    write_samples,
    iter_sample_batches,
    combine_samples,
    infer_format,
    CSVSampleWriter,
    HDF5SampleWriter,
)

h5py = pytest.importorskip("h5py", reason="HDF5 tests require h5py")


# Parametrise across every supported on-disk format using its extension.
FORMATS = [
    pytest.param("csv", ".csv", id="csv"),
    pytest.param("hdf5", ".h5", id="hdf5"),
]


def _make_samples(n_rows, n_dims, seed=0):
    """An (n_rows, n_dims + 1) matrix: n_dims params + 1 target column."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, n_dims + 1))


# --------------------------------------------------------------------------- #
# Format inference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path,expected",
    [
        ("a.csv", "csv"),
        ("a.CSV", "csv"),
        ("dir/b.h5", "hdf5"),
        ("b.hdf5", "hdf5"),
        ("b.he5", "hdf5"),
        ("no_extension", "csv"),
        ("weird.txt", "csv"),
    ],
)
def test_infer_format(path, expected):
    assert infer_format(path) == expected


def test_writer_factory_dispatch(tmp_path):
    assert isinstance(create_sample_writer(str(tmp_path / "x.csv")), CSVSampleWriter)
    assert isinstance(create_sample_writer(str(tmp_path / "x.h5")), HDF5SampleWriter)
    # Explicit override wins over the extension.
    assert isinstance(create_sample_writer(str(tmp_path / "x.csv"), fmt="hdf5"), HDF5SampleWriter)


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_round_trip_single_batch(tmp_path, fmt, ext):
    path = str(tmp_path / f"samples{ext}")
    data = _make_samples(50, 4)
    with create_sample_writer(path) as w:
        w.write_batch(data)

    back = read_samples(path)
    assert back.shape == data.shape
    # CSV keeps 10 significant digits; allow for that rounding.
    np.testing.assert_allclose(back, data, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_round_trip_multiple_batches_appends(tmp_path, fmt, ext):
    path = str(tmp_path / f"samples{ext}")
    b1 = _make_samples(30, 3, seed=1)
    b2 = _make_samples(70, 3, seed=2)
    with create_sample_writer(path) as w:
        w.write_batch(b1)
        w.write_batch(b2)

    back = read_samples(path)
    np.testing.assert_allclose(back, np.vstack([b1, b2]), rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Chunked iteration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_iter_chunks_respects_chunk_size(tmp_path, fmt, ext):
    path = str(tmp_path / f"samples{ext}")
    data = _make_samples(250, 2)
    with create_sample_writer(path) as w:
        w.write_batch(data)

    batches = list(iter_sample_batches(path, chunk_size=100))
    assert [len(b) for b in batches] == [100, 100, 50]
    np.testing.assert_allclose(np.vstack(batches), data, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# write_samples (one-shot inverse of read_samples)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_write_samples_round_trip(tmp_path, fmt, ext):
    path = str(tmp_path / f"out{ext}")
    data = _make_samples(120, 3)
    n = write_samples(data, path, chunk_size=50)
    assert n == 120
    np.testing.assert_allclose(read_samples(path), data, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# HDF5 specifics
# --------------------------------------------------------------------------- #
def test_hdf5_stores_minimal_metadata(tmp_path):
    path = str(tmp_path / "samples.h5")
    with create_sample_writer(path) as w:
        w.write_batch(_make_samples(5, 7))
    with h5py.File(path, "r") as f:
        dset = f["samples"]
        assert dset.attrs["n_dims"] == 7
        assert dset.shape == (5, 8)
        assert dset.maxshape == (None, 8)


# --------------------------------------------------------------------------- #
# Combination
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_combine_same_format(tmp_path, fmt, ext):
    paths = []
    parts = []
    for i in range(3):
        p = str(tmp_path / f"in_{i}{ext}")
        d = _make_samples(40, 3, seed=i)
        with create_sample_writer(p) as w:
            w.write_batch(d)
        paths.append(p)
        parts.append(d)

    out = str(tmp_path / f"combined{ext}")
    total = combine_samples(paths, out, chunk_size=17)
    assert total == 120

    back = read_samples(out)
    np.testing.assert_allclose(back, np.vstack(parts), rtol=1e-9, atol=1e-12)
