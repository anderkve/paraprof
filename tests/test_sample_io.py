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


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_reopen_appends_for_round_trip_warm_start(tmp_path, fmt, ext):
    """Re-opening the same path appends, mirroring output==warm_start round trips."""
    path = str(tmp_path / f"samples{ext}")
    b1 = _make_samples(10, 2, seed=3)
    with create_sample_writer(path) as w:
        w.write_batch(b1)
    b2 = _make_samples(15, 2, seed=4)
    with create_sample_writer(path) as w:
        w.write_batch(b2)

    back = read_samples(path)
    assert back.shape == (25, 3)
    np.testing.assert_allclose(back, np.vstack([b1, b2]), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_single_row_returns_2d(tmp_path, fmt, ext):
    path = str(tmp_path / f"samples{ext}")
    data = _make_samples(1, 4)
    with create_sample_writer(path) as w:
        w.write_batch(data)
    back = read_samples(path)
    assert back.ndim == 2
    assert back.shape == (1, 5)


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_no_samples_leaves_no_file(tmp_path, fmt, ext):
    path = tmp_path / f"samples{ext}"
    w = create_sample_writer(str(path))
    w.close()  # never wrote anything
    assert not path.exists()


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_empty_batch_is_noop(tmp_path, fmt, ext):
    path = tmp_path / f"samples{ext}"
    with create_sample_writer(str(path)) as w:
        w.write_batch(np.empty((0, 5)))
    assert not path.exists()


def test_read_missing_or_empty_csv_returns_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    out = read_samples(str(empty))
    assert out.shape == (0, 0)


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_write_after_close_raises(tmp_path, fmt, ext):
    w = create_sample_writer(str(tmp_path / f"x{ext}"))
    w.close()
    with pytest.raises(ValueError):
        w.write_batch(_make_samples(2, 2))


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


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_write_samples_refuses_existing_by_default(tmp_path, fmt, ext):
    path = str(tmp_path / f"out{ext}")
    original = _make_samples(40, 2, seed=1)
    write_samples(original, path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_samples(_make_samples(10, 2, seed=2), path)
    # The existing file must be left untouched.
    np.testing.assert_allclose(read_samples(path), original, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("fmt,ext", FORMATS)
def test_write_samples_overwrite_replaces(tmp_path, fmt, ext):
    path = str(tmp_path / f"out{ext}")
    write_samples(_make_samples(40, 2, seed=1), path)
    data2 = _make_samples(10, 2, seed=2)
    write_samples(data2, path, overwrite=True)  # replaces, does not append
    np.testing.assert_allclose(read_samples(path), data2, rtol=1e-9, atol=1e-12)


def test_write_samples_fmt_override(tmp_path):
    # .csv extension but force HDF5 output.
    path = str(tmp_path / "out.csv")
    data = _make_samples(8, 4)
    write_samples(data, path, fmt="hdf5")
    with h5py.File(path, "r") as f:
        assert "samples" in f
    np.testing.assert_allclose(read_samples(path, fmt="hdf5"), data, rtol=1e-9, atol=1e-12)


def test_write_samples_rejects_non_2d(tmp_path):
    with pytest.raises(ValueError, match="2D"):
        write_samples(np.arange(5.0), str(tmp_path / "out.csv"))


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


def test_hdf5_width_mismatch_raises(tmp_path):
    path = str(tmp_path / "samples.h5")
    with create_sample_writer(path) as w:
        w.write_batch(_make_samples(3, 4))
    with create_sample_writer(path) as w:
        with pytest.raises(ValueError, match="width"):
            w.write_batch(_make_samples(3, 6))


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


def test_combine_cross_format_csv_to_hdf5(tmp_path):
    csv1 = str(tmp_path / "a.csv")
    csv2 = str(tmp_path / "b.csv")
    d1 = _make_samples(20, 4, seed=10)
    d2 = _make_samples(35, 4, seed=11)
    with create_sample_writer(csv1) as w:
        w.write_batch(d1)
    with create_sample_writer(csv2) as w:
        w.write_batch(d2)

    out = str(tmp_path / "combined.h5")
    total = combine_samples([csv1, csv2], out)
    assert total == 55
    np.testing.assert_allclose(read_samples(out), np.vstack([d1, d2]), rtol=1e-9, atol=1e-12)


def test_combine_skips_missing_inputs(tmp_path):
    real = str(tmp_path / "real.csv")
    d = _make_samples(10, 2)
    with create_sample_writer(real) as w:
        w.write_batch(d)

    out = str(tmp_path / "out.csv")
    total = combine_samples([real, str(tmp_path / "ghost.csv")], out)
    assert total == 10
    np.testing.assert_allclose(read_samples(out), d, rtol=1e-9, atol=1e-12)


def test_combine_rejects_output_equal_to_input(tmp_path):
    p = str(tmp_path / "a.csv")
    with create_sample_writer(p) as w:
        w.write_batch(_make_samples(5, 2))
    with pytest.raises(ValueError, match="also one of the inputs"):
        combine_samples([p], p)


def test_combine_inconsistent_width_raises(tmp_path):
    a = str(tmp_path / "a.csv")
    b = str(tmp_path / "b.csv")
    with create_sample_writer(a) as w:
        w.write_batch(_make_samples(5, 2))
    with create_sample_writer(b) as w:
        w.write_batch(_make_samples(5, 4))
    with pytest.raises(ValueError, match="Inconsistent sample width"):
        combine_samples([a, b], str(tmp_path / "out.csv"))


def test_combine_no_inputs_writes_nothing(tmp_path):
    out = tmp_path / "out.csv"
    total = combine_samples([], str(out))
    assert total == 0
    assert not out.exists()
