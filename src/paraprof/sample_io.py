"""Pluggable read/write layer for ParaProf sample files.

A sample is one evaluated point: ``n_dims`` parameter values plus one target
(log-likelihood) value, stored as a row of length ``n_dims + 1``. The format
is chosen by file extension, so callers never handle it directly:

- ``.csv``                       -> plain text, ``%.10e`` columns
- ``.h5`` / ``.hdf5`` / ``.he5``  -> HDF5 binary (needs ``h5py``)

Other extensions default to CSV. Writers take one batch at a time and flush
per batch, so a crash loses at most the caller's still-buffered samples, and
they append when the file exists, so a run can extend and re-read its own file
(the warm-start round trip).
"""

import os

import numpy as np

__all__ = [
    "create_sample_writer",
    "read_samples",
    "iter_sample_batches",
    "combine_samples",
    "infer_format",
    "SampleWriter",
    "CSVSampleWriter",
    "HDF5SampleWriter",
]

# Default number of rows read/written per chunk during streaming operations.
DEFAULT_CHUNK_SIZE = 10000

# Output precision for the text format (kept identical to the legacy writer).
_CSV_FMT = "%.10e"
_CSV_DELIMITER = ", "

# HDF5 dataset name holding the (n_samples, n_dims + 1) sample matrix.
_HDF5_DATASET = "samples"

_HDF5_EXTENSIONS = (".h5", ".hdf5", ".he5")


def infer_format(path):
    """Return the format key (``'csv'`` or ``'hdf5'``) for ``path``, by extension.

    Unknown or missing extensions fall back to ``'csv'``.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _HDF5_EXTENSIONS:
        return "hdf5"
    return "csv"


def _import_h5py():
    """Import h5py with an actionable error message when it is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(
            "HDF5 sample files require the 'h5py' package, which is not "
            "installed. Install it with 'pip install h5py' or "
            "'pip install paraprof[hdf5]'."
        ) from exc
    return h5py


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
class SampleWriter:
    """Streaming writer base: write batches, then close (idempotent).

    The file is opened lazily on the first non-empty batch, so a run that
    records nothing leaves no file behind.
    """

    def write_batch(self, rows):
        """Append ``rows``, a 2D array of shape (n, n_dims + 1)."""
        raise NotImplementedError

    def flush(self):
        """Flush buffered data to disk."""

    def close(self):
        """Flush and release the file; idempotent."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CSVSampleWriter(SampleWriter):
    """Append samples as ``%.10e`` text, re-opening the file per batch."""

    def __init__(self, path):
        self.path = path
        self._closed = False

    def write_batch(self, rows):
        if self._closed:
            raise ValueError("write_batch called on a closed CSVSampleWriter")
        data = np.asarray(rows, dtype=float)
        if data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        with open(self.path, "a", buffering=1) as f:
            np.savetxt(f, data, fmt=_CSV_FMT, delimiter=_CSV_DELIMITER)
            f.flush()

    def close(self):
        self._closed = True


class HDF5SampleWriter(SampleWriter):
    """Append samples to a resizable, chunked ``samples`` dataset (float64)."""

    def __init__(self, path):
        self.path = path
        self._h5py = _import_h5py()
        self._file = None
        self._dset = None
        self._closed = False

    def _ensure_open(self, n_cols):
        if self._file is not None:
            return
        # Append mode: extend an existing dataset, else create a resizable one.
        self._file = self._h5py.File(self.path, "a")
        if _HDF5_DATASET in self._file:
            self._dset = self._file[_HDF5_DATASET]
            existing_cols = self._dset.shape[1]
            if existing_cols != n_cols:
                self.close()
                raise ValueError(
                    f"Existing HDF5 dataset in '{self.path}' has width "
                    f"{existing_cols}, cannot append rows of width {n_cols}."
                )
        else:
            self._dset = self._file.create_dataset(
                _HDF5_DATASET,
                shape=(0, n_cols),
                maxshape=(None, n_cols),
                chunks=(DEFAULT_CHUNK_SIZE, n_cols),
                dtype="float64",
            )
            self._dset.attrs["n_dims"] = n_cols - 1

    def write_batch(self, rows):
        if self._closed:
            raise ValueError("write_batch called on a closed HDF5SampleWriter")
        data = np.asarray(rows, dtype=np.float64)
        if data.size == 0:
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self._ensure_open(data.shape[1])
        start = self._dset.shape[0]
        self._dset.resize(start + data.shape[0], axis=0)
        self._dset[start:] = data
        self._file.flush()

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dset = None


_WRITERS = {
    "csv": CSVSampleWriter,
    "hdf5": HDF5SampleWriter,
}


def create_sample_writer(path, fmt=None):
    """Return a writer for ``path``; ``fmt`` overrides extension inference."""
    fmt = fmt or infer_format(path)
    try:
        writer_cls = _WRITERS[fmt]
    except KeyError:
        raise ValueError(f"Unknown sample format '{fmt}'. Supported: {sorted(_WRITERS)}.") from None
    return writer_cls(path)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _iter_csv_batches(path, chunk_size):
    """Yield (<=chunk_size, n_cols) float arrays from a CSV sample file."""
    with open(path) as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(np.fromstring(line, sep=","))
            if len(rows) >= chunk_size:
                yield np.vstack(rows)
                rows = []
        if rows:
            yield np.vstack(rows)


def _iter_hdf5_batches(path, chunk_size):
    """Yield (<=chunk_size, n_cols) float arrays from an HDF5 sample file."""
    h5py = _import_h5py()
    with h5py.File(path, "r") as f:
        if _HDF5_DATASET not in f:
            return
        dset = f[_HDF5_DATASET]
        n = dset.shape[0]
        for start in range(0, n, chunk_size):
            yield dset[start : start + chunk_size]


def iter_sample_batches(path, chunk_size=DEFAULT_CHUNK_SIZE, fmt=None):
    """Yield 2D arrays of up to ``chunk_size`` rows from a sample file.

    The streaming primitive behind warm-start loading and combination; lets
    large files be processed without loading them whole.
    """
    fmt = fmt or infer_format(path)
    if fmt == "hdf5":
        yield from _iter_hdf5_batches(path, chunk_size)
    else:
        yield from _iter_csv_batches(path, chunk_size)


def read_samples(path, fmt=None):
    """Read an entire sample file into a 2D ``(n_samples, n_dims + 1)`` array.

    Always returns a 2D array, even for single-row or empty files.
    """
    batches = [b for b in iter_sample_batches(path, fmt=fmt) if b.size]
    if not batches:
        return np.empty((0, 0), dtype=float)
    result = np.vstack(batches)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    return result


# --------------------------------------------------------------------------- #
# Combination
# --------------------------------------------------------------------------- #
def combine_samples(inputs, output, chunk_size=DEFAULT_CHUNK_SIZE, fmt=None):
    """Concatenate sample files into ``output``, streaming chunk by chunk.

    Scales to large files (nothing is held whole in memory) and may mix
    formats, since each path is dispatched on its own extension -- e.g.
    several CSVs into one HDF5. Missing inputs are skipped.

    Returns the number of samples written. Raises ValueError if ``output`` is
    also an input or the inputs disagree on column count.
    """
    inputs = list(inputs)
    out_abs = os.path.abspath(output)
    if any(os.path.abspath(p) == out_abs for p in inputs):
        raise ValueError(
            f"Output path '{output}' is also one of the inputs; refusing to "
            "read and write the same file."
        )

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = None
    n_cols = None
    total = 0
    try:
        for path in inputs:
            if not os.path.exists(path):
                continue
            for batch in iter_sample_batches(path, chunk_size):
                if batch.size == 0:
                    continue
                if writer is None:
                    n_cols = batch.shape[1]
                    writer = create_sample_writer(output, fmt=fmt)
                elif batch.shape[1] != n_cols:
                    raise ValueError(
                        f"Inconsistent sample width while combining: '{path}' "
                        f"has {batch.shape[1]} columns, expected {n_cols}."
                    )
                writer.write_batch(batch)
                total += batch.shape[0]
    finally:
        if writer is not None:
            writer.close()

    return total
