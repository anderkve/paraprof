"""Tests for the sample-log provenance phase column (paraprof.phases)."""
import numpy as np
import pytest

from paraprof import ProfileProjector, read_samples
from paraprof.phases import (
    PHASE_INITIAL,
    PHASE_SCAN,
    PHASE_UNKNOWN,
    PHASE_VOLUME,
    phase_for_job_type,
)


def test_phase_for_job_type():
    assert phase_for_job_type('INITIAL_POINT_EVAL') == PHASE_INITIAL
    assert phase_for_job_type('LBFGSB') == PHASE_SCAN
    assert phase_for_job_type('VOLUME') == PHASE_VOLUME
    # Unmapped / missing types fall back.
    assert phase_for_job_type('NOT_A_JOB') == PHASE_UNKNOWN
    assert phase_for_job_type(None) == PHASE_UNKNOWN


def test_phase_column_round_trips(tmp_path, simple_2d_function, simple_bounds_2d,
                                  basic_projection_1d):
    """The trailing phase column is written, defaults to UNKNOWN, and is
    ignored (params/logL read by position) when warm-starting."""
    path = tmp_path / "samples.csv"
    sampler = ProfileProjector(
        target_func=simple_2d_function, bounds=simple_bounds_2d,
        projections=[basic_projection_1d], samples_output_file=str(path))
    sampler._register_target_call(np.array([1.0, 2.0]), -3.0, phase=PHASE_SCAN)
    sampler._register_target_call(np.array([3.0, 4.0]), -0.5)  # default phase
    sampler._flush_samples_buffer()

    rows = read_samples(str(path))
    assert rows.shape == (2, 4)  # 2 params + logL + phase
    np.testing.assert_allclose(rows[0], [1.0, 2.0, -3.0, PHASE_SCAN])
    assert rows[1, -1] == PHASE_UNKNOWN

    reader = ProfileProjector(
        target_func=simple_2d_function, bounds=simple_bounds_2d,
        projections=[basic_projection_1d])
    reader._initialize_from_warm_start_file(str(path))
    assert reader.global_max_target_val >= -0.5
    assert any(np.allclose(m['point'], [3.0, 4.0])
               for m in reader.initial_maxima)
