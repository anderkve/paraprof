"""Tests for the sample-log provenance phase column (paraprof.phases)."""
import numpy as np
import pytest

from paraprof import ProfileProjector, read_samples
from paraprof.phases import (
    PHASE_INITIAL,
    PHASE_REFINE,
    PHASE_SCAN,
    PHASE_SUSPECT,
    PHASE_UNKNOWN,
    PHASE_VOLUME_PROBE,
    PHASE_VOLUME_SEARCH,
    phase_for_job_type,
)


class TestPhaseMapping:

    @pytest.mark.parametrize("job_type,phase", [
        ('INITIAL_POINT_EVAL', PHASE_INITIAL),
        ('INITIAL_OPTIMIZATION', PHASE_INITIAL),
        ('ACTIVATE_GRID_POINT', PHASE_SCAN),
        ('DE_GRID_POINT', PHASE_SCAN),
        ('LBFGSB', PHASE_SCAN),
        ('PATCHING_LBFGSB', PHASE_REFINE),
        ('REFINEMENT_LBFGSB', PHASE_REFINE),
        ('SUSPECT_RECHECK', PHASE_SUSPECT),
        ('VOLUME_PROBE', PHASE_VOLUME_PROBE),
        ('VOLUME_SEARCH', PHASE_VOLUME_SEARCH),
    ])
    def test_known_job_types(self, job_type, phase):
        assert phase_for_job_type(job_type) == phase

    def test_unknown_job_type_falls_back(self):
        assert phase_for_job_type('NOT_A_JOB') == PHASE_UNKNOWN
        assert phase_for_job_type(None) == PHASE_UNKNOWN


class TestSampleLogPhaseColumn:

    def make_sampler(self, path, simple_2d_function, simple_bounds_2d,
                     basic_projection_1d):
        return ProfileProjector(
            target_func=simple_2d_function,
            bounds=simple_bounds_2d,
            projections=[basic_projection_1d],
            samples_output_file=str(path),
        )

    def test_phase_written_as_trailing_column(self, tmp_path, simple_2d_function,
                                              simple_bounds_2d, basic_projection_1d):
        path = tmp_path / "samples.csv"
        sampler = self.make_sampler(path, simple_2d_function, simple_bounds_2d,
                                    basic_projection_1d)
        sampler._register_target_call(np.array([1.0, 2.0]), -3.0, phase=PHASE_SCAN)
        sampler._register_target_call(np.array([3.0, 4.0]), -1.0,
                                      phase=PHASE_VOLUME_PROBE)
        sampler._flush_samples_buffer()

        rows = read_samples(str(path))
        assert rows.shape == (2, 4)  # 2 params + logL + phase
        np.testing.assert_allclose(rows[0], [1.0, 2.0, -3.0, PHASE_SCAN])
        np.testing.assert_allclose(rows[1], [3.0, 4.0, -1.0, PHASE_VOLUME_PROBE])

    def test_default_phase_is_unknown(self, tmp_path, simple_2d_function,
                                      simple_bounds_2d, basic_projection_1d):
        path = tmp_path / "samples.csv"
        sampler = self.make_sampler(path, simple_2d_function, simple_bounds_2d,
                                    basic_projection_1d)
        sampler._register_target_call(np.array([1.0, 2.0]), -3.0)
        sampler._flush_samples_buffer()

        rows = read_samples(str(path))
        assert rows[0, -1] == PHASE_UNKNOWN

    def test_warm_start_ignores_phase_column(self, tmp_path, simple_2d_function,
                                             simple_bounds_2d, basic_projection_1d):
        # A round-tripped log (with a phase column) must warm-start correctly:
        # params and logL are read by position, the trailing phase is ignored.
        path = tmp_path / "samples.csv"
        sampler = self.make_sampler(path, simple_2d_function, simple_bounds_2d,
                                    basic_projection_1d)
        sampler._register_target_call(np.array([1.0, 2.0]), -0.5, phase=PHASE_SCAN)
        sampler._flush_samples_buffer()

        reader = self.make_sampler(tmp_path / "other.csv", simple_2d_function,
                                   simple_bounds_2d, basic_projection_1d)
        reader._initialize_from_warm_start_file(str(path))
        assert reader.global_max_target_val >= -0.5
        assert any(np.allclose(m['point'], [1.0, 2.0])
                   for m in reader.initial_maxima)
