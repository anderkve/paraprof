"""Tests for the volume-sampling stage jobs and bookkeeping.

Job-level tests feed synthetic results to ``process_result`` (no MPI), as in
the rest of the suite.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.jobs.volume_jobs import VolumeProbeJob, VolumeSearchJob
from paraprof.volume import (
    SOURCE_HARVEST,
    SOURCE_PROBE,
    SOURCE_SEARCH,
    AnchorSet,
    VolumeStageState,
    finalize_volume_stage,
    normalize_volume_config,
)

BAND_LO = -4.0
KAPPA = 1.0 / 16.0  # SEARCH_PENALTY_STRENGTH 1, roi_threshold 4


@pytest.fixture
def sampler(simple_2d_function, simple_bounds_2d, basic_projection_1d):
    return ProfileProjector(
        target_func=simple_2d_function,
        bounds=simple_bounds_2d,
        projections=[basic_projection_1d],
    )


def make_anchor_set(coverage_radius=0.1):
    """Four anchors in the corners of [0, 10]^2."""
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    anchors = np.array([
        [2.0, 2.0],
        [8.0, 2.0],
        [2.0, 8.0],
        [8.0, 8.0],
    ])
    return AnchorSet(anchors, bounds, coverage_radius=coverage_radius)


def drain_tangent(job, out, feed=-3.0):
    """Feed off-target results until the tangent phase exhausts its
    budget and the job finishes; returns the final (empty) task list."""
    for _ in range(25):
        if job.is_finished() or not out:
            break
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        out = job.process_result(result_for(out[0], feed))
    return out


def result_for(task, target_val, user_gradient=None):
    """Worker-result dict for a task, as the master would receive it."""
    return {
        'target_val': target_val,
        'params': np.asarray(task['params'], dtype=float),
        'context': dict(task['context']),
        'error': None,
        'user_gradient': user_gradient,
        'user_gradient_error': None,
    }


# --------------------------------------------------------------------------- #
# VolumeProbeJob
# --------------------------------------------------------------------------- #
class TestVolumeProbeJob:

    def test_probe_round_trip(self, sampler):
        anchor_set = make_anchor_set()
        job = VolumeProbeJob(0, sampler, anchor_set, [0, 1, 3],
                             BAND_LO)
        tasks = job.start()
        assert len(tasks) == 3
        np.testing.assert_allclose(tasks[0]['params'], [2.0, 2.0])
        np.testing.assert_allclose(tasks[2]['params'], [8.0, 8.0])

        # Anchor 0 in band, anchor 1 below, anchor 3 a failed eval (-inf).
        job.process_result(result_for(tasks[0], -1.0))
        job.process_result(result_for(tasks[1], -10.0))
        assert not job.is_finished()
        job.process_result(result_for(tasks[2], -np.inf))
        assert job.is_finished() and job.success
        job.on_finish(None)

        np.testing.assert_array_equal(anchor_set.probed,
                                      [True, True, False, True])
        assert anchor_set.probe_logls[0] == -1.0
        assert anchor_set.probe_logls[1] == -10.0
        assert anchor_set.probe_logls[3] == -np.inf
        # Only the in-band probe became its anchor's representative.
        assert anchor_set.covered[0]
        assert anchor_set.rep_dists[0] == 0.0
        assert anchor_set.rep_source[0] == SOURCE_PROBE
        assert not anchor_set.covered[1]
        assert not np.isfinite(anchor_set.rep_dists[1])

    def test_empty_probe_finishes_immediately(self, sampler):
        anchor_set = make_anchor_set()
        job = VolumeProbeJob(0, sampler, anchor_set, [], BAND_LO)
        assert job.start() == []
        assert job.is_finished() and job.success


# --------------------------------------------------------------------------- #
# VolumeSearchJob
# --------------------------------------------------------------------------- #
def make_search_job(sampler, anchor_set, anchor_index, start, **kwargs):
    return VolumeSearchJob(1, sampler, anchor_set, anchor_index,
                           BAND_LO, KAPPA,
                           np.asarray(start, dtype=float), **kwargs)


class TestVolumeSearchJob:

    def test_immediate_hit(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.5, 2.0])
        tasks = job.start()
        assert len(tasks) == 1
        assert tasks[0]['context']['sub_type'] == 'LBFGS_INITIAL_F'

        # In band, scaled dist 0.05 <= radius 0.1: covering point found.
        out = job.process_result(result_for(tasks[0], -1.0))
        assert out == []
        assert job.is_finished() and job.success and job.hit
        assert job.outcome() == 'hit'

    def test_out_of_band_transform_and_fd(self, sampler):
        anchor_set = make_anchor_set()
        start = np.array([2.0, 5.0])  # scaled dist 0.3 to anchor 0
        job = make_search_job(sampler, anchor_set, 0, start)
        tasks = job.start()

        # logL -10: violation 6 against band_lo -4.
        out = job.process_result(result_for(tasks[0], -10.0))
        # No user gradient and out of band: forward FD on both dims.
        assert len(out) == 2
        assert all(t['context']['sub_type'] == 'LBFGS_GRADIENT' for t in out)
        expected_fitness = -(0.3 ** 2 + KAPPA * 6.0 ** 2)
        assert job.current_fitness == pytest.approx(expected_fitness)
        # Closest-approach tracking.
        assert job.best_viol == pytest.approx(6.0)
        np.testing.assert_allclose(job.best_viol_point, start)
        assert job.outcome() == 'hole'  # nothing in band seen so far

    def test_in_band_beyond_radius_uses_analytic_gradient(self, sampler):
        anchor_set = make_anchor_set()
        start = np.array([2.0, 5.5])  # in band, scaled dist 0.35 > radius
        job = make_search_job(sampler, anchor_set, 0, start)
        tasks = job.start()

        out = job.process_result(result_for(tasks[0], -1.0))
        # Violation 0: fully analytic gradient, zero FD tasks, straight to
        # the line search.
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        # Objective-frame gradient = +∇dist² = 2(θ-a)/extent².
        expected = 2.0 * (start - np.array([2.0, 2.0])) / 10.0 ** 2
        np.testing.assert_allclose(job.current_gradient, expected)
        assert job.outcome() == 'projected'
        assert job.best_inband_dist == pytest.approx(0.35)

    def test_chain_rule_with_user_gradient(self, sampler):
        anchor_set = make_anchor_set()
        start = np.array([2.0, 5.0])
        job = make_search_job(sampler, anchor_set, 0, start)
        tasks = job.start()

        raw_grad = np.array([0.5, -1.5])  # ∇logL
        out = job.process_result(result_for(tasks[0], -10.0,
                                            user_gradient=raw_grad))
        # All dims covered by the chain-ruled gradient: no FD, straight to
        # the line search.
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        violation = 6.0
        ddist2 = 2.0 * (start - np.array([2.0, 2.0])) / 10.0 ** 2
        # Fitness frame: -∇dist² + 2κv∇logL (below the band, sign = -1);
        # the job stores the objective frame (negated).
        fitness_grad = -ddist2 + 2.0 * KAPPA * violation * raw_grad
        np.testing.assert_allclose(job.current_gradient, -fitness_grad)

    def test_failed_eval_gets_finite_penalty(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.0, 5.0])
        tasks = job.start()
        job.process_result(result_for(tasks[0], -np.inf))
        assert np.isfinite(job.current_fitness)

    def test_hole_outcome_after_line_search_failure(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.0, 5.0])
        tasks = job.start()
        pending = job.process_result(result_for(tasks[0], -10.0))
        # Feed the FD results (still out of band).
        for t in pending[:-1]:
            assert job.process_result(result_for(t, -10.0)) == []
        line_tasks = job.process_result(result_for(pending[-1], -10.0))
        assert len(line_tasks) == 1

        # Reject every line-search step (worse violation) until alpha
        # underflows and the job gives up.
        for _ in range(100):
            if job.is_finished():
                break
            out = job.process_result(result_for(line_tasks[0], -50.0))
            if out:
                line_tasks = out
        assert job.is_finished() and not job.success
        assert job.outcome() == 'hole'
        # Closest approach is still the best (least-violating) point seen.
        assert job.best_viol == pytest.approx(6.0)
        assert job.best_viol_logl == -10.0

    def test_max_iter_override(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.0, 5.0], max_iter=7)
        assert job.lbfgsb_max_iter == 7


# --------------------------------------------------------------------------- #
# VolumeStageState
# --------------------------------------------------------------------------- #
class TestVolumeStageState:

    def test_budget_and_eval_counting(self):
        state = VolumeStageState(make_anchor_set(), BAND_LO,
                                 eval_budget=2)
        assert state.budget_left()
        state.note_eval(np.array([2.0, 2.0]), -10.0)
        assert state.budget_left()
        state.note_eval(np.array([2.0, 2.0]), -10.0)
        assert not state.budget_left()
        assert state.evals_used == 2

    def test_note_eval_offers_in_band_to_nearest_anchor(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        state.note_eval(np.array([2.1, 2.0]), -1.0)
        assert anchor_set.covered[0]
        assert anchor_set.rep_source[0] == SOURCE_SEARCH
        # Out-of-band and offer=False evals leave the records alone.
        state.note_eval(np.array([8.0, 2.0]), -10.0)
        state.note_eval(np.array([8.05, 2.0]), -1.0, offer=False)
        assert not anchor_set.covered[1]

    def test_record_search_job(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        job = SimpleNamespace(
            anchor_index=2, hit=False,
            best_inband_point=np.array([2.0, 5.5]), best_inband_logl=-1.0,
            best_inband_dist=0.25,
            best_viol_point=np.array([2.0, 6.0]), best_viol_logl=-5.0,
            best_viol_dist=0.2, best_viol=1.0,
        )
        state.record_search_job(job)
        assert state.searched[2]
        # The in-band point became anchor 2's representative (projected).
        assert anchor_set.rep_dists[2] == pytest.approx(0.25)
        assert anchor_set.rep_source[2] == SOURCE_SEARCH
        assert state.closest_violations[2] == pytest.approx(1.0)
        np.testing.assert_allclose(state.closest_points[2], [2.0, 6.0])


# --------------------------------------------------------------------------- #
# finalize_volume_stage
# --------------------------------------------------------------------------- #
def roi_config(**overrides):
    return normalize_volume_config(overrides, roi_threshold=4.0)


class TestFinalize:

    def make_state(self):
        anchor_set = make_anchor_set()
        anchor_set.n_draws = 1000
        anchor_set.n_prefilter_accepted = 200
        state = VolumeStageState(anchor_set, BAND_LO)
        return anchor_set, state

    def test_status_partition_and_stats(self):
        anchor_set, state = self.make_state()

        # Anchor 0: covered by harvest. Anchor 1: projected (in-band rep
        # beyond the radius, found by search). Anchor 2: searched, hole.
        # Anchor 3: never probed (budget).
        anchor_set.offer_to_anchor(0, np.array([2.1, 2.0]), -1.0, 0.01,
                                   SOURCE_HARVEST)
        anchor_set.offer_to_anchor(1, np.array([8.0, 4.0]), -2.0, 0.2,
                                   SOURCE_SEARCH)
        anchor_set.probed[:3] = True
        anchor_set.probe_logls[:3] = [-1.0, -10.0, -10.0]
        state.searched[1] = True
        state.searched[2] = True
        state.closest_violations[2] = 3.0
        state.unbudgeted[3] = True

        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)

        np.testing.assert_array_equal(
            result['anchor_status'],
            ['covered', 'projected', 'hole', 'unbudgeted'])
        stats = result['stats']
        assert stats['n_covered'] == 1
        assert stats['n_covered_harvest'] == 1
        assert stats['n_projected'] == 1
        assert stats['n_holes'] == 1
        assert stats['n_unbudgeted'] == 1
        assert stats['n_uncovered'] == 0
        assert stats['n_probed'] == 3
        assert stats['n_probe_hits'] == 1
        assert stats['probe_acceptance'] == pytest.approx(1.0 / 3.0)
        # Uniform subset = in-band probes only.
        np.testing.assert_array_equal(result['uniform_subset'],
                                      [True, False, False, False])

    def test_volume_estimate(self):
        anchor_set, state = self.make_state()
        anchor_set.probed[:] = True
        anchor_set.probe_logls[:] = [-1.0, -2.0, -10.0, -10.0]

        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        stats = result['stats']
        # Box volume 100, prefilter acceptance 0.2, probe acceptance 0.5.
        assert stats['volume_estimate'] == pytest.approx(10.0)
        expected_err = 20.0 * np.sqrt(0.5 * 0.5 / 4)
        assert stats['volume_estimate_err'] == pytest.approx(expected_err)

    def test_no_volume_estimate_without_probe_all(self):
        anchor_set, state = self.make_state()
        anchor_set.probed[:] = True
        anchor_set.probe_logls[:] = -1.0
        result = finalize_volume_stage(
            state, roi_config(probe_all_anchors=False),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        assert result['stats']['volume_estimate'] is None

    def test_drift_reclassifies_band_membership(self):
        anchor_set, state = self.make_state()
        # Representative at logL -3: inside the initial band (max 0,
        # threshold 4) but outside the final band once the max drifts to 2.
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -3.0, 0.0,
                                   SOURCE_PROBE)
        anchor_set.probed[0] = True
        anchor_set.probe_logls[0] = -3.0

        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=2.0,
            search_enabled=True)
        assert result['anchor_status'][0] == 'uncovered'
        assert result['stats']['n_probe_hits'] == 0
        assert result['stats']['global_max_drift'] == pytest.approx(2.0)
        assert result['band_lo_final'] == pytest.approx(-2.0)

    def test_widened_band_finalize(self):
        # A larger stage roi_threshold reaches deeper: a logL -10 point that
        # the default band (lo -4) would reject is now in-band.
        anchor_set, state = self.make_state()
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -10.0, 0.0,
                                   SOURCE_PROBE)
        cfg = normalize_volume_config(
            {'roi_threshold': 25.0}, roi_threshold=4.0)
        result = finalize_volume_stage(
            state, cfg,
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        assert result['anchor_status'][0] == 'covered'
        assert result['band_lo_final'] == pytest.approx(-25.0)


# --------------------------------------------------------------------------- #
# Output file and JSON summary
# --------------------------------------------------------------------------- #
import json

from paraprof.sample_io import read_samples
from paraprof.volume import (
    TAG_HARVEST,
    TAG_HOLE,
    TAG_PROBE,
    TAG_SEARCH,
    assemble_volume_rows,
    default_summary_file,
    write_volume_output,
)


def make_finalized_result():
    """A finalized stage result with one anchor per status/tag combination."""
    anchor_set = make_anchor_set()
    state = VolumeStageState(anchor_set, BAND_LO)

    # Anchor 0: covered by harvest. Anchor 1: covered by its probe.
    # Anchor 2: projected (search). Anchor 3: hole with closest approach.
    anchor_set.offer_to_anchor(0, np.array([2.1, 2.0]), -1.0, 0.01,
                               SOURCE_HARVEST)
    anchor_set.offer_to_anchor(1, np.array([8.0, 2.0]), -2.0, 0.0,
                               SOURCE_PROBE)
    anchor_set.offer_to_anchor(2, np.array([2.0, 5.5]), -3.0, 0.25,
                               SOURCE_SEARCH)
    anchor_set.probed[:] = True
    anchor_set.probe_logls[:] = [-10.0, -2.0, -10.0, -10.0]
    state.searched[2:] = True
    state.closest_points[3] = [7.0, 7.0]
    state.closest_logls[3] = -6.0
    state.closest_dists[3] = 0.14
    state.closest_violations[3] = 2.0

    return finalize_volume_stage(
        state, roi_config(),
        global_max_start=0.0, global_max_final=0.0,
        search_enabled=True)


class TestAssembleVolumeRows:

    def test_rows_and_tags(self):
        result = make_finalized_result()
        np.testing.assert_array_equal(
            result['anchor_status'],
            ['covered', 'covered', 'projected', 'hole'])

        rows = assemble_volume_rows(result)
        assert rows.shape == (4, 4)  # 2 dims + logL + tag

        by_tag = {row[-1]: row for row in rows}
        np.testing.assert_allclose(by_tag[TAG_HARVEST], [2.1, 2.0, -1.0, 0.0])
        np.testing.assert_allclose(by_tag[TAG_PROBE], [8.0, 2.0, -2.0, 1.0])
        np.testing.assert_allclose(by_tag[TAG_SEARCH], [2.0, 5.5, -3.0, 2.0])
        np.testing.assert_allclose(by_tag[TAG_HOLE], [7.0, 7.0, -6.0, 3.0])

    def test_hole_without_closest_approach_is_skipped(self):
        result = make_finalized_result()
        # Erase anchor 3's closest-approach record.
        result['closest_violations'][3] = np.inf
        rows = assemble_volume_rows(result)
        assert rows.shape == (3, 4)
        assert TAG_HOLE not in rows[:, -1]

    def test_empty_result(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        rows = assemble_volume_rows(result)
        assert rows.shape == (0, 4)


class TestWriteVolumeOutput:

    @pytest.mark.parametrize("ext", [".csv", ".h5"])
    def test_round_trip(self, tmp_path, ext):
        if ext == ".h5":
            pytest.importorskip("h5py")
        result = make_finalized_result()
        cfg = roi_config(output_file=str(tmp_path / f"vol{ext}"))

        out_path, summary_path, rows_by_tag = write_volume_output(result, cfg)

        samples = read_samples(out_path)
        assert samples.shape == (4, 4)
        assert rows_by_tag == {0: 1, 1: 1, 2: 1, 3: 1}
        # Annotations on the result dict.
        assert result['output_file'] == out_path
        assert result['summary_file'] == summary_path
        assert result['rows_by_tag'] == rows_by_tag

        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['n_rows'] == 4
        assert summary['rows_by_tag'] == {'0': 1, '1': 1, '2': 1, '3': 1}
        assert summary['stats']['n_covered'] == 2
        assert summary['uniform_subset_valid'] is True
        assert summary['band_lo_final'] == -4.0
        assert summary['tag_legend']['3'].startswith('hole')

    def test_default_summary_path(self, tmp_path):
        result = make_finalized_result()
        out = str(tmp_path / "sub" / "vol.csv")
        cfg = roi_config(output_file=out)
        _, summary_path, _ = write_volume_output(result, cfg)
        assert summary_path == str(tmp_path / "sub" / "vol_summary.json")
        assert default_summary_file("a/b.h5") == "a/b_summary.json"

    def test_explicit_summary_path(self, tmp_path):
        result = make_finalized_result()
        cfg = roi_config(output_file=str(tmp_path / "vol.csv"),
                         summary_file=str(tmp_path / "s.json"))
        _, summary_path, _ = write_volume_output(result, cfg)
        assert summary_path == str(tmp_path / "s.json")
        assert (tmp_path / "s.json").exists()

    def test_overwrites_with_warning(self, tmp_path):
        result = make_finalized_result()
        out = tmp_path / "vol.csv"
        out.write_text("old content\n")
        cfg = roi_config(output_file=str(out))
        write_volume_output(result, cfg)
        assert read_samples(str(out)).shape == (4, 4)

    def test_empty_result_writes_summary_only(self, tmp_path):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        cfg = roi_config(output_file=str(tmp_path / "vol.csv"))

        out_path, summary_path, rows_by_tag = write_volume_output(result, cfg)
        assert out_path is None
        assert not (tmp_path / "vol.csv").exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['output_file'] is None
        assert summary['n_rows'] == 0

    def test_nan_stats_become_null(self, tmp_path):
        anchor_set = make_anchor_set()  # no draws: prefilter acceptance NaN
        state = VolumeStageState(anchor_set, BAND_LO)
        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        cfg = roi_config(output_file=str(tmp_path / "vol.csv"))
        _, summary_path, _ = write_volume_output(result, cfg)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['stats']['prefilter_acceptance'] is None
        assert summary['stats']['probe_acceptance'] is None
        assert summary['stats']['volume_estimate'] is None


# --------------------------------------------------------------------------- #
# Interior steps
# --------------------------------------------------------------------------- #
from paraprof.exceptions import ConfigurationError


class TestInteriorSteps:

    def make_walk_job(self, sampler, anchor_set, steps=8):
        return VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                               KAPPA, np.array([2.5, 2.0]),
                               interior_steps=steps)

    def test_config_validation(self):
        cfg = normalize_volume_config({'interior_steps': 3}, roi_threshold=4.0)
        assert cfg['interior_steps'] == 3
        assert normalize_volume_config({}, 4.0)['interior_steps'] == 8
        for bad in (-1, 1.5, True, 'x'):
            with pytest.raises(ConfigurationError, match="interior_steps"):
                normalize_volume_config({'interior_steps': bad}, 4.0)

    def test_hit_starts_walk_along_inward_direction(self, sampler,
                                                    monkeypatch):
        anchor_set = make_anchor_set()
        job = self.make_walk_job(sampler, anchor_set, steps=8)
        # U -> 0 gives a depth target at the very top of the band, so the
        # walk runs until the step cap or a non-improving step.
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 1e-12)
        tasks = job.start()

        out = job.process_result(result_for(tasks[0], -1.0))
        # Hit registered, but the job keeps running an interior walk.
        assert job.hit and not job.is_finished()
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'VOLUME_INTERIOR'
        # Direction: inward continuation of (entry - anchor); entry
        # (2.5, 2.0) - anchor (2.0, 2.0) -> +x. Step = radius/steps = 0.025
        # scaled = 0.25 unscaled.
        np.testing.assert_allclose(out[0]['params'], [2.75, 2.0])

        # Accept two improving in-band steps; positions beyond [3.0, 2.0]
        # leave the cap, so the walk works through its bounded shrink/
        # re-aim ladder and finishes with the deepest in-band point.
        out = job.process_result(result_for(out[0], -0.8))
        np.testing.assert_allclose(out[0]['params'], [3.0, 2.0])
        out = job.process_result(result_for(out[0], -0.5))
        assert len(out) == 1
        for _ in range(12):
            if job.is_finished():
                break
            out = job.process_result(result_for(out[0], -0.9))
        assert job.is_finished() and job.success
        np.testing.assert_allclose(job.interior_point, [3.0, 2.0])
        assert job.interior_logl == -0.5
        assert job.outcome() == 'hit'

    def test_walk_respects_distance_cap(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job = self.make_walk_job(sampler, anchor_set, steps=1)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 1e-12)
        tasks = job.start()
        out = job.process_result(result_for(tasks[0], -1.0))
        # Single step of 2*radius: lands at scaled dist 0.25 > cap 0.1,
        # so the step is rejected even though it is deeper.
        out2 = job.process_result(result_for(out[0], -0.1))
        assert out2 == [] and job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])

    def test_walk_bisects_toward_depth_target(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job = self.make_walk_job(sampler, anchor_set, steps=8)
        # d=2, band_depth=4: U=0.25 -> target logL = 0 - 4*0.25 = -1.0;
        # entry at -2.0 is shallower, so the walk runs.
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        tasks = job.start()
        out = job.process_result(result_for(tasks[0], -2.0))
        assert len(out) == 1

        # First step overshoots the target by 0.5 > tol (0.05*4): a
        # bisection task between the entry and the overshot point follows.
        out = job.process_result(result_for(out[0], -0.5))
        assert len(out) == 1
        np.testing.assert_allclose(out[0]['params'], [2.625, 2.0])
        # Midpoint still below the target: becomes the bracket's lower end.
        out = job.process_result(result_for(out[0], -1.05))
        assert len(out) == 1
        np.testing.assert_allclose(out[0]['params'], [2.6875, 2.0])
        # Within tolerance of the target: the depth search is done and the
        # leftover budget goes to tangential randomization; off-target
        # tangent feeds leave the refined representative in place.
        out = job.process_result(result_for(out[0], -0.95))
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        drain_tangent(job, out)
        assert job.is_finished() and job.success
        assert job.interior_logl == -0.95
        np.testing.assert_allclose(job.interior_point, [2.6875, 2.0])

    def test_shallow_depth_target_finishes_at_entry(self, sampler,
                                                     monkeypatch):
        anchor_set = make_anchor_set()
        job = self.make_walk_job(sampler, anchor_set, steps=8)
        # U -> 1 gives a depth target at the band edge, which the entry
        # point already satisfies: no walk.
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 1.0)
        tasks = job.start()
        out = job.process_result(result_for(tasks[0], -1.0))
        assert out == [] and job.is_finished() and job.success and job.hit
        # The entry point still becomes the interior representative.
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])

    def test_interior_steps_off_behaves_as_before(self, sampler):
        anchor_set = make_anchor_set()
        job = self.make_walk_job(sampler, anchor_set, steps=0)
        tasks = job.start()
        out = job.process_result(result_for(tasks[0], -1.0))
        assert out == [] and job.is_finished() and job.hit
        assert job.interior_point is None

    def test_projected_termination_detours_through_walk(self, sampler,
                                                        monkeypatch):
        anchor_set = make_anchor_set()
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.0, 5.5]), max_iter=1,
                              interior_steps=2)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 1e-12)
        tasks = job.start()
        # In band beyond the radius: analytic gradient -> line search.
        out = job.process_result(result_for(tasks[0], -1.0))
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        # Accept a line-search step that stays in band but beyond radius;
        # max_iter=1 then terminates the base machinery -> walk detour.
        ls_result = result_for(out[0], -0.9)
        ls_result['params'] = np.array([2.0, 5.0])
        out = job.process_result(ls_result)
        assert not job.is_finished()
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'VOLUME_INTERIOR'
        # Reject the step: the walk re-aims once (empty pool: same
        # direction); a second rejection finishes the job with the walk
        # origin (the best in-band point) as the interior point.
        out = job.process_result(result_for(out[0], -10.0))
        assert len(out) == 1 and not job.is_finished()      # re-aimed
        out = job.process_result(result_for(out[0], -10.0))
        assert out == [] and job.is_finished()
        assert job.outcome() == 'projected'
        assert job.interior_point is not None

    def test_record_search_job_prefers_interior_point(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        # The closer band-edge entry point is already registered...
        anchor_set.offer_to_anchor(0, np.array([2.5, 2.0]), -3.9, 0.05,
                                   SOURCE_SEARCH)
        # ...but the job's deeper interior point wins the rep slot.
        job = SimpleNamespace(
            anchor_index=0, hit=True,
            best_inband_point=np.array([2.5, 2.0]), best_inband_logl=-3.9,
            best_inband_dist=0.05,
            best_viol_point=None, best_viol_logl=-np.inf,
            best_viol_dist=np.inf, best_viol=np.inf,
            interior_point=np.array([2.9, 2.0]), interior_logl=-1.2,
            interior_dist=0.09,
        )
        state.record_search_job(job)
        np.testing.assert_allclose(anchor_set.rep_points[0], [2.9, 2.0])
        assert anchor_set.rep_logls[0] == -1.2
        assert anchor_set.covered[0]  # 0.09 <= radius 0.1


class TestDepthLaw:

    def test_config_validation(self):
        cfg = normalize_volume_config({}, roi_threshold=4.0)
        assert cfg['depth_law'] == 'uniform_dlnl'
        for law in ('volume', 'uniform_dlnl', 'uniform_sigma'):
            cfg = normalize_volume_config({'depth_law': law}, 4.0)
            assert cfg['depth_law'] == law
        with pytest.raises(ConfigurationError, match="depth_law"):
            normalize_volume_config({'depth_law': 'posterior'}, 4.0)

    def test_exponent_mapping(self):
        from paraprof.volume import depth_law_exponent
        assert depth_law_exponent('uniform_dlnl', 8) == 1.0
        assert depth_law_exponent('uniform_sigma', 8) == 2.0
        assert depth_law_exponent('volume', 8) == pytest.approx(0.25)
        assert depth_law_exponent('volume', 2) == 1.0

    def test_job_uses_depth_exponent(self, sampler, monkeypatch):
        """Same U, different laws -> different targets -> different
        walk/no-walk decisions for the same entry point."""
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        # band_depth 4, entry logL -1.0:
        # gamma=1 (uniform_dlnl): target = -4*0.25 = -1.0 -> entry already
        # reaches it, no walk.
        anchor_set = make_anchor_set()
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]), interior_steps=4,
                              depth_exponent=1.0)
        out = job.process_result(result_for(job.start()[0], -1.0))
        # The at-target entry skips the depth search; the untouched budget
        # funds tangential randomization instead.
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        drain_tangent(job, out)
        assert job.is_finished()
        assert job.interior_logl == -1.0

        # gamma=2 (uniform_sigma): target = -4*0.0625 = -0.25 -> deeper
        # than the entry, walk runs.
        anchor_set = make_anchor_set()
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]), interior_steps=4,
                              depth_exponent=2.0)
        out = job.process_result(result_for(job.start()[0], -1.0))
        assert len(out) == 1 and not job.is_finished()
        assert job._walk_target_logl == pytest.approx(-0.25)

    def test_finalize_reports_depth_histogram(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        # Depths 0.5, 1.5, 3.5 below a final max of 0.
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -0.5, 0.0,
                                   SOURCE_PROBE)
        anchor_set.offer_to_anchor(1, np.array([8.0, 2.0]), -1.5, 0.0,
                                   SOURCE_PROBE)
        anchor_set.offer_to_anchor(2, np.array([2.0, 8.0]), -3.5, 0.2,
                                   SOURCE_SEARCH)
        result = finalize_volume_stage(
            state, roi_config(),
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        hist = result['stats']['rep_depth_histogram']
        assert hist['bin_edges'][0] == 0.0
        assert hist['bin_edges'][-1] == 4.0
        assert sum(hist['counts']) == 3
        # 10 bins of width 0.4: depths 0.5, 1.5, 3.5 -> bins 1, 3, 8.
        assert hist['counts'][1] == 1
        assert hist['counts'][3] == 1
        assert hist['counts'][8] == 1

class TestWalkAiming:
    """The walk aims at the nearest known point at least as deep as the
    target (scan pool / initial maxima), falling back to the inward
    (entry - anchor) ray when none qualifies."""

    def inject_pool_point(self, sampler, point, logl):
        sampler.global_solution_pool.append(
            (logl, 0, {'full_params': np.asarray(point, dtype=float),
                       'fitness': logl, 'grid_idx': None}))

    def test_walk_aims_at_pool_point(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        # A known deep point straight "north" of the entry, inside the
        # anchor's cap ball; the naive inward ray would go "east"
        # (entry - anchor = +x).
        self.inject_pool_point(sampler, [2.5, 2.8], 0.0)
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]), interior_steps=8)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        out = job.process_result(result_for(job.start()[0], -2.0))
        # Step radius/4 = 0.025 scaled = 0.25 unscaled, toward +y.
        np.testing.assert_allclose(out[0]['params'], [2.5, 2.25])

    def test_out_of_band_passthrough_within_cap(self, sampler, monkeypatch):
        """Out-of-band points within the cap advance the march (thin
        out-of-band slivers across straight chords are crossed, never
        adopted as representatives)."""
        anchor_set = make_anchor_set()
        self.inject_pool_point(sampler, [2.5, 2.8], 0.0)
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]), interior_steps=6)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        out = job.process_result(result_for(job.start()[0], -2.0))
        # Step lands out of band but within the cap: march continues.
        out = job.process_result(result_for(out[0], -10.0))
        assert len(out) == 1 and not job.is_finished()
        np.testing.assert_allclose(out[0]['params'], [2.5, 2.0 + 4 * 6/36])
        # The out-of-band point never became the representative.
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])
        # Crossing the target within tolerance hands the leftover budget
        # to the tangent phase.
        out = job.process_result(result_for(out[0], -1.0))
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        drain_tangent(job, out)
        assert job.is_finished()
        assert job.interior_logl == -1.0

    def test_out_of_cap_steps_shrink_then_reaim(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        # Entry close to the cap rim, empty pool: the fallback +x ray
        # leaves the cap immediately, so the ladder fires: two halvings,
        # one re-aim (same direction, step reset), two more halvings.
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.99, 2.0]), interior_steps=6)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        out = job.process_result(result_for(job.start()[0], -2.0))
        step = 2.0 * 0.1 / 6 * 10.0   # unscaled step length
        np.testing.assert_allclose(out[0]['params'], [2.99 + step, 2.0])
        seq = [2.99 + 0.5 * step, 2.99 + 0.25 * step,   # two halvings
               2.99 + step,                              # re-aim, reset
               2.99 + 0.5 * step, 2.99 + 0.25 * step]    # two halvings
        for expected_x in seq:
            out = job.process_result(result_for(out[0], -1.5))
            assert len(out) == 1
            np.testing.assert_allclose(out[0]['params'], [expected_x, 2.0])
        # Budget exhausted: finish at the entry point.
        out = job.process_result(result_for(out[0], -1.5))
        assert out == [] and job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.99, 2.0])

    def test_out_of_reach_aim_projected_onto_cap_sphere(self, sampler,
                                                        monkeypatch):
        """An aim candidate beyond the walk's distance cap is replaced by
        its projection onto the cap sphere around the anchor, so the walk
        heads for the deepest reachable location instead of exiting the
        cap immediately."""
        anchor_set = make_anchor_set()
        # Candidate straight north of anchor 0 at scaled distance 0.2,
        # outside the hit cap (= coverage radius 0.1): projected aim is
        # anchor + 0.1 * (0, 1) = (2.0, 3.0).
        self.inject_pool_point(sampler, [2.0, 4.0], 0.0)
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]), interior_steps=8)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        out = job.process_result(result_for(job.start()[0], -2.0))
        # Direction from entry (2.5, 2.0) toward (2.0, 3.0), scaled:
        # (-0.05, 0.1)/|.| ; one step of 0.025.
        d = np.array([-0.05, 0.1]) / np.linalg.norm([-0.05, 0.1])
        expected = np.array([2.5, 2.0]) + 0.025 * 10.0 * d
        np.testing.assert_allclose(out[0]['params'], expected)


class TestTangentPhase:
    """Tangential randomization: leftover walk budget moves the on-target
    representative along its iso-likelihood shell."""

    def make_at_target_job(self, sampler, anchor_set, monkeypatch, steps=8):
        """A hit job whose entry sits exactly at the drawn depth target
        (-1.0), so the whole budget goes to tangent rounds."""
        job = VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO,
                              KAPPA, np.array([2.5, 2.0]),
                              interior_steps=steps, depth_exponent=1.0)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        # Tangent hops always propose "north" (perpendicular to the
        # entry-anchor secant normal, which is +x here).
        monkeypatch.setattr(np.random, 'standard_normal',
                            lambda *a, **k: np.array([0.0, 1.0]))
        out = job.process_result(result_for(job.start()[0], -1.0))
        assert out and out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        return job, out

    def test_accepted_hop_moves_representative(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        # Hop scale: max(travel=0, 2*step) = 2 * (2*0.1/8) = 0.05 scaled
        # = 0.5 unscaled, perpendicular to +x: due north.
        np.testing.assert_allclose(out[0]['params'], [2.5, 2.5])
        # On-target result (within tol 0.2 of -1.0): the rep moves.
        out = job.process_result(result_for(out[0], -1.1))
        assert not job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.5, 2.5])
        assert job.interior_logl == -1.1
        assert job.tangent_moves == 1
        # Next round proposes from the new position, shrinking the hop
        # as needed to stay inside the cap ball.
        assert out[0]['params'][1] > 2.5
        assert np.linalg.norm(
            (out[0]['params'] - np.array([2.0, 2.0])) / 10.0) <= 0.1 + 1e-9

    def test_drifted_hop_newton_corrected(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        # Off-target proposal at [2.5, 2.5] (drift -1.5 over a hop of 0.05
        # scaled): the Broyden update turns the initial unit +x gradient
        # estimate into [1, -30], and the correction is a Newton step
        # along it — preserving the tangential displacement instead of
        # bisecting back toward [2.5, 2.0].
        out = job.process_result(result_for(out[0], -2.5))
        np.testing.assert_allclose(job._tan_grad, [1.0, -30.0])
        g = np.array([1.0, -30.0])
        step = g * (1.5 / float(g @ g))      # Newton: target - logl = 1.5
        expected = np.array([2.5, 2.5]) + step * 10.0
        np.testing.assert_allclose(out[0]['params'], expected)
        assert abs(out[0]['params'][1] - 2.25) > 0.2  # not the midpoint
        # The corrected point is on-target: accepted as the new rep.
        out = job.process_result(result_for(out[0], -0.9))
        np.testing.assert_allclose(job.interior_point, expected)
        assert job.tangent_moves == 1

    def test_failed_round_reverts_and_shrinks(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        h0 = job._tan_h
        # Proposal and both corrections miss: round reverts, hop halves.
        for _ in range(3):
            out = job.process_result(result_for(out[0], -3.5))
        assert not job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])
        assert job.interior_logl == -1.0
        assert job._tan_h == pytest.approx(0.5 * h0)
        assert job.tangent_moves == 0

    def test_budget_exhaustion_finishes_with_success(self, sampler,
                                                     monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch,
                                           steps=2)
        out = job.process_result(result_for(out[0], -1.1))   # eval 1
        out = job.process_result(result_for(out[0], -1.05))  # eval 2
        assert out == [] and job.is_finished() and job.success and job.hit
        assert job.interior_logl == -1.05

