"""Tests for the volume-sampling stage jobs and bookkeeping (Phase 3).

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

BAND_LO, BAND_HI = -4.0, np.inf
KAPPA = 1.0 / 16.0  # penalty_strength 1, roi_threshold 4


@pytest.fixture
def sampler(simple_2d_function, simple_bounds_2d, basic_projection_1d):
    return ProfileProjector(
        target_func=simple_2d_function,
        bounds=simple_bounds_2d,
        projections=[basic_projection_1d],
    )


def make_anchor_set(coverage_radius=0.1):
    """Four anchors in the corners of [0, 10]^2 (the job tests reuse the
    same geometry as the Phase-2 harvest tests)."""
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    anchors = np.array([
        [2.0, 2.0],
        [8.0, 2.0],
        [2.0, 8.0],
        [8.0, 8.0],
    ])
    return AnchorSet(anchors, bounds, coverage_radius=coverage_radius)


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
                             BAND_LO, BAND_HI)
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
        job = VolumeProbeJob(0, sampler, anchor_set, [], BAND_LO, BAND_HI)
        assert job.start() == []
        assert job.is_finished() and job.success


# --------------------------------------------------------------------------- #
# VolumeSearchJob
# --------------------------------------------------------------------------- #
def make_search_job(sampler, anchor_set, anchor_index, start, **kwargs):
    return VolumeSearchJob(1, sampler, anchor_set, anchor_index,
                           BAND_LO, BAND_HI, KAPPA,
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

    def test_nan_gradient_component_falls_back_to_fd(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.0, 5.0])
        tasks = job.start()

        raw_grad = np.array([0.5, np.nan])
        out = job.process_result(result_for(tasks[0], -10.0,
                                            user_gradient=raw_grad))
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_GRADIENT'
        assert out[0]['context']['dim'] == 1

    def test_hit_during_gradient_phase_stops_job(self, sampler):
        anchor_set = make_anchor_set()
        job = make_search_job(sampler, anchor_set, 0, [2.0, 5.0])
        tasks = job.start()
        fd_tasks = job.process_result(result_for(tasks[0], -10.0))
        assert len(fd_tasks) == 2

        # First FD eval happens to land in band within the radius.
        hit_result = result_for(fd_tasks[0], -1.0)
        hit_result['params'] = np.array([2.1, 2.0])
        out = job.process_result(hit_result)
        assert out == []
        assert job.is_finished() and job.success and job.hit

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
        state = VolumeStageState(make_anchor_set(), BAND_LO, BAND_HI,
                                 eval_budget=2)
        assert state.budget_left()
        state.note_eval(np.array([2.0, 2.0]), -10.0)
        assert state.budget_left()
        state.note_eval(np.array([2.0, 2.0]), -10.0)
        assert not state.budget_left()
        assert state.evals_used == 2

    def test_note_eval_offers_in_band_to_nearest_anchor(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
        state.note_eval(np.array([2.1, 2.0]), -1.0)
        assert anchor_set.covered[0]
        assert anchor_set.rep_source[0] == SOURCE_SEARCH
        # Out-of-band and offer=False evals leave the records alone.
        state.note_eval(np.array([8.0, 2.0]), -10.0)
        state.note_eval(np.array([8.05, 2.0]), -1.0, offer=False)
        assert not anchor_set.covered[1]
        assert state.max_logl_seen == -1.0

    def test_record_search_job(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
        job = SimpleNamespace(
            anchor_index=2, hit=False,
            best_inband_point=np.array([2.0, 5.5]), best_inband_logl=-1.0,
            best_inband_dist=0.25,
            best_viol_point=np.array([2.0, 6.0]), best_viol_logl=-5.0,
            best_viol_dist=0.2, best_viol=1.0,
        )
        state.record_search_job(job)
        assert state.searched[2] and not state.search_hit[2]
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
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
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
            state, roi_config(), roi_threshold=4.0,
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
            state, roi_config(), roi_threshold=4.0,
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
            state, roi_config(probe_all_anchors=False), roi_threshold=4.0,
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
            state, roi_config(), roi_threshold=4.0,
            global_max_start=0.0, global_max_final=2.0,
            search_enabled=True)
        assert result['anchor_status'][0] == 'uncovered'
        assert result['stats']['n_probe_hits'] == 0
        assert result['stats']['global_max_drift'] == pytest.approx(2.0)
        assert result['band_final'][0] == pytest.approx(-2.0)

    def test_shell_band_finalize(self):
        anchor_set, state = self.make_state()
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -10.0, 0.0,
                                   SOURCE_PROBE)
        cfg = normalize_volume_config(
            {'mode': 'shell', 'shell_threshold': 25.0}, roi_threshold=4.0)
        result = finalize_volume_stage(
            state, cfg, roi_threshold=4.0,
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        assert result['anchor_status'][0] == 'covered'
        assert result['band_final'] == (-25.0, -4.0)


# --------------------------------------------------------------------------- #
# Phase 4: output file and JSON summary
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
    state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)

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
        state, roi_config(), roi_threshold=4.0,
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
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
        result = finalize_volume_stage(
            state, roi_config(), roi_threshold=4.0,
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
        assert summary['mode'] == 'roi'
        assert summary['n_rows'] == 4
        assert summary['rows_by_tag'] == {'0': 1, '1': 1, '2': 1, '3': 1}
        assert summary['stats']['n_covered'] == 2
        assert summary['uniform_subset_valid'] is True
        # JSON portability: the roi band's +inf upper edge becomes null.
        assert summary['band_final'][1] is None
        assert summary['band_final'][0] == -4.0
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
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
        result = finalize_volume_stage(
            state, roi_config(), roi_threshold=4.0,
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
        state = VolumeStageState(anchor_set, BAND_LO, BAND_HI)
        result = finalize_volume_stage(
            state, roi_config(), roi_threshold=4.0,
            global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        cfg = roi_config(output_file=str(tmp_path / "vol.csv"))
        _, summary_path, _ = write_volume_output(result, cfg)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary['stats']['prefilter_acceptance'] is None
        assert summary['stats']['probe_acceptance'] is None
        assert summary['stats']['volume_estimate'] is None


class TestVolumeSearchJobShellBand:
    """Shell mode: the band has a finite upper edge with the opposite
    hinge sign."""

    def make_shell_job(self, sampler, anchor_set, start):
        return VolumeSearchJob(1, sampler, anchor_set, 0, -25.0, -4.0,
                               KAPPA, np.asarray(start, dtype=float))

    def test_above_band_violation_and_chain_rule(self, sampler):
        anchor_set = make_anchor_set()
        start = np.array([2.0, 5.0])  # scaled dist 0.3 to anchor 0
        job = self.make_shell_job(sampler, anchor_set, start)
        tasks = job.start()

        # logL -1 is ABOVE the shell band [-25, -4]: violation 3.
        raw_grad = np.array([0.5, -1.5])
        out = job.process_result(result_for(tasks[0], -1.0,
                                            user_gradient=raw_grad))
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        violation = 3.0
        expected_fitness = -(0.3 ** 2 + KAPPA * violation ** 2)
        assert job.current_fitness == pytest.approx(expected_fitness)
        # Above the band the hinge sign flips: ∇F = -∇dist² - 2κv∇logL.
        ddist2 = 2.0 * (start - np.array([2.0, 2.0])) / 10.0 ** 2
        fitness_grad = -ddist2 - 2.0 * KAPPA * violation * raw_grad
        np.testing.assert_allclose(job.current_gradient, -fitness_grad)
        # An above-band point is closest-approach material, not in-band.
        assert job.outcome() == 'hole'
        assert job.best_viol == pytest.approx(3.0)

    def test_inside_shell_band_hits(self, sampler):
        anchor_set = make_anchor_set()
        job = self.make_shell_job(sampler, anchor_set, [2.5, 2.0])
        tasks = job.start()
        out = job.process_result(result_for(tasks[0], -10.0))
        assert out == []
        assert job.is_finished() and job.hit
