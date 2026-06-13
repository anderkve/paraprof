"""Tests for the volume-sampling stage jobs and bookkeeping.

Job-level tests feed synthetic results to ``process_result`` (no MPI), as in
the rest of the suite.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from paraprof import ProfileProjector
from paraprof.exceptions import ConfigurationError
from paraprof.jobs.volume_jobs import VolumeProbeJob, VolumeSearchJob
from paraprof.sample_io import read_samples
from paraprof.volume import (
    SOURCE_HARVEST,
    SOURCE_PROBE,
    SOURCE_SEARCH,
    TAG_HARVEST,
    TAG_HOLE,
    TAG_PROBE,
    TAG_SEARCH,
    AnchorSet,
    VolumeStageState,
    assemble_volume_rows,
    default_summary_file,
    depth_law_exponent,
    finalize_volume_stage,
    normalize_volume_config,
    write_volume_output,
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
    anchors = np.array([[2.0, 2.0], [8.0, 2.0], [2.0, 8.0], [8.0, 8.0]])
    return AnchorSet(anchors, bounds, coverage_radius=coverage_radius)


def roi_config(**overrides):
    return normalize_volume_config(overrides, roi_threshold=4.0)


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


def make_search_job(sampler, anchor_set, anchor_index, start, **kwargs):
    return VolumeSearchJob(1, sampler, anchor_set, anchor_index,
                           BAND_LO, KAPPA,
                           np.asarray(start, dtype=float), **kwargs)


def drain_tangent(job, out, feed=-3.0):
    """Feed off-target results until the tangent phase exhausts its budget."""
    for _ in range(25):
        if job.is_finished() or not out:
            break
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        out = job.process_result(result_for(out[0], feed))
    return out


# --------------------------------------------------------------------------- #
# VolumeProbeJob
# --------------------------------------------------------------------------- #
def test_probe_round_trip(sampler):
    anchor_set = make_anchor_set()
    job = VolumeProbeJob(0, sampler, anchor_set, [0, 1, 3], BAND_LO)
    tasks = job.start()
    assert len(tasks) == 3
    np.testing.assert_allclose(tasks[0]['params'], [2.0, 2.0])

    # Anchor 0 in band, anchor 1 below, anchor 3 a failed eval (-inf).
    job.process_result(result_for(tasks[0], -1.0))
    job.process_result(result_for(tasks[1], -10.0))
    assert not job.is_finished()
    job.process_result(result_for(tasks[2], -np.inf))
    assert job.is_finished() and job.success
    job.on_finish(None)

    np.testing.assert_array_equal(anchor_set.probed, [True, True, False, True])
    np.testing.assert_array_equal(anchor_set.probe_logls[[0, 1, 3]],
                                  [-1.0, -10.0, -np.inf])
    # Only the in-band probe became its anchor's representative (distance 0).
    assert anchor_set.covered[0]
    assert anchor_set.rep_dists[0] == 0.0
    assert anchor_set.rep_source[0] == SOURCE_PROBE
    assert not anchor_set.covered[1]


# --------------------------------------------------------------------------- #
# VolumeSearchJob: objective transform and gradients
# --------------------------------------------------------------------------- #
class TestVolumeSearchJob:

    def test_immediate_hit(self, sampler):
        job = make_search_job(sampler, make_anchor_set(), 0, [2.5, 2.0])
        tasks = job.start()
        assert tasks[0]['context']['sub_type'] == 'LBFGS_INITIAL_F'
        # In band, scaled dist 0.05 <= radius 0.1: covering point found.
        assert job.process_result(result_for(tasks[0], -1.0)) == []
        assert job.is_finished() and job.success and job.hit
        assert job.outcome() == 'hit'

    def test_out_of_band_transform_and_fd(self, sampler):
        start = np.array([2.0, 5.0])  # scaled dist 0.3 to anchor 0
        job = make_search_job(sampler, make_anchor_set(), 0, start)
        # logL -10: violation 6 against band_lo -4. No user gradient: FD.
        out = job.process_result(result_for(job.start()[0], -10.0))
        assert len(out) == 2
        assert all(t['context']['sub_type'] == 'LBFGS_GRADIENT' for t in out)
        assert job.current_fitness == pytest.approx(-(0.3 ** 2 + KAPPA * 6.0 ** 2))
        assert job.best_viol == pytest.approx(6.0)
        np.testing.assert_allclose(job.best_viol_point, start)
        assert job.outcome() == 'hole'

    def test_in_band_beyond_radius_uses_analytic_gradient(self, sampler):
        start = np.array([2.0, 5.5])  # in band, scaled dist 0.35 > radius
        job = make_search_job(sampler, make_anchor_set(), 0, start)
        out = job.process_result(result_for(job.start()[0], -1.0))
        # Violation 0: fully analytic gradient, zero FD tasks, straight to LS.
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        expected = 2.0 * (start - np.array([2.0, 2.0])) / 10.0 ** 2
        np.testing.assert_allclose(job.current_gradient, expected)
        assert job.outcome() == 'projected'
        assert job.best_inband_dist == pytest.approx(0.35)

    def test_chain_rule_with_user_gradient(self, sampler):
        start = np.array([2.0, 5.0])
        job = make_search_job(sampler, make_anchor_set(), 0, start)
        raw_grad = np.array([0.5, -1.5])  # ∇logL
        out = job.process_result(
            result_for(job.start()[0], -10.0, user_gradient=raw_grad))
        # All dims chain-ruled: no FD, straight to the line search.
        assert len(out) == 1
        assert out[0]['context']['sub_type'] == 'LBFGS_LINE_SEARCH'
        ddist2 = 2.0 * (start - np.array([2.0, 2.0])) / 10.0 ** 2
        fitness_grad = -ddist2 + 2.0 * KAPPA * 6.0 * raw_grad
        np.testing.assert_allclose(job.current_gradient, -fitness_grad)

    def test_hole_outcome_after_line_search_failure(self, sampler):
        job = make_search_job(sampler, make_anchor_set(), 0, [2.0, 5.0])
        pending = job.process_result(result_for(job.start()[0], -10.0))
        for t in pending[:-1]:
            assert job.process_result(result_for(t, -10.0)) == []
        line_tasks = job.process_result(result_for(pending[-1], -10.0))
        # Reject every line-search step until alpha underflows and it gives up.
        for _ in range(100):
            if job.is_finished():
                break
            out = job.process_result(result_for(line_tasks[0], -50.0))
            if out:
                line_tasks = out
        assert job.is_finished() and not job.success
        assert job.outcome() == 'hole'
        assert job.best_viol == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# VolumeStageState and finalize_volume_stage
# --------------------------------------------------------------------------- #
class TestStageState:

    def test_budget_and_in_band_offer(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO, eval_budget=2)
        # In-band eval offered to nearest anchor; budget counts down.
        state.note_eval(np.array([2.1, 2.0]), -1.0)
        assert anchor_set.covered[0] and anchor_set.rep_source[0] == SOURCE_SEARCH
        assert state.budget_left()
        # Out-of-band / offer=False evals leave records alone but still count.
        state.note_eval(np.array([8.0, 2.0]), -10.0)
        assert not anchor_set.covered[1]
        assert not state.budget_left() and state.evals_used == 2

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
        assert anchor_set.rep_dists[2] == pytest.approx(0.25)  # projected
        assert state.closest_violations[2] == pytest.approx(1.0)
        np.testing.assert_allclose(state.closest_points[2], [2.0, 6.0])


class TestFinalize:

    def make_state(self):
        anchor_set = make_anchor_set()
        anchor_set.n_draws = 1000
        anchor_set.n_prefilter_accepted = 200
        return anchor_set, VolumeStageState(anchor_set, BAND_LO)

    def test_status_partition_and_volume_estimate(self):
        anchor_set, state = self.make_state()
        # Anchor 0 covered (harvest), 1 projected (search), 2 hole, 3 unbudgeted.
        anchor_set.offer_to_anchor(0, np.array([2.1, 2.0]), -1.0, 0.01,
                                   SOURCE_HARVEST)
        anchor_set.offer_to_anchor(1, np.array([8.0, 4.0]), -2.0, 0.2,
                                   SOURCE_SEARCH)
        anchor_set.probed[:] = True
        anchor_set.probe_logls[:] = [-1.0, -2.0, -10.0, -10.0]
        state.searched[1] = True
        state.searched[2] = True
        state.closest_violations[2] = 3.0
        state.unbudgeted[3] = True

        result = finalize_volume_stage(
            state, roi_config(), global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        np.testing.assert_array_equal(
            result['anchor_status'],
            ['covered', 'projected', 'hole', 'unbudgeted'])
        stats = result['stats']
        assert (stats['n_covered'], stats['n_projected'], stats['n_holes'],
                stats['n_unbudgeted']) == (1, 1, 1, 1)
        assert stats['n_covered_harvest'] == 1
        # Box volume 100, prefilter acceptance 0.2, probe acceptance 0.5.
        assert stats['probe_acceptance'] == pytest.approx(0.5)
        assert stats['volume_estimate'] == pytest.approx(10.0)
        assert stats['volume_estimate_err'] == pytest.approx(
            20.0 * np.sqrt(0.5 * 0.5 / 4))
        # Uniform subset = in-band probes only.
        np.testing.assert_array_equal(result['uniform_subset'],
                                      [True, True, False, False])

    def test_no_volume_estimate_without_probe_all(self):
        anchor_set, state = self.make_state()
        anchor_set.probed[:] = True
        anchor_set.probe_logls[:] = -1.0
        result = finalize_volume_stage(
            state, roi_config(probe_all_anchors=False),
            global_max_start=0.0, global_max_final=0.0, search_enabled=True)
        assert result['stats']['volume_estimate'] is None

    def test_drift_reclassifies_band_membership(self):
        anchor_set, state = self.make_state()
        # Rep at logL -3: inside the initial band (max 0) but outside the
        # final band once the max drifts to 2.
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -3.0, 0.0,
                                   SOURCE_PROBE)
        anchor_set.probed[0] = True
        anchor_set.probe_logls[0] = -3.0
        result = finalize_volume_stage(
            state, roi_config(), global_max_start=0.0, global_max_final=2.0,
            search_enabled=True)
        assert result['anchor_status'][0] == 'uncovered'
        assert result['stats']['n_probe_hits'] == 0
        assert result['band_lo_final'] == pytest.approx(-2.0)

    def test_widened_band_finalize(self):
        # A larger stage roi_threshold reaches deeper: a logL -10 rep the
        # default band (lo -4) rejects is now in-band.
        anchor_set, state = self.make_state()
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -10.0, 0.0,
                                   SOURCE_PROBE)
        result = finalize_volume_stage(
            state, roi_config(roi_threshold=25.0),
            global_max_start=0.0, global_max_final=0.0, search_enabled=True)
        assert result['anchor_status'][0] == 'covered'
        assert result['band_lo_final'] == pytest.approx(-25.0)

    def test_reports_depth_histogram(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        # Depths 0.5, 1.5, 3.5 below a final max of 0 -> bins 1, 3, 8.
        anchor_set.offer_to_anchor(0, np.array([2.0, 2.0]), -0.5, 0.0, SOURCE_PROBE)
        anchor_set.offer_to_anchor(1, np.array([8.0, 2.0]), -1.5, 0.0, SOURCE_PROBE)
        anchor_set.offer_to_anchor(2, np.array([2.0, 8.0]), -3.5, 0.2, SOURCE_SEARCH)
        result = finalize_volume_stage(
            state, roi_config(), global_max_start=0.0, global_max_final=0.0,
            search_enabled=True)
        hist = result['stats']['rep_depth_histogram']
        assert hist['bin_edges'][-1] == 4.0
        assert sum(hist['counts']) == 3
        assert [hist['counts'][i] for i in (1, 3, 8)] == [1, 1, 1]


# --------------------------------------------------------------------------- #
# Output file and JSON summary
# --------------------------------------------------------------------------- #
def make_finalized_result():
    """A finalized stage result with one anchor per status/tag combination."""
    anchor_set = make_anchor_set()
    state = VolumeStageState(anchor_set, BAND_LO)
    # Anchor 0 covered (harvest), 1 covered (probe), 2 projected (search),
    # 3 hole with a closest approach.
    anchor_set.offer_to_anchor(0, np.array([2.1, 2.0]), -1.0, 0.01, SOURCE_HARVEST)
    anchor_set.offer_to_anchor(1, np.array([8.0, 2.0]), -2.0, 0.0, SOURCE_PROBE)
    anchor_set.offer_to_anchor(2, np.array([2.0, 5.5]), -3.0, 0.25, SOURCE_SEARCH)
    anchor_set.probed[:] = True
    anchor_set.probe_logls[:] = [-10.0, -2.0, -10.0, -10.0]
    state.searched[2:] = True
    state.closest_points[3] = [7.0, 7.0]
    state.closest_logls[3] = -6.0
    state.closest_violations[3] = 2.0
    return finalize_volume_stage(
        state, roi_config(), global_max_start=0.0, global_max_final=0.0,
        search_enabled=True)


def test_assemble_volume_rows():
    result = make_finalized_result()
    np.testing.assert_array_equal(
        result['anchor_status'], ['covered', 'covered', 'projected', 'hole'])
    rows = assemble_volume_rows(result)
    assert rows.shape == (4, 4)  # 2 dims + logL + tag
    by_tag = {row[-1]: row for row in rows}
    np.testing.assert_allclose(by_tag[TAG_HARVEST], [2.1, 2.0, -1.0, 0.0])
    np.testing.assert_allclose(by_tag[TAG_PROBE], [8.0, 2.0, -2.0, 1.0])
    np.testing.assert_allclose(by_tag[TAG_SEARCH], [2.0, 5.5, -3.0, 2.0])
    np.testing.assert_allclose(by_tag[TAG_HOLE], [7.0, 7.0, -6.0, 3.0])


def test_write_volume_output_round_trip(tmp_path):
    result = make_finalized_result()
    cfg = roi_config(output_file=str(tmp_path / "vol.csv"))
    out_path, summary_path, rows_by_tag = write_volume_output(result, cfg)

    assert read_samples(out_path).shape == (4, 4)
    assert rows_by_tag == {0: 1, 1: 1, 2: 1, 3: 1}
    assert result['output_file'] == out_path
    assert summary_path == default_summary_file(cfg['output_file'])
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary['n_rows'] == 4
    assert summary['stats']['n_covered'] == 2
    assert summary['band_lo_final'] == -4.0
    assert summary['uniform_subset_valid'] is True
    assert summary['tag_legend']['3'].startswith('hole')


def test_write_volume_output_empty_writes_summary_only(tmp_path):
    state = VolumeStageState(make_anchor_set(), BAND_LO)
    result = finalize_volume_stage(
        state, roi_config(), global_max_start=0.0, global_max_final=0.0,
        search_enabled=True)
    cfg = roi_config(output_file=str(tmp_path / "vol.csv"))
    out_path, summary_path, _ = write_volume_output(result, cfg)
    assert out_path is None
    assert not (tmp_path / "vol.csv").exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary['n_rows'] == 0
    # Non-finite stats are JSON-nulled.
    assert summary['stats']['prefilter_acceptance'] is None


# --------------------------------------------------------------------------- #
# Interior walk and depth law
# --------------------------------------------------------------------------- #
def test_depth_law_exponent():
    assert depth_law_exponent('uniform_dlnl', 8) == 1.0
    assert depth_law_exponent('uniform_sigma', 8) == 2.0
    assert depth_law_exponent('volume', 8) == pytest.approx(0.25)
    assert depth_law_exponent('volume', 2) == 1.0


def walk_job(sampler, anchor_set, start=(2.5, 2.0), steps=8, **kwargs):
    return VolumeSearchJob(1, sampler, anchor_set, 0, BAND_LO, KAPPA,
                           np.array(start, dtype=float),
                           interior_steps=steps, **kwargs)


class TestInteriorWalk:

    def test_off_behaves_as_plain_hit(self, sampler):
        job = walk_job(sampler, make_anchor_set(), steps=0)
        assert job.process_result(result_for(job.start()[0], -1.0)) == []
        assert job.is_finished() and job.hit
        assert job.interior_point is None

    def test_hit_marches_inward_and_respects_cap(self, sampler, monkeypatch):
        # U -> 0: depth target at the band top, so the walk runs to the cap.
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 1e-12)
        job = walk_job(sampler, make_anchor_set(), steps=1)
        out = job.process_result(result_for(job.start()[0], -1.0))
        assert job.hit and not job.is_finished()
        assert out[0]['context']['sub_type'] == 'VOLUME_INTERIOR'
        # Step = 2*radius along +x (entry - anchor). Single step lands at
        # scaled dist 0.25 > cap 0.1: rejected even though deeper.
        out = job.process_result(result_for(out[0], -0.1))
        assert out == [] and job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])

    def test_bisects_toward_depth_target(self, sampler, monkeypatch):
        # d=2, band_depth=4, U=0.25 -> target logL -1.0; entry -2.0 is shallower.
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        job = walk_job(sampler, make_anchor_set(), steps=8)
        out = job.process_result(result_for(job.start()[0], -2.0))
        # First step overshoots (-0.5): bisect between entry and overshoot.
        out = job.process_result(result_for(out[0], -0.5))
        np.testing.assert_allclose(out[0]['params'], [2.625, 2.0])
        out = job.process_result(result_for(out[0], -1.05))  # lower bracket
        np.testing.assert_allclose(out[0]['params'], [2.6875, 2.0])
        # On target: leftover budget hands off to tangential randomization.
        out = job.process_result(result_for(out[0], -0.95))
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        drain_tangent(job, out)
        assert job.is_finished() and job.success
        np.testing.assert_allclose(job.interior_point, [2.6875, 2.0])

    def test_walk_aims_at_known_deep_point(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        # A known deep point "north" of the entry, inside the cap ball; the
        # naive inward ray would head "east".
        sampler.global_solution_pool.append(
            (0.0, 0, {'full_params': np.array([2.5, 2.8]), 'grid_idx': None}))
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        job = walk_job(sampler, anchor_set, steps=8)
        out = job.process_result(result_for(job.start()[0], -2.0))
        # Step radius/4 = 0.025 scaled = 0.25 unscaled, toward +y.
        np.testing.assert_allclose(out[0]['params'], [2.5, 2.25])

    def test_record_search_job_prefers_interior_point(self):
        anchor_set = make_anchor_set()
        state = VolumeStageState(anchor_set, BAND_LO)
        anchor_set.offer_to_anchor(0, np.array([2.5, 2.0]), -3.9, 0.05,
                                   SOURCE_SEARCH)
        # The job's deeper interior point wins the rep slot over the edge entry.
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


class TestTangentPhase:
    """Leftover walk budget moves the on-target representative along its
    iso-likelihood shell."""

    def make_at_target_job(self, sampler, anchor_set, monkeypatch, steps=8):
        """A hit job whose entry sits exactly at the depth target (-1.0), so
        the whole budget goes to tangent rounds; hops always propose north."""
        job = walk_job(sampler, anchor_set, steps=steps, depth_exponent=1.0)
        monkeypatch.setattr(np.random, 'random', lambda *a, **k: 0.25)
        monkeypatch.setattr(np.random, 'standard_normal',
                            lambda *a, **k: np.array([0.0, 1.0]))
        out = job.process_result(result_for(job.start()[0], -1.0))
        assert out[0]['context']['sub_type'] == 'VOLUME_TANGENT'
        return job, out

    def test_accepted_hop_moves_representative(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        # Hop 2*step = 0.05 scaled = 0.5 unscaled, due north.
        np.testing.assert_allclose(out[0]['params'], [2.5, 2.5])
        out = job.process_result(result_for(out[0], -1.1))  # on target
        assert not job.is_finished()
        np.testing.assert_allclose(job.interior_point, [2.5, 2.5])
        assert job.tangent_moves == 1
        # Subsequent proposals stay inside the cap ball.
        assert np.linalg.norm(
            (out[0]['params'] - np.array([2.0, 2.0])) / 10.0) <= 0.1 + 1e-9

    def test_drifted_hop_newton_corrected(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        # Off-target proposal: the Broyden update turns the unit +x normal
        # into [1, -30] and the correction is a Newton step along it.
        out = job.process_result(result_for(out[0], -2.5))
        np.testing.assert_allclose(job._tan_grad, [1.0, -30.0])
        g = np.array([1.0, -30.0])
        expected = np.array([2.5, 2.5]) + g * (1.5 / float(g @ g)) * 10.0
        np.testing.assert_allclose(out[0]['params'], expected)
        out = job.process_result(result_for(out[0], -0.9))  # accepted
        np.testing.assert_allclose(job.interior_point, expected)
        assert job.tangent_moves == 1

    def test_failed_round_reverts_and_shrinks(self, sampler, monkeypatch):
        anchor_set = make_anchor_set()
        job, out = self.make_at_target_job(sampler, anchor_set, monkeypatch)
        h0 = job._tan_h
        for _ in range(3):  # proposal + both corrections miss
            out = job.process_result(result_for(out[0], -3.5))
        np.testing.assert_allclose(job.interior_point, [2.5, 2.0])
        assert job._tan_h == pytest.approx(0.5 * h0)
        assert job.tangent_moves == 0


def test_interior_steps_config_validation():
    assert normalize_volume_config({}, 4.0)['interior_steps'] == 8
    assert normalize_volume_config({'depth_law': 'volume'}, 4.0)['depth_law'] \
        == 'volume'
    for bad in (-1, 1.5, True, 'x'):
        with pytest.raises(ConfigurationError, match="interior_steps"):
            normalize_volume_config({'interior_steps': bad}, 4.0)
