"""Stratified volume sampling (see docs/volume_sampling_plan.md).

After the projections complete, the volume-sampling stage collects a
stratified, well-spread set of samples inside the stage's region of
interest — ``{logL > global_max - roi_threshold}`` — via a three-tier
funnel: harvest of existing samples, direct probes at scrambled-Sobol
anchor points, and an anchored search for anchors whose probe failed.

The stage's ``roi_threshold`` defaults to the projection's but can be set
larger to reach into the shell around the projection ROI (the band is
always one-sided: every point down to ``global_max - roi_threshold``).

This module holds the master-side building blocks (no MPI):

- :func:`normalize_volume_config` — validate the ``volume_sampling`` config.
- :func:`volume_band` — the stage's logL band and prefilter threshold.
- :class:`ProjectionEnvelope` — the projection-grid prefilter. Converged
  profile values upper-bound logL over the cylinder of points projecting
  into a cell, so a point whose projection falls in a below-threshold (or
  never-activated) cell of *any* computed projection cannot be in the band.
- :class:`AnchorSet` / :func:`generate_anchors` — scrambled-Sobol anchors
  filtered through the envelope (tier-2 probes at these anchors are
  uniform over the prefilter region, which is what makes the in-band
  probes a valid uniform-on-ROI subset and the acceptance fractions an
  unbiased volume estimate).
- :func:`resolve_harvest_files` / :func:`harvest_existing_samples` —
  tier 1: cover anchors from already-evaluated samples at zero cost.

All distances are Euclidean in bounds-scaled coordinates (each dimension
mapped to [0, 1]).
"""
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .exceptions import ConfigurationError
from .logger import get_logger

VOLUME_CONFIG_DEFAULTS = {
    'roi_threshold': None,      # ΔlnL band depth; None = projection's roi_threshold.
                                # Larger reaches into the shell outside the ROI.
    'n_anchors': 1000,          # target number of stratified anchor points
    'min_spacing': None,        # optional Poisson-disk radius (bounds-scaled units)
    'eval_budget': None,        # hard cap on stage evaluations (None = unlimited)
    'search': 'lbfgsb',         # tier-3 method: 'lbfgsb' or 'none' (probe-only)
    'probe_all_anchors': True,  # probe even harvest-covered anchors (uniform subset)
    'search_max_iter': None,    # per-anchor L-BFGS-B cap (None = lbfgsb_max_iter)
    'interior_steps': 8,        # post-entry steps into the band (0 = off)
    'depth_law': 'uniform_dlnl',  # interior-walk depth target: 'volume',
                                  # 'uniform_dlnl' or 'uniform_sigma'
    'harvest_files': None,      # extra sample files for tier 1
    'output_file': 'volume_samples.csv',
    'summary_file': None,       # None = derived from output_file
}

# Sobol draws come in power-of-2 batches to keep the sequence's balance
# properties (scipy warns otherwise).
ANCHOR_DRAW_BATCH = 4096
# Default cap on envelope-tested draws: this many per requested anchor.
DEFAULT_DRAW_CAP_FACTOR = 10_000

# Funnel tier that found an anchor's representative (SOURCE_*) and the
# corresponding tag value in the volume output file (TAG_*).
# TAG_HOLE rows are NOT in-band; they are closest-approach diagnostics.
SOURCE_NONE = 0
SOURCE_HARVEST = 1
SOURCE_PROBE = 2
SOURCE_SEARCH = 3

TAG_HARVEST = 0.0
TAG_PROBE = 1.0    # the uniform subset (in-band anchor probes)
TAG_SEARCH = 2.0
TAG_HOLE = 3.0

_SOURCE_TO_TAG = {
    SOURCE_HARVEST: TAG_HARVEST,
    SOURCE_PROBE: TAG_PROBE,
    SOURCE_SEARCH: TAG_SEARCH,
}

TAG_LEGEND = {
    int(TAG_HARVEST): 'harvested from existing samples (in-band)',
    int(TAG_PROBE): 'direct anchor probe (in-band; the uniform subset)',
    int(TAG_SEARCH): 'anchored search result (in-band)',
    int(TAG_HOLE): 'hole closest approach (NOT in-band; diagnostic)',
}


def _check_positive_int(cfg, key, allow_none=False):
    val = cfg[key]
    if val is None and allow_none:
        return
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < 1:
        raise ConfigurationError(
            f"volume_sampling['{key}'] must be a positive integer"
            + (" or None" if allow_none else ""),
            parameter=f"volume_sampling.{key}", value=val,
        )
    cfg[key] = int(val)


def normalize_volume_config(config, roi_threshold):
    """Validate a ``volume_sampling`` config dict and fill in defaults.

    Returns a new dict with every key from :data:`VOLUME_CONFIG_DEFAULTS`
    present (``harvest_files`` normalized to a list or None). Raises
    :class:`ConfigurationError` on unknown keys or invalid values.
    """
    if not isinstance(config, dict):
        raise ConfigurationError(
            "volume_sampling must be a dict",
            parameter="volume_sampling", value=config,
        )
    unknown = set(config) - set(VOLUME_CONFIG_DEFAULTS)
    if unknown:
        raise ConfigurationError(
            f"volume_sampling has unknown keys {sorted(unknown)}; "
            f"allowed keys are {sorted(VOLUME_CONFIG_DEFAULTS)}",
            parameter="volume_sampling", value=config,
        )

    cfg = dict(VOLUME_CONFIG_DEFAULTS)
    cfg.update(config)

    if cfg['roi_threshold'] is None:
        cfg['roi_threshold'] = roi_threshold
    thresh = cfg['roi_threshold']
    if isinstance(thresh, bool) or not isinstance(thresh, (int, float, np.floating, np.integer)) \
            or not np.isfinite(thresh) or thresh <= 0:
        raise ConfigurationError(
            "volume_sampling['roi_threshold'] must be a positive finite number or None",
            parameter="volume_sampling.roi_threshold", value=thresh,
        )
    cfg['roi_threshold'] = float(thresh)

    _check_positive_int(cfg, 'n_anchors')
    _check_positive_int(cfg, 'eval_budget', allow_none=True)
    _check_positive_int(cfg, 'search_max_iter', allow_none=True)

    steps = cfg['interior_steps']
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) \
            or steps < 0:
        raise ConfigurationError(
            "volume_sampling['interior_steps'] must be a non-negative integer",
            parameter="volume_sampling.interior_steps", value=steps,
        )
    cfg['interior_steps'] = int(steps)

    if cfg['depth_law'] not in ('volume', 'uniform_dlnl', 'uniform_sigma'):
        raise ConfigurationError(
            f"volume_sampling['depth_law'] must be 'volume', 'uniform_dlnl' "
            f"or 'uniform_sigma', got {cfg['depth_law']!r}",
            parameter="volume_sampling.depth_law", value=cfg['depth_law'],
        )

    spacing = cfg['min_spacing']
    if spacing is not None:
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float, np.floating, np.integer)) \
                or not np.isfinite(spacing) or spacing <= 0:
            raise ConfigurationError(
                "volume_sampling['min_spacing'] must be a positive finite number or None",
                parameter="volume_sampling.min_spacing", value=spacing,
            )
        cfg['min_spacing'] = float(spacing)

    if cfg['search'] not in ('lbfgsb', 'none'):
        raise ConfigurationError(
            f"volume_sampling['search'] must be 'lbfgsb' or 'none', got {cfg['search']!r}",
            parameter="volume_sampling.search", value=cfg['search'],
        )

    if not isinstance(cfg['probe_all_anchors'], bool):
        raise ConfigurationError(
            "volume_sampling['probe_all_anchors'] must be a bool",
            parameter="volume_sampling.probe_all_anchors", value=cfg['probe_all_anchors'],
        )

    files = cfg['harvest_files']
    if files is not None:
        if isinstance(files, str):
            files = [files]
        if not isinstance(files, (list, tuple)) or not all(isinstance(f, str) for f in files):
            raise ConfigurationError(
                "volume_sampling['harvest_files'] must be a path or a list of paths",
                parameter="volume_sampling.harvest_files", value=cfg['harvest_files'],
            )
        cfg['harvest_files'] = list(files)

    if not isinstance(cfg['output_file'], str) or not cfg['output_file']:
        raise ConfigurationError(
            "volume_sampling['output_file'] must be a non-empty path",
            parameter="volume_sampling.output_file", value=cfg['output_file'],
        )
    if cfg['summary_file'] is not None and not isinstance(cfg['summary_file'], str):
        raise ConfigurationError(
            "volume_sampling['summary_file'] must be a path or None",
            parameter="volume_sampling.summary_file", value=cfg['summary_file'],
        )

    return cfg


def depth_law_exponent(depth_law, n_dims):
    """Exponent γ of the interior-walk depth-target draw t = c·U^γ.

    The walk targets a depth t (ΔlnL below the band top) drawn by inverse
    CDF: t = c·U^γ with U ~ Uniform(0, 1). The presets:

    - ``'uniform_dlnl'`` (γ = 1): depths uniform in ΔlnL — equal
      representation at every fit-quality level (the default).
    - ``'uniform_sigma'`` (γ = 2): depths uniform in Z = sqrt(2·ΔlnL),
      the one-dof Wilks significance — extra resolution near the top.
    - ``'volume'`` (γ = 2/d): the uniform-in-parameter-volume law for a
      locally quadratic d-dimensional basin, where the volume at depth
      <= t grows like t^(d/2) — concentrates near the band edge in high
      dimensions.
    """
    return {
        'volume': 2.0 / max(int(n_dims), 1),
        'uniform_dlnl': 1.0,
        'uniform_sigma': 2.0,
    }[depth_law]


def volume_band(config, global_max):
    """The stage's logL lower edge and prefilter threshold for a config.

    Returns ``(band_lo, prefilter_delta)``: a point is in-band iff
    ``logL >= band_lo``. The band is one-sided (no upper edge — the ROI
    runs all the way up to the global max), and the envelope prefilter uses
    the same threshold, since profile values upper-bound logL and so can
    never exclude a point from above.
    """
    return (global_max - config['roi_threshold'], config['roi_threshold'])


class ProjectionEnvelope:
    """Necessary-condition prefilter built from converged projection grids.

    For any point θ and any computed projection, ``logL(θ) <=
    profile(proj(θ))`` by definition of profiling, so a point whose
    projection falls in a cell with profile value below ``global_max −
    threshold_delta`` — or in a cell that was never activated, which the
    scan's dynamic expansion only leaves outside the ROI — cannot satisfy
    ``logL >= global_max − threshold_delta``. The true band region is
    therefore contained in the intersection of the per-projection cylinder
    sets this class tests against.

    Built from ``export_grid_solution()``-style records and the *final*
    global maximum, so per-cell membership honors improvements made by
    later projections.
    """

    def __init__(self, records, global_max, n_dims):
        self.global_max = float(global_max)
        self.n_dims = int(n_dims)
        self._records = []
        for i, rec in enumerate(records):
            dims = np.asarray(rec['projection_dims'], dtype=int)
            axes = [np.asarray(ax, dtype=float) for ax in rec['grid_axes']]
            if len(axes) != len(dims):
                raise ValueError(
                    f"Projection record {i}: {len(dims)} projection dims "
                    f"but {len(axes)} grid axes."
                )
            if dims.size == 0 or dims.min() < 0 or dims.max() >= self.n_dims:
                raise ValueError(
                    f"Projection record {i}: projection_dims {dims.tolist()} "
                    f"out of range for n_dims={self.n_dims}."
                )
            if any(len(ax) < 2 for ax in axes):
                raise ValueError(
                    f"Projection record {i}: every grid axis needs at least 2 points."
                )
            shape = tuple(len(ax) for ax in axes)
            # Never-activated cells stay at -inf: presumed outside the ROI,
            # matching the scan's own expansion logic.
            values = np.full(shape, -np.inf)
            for grid_idx, sol in rec['solutions'].items():
                values[tuple(grid_idx)] = sol['likelihood']
            self._records.append({'dims': dims, 'axes': axes, 'values': values})

    @classmethod
    def from_projection_results(cls, results, global_max, n_dims):
        """Build from the result list returned by ``run_all_projections``.

        Each projection contributes its finest available grid
        (``refined_solution`` if present, else ``coarse_solution``).
        """
        records = []
        for i, res in enumerate(results):
            rec = res.get('refined_solution') or res.get('coarse_solution')
            if rec is None:
                raise ValueError(f"Projection result {i} has no exported grid solution.")
            records.append(rec)
        return cls(records, global_max, n_dims)

    @property
    def n_projections(self):
        return len(self._records)

    @property
    def covers_full_space(self):
        """True if any projection gridded the full parameter space directly.

        In that case (direct-eval mode) the grid already covers the volume
        and the sampling stage adds only resolution; callers should skip it.
        """
        return any(len(rec['dims']) == self.n_dims for rec in self._records)

    def cell_indices(self, record_index, points):
        """Grid indices of ``points`` on one projection's grid.

        Same nearest-cell arithmetic as the sampler's
        ``_get_grid_indices_from_point`` (round to nearest grid node, clip
        to the grid). Returns a tuple of index arrays, one per grid axis.
        """
        rec = self._records[record_index]
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        coords = pts[:, rec['dims']]
        indices = []
        for j, axis in enumerate(rec['axes']):
            normalized = (coords[:, j] - axis[0]) / (axis[-1] - axis[0])
            idx = np.rint(normalized * (len(axis) - 1)).astype(int)
            np.clip(idx, 0, len(axis) - 1, out=idx)
            indices.append(idx)
        return tuple(indices)

    def test(self, points, threshold_delta):
        """Boolean mask: which points *may* lie in the band.

        ``True`` means the point passes every projection's necessary
        condition (profile value at its cell >= ``global_max −
        threshold_delta``); ``False`` means it provably cannot be in the
        band. Accepts a single point or an ``(n, n_dims)`` array.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != self.n_dims:
            raise ValueError(
                f"points must have shape (n, {self.n_dims}); got {pts.shape}."
            )
        cutoff = self.global_max - threshold_delta
        mask = np.ones(len(pts), dtype=bool)
        for i, rec in enumerate(self._records):
            values = rec['values'][self.cell_indices(i, pts)]
            mask &= values >= cutoff
        return mask


class AnchorSet:
    """Anchors plus prefilter statistics and per-anchor sample records.

    ``rep_points/_logls/_dists/_source`` hold each anchor's best in-band
    representative (nearest first; higher logL on ties). Within
    ``coverage_radius`` it marks the anchor as covered; beyond it provides a
    warm start for the tier-3 search. ``probed``/``probe_logls`` record tier-2
    probes unconditionally so the uniform subset and volume estimate remain
    re-derivable after global-max drift.
    """

    def __init__(self, anchors, bounds, coverage_radius,
                 n_draws=0, n_prefilter_accepted=0):
        self.bounds = np.asarray(bounds, dtype=float)
        n_dims = len(self.bounds)
        self.anchors = np.asarray(anchors, dtype=float).reshape(-1, n_dims)
        self.coverage_radius = float(coverage_radius)
        self.n_draws = int(n_draws)
        self.n_prefilter_accepted = int(n_prefilter_accepted)

        self._lo = self.bounds[:, 0]
        self._extent = self.bounds[:, 1] - self.bounds[:, 0]
        self.scaled_anchors = self.scale(self.anchors)
        self._tree = None

        n = len(self.anchors)
        self.rep_points = np.full((n, n_dims), np.nan)
        self.rep_logls = np.full(n, -np.inf)
        self.rep_dists = np.full(n, np.inf)
        self.rep_source = np.full(n, SOURCE_NONE, dtype=np.int8)
        # Depth-targeted (walked) reps are locked against replacement by
        # passive closest-wins offers, which concentrate at the band edge.
        self.rep_walked = np.zeros(n, dtype=bool)
        self.probed = np.zeros(n, dtype=bool)
        self.probe_logls = np.full(n, np.nan)

    @property
    def n_anchors(self):
        return len(self.anchors)

    @property
    def tree(self):
        """KD-tree over the scaled anchors (built lazily, cached)."""
        if self._tree is None:
            self._tree = cKDTree(self.scaled_anchors)
        return self._tree

    @property
    def prefilter_acceptance(self):
        """Fraction of Sobol draws that passed the envelope (NaN if no draws)."""
        if self.n_draws == 0:
            return np.nan
        return self.n_prefilter_accepted / self.n_draws

    @property
    def covered(self):
        """Boolean mask of anchors covered by an in-band representative."""
        return self.rep_dists <= self.coverage_radius

    def scale(self, points):
        """Map points to bounds-scaled ([0, 1] per dim) coordinates."""
        return (np.asarray(points, dtype=float) - self._lo) / self._extent

    def offer_to_anchor(self, anchor_index, point, logl, dist, source):
        """Offer an in-band sample as anchor ``anchor_index``'s representative.

        Accepted if it beats the current record (closer; higher logL on an
        exact distance tie). Returns True if accepted. The caller is
        responsible for the band check.
        """
        if self.rep_walked[anchor_index]:
            return False
        better = (dist < self.rep_dists[anchor_index]
                  or (dist == self.rep_dists[anchor_index]
                      and logl > self.rep_logls[anchor_index]))
        if better:
            self.rep_points[anchor_index] = point
            self.rep_logls[anchor_index] = logl
            self.rep_dists[anchor_index] = dist
            self.rep_source[anchor_index] = source
        return better

    def offer_sample(self, point, logl, source):
        """Offer an in-band sample to its nearest anchor (and only that one,
        preserving stratification). Returns True if it became that anchor's
        representative. The caller is responsible for the band check."""
        if self.n_anchors == 0:
            return False
        dist, idx = self.tree.query(self.scale(point))
        return self.offer_to_anchor(int(idx), np.asarray(point, dtype=float),
                                    float(logl), float(dist), source)


def generate_anchors(envelope, bounds, n_anchors, threshold_delta,
                     min_spacing=None, seed=None, draw_cap=None):
    """Draw scrambled-Sobol anchors inside the projection envelope.

    Draws Sobol points over the bounds box (in power-of-2 batches) and
    keeps those passing ``envelope.test(..., threshold_delta)`` — in draw
    order, so probes at the kept anchors stay uniform over the prefilter
    region — until ``n_anchors`` anchors are kept or ``draw_cap`` draws have
    been tested (default ``DEFAULT_DRAW_CAP_FACTOR * n_anchors``; a warning
    is logged if the cap is hit). With ``min_spacing`` set, anchors closer
    than that (bounds-scaled) to an already-kept anchor are skipped;
    prefilter-acceptance counting is unaffected.

    The :class:`AnchorSet`'s ``coverage_radius`` is ``min_spacing`` when
    given, else the median nearest-neighbor distance among the anchors.
    """
    logger = get_logger()
    bounds = np.asarray(bounds, dtype=float)
    n_dims = len(bounds)
    if draw_cap is None:
        draw_cap = DEFAULT_DRAW_CAP_FACTOR * n_anchors

    lo, hi = bounds[:, 0], bounds[:, 1]
    extent = hi - lo
    sobol = qmc.Sobol(d=n_dims, scramble=True, seed=seed)

    kept = np.empty((n_anchors, n_dims))
    kept_scaled = np.empty((n_anchors, n_dims))
    n_kept = 0
    n_draws = 0
    n_accepted = 0

    while n_kept < n_anchors and n_draws < draw_cap:
        batch = qmc.scale(sobol.random(ANCHOR_DRAW_BATCH), lo, hi)
        n_draws += len(batch)
        passed = envelope.test(batch, threshold_delta)
        n_accepted += int(np.count_nonzero(passed))

        for point in batch[passed]:
            if n_kept >= n_anchors:
                break
            point_scaled = (point - lo) / extent
            if min_spacing is not None and n_kept > 0:
                d2_min = np.min(np.sum(
                    (kept_scaled[:n_kept] - point_scaled) ** 2, axis=1))
                if d2_min < min_spacing ** 2:
                    continue
            kept[n_kept] = point
            kept_scaled[n_kept] = point_scaled
            n_kept += 1

    if n_kept < n_anchors:
        acceptance = n_accepted / n_draws if n_draws else 0.0
        logger.warning(
            f"Volume sampling: anchor draw cap hit after {n_draws} draws with "
            f"{n_kept}/{n_anchors} anchors kept (prefilter acceptance "
            f"{acceptance:.2e}). The prefilter region is very small or "
            f"min_spacing is too large for n_anchors."
        )

    anchors = kept[:n_kept]
    if min_spacing is not None:
        coverage_radius = float(min_spacing)
    elif n_kept >= 2:
        nn_dists, _ = cKDTree(kept_scaled[:n_kept]).query(kept_scaled[:n_kept], k=2)
        coverage_radius = float(np.median(nn_dists[:, 1]))
    else:
        coverage_radius = np.nan
        logger.warning(
            "Volume sampling: fewer than 2 anchors and no min_spacing; "
            "coverage radius is undefined (NaN) and no anchor can be "
            "marked covered."
        )

    return AnchorSet(anchors, bounds, coverage_radius,
                     n_draws=n_draws, n_prefilter_accepted=n_accepted)


def resolve_harvest_files(volume_config, samples_output_file=None):
    """The sample files tier 1 reads: this run's output plus harvest_files.

    The run's own ``samples_output_file`` is included only if it exists
    (logged otherwise); explicitly listed ``harvest_files`` must exist.
    Duplicate paths are dropped, order preserved.
    """
    logger = get_logger()
    candidates = []
    if samples_output_file:
        if os.path.exists(samples_output_file):
            candidates.append(samples_output_file)
        else:
            logger.info(
                f"Volume sampling: samples_output_file "
                f"'{samples_output_file}' not found; not harvesting from it."
            )
    for path in volume_config.get('harvest_files') or []:
        if not os.path.exists(path):
            raise ConfigurationError(
                f"volume_sampling['harvest_files'] entry not found: '{path}'",
                parameter="volume_sampling.harvest_files", value=path,
            )
        candidates.append(path)

    files = []
    seen = set()
    for path in candidates:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            files.append(path)
    return files


def harvest_existing_samples(anchor_set, sample_files, band_lo,
                             chunk_size=None):
    """Tier 1: fill the anchor set's harvest records from existing samples.

    Streams every file batch by batch (memory stays O(n_anchors)
    regardless of file size); each in-band sample is assigned to its
    nearest anchor only — one sample never covers several anchors, which
    preserves the stratification — and per anchor the closest sample wins,
    with higher logL breaking distance ties.

    Returns a stats dict with keys ``n_files``, ``n_samples``,
    ``n_in_band``, ``n_covered``, ``n_with_warm_start``.
    """
    from .sample_io import iter_sample_batches

    logger = get_logger()
    stats = {'n_files': len(sample_files), 'n_samples': 0, 'n_in_band': 0,
             'n_covered': 0, 'n_with_warm_start': 0}
    if anchor_set.n_anchors == 0 or not sample_files:
        if not sample_files:
            logger.info(
                "Volume sampling: no sample files to harvest from; "
                "skipping tier 1."
            )
        return stats

    n_dims = anchor_set.anchors.shape[1]
    tree = anchor_set.tree
    iter_kwargs = {} if chunk_size is None else {'chunk_size': chunk_size}

    for path in sample_files:
        for batch in iter_sample_batches(path, **iter_kwargs):
            if batch.size == 0:
                continue
            # Sample rows are [params..., logL, phase] (width n_dims + 2);
            # harvest reads logL at column n_dims and ignores the phase.
            if batch.shape[1] != n_dims + 2:
                raise ConfigurationError(
                    f"Sample file '{path}' has rows of width {batch.shape[1]}; "
                    f"expected n_dims + 2 = {n_dims + 2} "
                    f"([params..., logL, phase]).",
                    parameter="volume_sampling.harvest_files", value=path,
                )
            stats['n_samples'] += len(batch)

            logls = batch[:, n_dims]
            in_band = np.isfinite(logls) & (logls >= band_lo)
            if not in_band.any():
                continue
            stats['n_in_band'] += int(np.count_nonzero(in_band))

            params = batch[in_band, :n_dims]
            logls = logls[in_band]
            dists, anchor_idx = tree.query(anchor_set.scale(params))

            # Per-anchor batch winner, fully vectorized: minimum distance
            # per anchor, then maximum logL among the rows achieving it.
            n_anchors = anchor_set.n_anchors
            batch_min_dist = np.full(n_anchors, np.inf)
            np.minimum.at(batch_min_dist, anchor_idx, dists)
            at_min = dists <= batch_min_dist[anchor_idx]
            batch_best_logl = np.full(n_anchors, -np.inf)
            np.maximum.at(batch_best_logl, anchor_idx[at_min], logls[at_min])
            winner = at_min & (logls >= batch_best_logl[anchor_idx])
            winner_rows = np.flatnonzero(winner)
            _, first = np.unique(anchor_idx[winner_rows], return_index=True)
            rows = winner_rows[first]

            a = anchor_idx[rows]
            d = dists[rows]
            l = logls[rows]
            better = (d < anchor_set.rep_dists[a]) | (
                (d == anchor_set.rep_dists[a]) & (l > anchor_set.rep_logls[a])
            )
            upd = a[better]
            anchor_set.rep_dists[upd] = d[better]
            anchor_set.rep_logls[upd] = l[better]
            anchor_set.rep_points[upd] = params[rows[better]]
            anchor_set.rep_source[upd] = SOURCE_HARVEST

    stats['n_covered'] = int(np.count_nonzero(anchor_set.covered))
    stats['n_with_warm_start'] = int(np.count_nonzero(np.isfinite(anchor_set.rep_dists)))
    logger.info(
        f"Volume sampling harvest: {stats['n_in_band']} in-band samples out of "
        f"{stats['n_samples']} read from {stats['n_files']} file(s); "
        f"{stats['n_covered']}/{anchor_set.n_anchors} anchors covered, "
        f"{stats['n_with_warm_start']} have a warm start."
    )
    return stats


class VolumeStageState:
    """Mutable bookkeeping for one volume-sampling stage run (MPI-free).

    Fed by the orchestrator via :meth:`note_eval` (budget counting, opportunistic
    representative updates) and :meth:`record_search_job`. The initial band is
    stored; :func:`finalize_volume_stage` re-derives final membership from stored
    logL values against the final global max, so mid-stage drift is handled.
    """

    def __init__(self, anchor_set, band_lo, eval_budget=None):
        self.anchor_set = anchor_set
        self.band_lo = float(band_lo)
        self.eval_budget = eval_budget
        self.evals_used = 0

        n = anchor_set.n_anchors
        n_dims = len(anchor_set.bounds)
        self.searched = np.zeros(n, dtype=bool)
        self.unbudgeted = np.zeros(n, dtype=bool)
        # Adaptive depth-quota state (armed by init_depth_quota).
        self._quota_probs = None
        self._quota_achieved = None
        # Closest-approach records for anchors whose search never reached the
        # band (the "hole" diagnostic): the evaluation with the smallest band
        # violation, with its logL and scaled distance to the anchor.
        self.closest_points = np.full((n, n_dims), np.nan)
        self.closest_logls = np.full(n, -np.inf)
        self.closest_dists = np.full(n, np.inf)
        self.closest_violations = np.full(n, np.inf)

    def in_band(self, logl):
        return logl >= self.band_lo

    def init_depth_quota(self, band_depth, exponent, n_bins=10):
        """Arm adaptive depth-target drawing for the interior walks.

        Each draw picks a bin in proportion to the law's residual need and
        draws from the law restricted to that bin, so under-filled (hard)
        depth bins are retried until enough anchors can reach them.
        """
        edges = np.linspace(0.0, band_depth, n_bins + 1)
        cdf = (edges / band_depth) ** (1.0 / exponent)
        self._quota_band_depth = float(band_depth)
        self._quota_exponent = float(exponent)
        self._quota_edges = edges
        self._quota_cdf = cdf
        self._quota_probs = np.diff(cdf)
        self._quota_achieved = np.zeros(n_bins)

    def draw_depth_target(self):
        """Depth target (ΔlnL below the band top) for the next walk."""
        need = (self._quota_probs * (self._quota_achieved.sum() + 1.0)
                - self._quota_achieved)
        np.clip(need, 0.0, None, out=need)
        weights = need if need.sum() > 0 else self._quota_probs
        b = int(np.random.choice(len(weights), p=weights / weights.sum()))
        # Exact law restricted to the chosen bin (inverse CDF on the
        # bin's CDF segment).
        u = np.random.uniform(self._quota_cdf[b], self._quota_cdf[b + 1])
        return self._quota_band_depth * u ** self._quota_exponent

    def record_rep_depth(self, logl):
        """Count a representative's depth toward the quota.

        Called for walked reps and passively covered anchors alike, so the
        adaptive draws compensate for edge-concentrated passive hits and the
        combined representative set converges to the requested depth law.
        """
        if self._quota_probs is None or not np.isfinite(logl):
            return
        depth = self.band_lo + self._quota_band_depth - logl
        width = self._quota_edges[1] - self._quota_edges[0]
        b = int(np.clip(depth // width, 0, len(self._quota_achieved) - 1))
        self._quota_achieved[b] += 1

    def budget_left(self):
        return self.eval_budget is None or self.evals_used < self.eval_budget

    def note_eval(self, params, logl, offer=True):
        """Account for one stage evaluation; opportunistically offer in-band
        results to their nearest anchor (probe jobs do their own exact
        bookkeeping and pass ``offer=False``)."""
        self.evals_used += 1
        if offer and np.isfinite(logl) and self.in_band(logl):
            self.anchor_set.offer_sample(params, logl, SOURCE_SEARCH)

    def record_search_job(self, job):
        """Fold a finished VolumeSearchJob's outcome into the per-anchor records."""
        k = job.anchor_index
        self.searched[k] = True
        if job.best_inband_point is not None:
            # The job's best in-band point may be nearest to a *different*
            # anchor (note_eval offered it there); record it against the
            # job's own anchor too, so 'projected' classification sees it.
            self.anchor_set.offer_to_anchor(
                k, job.best_inband_point, job.best_inband_logl,
                job.best_inband_dist, SOURCE_SEARCH,
            )
        if getattr(job, 'interior_point', None) is not None:
            # Interior-steps mode: the job deliberately walked away from
            # the band edge, so depth beats the closest-wins rule for this
            # anchor's representative. Coverage is unaffected: the walk is
            # distance-capped (within the radius for hits).
            aset = self.anchor_set
            aset.rep_points[k] = job.interior_point
            aset.rep_logls[k] = job.interior_logl
            aset.rep_dists[k] = job.interior_dist
            aset.rep_source[k] = SOURCE_SEARCH
            aset.rep_walked[k] = True
            self.record_rep_depth(job.interior_logl)
        if job.best_viol_point is not None and \
                job.best_viol < self.closest_violations[k]:
            self.closest_points[k] = job.best_viol_point
            self.closest_logls[k] = job.best_viol_logl
            self.closest_dists[k] = job.best_viol_dist
            self.closest_violations[k] = job.best_viol


def finalize_volume_stage(state, config,
                          global_max_start, global_max_final,
                          search_enabled):
    """Classify every anchor and assemble the stage-result dict.

    Band membership is re-derived from stored logL values against
    ``global_max_final``. Anchor statuses: ``covered`` (in-band rep within
    coverage radius), ``projected`` (in-band rep beyond radius), ``hole``
    (searched but never in-band), ``unbudgeted`` (budget ran out before probing),
    ``uncovered`` (everything else). The volume estimate is only computed when
    ``probe_all_anchors`` kept the probe set uniform.
    """
    aset = state.anchor_set
    n = aset.n_anchors
    band_lo_f, _ = volume_band(config, global_max_final)

    rep_in_band = (np.isfinite(aset.rep_logls)
                   & (aset.rep_logls >= band_lo_f))
    covered = rep_in_band & (aset.rep_dists <= aset.coverage_radius)
    projected = rep_in_band & ~covered
    hole = ~rep_in_band & state.searched
    budget_exhausted = not state.budget_left()
    unbudgeted = ~rep_in_band & ~state.searched & (
        state.unbudgeted
        | (search_enabled & budget_exhausted & aset.probed)
    )
    status = np.full(n, 'uncovered', dtype=object)
    status[unbudgeted] = 'unbudgeted'
    status[hole] = 'hole'
    status[projected] = 'projected'
    status[covered] = 'covered'

    probe_in_band = (aset.probed
                     & np.isfinite(aset.probe_logls)
                     & (aset.probe_logls >= band_lo_f))
    n_probed = int(np.count_nonzero(aset.probed))
    n_probe_hits = int(np.count_nonzero(probe_in_band))
    probe_acceptance = n_probe_hits / n_probed if n_probed else np.nan

    volume_estimate = None
    volume_estimate_err = None
    if config['probe_all_anchors'] and n_probed > 0 \
            and np.isfinite(aset.prefilter_acceptance):
        box_volume = float(np.prod(aset.bounds[:, 1] - aset.bounds[:, 0]))
        envelope_volume = box_volume * aset.prefilter_acceptance
        volume_estimate = envelope_volume * probe_acceptance
        # Binomial uncertainty from the probe stage (the prefilter-acceptance
        # term's error is typically negligible given the draw count, and the
        # scrambled-Sobol anchors make both conservative).
        p = probe_acceptance
        volume_estimate_err = envelope_volume * float(np.sqrt(p * (1.0 - p) / n_probed))

    def _count(mask):
        return int(np.count_nonzero(mask))

    # Realized depth distribution of the in-band representatives: lets users
    # check the achieved depth law against the requested one (walks are
    # censored by the distance cap and the locally reachable likelihood, so
    # the realized distribution can deviate).
    rep_depth_histogram = None
    if rep_in_band.any():
        depths = global_max_final - aset.rep_logls[rep_in_band]
        edges = np.linspace(0.0, config['roi_threshold'], 11)
        counts, _ = np.histogram(depths, bins=edges)
        rep_depth_histogram = {'bin_edges': edges.tolist(),
                               'counts': counts.tolist()}

    stats = {
        'n_anchors': n,
        'n_covered': _count(covered),
        'n_covered_harvest': _count(covered & (aset.rep_source == SOURCE_HARVEST)),
        'n_covered_probe': _count(covered & (aset.rep_source == SOURCE_PROBE)),
        'n_covered_search': _count(covered & (aset.rep_source == SOURCE_SEARCH)),
        'n_projected': _count(projected),
        'n_holes': _count(hole),
        'n_unbudgeted': _count(unbudgeted),
        'n_uncovered': _count(status == 'uncovered'),
        'evals_used': state.evals_used,
        'n_draws': aset.n_draws,
        'prefilter_acceptance': aset.prefilter_acceptance,
        'n_probed': n_probed,
        'n_probe_hits': n_probe_hits,
        'probe_acceptance': probe_acceptance,
        'volume_estimate': volume_estimate,
        'volume_estimate_err': volume_estimate_err,
        'global_max_drift': global_max_final - global_max_start,
        'coverage_radius': aset.coverage_radius,
        'rep_depth_histogram': rep_depth_histogram,
    }

    return {
        'skipped': False,
        'reason': None,
        'anchor_set': aset,
        'anchors': aset.anchors,
        'anchor_status': status,
        'rep_points': aset.rep_points,
        'rep_logls': aset.rep_logls,
        'rep_dists': aset.rep_dists,
        'rep_source': aset.rep_source,
        'probed': aset.probed,
        'probe_logls': aset.probe_logls,
        'uniform_subset': probe_in_band,
        'closest_points': state.closest_points,
        'closest_logls': state.closest_logls,
        'closest_dists': state.closest_dists,
        'closest_violations': state.closest_violations,
        'band_lo_initial': state.band_lo,
        'band_lo_final': band_lo_f,
        'stats': stats,
    }


def assemble_volume_rows(result):
    """The stage's output rows: ``[params..., logL, tag]`` per anchor.

    One row per anchor with an in-band representative (status ``covered``
    or ``projected``), tagged by the tier that found it (``TAG_HARVEST`` /
    ``TAG_PROBE`` / ``TAG_SEARCH``), plus one ``TAG_HOLE`` row per hole
    anchor whose search recorded a closest-approach point (those rows are
    *not* in-band). Returns an ``(n_rows, n_dims + 2)`` float array.
    """
    status = result['anchor_status']
    n_dims = result['anchors'].shape[1] if len(result['anchors']) else 0

    blocks = []
    rep_mask = np.isin(status, ('covered', 'projected'))
    if rep_mask.any():
        tags = np.array([_SOURCE_TO_TAG[s]
                         for s in result['rep_source'][rep_mask]])
        blocks.append(np.column_stack([
            result['rep_points'][rep_mask],
            result['rep_logls'][rep_mask],
            tags,
        ]))

    hole_mask = (status == 'hole') & np.isfinite(result['closest_violations'])
    if hole_mask.any():
        blocks.append(np.column_stack([
            result['closest_points'][hole_mask],
            result['closest_logls'][hole_mask],
            np.full(int(np.count_nonzero(hole_mask)), TAG_HOLE),
        ]))

    if not blocks:
        return np.empty((0, n_dims + 2))
    return np.vstack(blocks)


def _json_safe(obj):
    """Recursively convert a result structure to JSON-portable types.

    NumPy scalars become Python scalars; non-finite floats become None
    (JSON has no inf/nan); tuples become lists.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def default_summary_file(output_file):
    """The summary path derived from ``output_file``: same base,
    ``_summary.json`` suffix."""
    base, _ = os.path.splitext(output_file)
    return base + '_summary.json'


def write_volume_output(result, config):
    """Write the stage's sample file and JSON summary.

    The sample file (``config['output_file']``; format from the extension
    via sample_io, replaced if it exists) holds the rows from
    :func:`assemble_volume_rows`; nothing is written when there are no
    rows. The JSON summary (``config['summary_file']``, default derived
    via :func:`default_summary_file`) holds the run configuration, bands,
    statistics, per-tag row counts, and the tag legend.

    Returns ``(output_path_or_None, summary_path, rows_by_tag)`` and
    annotates ``result`` with the same under the keys ``output_file``,
    ``summary_file`` and ``rows_by_tag``.
    """
    import json

    from .sample_io import write_samples

    logger = get_logger()
    rows = assemble_volume_rows(result)
    rows_by_tag = {
        int(tag): int(np.count_nonzero(rows[:, -1] == tag))
        for tag in (TAG_HARVEST, TAG_PROBE, TAG_SEARCH, TAG_HOLE)
    }

    output_path = config['output_file']
    if len(rows):
        if os.path.exists(output_path):
            logger.warning(
                f"Volume sampling: replacing existing file '{output_path}'."
            )
        write_samples(rows, output_path, overwrite=True)
        logger.info(
            f"Volume sampling: wrote {len(rows)} tagged samples to "
            f"'{output_path}'."
        )
    else:
        output_path = None
        logger.info(
            "Volume sampling: no in-band representatives or closest-approach "
            "points; no sample file written."
        )

    summary_path = config['summary_file'] or default_summary_file(
        config['output_file'])
    summary = {
        'band_lo_initial': result['band_lo_initial'],
        'band_lo_final': result['band_lo_final'],
        'config': config,
        'stats': result['stats'],
        'output_file': output_path,
        'n_rows': len(rows),
        'rows_by_tag': rows_by_tag,
        'tag_legend': TAG_LEGEND,
        'uniform_subset_valid': config['probe_all_anchors'],
    }
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(_json_safe(summary), f, indent=2)
    logger.info(f"Volume sampling: wrote summary to '{summary_path}'.")

    result['output_file'] = output_path
    result['summary_file'] = summary_path
    result['rows_by_tag'] = rows_by_tag
    return output_path, summary_path, rows_by_tag
