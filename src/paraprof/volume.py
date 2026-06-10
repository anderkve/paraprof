"""Stratified ROI/shell volume sampling (see docs/volume_sampling_plan.md).

After the projections complete, the volume-sampling stage collects a
stratified, well-spread set of samples inside the region of interest
(``mode='roi'``) or in a shell around it (``mode='shell'``), via a
three-tier funnel: harvest of existing samples, direct probes at
scrambled-Sobol anchor points, and an anchored search for anchors whose
probe failed.

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
    'mode': 'roi',              # 'roi' or 'shell'
    'shell_threshold': 25.0,    # outer ΔlnL; inner edge is roi_threshold (shell mode)
    'n_points': 1000,           # target number of anchors
    'min_spacing': None,        # optional Poisson-disk radius (bounds-scaled units)
    'eval_budget': None,        # hard cap on stage evaluations (None = unlimited)
    'search': 'lbfgsb',         # tier-3 method: 'lbfgsb' or 'none' (probe-only)
    'probe_all_anchors': True,  # probe even harvest-covered anchors (uniform subset)
    'search_max_iter': None,    # per-anchor L-BFGS-B cap (None = lbfgsb_max_iter)
    'harvest_files': None,      # extra sample files for tier 1
    'output_file': 'volume_samples.csv',
    'summary_file': None,       # None = derived from output_file
}

# Sobol draws come in power-of-2 batches to keep the sequence's balance
# properties (scipy warns otherwise).
ANCHOR_DRAW_BATCH = 4096
# Default cap on envelope-tested draws: this many per requested anchor.
DEFAULT_DRAW_CAP_FACTOR = 10_000


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

    if cfg['mode'] not in ('roi', 'shell'):
        raise ConfigurationError(
            f"volume_sampling['mode'] must be 'roi' or 'shell', got {cfg['mode']!r}",
            parameter="volume_sampling.mode", value=cfg['mode'],
        )

    shell = cfg['shell_threshold']
    if isinstance(shell, bool) or not isinstance(shell, (int, float, np.floating, np.integer)) \
            or not np.isfinite(shell) or shell <= 0:
        raise ConfigurationError(
            "volume_sampling['shell_threshold'] must be a positive finite number",
            parameter="volume_sampling.shell_threshold", value=shell,
        )
    cfg['shell_threshold'] = float(shell)
    if cfg['mode'] == 'shell' and cfg['shell_threshold'] <= roi_threshold:
        raise ConfigurationError(
            f"volume_sampling['shell_threshold'] ({cfg['shell_threshold']}) must exceed "
            f"roi_threshold ({roi_threshold}) in shell mode (it is the outer edge of the band)",
            parameter="volume_sampling.shell_threshold", value=cfg['shell_threshold'],
        )

    _check_positive_int(cfg, 'n_points')
    _check_positive_int(cfg, 'eval_budget', allow_none=True)
    _check_positive_int(cfg, 'search_max_iter', allow_none=True)

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


def volume_band(config, roi_threshold, global_max):
    """The stage's logL band and prefilter threshold for a normalized config.

    Returns ``(band_lo, band_hi, prefilter_delta)``: a point belongs to the
    band iff ``band_lo <= logL <= band_hi``. The envelope prefilter uses
    ``prefilter_delta`` — the *outer* threshold only, since profile values
    upper-bound logL and therefore can never exclude a point from above
    (the point under a high-profile cell may still be low-likelihood).
    """
    if config['mode'] == 'shell':
        return (global_max - config['shell_threshold'],
                global_max - roi_threshold,
                config['shell_threshold'])
    return (global_max - roi_threshold, np.inf, roi_threshold)


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
    """Anchors plus prefilter statistics and per-anchor harvest records.

    ``harvest_points/_logls/_dists`` hold each anchor's best in-band sample
    so far — nearest first, higher logL on distance ties — at any distance.
    The same record serves two roles: within ``coverage_radius`` it covers
    the anchor (tier 1 done), at any distance it is the warm start for the
    anchor's tier-3 search.
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

        n = len(self.anchors)
        self.harvest_points = np.full((n, n_dims), np.nan)
        self.harvest_logls = np.full(n, -np.inf)
        self.harvest_dists = np.full(n, np.inf)

    @property
    def n_anchors(self):
        return len(self.anchors)

    @property
    def prefilter_acceptance(self):
        """Fraction of Sobol draws that passed the envelope (NaN if no draws)."""
        if self.n_draws == 0:
            return np.nan
        return self.n_prefilter_accepted / self.n_draws

    @property
    def covered(self):
        """Boolean mask of anchors covered by a harvested in-band sample."""
        return self.harvest_dists <= self.coverage_radius

    def scale(self, points):
        """Map points to bounds-scaled ([0, 1] per dim) coordinates."""
        return (np.asarray(points, dtype=float) - self._lo) / self._extent


def generate_anchors(envelope, bounds, n_points, threshold_delta,
                     min_spacing=None, seed=None, draw_cap=None):
    """Draw scrambled-Sobol anchors inside the projection envelope.

    Draws Sobol points over the bounds box (in power-of-2 batches) and
    keeps those passing ``envelope.test(..., threshold_delta)`` — in draw
    order, so probes at the kept anchors stay uniform over the prefilter
    region — until ``n_points`` anchors are kept or ``draw_cap`` draws have
    been tested (default ``DEFAULT_DRAW_CAP_FACTOR * n_points``; a warning
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
        draw_cap = DEFAULT_DRAW_CAP_FACTOR * n_points

    lo, hi = bounds[:, 0], bounds[:, 1]
    extent = hi - lo
    sobol = qmc.Sobol(d=n_dims, scramble=True, seed=seed)

    kept = np.empty((n_points, n_dims))
    kept_scaled = np.empty((n_points, n_dims))
    n_kept = 0
    n_draws = 0
    n_accepted = 0

    while n_kept < n_points and n_draws < draw_cap:
        batch = qmc.scale(sobol.random(ANCHOR_DRAW_BATCH), lo, hi)
        n_draws += len(batch)
        passed = envelope.test(batch, threshold_delta)
        n_accepted += int(np.count_nonzero(passed))

        for point in batch[passed]:
            if n_kept >= n_points:
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

    if n_kept < n_points:
        acceptance = n_accepted / n_draws if n_draws else 0.0
        logger.warning(
            f"Volume sampling: anchor draw cap hit after {n_draws} draws with "
            f"{n_kept}/{n_points} anchors kept (prefilter acceptance "
            f"{acceptance:.2e}). The prefilter region is very small or "
            f"min_spacing is too large for n_points."
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


def harvest_existing_samples(anchor_set, sample_files, band_lo, band_hi,
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
    tree = cKDTree(anchor_set.scaled_anchors)
    iter_kwargs = {} if chunk_size is None else {'chunk_size': chunk_size}

    for path in sample_files:
        for batch in iter_sample_batches(path, **iter_kwargs):
            if batch.size == 0:
                continue
            if batch.shape[1] != n_dims + 1:
                raise ConfigurationError(
                    f"Sample file '{path}' has rows of width {batch.shape[1]}; "
                    f"expected n_dims + 1 = {n_dims + 1}.",
                    parameter="volume_sampling.harvest_files", value=path,
                )
            stats['n_samples'] += len(batch)

            logls = batch[:, -1]
            in_band = np.isfinite(logls) & (logls >= band_lo) & (logls <= band_hi)
            if not in_band.any():
                continue
            stats['n_in_band'] += int(np.count_nonzero(in_band))

            params = batch[in_band, :-1]
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
            better = (d < anchor_set.harvest_dists[a]) | (
                (d == anchor_set.harvest_dists[a]) & (l > anchor_set.harvest_logls[a])
            )
            upd = a[better]
            anchor_set.harvest_dists[upd] = d[better]
            anchor_set.harvest_logls[upd] = l[better]
            anchor_set.harvest_points[upd] = params[rows[better]]

    stats['n_covered'] = int(np.count_nonzero(anchor_set.covered))
    stats['n_with_warm_start'] = int(np.count_nonzero(np.isfinite(anchor_set.harvest_dists)))
    logger.info(
        f"Volume sampling harvest: {stats['n_in_band']} in-band samples out of "
        f"{stats['n_samples']} read from {stats['n_files']} file(s); "
        f"{stats['n_covered']}/{anchor_set.n_anchors} anchors covered, "
        f"{stats['n_with_warm_start']} have a warm start."
    )
    return stats
