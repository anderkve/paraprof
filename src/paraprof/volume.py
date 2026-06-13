"""Volume sampling (see docs/volume_sampling_plan.md).

After the projections complete, the volume-sampling stage populates the
region of interest — ``{logL > global_max - roi_threshold}`` — with a set of
samples that balances space-filling and lnL-filling, using an affine-invariant
ensemble of umbrella walkers. Each walker carries a fixed *home level* spanning
the band uniformly in ΔlnL and targets a log-Gaussian in logL around it; the
ensemble is moved with the Goodman-Weare stretch move (partners pooled across
the whole ensemble by default), so curved/stretched geometries need no step,
gradient, or covariance tuning.

The stage's ``roi_threshold`` defaults to the projection's but can be set
larger to reach into the shell around the ROI (the band is one-sided, running
up to the global max).

This module holds the MPI-free building blocks: config validation, the
:class:`ProjectionEnvelope` prefilter (which seeds walkers and flags
direct-eval mode), the ensemble primitives the master loop drives, and the
output writer. Coordinates are in the input-parameter space; the affine
stretch move needs no bounds-scaling.
"""
import os

import numpy as np
from scipy.stats import qmc

from .exceptions import ConfigurationError
from .logger import get_logger

VOLUME_CONFIG_DEFAULTS = {
    'roi_threshold': None,         # ΔlnL band depth; None = projection's roi_threshold.
                                   # Larger reaches into the shell outside the ROI.
    'n_walkers': 1000,             # ensemble size (each carries a home level)
    'n_steps': 30,                 # stretch sweeps per walker
    'eval_budget': None,           # optional cap; None = n_walkers * (n_steps + 1)
    'sigma_frac': 0.05,            # umbrella width sigma / roi_threshold
    'partner_level_window': None,  # lnL window for stretch partners; None = full pool
    'warm_start': True,            # seed near each level from scan samples
    'output_file': 'volume_samples.csv',
    'summary_file': None,          # None = derived from output_file
}

# Sobol seeds come in power-of-2 batches (scipy warns otherwise).
SEED_DRAW_BATCH = 4096
# Cap on envelope-tested seed draws: this many per requested walker.
DEFAULT_DRAW_CAP_FACTOR = 10_000
# Goodman-Weare stretch scale parameter (a=2 is the standard default).
STRETCH_A = 2.0


def _check_positive_int(cfg, key, minimum=1, allow_none=False):
    val = cfg[key]
    if val is None and allow_none:
        return
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)) \
            or val < minimum:
        raise ConfigurationError(
            f"volume_sampling['{key}'] must be an integer >= {minimum}"
            + (" or None" if allow_none else ""),
            parameter=f"volume_sampling.{key}", value=val,
        )
    cfg[key] = int(val)


def _check_positive_float(cfg, key, allow_none=False):
    val = cfg[key]
    if val is None and allow_none:
        return
    if isinstance(val, bool) or not isinstance(val, (int, float, np.floating,
                                                     np.integer)) \
            or not np.isfinite(val) or val <= 0:
        raise ConfigurationError(
            f"volume_sampling['{key}'] must be a positive finite number"
            + (" or None" if allow_none else ""),
            parameter=f"volume_sampling.{key}", value=val,
        )
    cfg[key] = float(val)


def normalize_volume_config(config, roi_threshold):
    """Validate a ``volume_sampling`` config dict and fill in defaults.

    Returns a new dict with every key from :data:`VOLUME_CONFIG_DEFAULTS`
    present. ``roi_threshold`` is filled from the projection's value when
    unset. Raises :class:`ConfigurationError` on unknown keys or bad values.
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
    _check_positive_float(cfg, 'roi_threshold')
    _check_positive_float(cfg, 'sigma_frac')
    _check_positive_float(cfg, 'partner_level_window', allow_none=True)
    _check_positive_int(cfg, 'n_walkers', minimum=2)
    _check_positive_int(cfg, 'n_steps')
    _check_positive_int(cfg, 'eval_budget', allow_none=True)

    if not isinstance(cfg['warm_start'], bool):
        raise ConfigurationError(
            "volume_sampling['warm_start'] must be a bool",
            parameter="volume_sampling.warm_start", value=cfg['warm_start'],
        )
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


def volume_band(config, global_max):
    """The stage's logL lower edge and prefilter threshold for a config.

    Returns ``(band_lo, prefilter_delta)``: a point is in-band iff
    ``logL >= band_lo``. The band is one-sided (no upper edge), and the
    envelope prefilter uses the same threshold (profile values upper-bound
    logL and so can never exclude a point from above).
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


# --------------------------------------------------------------------------- #
# Ensemble primitives
# --------------------------------------------------------------------------- #
def assign_levels(n_walkers, global_max, roi_threshold):
    """Home levels for the ensemble: centres uniform in ΔlnL across the band.

    Walker ``k`` gets ``global_max - (k + 0.5)/n_walkers * roi_threshold``, so
    the levels tile ``[global_max - roi_threshold, global_max]`` evenly.
    """
    dlnl = (np.arange(n_walkers) + 0.5) / n_walkers * roi_threshold
    return global_max - dlnl


def umbrella_logpi(logl, level, sigma):
    """Unnormalised log umbrella density ``-(logL - level)^2 / (2 sigma^2)``.

    Non-finite ``logL`` (a failed evaluation) maps to ``-inf`` (always
    rejected). Works on scalars or arrays.
    """
    logl = np.asarray(logl, dtype=float)
    out = -0.5 * ((logl - level) / sigma) ** 2
    return np.where(np.isfinite(logl), out, -np.inf)


def draw_stretch(rng, size, a=STRETCH_A):
    """Draw the stretch factor ``Z ~ g(z) ∝ 1/sqrt(z)`` on ``[1/a, a]``."""
    u = rng.uniform(size=size)
    return ((a - 1.0) * u + 1.0) ** 2 / a


def draw_envelope_seeds(envelope, bounds, n_seeds, threshold_delta,
                        seed=None, draw_cap=None):
    """Scrambled-Sobol points inside the projection envelope (walker seeds).

    Draws Sobol points over the bounds box in power-of-2 batches and keeps
    those passing ``envelope.test(..., threshold_delta)`` until ``n_seeds``
    are kept or ``draw_cap`` draws are exhausted (default
    ``DEFAULT_DRAW_CAP_FACTOR * n_seeds``; a warning is logged if hit).
    Returns an ``(n_kept, n_dims)`` array (``n_kept`` may be < ``n_seeds``).
    """
    logger = get_logger()
    bounds = np.asarray(bounds, dtype=float)
    n_dims = len(bounds)
    if draw_cap is None:
        draw_cap = DEFAULT_DRAW_CAP_FACTOR * n_seeds
    lo, hi = bounds[:, 0], bounds[:, 1]
    sobol = qmc.Sobol(d=n_dims, scramble=True, seed=seed)

    kept = []
    n_kept = n_draws = 0
    while n_kept < n_seeds and n_draws < draw_cap:
        batch = qmc.scale(sobol.random(SEED_DRAW_BATCH), lo, hi)
        n_draws += len(batch)
        passed = batch[envelope.test(batch, threshold_delta)]
        take = passed[:n_seeds - n_kept]
        if len(take):
            kept.append(take)
            n_kept += len(take)

    seeds = np.vstack(kept) if kept else np.empty((0, n_dims))
    if n_kept < n_seeds:
        logger.warning(
            f"Volume sampling: seed draw cap hit after {n_draws} draws with "
            f"{n_kept}/{n_seeds} seeds kept. The prefilter region is very "
            f"small relative to the bounds box."
        )
    return seeds


def warm_start_positions(levels, sigma, scan_params, scan_logls,
                         fallback_seeds, rng):
    """Start each walker near its home level using existing scan samples.

    For each walker, pick a random scan sample whose logL is within ``sigma``
    of the walker's level (so high-level walkers start near the peak the scan
    already found); where the scan has no nearby sample, use the next
    envelope ``fallback_seeds`` point. Returns an ``(n_walkers, n_dims)``
    array.
    """
    n = len(levels)
    n_dims = fallback_seeds.shape[1]
    out = np.empty((n, n_dims))
    finite = np.isfinite(scan_logls)
    s_p, s_l = scan_params[finite], scan_logls[finite]
    fb = 0
    for k in range(n):
        cand = np.flatnonzero(np.abs(s_l - levels[k]) <= sigma)
        if len(cand):
            out[k] = s_p[rng.choice(cand)]
        elif fb < len(fallback_seeds):
            out[k] = fallback_seeds[fb]
            fb += 1
        else:
            out[k] = fallback_seeds[rng.integers(len(fallback_seeds))]
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
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


def lnl_histogram(logls, band_lo, global_max, n_bins=20):
    """Histogram of in-band logL values (to check the achieved lnL spread)."""
    edges = np.linspace(band_lo, global_max, n_bins + 1)
    counts, _ = np.histogram(logls, bins=edges)
    return {'bin_edges': edges.tolist(), 'counts': counts.tolist()}


def write_volume_output(result, config):
    """Write the in-band sample file and the JSON summary.

    ``result`` must have ``samples`` (an ``(M, n_dims + 1)`` array of in-band
    ``[params..., logL]`` rows), ``band_lo_final``, ``global_max`` and
    ``stats``. The sample file (``config['output_file']``, format from the
    extension, replaced if present) is skipped when there are no rows. Returns
    ``(output_path_or_None, summary_path)`` and annotates ``result``.
    """
    import json

    from .sample_io import write_samples

    logger = get_logger()
    samples = result['samples']
    output_path = config['output_file']
    if len(samples):
        if os.path.exists(output_path):
            logger.warning(
                f"Volume sampling: replacing existing file '{output_path}'.")
        write_samples(samples, output_path, overwrite=True)
        logger.info(f"Volume sampling: wrote {len(samples)} in-band samples "
                    f"to '{output_path}'.")
    else:
        output_path = None
        logger.info("Volume sampling: no in-band samples; no file written.")

    summary_path = config['summary_file'] or default_summary_file(
        config['output_file'])
    summary = {
        'band_lo_final': result['band_lo_final'],
        'global_max': result['global_max'],
        'config': config,
        'stats': result['stats'],
        'output_file': output_path,
        'n_samples': len(samples),
    }
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(_json_safe(summary), f, indent=2)
    logger.info(f"Volume sampling: wrote summary to '{summary_path}'.")

    result['output_file'] = output_path
    result['summary_file'] = summary_path
    return output_path, summary_path
