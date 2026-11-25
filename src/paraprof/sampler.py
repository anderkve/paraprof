"""Profile Likelihood Projector with Grid-Based Optimization."""
import os
import math
import collections
import functools
import numpy as np
import itertools
from scipy.stats.qmc import LatinHypercube as LHS
from .logger import get_logger
from .exceptions import (
    InvalidBoundsError, InvalidProjectionError, ConfigurationError,
)
from .sample_io import create_sample_writer, read_samples
from .phases import PHASE_UNKNOWN
from .volume import normalize_volume_config
from .jobs.lbfgsb_job import LBFGSBJob
from .jobs.activation_job import ActivationJob
from .jobs.de_job import DEGridPointJob


# n_initial_optimizations default: min(MAX, MULTIPLIER * n_dims). A ceiling;
# the Bayesian stopping rule sets the actual number of starts.
DEFAULT_BASIN_CAP_MULTIPLIER = 50
DEFAULT_BASIN_CAP_MAX = 400

# Basin detection for the initial-optimization stage: online single-linkage
# clustering of converged optima feeds a Boender-Rinnooy Kan Bayesian stopping
# rule over ROI optima.
DEFAULT_BASIN_BATCH_SIZE = None             # None -> FD-aware auto (see resolve_initial_opt_batch_size)
# Not a user knob: it tracks internal scales (bounds normalization, L-BFGS-B
# tolerance), not the target. Larger values merge distinct optima.
DEFAULT_BASIN_MERGE_TOL = 0.02              # RMS bounds-normalized param distance to merge optima
DEFAULT_BASIN_UNDISCOVERED_THRESHOLD = 0.5  # stop when E[undiscovered ROI optima] < this
DEFAULT_BASIN_MIN_STARTS_MULTIPLIER = 3     # min starts before the rule applies = mult * n_dims
DEFAULT_BASIN_MIN_STARTS_FLOOR = 10         # ... floored at this many

# Global solution pool size: clip(n_dims * PER_DIM, POOL_SIZE, POOL_MAX).
# The ceiling caps master memory and the proximity-cache rebuild cost.
DEFAULT_GLOBAL_POOL_SIZE = 10000
DEFAULT_GLOBAL_POOL_PER_DIM = 2500
DEFAULT_GLOBAL_POOL_MAX = 100000

MEMORY_SIZE_MULTIPLIER = 25  # DE memory size = max_grid_size * multiplier

# DE per-cell convergence cutoff: mean logL improvement per generation over
# `de.convergence_window` generations.
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6

# DE mutation settings. Not user-tunable.
DEFAULT_DE_MUTATION_STRATEGY = 'current-to-pbest/1'
DEFAULT_DE_PBEST_FRACTION = 0.1
DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY = 0.5
DEFAULT_DE_CONVERGENCE_WINDOW = 3
DEFAULT_DE_NUM_GENERATIONS = 100000

# allow_early_DE_exit: when a fresh cell's neighbours agree on the profiled
# argmax and the neighbour warm start was the best activation seed, the cell
# runs one DE generation and goes to the L-BFGS-B polish, instead of waiting out
# `de.convergence_window`. A cell that improves in that generation keeps
# evolving. Disable it when the profiled parameters have a multimodal inner
# problem, where one generation under-explores the modes.
DEFAULT_DE_ALLOW_EARLY_DE_EXIT = True
# Reduced per-cell convergence window applied to skip-DE-eligible cells.
SKIP_DE_WINDOW = 1
# Max per-profiled-dim deviation of the neighbours' argmax from their mean (as a
# fraction of the profiled-dim bounds extent) for a cell to be skip-DE-eligible.
SKIP_DE_PHI_SPREAD = 0.05

DEFAULT_LBFGSB_FTOL = 1e-9
# Finite-difference step of the L-BFGS-B gradient: a fraction of each
# parameter's bound extent, or 'auto' (from a noise estimate of the target,
# see jobs/noise_job.py).
DEFAULT_LBFGSB_FD_STEP = 'auto'
# Bounds on the automatic step, and the step used when the estimate fails.
LBFGSB_FD_STEP_MIN = 1e-8
LBFGSB_FD_STEP_MAX = 1e-3
LBFGSB_FD_STEP_FALLBACK = 1e-6
# Accepted curvature scales remembered per kind of run (global start / cell).
LBFGSB_SCALE_MEMORY = 64
# Base points of the noise estimate.
NOISE_BASE_POINTS = 3
# Base points at which the target is compared at the two bounds of a periodic
# parameter (two evaluations per base point and periodic parameter).
PERIODIC_CHECK_BASE_POINTS = 3
# Bounds on the step of the second differences (fraction of the bound extent).
LBFGSB_HESSIAN_STEP_MIN = 1e-4
LBFGSB_HESSIAN_STEP_MAX = 0.05

# Patching: two neighbouring grid points hold different optima if a profiled
# parameter differs between them by more than median + k * MAD of its
# differences over all neighbour pairs, and by more than the floor (a fraction
# of the bound extent).
PATCHING_JUMP_K = 3.0
PATCHING_JUMP_FLOOR = 1e-3

# Suspect-cell recheck defaults
DEFAULT_SUSPECT_RECHECK_ENABLED = True
DEFAULT_SUSPECT_MAX_WAVES = 10
DEFAULT_SUSPECT_PARAM_K = 3.0          # MAD multiplier on profiled-param discontinuity
DEFAULT_SUSPECT_MAX_FRACTION = 0.25    # safety cap: max fraction of ROI cells per wave
DEFAULT_SUSPECT_SEEDS_K_RING = 3       # Chebyshev radius for extended-neighbour seeds
DEFAULT_SUSPECT_SEEDS_FROM_POOL = 3
DEFAULT_SUSPECT_POLISH_THRESHOLD = 1e-3  # min improvement over current to trigger LBFGSB

# Activation-population mix ratios (neighbours / global pool / random LHS).
# Not user-tunable.
DEFAULT_ACTIVATION_MIX_RATIOS = {
    'neighbors': 0.5,
    'global':    0.25,
    'random':    0.25,
}

DEFAULT_CLUSTERING_EPS_MULTIPLIER = 3.0
DEFAULT_CLUSTERING_PROJECTION_WEIGHT = 1.0


@functools.lru_cache(maxsize=None)
def _offsets_within(ndim, depth):
    """Non-zero grid offsets up to Chebyshev distance ``depth``, nearest first."""
    offsets = [o for o in itertools.product(range(-depth, depth + 1), repeat=ndim) if any(o)]
    return sorted(offsets, key=lambda o: max(abs(v) for v in o))


class ProfileProjector:
    """
    State and configuration of a grid-based profile likelihood scan.

    Cells are optimised with differential evolution (DE) or L-BFGS-B. The
    execution logic is in the Job classes and master_main.
    """
    def __init__(self,
                 target_func,
                 bounds,
                 projections,
                 # Core tuning parameters (commonly adjusted)
                 roi_threshold=4.0,
                 pop_per_grid_point=3,
                 max_patching_waves=None,
                 lbfgsb_max_iter=50,
                 lbfgsb_polish=True,
                 n_initial_optimizations=None,
                 n_optima=None,
                 initial_points=None,
                 refine_initial_points=False,
                 # Optional user-supplied gradient
                 grad_func=None,
                 # Feature toggles
                 use_clustering=True,
                 refinement_direct_eval=False,
                 # I/O
                 samples_output_file=None,
                 warm_start_file=None,
                 # Post-projection volume-sampling stage (optional)
                 volume_sampling=None,
                 # Parameter naming (optional, enables string dims in projections)
                 parameter_names=None,
                 # Advanced configuration (optional)
                 advanced_config=None,
                 # Parameters whose bounds span one period (optional)
                 periodic=None):
        """
        Parameters
        ----------
        target_func : callable
            The target function to maximize (e.g., log-likelihood)
        bounds : array-like, shape (n_dims, 2)
            Parameter bounds for each dimension
        projections : list of dict
            List of projection configurations, each with 'dims' and 'grid_points'
        grad_func : callable, optional
            Gradient of ``target_func``, the function being maximised
            (default: None). Used only by L-BFGS-B. Two return formats:

            - length-``n_dims`` array of floats; entries that are ``NaN``,
              ``+inf`` or ``-inf`` are treated as "not provided" and filled
              in by finite differences.
            - ``{dim_index: value}`` dict of known components; any dim not
              in the dict is filled in by finite differences.

            Provided components replace finite-difference evaluations; the
            saved calls are counted in
            ``sampler.target_calls_saved_by_user_gradient``.

        Core Tuning Parameters
        ----------------------
        roi_threshold : float, optional
            Depth of the region of interest below the global maximum, in units
            of the target (default: 4.0). For a log-likelihood this is half a
            chi-squared difference: 3 sigma for two degrees of freedom is 5.92.
        pop_per_grid_point : int, optional
            DE population size per grid point (default: 3). Typical values: 1-5.
        max_patching_waves : int, optional
            Maximum number of patching waves. A wave moves the border between
            two optima by one grid point. Default: the largest number of grid
            points along a projection dimension, so that patching ends when a
            wave updates nothing.
        lbfgsb_max_iter : int, optional
            Maximum L-BFGS-B iterations per optimization (default: 50).
        lbfgsb_polish : bool, optional
            Polish DE results with L-BFGS-B (default: True).
        n_initial_optimizations : int, optional
            Maximum number of global L-BFGS-B optimizations that find the
            initial maxima (default: None, which gives min(400, 50 * n_dims)).
            The Bayesian stopping rule usually ends the stage earlier, so set
            it generously. See ``advanced_config['basin_detection']``.
        n_optima : int or dict, optional
            Known number of optima of the target over the full parameter
            space (default: None). An ``int`` (exact) or
            ``{'min': int, 'max': int}``. The initial multistart stops once
            ``max`` distinct optima are found, skipping the
            ``basin_detection.min_starts`` floor (``n_optima=1`` stops after the
            first converged start), and runs until ``min`` are found. If the
            true count exceeds ``max``, the stage can stop before finding the
            global maximum.
        initial_points : array-like, shape (n_points, n_dims), optional
            Points in the full parameter space (default: None), e.g.
            ``[[3.0, 0.0, -3.0, 0.0]]`` in 4D. Each activates its nearest grid
            point. The points replace the global L-BFGS-B starts. They are
            evaluated, not optimised: unless a grid point later exceeds it, the
            best of them is the maximum the region of interest is measured from.
        refine_initial_points : bool, optional
            Run one L-BFGS-B optimisation over all parameters from every
            entry of ``initial_points`` before the grid points are activated
            (default: False). The endpoints join the initial maxima. Use it
            when the points are near the maximum but not known to be at it,
            e.g. the best fit of an earlier run with a different target.

        Feature Toggles
        ---------------
        use_clustering : bool, optional
            Cluster solutions into modes during grid refinement (default: True).
        refinement_direct_eval : bool, optional
            During refinement, evaluate interpolated points instead of
            optimising them (default: False).

        I/O
        ---
        samples_output_file : str, optional
            Path to save all evaluated points (default: None). Each row is
            ``[params..., logL, phase]``, where ``phase`` tags the stage that
            produced the point (see :mod:`paraprof.phases`). ``.h5``/``.hdf5``
            writes HDF5 (needs ``h5py``); anything else writes CSV.
        warm_start_file : str, optional
            Sample file from a previous run (default: None). When set and
            warm start is allowed for the projection, ``initial_maxima`` are
            read from it and the global L-BFGS-B starts are skipped. It may be
            the same path as ``samples_output_file``.
        volume_sampling : dict, optional
            Configuration of the volume-sampling stage after the projections
            (default: None, disabled). It collects a stratified sample inside
            ``{logL > global_max - roi_threshold}``, where the stage's
            ``roi_threshold`` defaults to the projection's. See
            ``volume.VOLUME_CONFIG_DEFAULTS`` for the keys.

        Parameter Naming
        ----------------
        parameter_names : list of str, optional
            One name per parameter (default: None). Projection ``dims`` may
            then hold names, indices, or a mix.

        Periodic Parameters
        -------------------
        periodic : list of int or str, optional
            Parameters whose bounds span exactly one period, so the lower and
            the upper bound are the same point (default: None). Entries are
            parameter indices, or names if ``parameter_names`` is given.

            As a projection dimension, the first and the last grid point of
            such a parameter are neighbours of each other and of the grid
            points next to either, so the region of interest is followed
            through the bound. Both are still evaluated.

            As a profiled parameter, a value that leaves the bounds comes
            back in on the other side instead of stopping at the bound, and
            differences between two values are taken the short way round.

            Before the scan the target is evaluated at the two bounds of
            each periodic parameter and a difference is logged as a warning.
            With bounds that do not span one period the scan still returns
            true target values, but cells and optimisers are continued from
            a point that is not the same point.

        Advanced Configuration
        ----------------------
        advanced_config : dict, optional
            Expert settings. Structure:
            {
                'memory_size': int,                # Default: max(grid_sizes) * 25
                'convergence_threshold': float,    # Default: 1e-6

                'de': {
                    'convergence_window': int,     # Default: 3
                    'num_generations': int,        # Default: 100000
                    'max_num_to_evolve': int,      # Default: None (all grid points)
                    'allow_early_DE_exit': bool,        # Default: True
                },

                'lbfgsb': {
                    'ftol': float,                 # Default: 1e-9
                    'gradient_method': str,        # Default: 'forward'
                    'fd_step': 'auto'|float,       # Default: 'auto'
                    'activation_depth': int,       # Default: 1
                },

                'clustering': {
                    'method': str,                 # Default: 'dbscan'
                    'eps': float,                  # Default: None (auto-estimated)
                    'min_samples': int,            # Default: None (auto: max(2, n_prof_dims))
                    'eps_multiplier': float,       # Default: 3.0
                    'projection_weight': float,    # Default: 1.0
                },

                'cross_projection': {
                    'proximity_warm_start': bool,        # Default: True
                    'pool_seeded_initial_maxima': bool,  # Default: True
                    'eager_start': bool,                 # Default: True
                },

                'suspect_recheck': {
                    'enabled': bool,                # Default: True
                    'max_waves': int,               # Default: 10
                    'param_k': float,               # Default: 3.0
                    'max_fraction': float,          # Default: 0.25
                    'seeds_k_ring': int,            # Default: 3
                    'seeds_from_pool': int,         # Default: 3
                    'polish_threshold': float,      # Default: 1e-3
                },

                'basin_detection': {
                    'batch_size': int,              # Default: None (FD-aware auto)
                    'undiscovered_threshold': float,# Default: 0.5 (0 disables early stop)
                    'min_starts': int,              # Default: None (auto)
                },
            }

        The ``lbfgsb`` sub-dict controls the L-BFGS-B runs (global starts,
        cell optimisations, polish).

        ``fd_step`` is the finite-difference step of the gradient, as a
        fraction of each parameter's bound extent. ``'auto'`` estimates the
        numerical noise of the target before the first optimisation (about
        60 evaluations, logged) and sets the step to about
        ``sqrt(noise / curvature)``. A run stops once an accepted step
        changes the target by less than ``max(ftol, 4 * noise)``.

        In an ``optimization_method='lbfgsb'`` projection a cell is optimised
        as soon as its activation returns and activates its neighbours as
        soon as it finishes inside the region of interest. While workers are
        idle, a cell whose activation samples put it inside the region of
        interest activates its neighbours before its own optimisation.
        ``activation_depth`` is how many cells ahead it then activates (1:
        its neighbours). Cells further out get only idle workers and are
        dropped unless they or a neighbour turn out to lie inside the region
        of interest, so the result holds the same cells as with depth 1;
        their evaluations are lost.

        The ``basin_detection`` sub-dict controls the initial-optimization
        stage: a rolling Latin-hypercube multistart whose converged optima are
        clustered into basins. A Boender-Rinnooy Kan Bayesian stopping rule
        over the optima inside the region of interest ends the stage once the
        expected number of undiscovered ones falls below
        ``undiscovered_threshold`` (after at least ``min_starts`` starts),
        aborting running optimizations. At the ``n_initial_optimizations``
        cap, running optimizations finish instead. ``batch_size`` sets how
        many run concurrently (see :meth:`resolve_initial_opt_batch_size`).
        ``undiscovered_threshold=0`` disables early stopping; set
        ``n_initial_optimizations`` explicitly then.

        The ``cross_projection`` sub-dict toggles the hooks that pass
        results from one projection to the next. All default to ``True``.

        ``eager_start`` (default ``True``) starts the next projection on
        workers the current one leaves idle, once a tenth of the workers is
        idle and the projection started before it has finished its first
        cell optimisation (or DE generation). It needs
        ``pool_seeded_initial_maxima``. Earlier projections are served
        first. Projections with ``grid_refinement_factor`` or without
        profiled parameters are not overlapped. A projection that starts
        early is seeded from a smaller pool than in a sequential run. On a
        target with several optima single cells can then end at a different
        optimum, as they do between two random seeds. Set it to ``False``
        to run the projections one after another.

        While projections overlap, log lines are prefixed with the number
        of the projection they belong to (``[P2]``). The call counts in the
        result of a projection are those of the whole scan at the time that
        projection ended. The projection listed last need not end last, so
        the total of the scan is the sampler's ``target_calls``.

        The DE mutation settings, global pool size, patching jump threshold,
        activation mix ratios and basin ``merge_tol`` are module-level
        constants, not options.
        """
        # --- Input validation ---

        # Validate target function
        if not callable(target_func):
            raise ConfigurationError("target_func must be callable", parameter="target_func", value=target_func)

        # Validate optional user gradient function
        if grad_func is not None and not callable(grad_func):
            raise ConfigurationError(
                "grad_func must be callable or None",
                parameter="grad_func", value=grad_func,
            )

        # Validate and set bounds
        self.bounds = np.array(bounds)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise InvalidBoundsError(
                f"bounds must be shape (n_dims, 2), got {self.bounds.shape}",
                bounds=bounds
            )
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise InvalidBoundsError(
                "Lower bounds must be < upper bounds for all dimensions",
                bounds=bounds
            )

        self.target_func = target_func
        self.grad_func = grad_func
        self.dims = len(self.bounds)

        # Validate parameter_names (optional) and store as a name->index map
        if parameter_names is not None:
            if (not isinstance(parameter_names, (list, tuple))
                    or len(parameter_names) != self.dims
                    or not all(isinstance(n, str) for n in parameter_names)):
                raise ConfigurationError(
                    f"parameter_names must be a list of {self.dims} strings, "
                    f"one per parameter dimension",
                    parameter="parameter_names",
                    value=parameter_names,
                )
            if len(set(parameter_names)) != len(parameter_names):
                raise ConfigurationError(
                    "parameter_names must not contain duplicates",
                    parameter="parameter_names",
                    value=parameter_names,
                )
            self.parameter_names = list(parameter_names)
            self._name_to_dim = {n: i for i, n in enumerate(self.parameter_names)}
        else:
            self.parameter_names = None
            self._name_to_dim = None

        # Periodic parameters: sorted indices and a mask over all parameters.
        self.periodic_dims = self._resolve_periodic(periodic)
        self._periodic_mask = np.zeros(self.dims, dtype=bool)
        self._periodic_mask[self.periodic_dims] = True
        self._has_periodic = bool(self.periodic_dims)
        self._periodic_checked = False       # see PeriodicCheckJob

        # Validate projections
        if not isinstance(projections, list) or len(projections) == 0:
            raise InvalidProjectionError("projections must be a non-empty list")

        # Rewrite string dims to indices in place.
        for i, proj in enumerate(projections):
            if isinstance(proj, dict) and 'dims' in proj:
                resolved = self._resolve_dims(proj['dims'], context=f"projection {i}")
                if resolved is not None:
                    proj['dims'] = resolved

        for i, proj in enumerate(projections):
            if not isinstance(proj, dict):
                raise InvalidProjectionError(f"Projection {i} must be a dictionary", projection=proj)
            if 'dims' not in proj:
                raise InvalidProjectionError(f"Projection {i} missing 'dims' key", projection=proj)
            if 'grid_points' not in proj:
                raise InvalidProjectionError(f"Projection {i} missing 'grid_points' key", projection=proj)

            dims = proj['dims']
            grid_points = proj['grid_points']

            if not isinstance(dims, (list, tuple)) or len(dims) == 0:
                raise InvalidProjectionError(
                    f"Projection {i} 'dims' must be a non-empty list/tuple",
                    projection=proj
                )
            if not isinstance(grid_points, (list, tuple)) or len(grid_points) != len(dims):
                raise InvalidProjectionError(
                    f"Projection {i} 'grid_points' length must match 'dims' length",
                    projection=proj
                )

            # Check dimension indices are valid
            for dim in dims:
                if not isinstance(dim, int) or dim < 0 or dim >= self.dims:
                    raise InvalidProjectionError(
                        f"Projection {i} has invalid dimension index {dim} (must be 0-{self.dims-1})",
                        projection=proj
                    )

            # Check grid points are positive integers
            for gp in grid_points:
                if not isinstance(gp, int) or gp <= 0:
                    raise InvalidProjectionError(
                        f"Projection {i} grid_points must be positive integers, got {gp}",
                        projection=proj
                    )

        self.projections = projections

        # Validate core parameters
        if not isinstance(pop_per_grid_point, int) or pop_per_grid_point < 1:
            raise ConfigurationError(
                "pop_per_grid_point must be a positive integer",
                parameter="pop_per_grid_point",
                value=pop_per_grid_point
            )

        if not isinstance(roi_threshold, (int, float)) or roi_threshold <= 0:
            raise ConfigurationError(
                "roi_threshold must be a positive number",
                parameter="roi_threshold",
                value=roi_threshold
            )

        # Validate and process initial_points
        if initial_points is not None:
            initial_points = np.array(initial_points)
            if initial_points.ndim == 1:
                # Single point provided - reshape to (1, n_dims)
                initial_points = initial_points.reshape(1, -1)
            if initial_points.ndim != 2 or initial_points.shape[1] != self.dims:
                raise ConfigurationError(
                    f"initial_points must have shape (n_points, {self.dims}), got {initial_points.shape}",
                    parameter="initial_points",
                    value=initial_points
                )
            # Check that points are within bounds
            for i, point in enumerate(initial_points):
                for j, (val, (lb, ub)) in enumerate(zip(point, self.bounds)):
                    if not (lb <= val <= ub):
                        raise ConfigurationError(
                            f"initial_points[{i}][{j}] = {val} is outside bounds [{lb}, {ub}]",
                            parameter="initial_points",
                            value=initial_points
                        )

        # --- Defaults, overridden by advanced_config ---
        max_grid_size = max(max(proj['grid_points']) for proj in projections)

        config = {
            'memory_size': max_grid_size * MEMORY_SIZE_MULTIPLIER,
            'convergence_threshold': DEFAULT_CONVERGENCE_THRESHOLD,

            'de': {
                'convergence_window': DEFAULT_DE_CONVERGENCE_WINDOW,
                'num_generations': DEFAULT_DE_NUM_GENERATIONS,
                'max_num_to_evolve': None,
                'allow_early_DE_exit': DEFAULT_DE_ALLOW_EARLY_DE_EXIT,
            },

            'lbfgsb': {
                'ftol': DEFAULT_LBFGSB_FTOL,
                'gradient_method': 'forward',
                'fd_step': DEFAULT_LBFGSB_FD_STEP,
                'activation_depth': 1,
            },

            'clustering': {
                'method': 'dbscan',
                'eps': None,
                'min_samples': None,
                'eps_multiplier': DEFAULT_CLUSTERING_EPS_MULTIPLIER,
                'projection_weight': DEFAULT_CLUSTERING_PROJECTION_WEIGHT,
            },

            'cross_projection': {
                'proximity_warm_start': True,
                'pool_seeded_initial_maxima': True,
                'eager_start': True,
            },

            'suspect_recheck': {
                'enabled': DEFAULT_SUSPECT_RECHECK_ENABLED,
                'max_waves': DEFAULT_SUSPECT_MAX_WAVES,
                'param_k': DEFAULT_SUSPECT_PARAM_K,
                'max_fraction': DEFAULT_SUSPECT_MAX_FRACTION,
                'seeds_k_ring': DEFAULT_SUSPECT_SEEDS_K_RING,
                'seeds_from_pool': DEFAULT_SUSPECT_SEEDS_FROM_POOL,
                'polish_threshold': DEFAULT_SUSPECT_POLISH_THRESHOLD,
            },

            'basin_detection': {
                'batch_size': DEFAULT_BASIN_BATCH_SIZE,
                'undiscovered_threshold': DEFAULT_BASIN_UNDISCOVERED_THRESHOLD,
                'min_starts': None,  # None -> max(floor, mult*n_dims), capped at n_initial_optimizations
            },
        }

        # Merge with advanced_config if provided
        if advanced_config:
            self._deep_update(config, advanced_config)

        if n_initial_optimizations is None:
            n_initial_optimizations = min(
                DEFAULT_BASIN_CAP_MAX,
                DEFAULT_BASIN_CAP_MULTIPLIER * self.dims,
            )

        # --- Store configuration as instance variables ---
        self.pop_per_grid_point = pop_per_grid_point
        self.roi_threshold = roi_threshold

        # Validated here; the stage runs after the projections (volume.py).
        if volume_sampling is not None:
            self.volume_sampling_config = normalize_volume_config(
                volume_sampling, self.roi_threshold)
        else:
            self.volume_sampling_config = None
        # Result dict of the last run_volume_sampling call (None until then).
        self.volume_stage_result = None
        self.max_patching_waves = max_patching_waves
        self.lbfgsb_max_iter = lbfgsb_max_iter
        self.n_initial_optimizations = n_initial_optimizations
        self.initial_points = initial_points
        self.refine_initial_points = bool(refine_initial_points)
        self.refinement_direct_eval = refinement_direct_eval
        self.use_clustering = use_clustering

        # Store advanced config values
        self.memory_size = config['memory_size']
        self.convergence_threshold = config['convergence_threshold']

        # DE configuration
        self.convergence_window = config['de']['convergence_window']
        self.de_num_generations = config['de']['num_generations']
        self.de_max_num_to_evolve = config['de']['max_num_to_evolve']
        self.de_allow_early_DE_exit = config['de']['allow_early_DE_exit']

        # L-BFGS-B configuration
        lb = config['lbfgsb']
        self.lbfgsb_ftol = lb['ftol']
        self.lbfgsb_gradient_method = lb['gradient_method']
        fd_step = lb['fd_step']
        if not (fd_step == 'auto'
                or (isinstance(fd_step, (int, float, np.floating)) and not isinstance(fd_step, bool)
                    and 0.0 < float(fd_step) < 1.0)):
            raise ConfigurationError(
                "lbfgsb fd_step must be 'auto' or a fraction of the bound extent in (0, 1)",
                parameter="advanced_config['lbfgsb']['fd_step']", value=fd_step,
            )
        self.lbfgsb_fd_step = fd_step
        # The step the jobs use. With 'auto' it is the fallback until the
        # noise-estimate stage sets it.
        self.lbfgsb_fd_step_value = (LBFGSB_FD_STEP_FALLBACK if fd_step == 'auto'
                                     else float(fd_step))
        # Noise of the target (rms, same units as the target), from the
        # noise-estimate stage; 0 means unknown.
        self.lbfgsb_noise = 0.0
        self._noise_estimated = False
        depth = lb['activation_depth']
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 1:
            raise ConfigurationError(
                "lbfgsb activation_depth must be an integer >= 1",
                parameter="advanced_config['lbfgsb']['activation_depth']", value=depth)
        self.lbfgsb_activation_depth = int(depth)
        # Curvature scales (gamma = s.y / y.y) of finished global starts (see
        # lbfgsb_scale_hint). Those of cell runs are per projection.
        self._lbfgsb_global_scales = collections.deque(maxlen=LBFGSB_SCALE_MEMORY)

        # Fixed settings, from module-level constants.
        self.global_pool_size = min(
            DEFAULT_GLOBAL_POOL_MAX,
            max(DEFAULT_GLOBAL_POOL_SIZE,
                self.dims * DEFAULT_GLOBAL_POOL_PER_DIM),
        )
        self.mutation_strategy = DEFAULT_DE_MUTATION_STRATEGY
        self.pbest_fraction = DEFAULT_DE_PBEST_FRACTION
        self.neighbor_pull_probability = DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY
        self.activation_mix_ratios = dict(DEFAULT_ACTIVATION_MIX_RATIOS)

        # Clustering configuration
        self.clustering_method = config['clustering']['method']
        self.clustering_eps = config['clustering']['eps']
        self.clustering_min_samples = config['clustering']['min_samples']
        self.clustering_eps_multiplier = config['clustering']['eps_multiplier']
        self.clustering_projection_weight = config['clustering']['projection_weight']

        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        self.warm_start_file = warm_start_file
        # Always initialize the buffer
        self.samples_buffer = []
        self.sample_buffer_size = 1000

        # The writer picks the format from the extension and opens the file on
        # the first flush.
        if self.samples_output_file:
            # Create the directory up front so the first flush succeeds.
            samples_dir = os.path.dirname(self.samples_output_file)
            if samples_dir:
                os.makedirs(samples_dir, exist_ok=True)
            self._sample_writer = create_sample_writer(self.samples_output_file)
            self._file_closed = False
        else:
            self._sample_writer = None
            self._file_closed = True

        # --- Persistent State (across projections) ---
        self.target_calls = 0
        self.target_call_errors = 0
        # Cells that took the allow_early_DE_exit path (all projections).
        self.de_cells_skipped = 0
        # grad_func counters (all projections).
        self.target_calls_saved_by_user_gradient = 0
        self.user_gradient_errors = 0
        self.global_max_target_val = -np.inf
        self.global_solution_pool = []  # Min-heap of (fitness, count, entry) tuples
        self.global_pool_counter = 0  # Unique counter for tiebreaking in heap

        # Cross-projection hooks, all using global_solution_pool:
        # 1. proximity_warm_start: a cell's activation population swaps one
        #    random LHS seed for the best past evaluation closest to the cell
        #    in the projection dims.
        # 2. pool_seeded_initial_maxima: projections after the first seed
        #    `initial_maxima` from the pool and skip the global L-BFGS-B starts.
        self.proximity_warm_start = config['cross_projection']['proximity_warm_start']
        self.pool_seeded_initial_maxima = config['cross_projection']['pool_seeded_initial_maxima']
        # 3. eager_start: the next projection starts on workers the current
        #    one leaves idle (run_all_projections, master_main).
        self.eager_projection_start = bool(config['cross_projection']['eager_start'])

        # Suspect-cell recheck: after patching, recheck cells whose profiled
        # parameters are far from the median of their neighbours, with seeds
        # from further away and from the pool.
        sc = config['suspect_recheck']
        self.suspect_recheck_enabled = sc['enabled']
        self.max_suspect_waves = sc['max_waves']
        self.suspect_param_k = sc['param_k']
        self.suspect_max_fraction = sc['max_fraction']
        self.suspect_seeds_k_ring = sc['seeds_k_ring']
        self.suspect_seeds_from_pool = sc['seeds_from_pool']
        self.suspect_polish_threshold = sc['polish_threshold']

        # Basin-detection configuration for the initial-optimization stage.
        bd = config['basin_detection']
        self.basin_batch_size = bd['batch_size']
        self.basin_merge_tol = DEFAULT_BASIN_MERGE_TOL
        self.basin_undiscovered_threshold = bd['undiscovered_threshold']
        if bd['min_starts'] is None:
            self.basin_min_starts = min(
                self.n_initial_optimizations,
                max(DEFAULT_BASIN_MIN_STARTS_FLOOR,
                    DEFAULT_BASIN_MIN_STARTS_MULTIPLIER * self.dims),
            )
        else:
            self.basin_min_starts = int(bd['min_starts'])
        # n_optima as (min, max) bounds for basin_detection_should_stop.
        self.basin_min_optima, self.basin_max_optima = (
            self._parse_n_optima(n_optima)
        )
        # Pre-generated LHS start pool for rolling multistart (lazily filled).
        self._initial_opt_start_points = None
        self._initial_opt_lhs_idx = 0
        # (proj_coords, profiled_coords, extent) of the global pool, rebuilt
        # at most once per projection (see _sample_proximity_from_global_pool).
        self._proximity_pool_cache = None

        # --- Initial points tracking ---
        self._initial_points_evaluated = False
        self._initial_points_refined = False
        self._initial_point_values = None     # target at each initial point
        self._refine_best_before = None

        # --- Refinement State ---
        self.is_refinement_run = False
        self.grid_refinement_factor = None
        self.coarse_grid_solution = None
        self.refinement_interpolator = None
        self.cluster_labels = None
        self.cluster_info = None
        self.boundary_points = None

        # Per-projection state (reset on every new projection).
        self.projection_dims = None
        self.grid_points_per_dim = None
        self.initial_maxima = []
        # Distinct optima of the initial-optimization stage, as
        # {'point', 'point_norm', 'target_val', 'count'}; filled by
        # register_initial_optimum.
        self.initial_optima_registry = []
        self.population = {}  # {grid_idx: state_dict}
        self.activated_grid_indices = set()
        self.pending_activation_indices = set()
        self.ahead_cells = set()     # activated ahead, not confirmed (activation_depth > 1)
        self._patching_jump_thresholds = None   # set by the first patching wave
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.optimization_method = 'de'
        self.lbfgsb_polish = lbfgsb_polish
        self.patch_coarse_grid = True
        self.patch_refined_grid = False

        self.logger = get_logger()

        self._reset_for_new_projection(self.projections[0])

    def _resolve_dims(self, dims, context="projection"):
        """Translate a mixed list of ints/parameter names to integer indices.

        Returns None when no translation is needed (all entries already ints)
        so the caller can avoid mutating. Raises InvalidProjectionError on
        unknown names.
        """
        if not isinstance(dims, (list, tuple)):
            return None

        if not any(isinstance(d, str) for d in dims):
            return None

        if self._name_to_dim is None:
            raise InvalidProjectionError(
                f"{context} 'dims' contains parameter names but ProfileProjector "
                f"was constructed without parameter_names",
                projection={'dims': list(dims)},
            )

        resolved = []
        for d in dims:
            if isinstance(d, str):
                if d not in self._name_to_dim:
                    raise InvalidProjectionError(
                        f"{context} 'dims' references unknown parameter name "
                        f"{d!r}; known names: {self.parameter_names}",
                        projection={'dims': list(dims)},
                    )
                resolved.append(self._name_to_dim[d])
            else:
                resolved.append(d)
        return resolved

    def _resolve_periodic(self, periodic):
        """Sorted parameter indices for the ``periodic`` argument (indices or names)."""
        if periodic is None:
            return []
        if isinstance(periodic, (str, int, np.integer)) and not isinstance(periodic, bool):
            periodic = [periodic]
        if not isinstance(periodic, (list, tuple)):
            raise ConfigurationError(
                "periodic must be a list of parameter indices or names",
                parameter="periodic", value=periodic)
        dims = set()
        for entry in periodic:
            if isinstance(entry, str):
                if self._name_to_dim is None or entry not in self._name_to_dim:
                    raise ConfigurationError(
                        f"periodic names the unknown parameter {entry!r}"
                        + ("; names need parameter_names" if self._name_to_dim is None
                           else f"; known names: {self.parameter_names}"),
                        parameter="periodic", value=periodic)
                dims.add(self._name_to_dim[entry])
            elif (isinstance(entry, (int, np.integer)) and not isinstance(entry, bool)
                    and 0 <= entry < self.dims):
                dims.add(int(entry))
            else:
                raise ConfigurationError(
                    f"periodic entry {entry!r} is not a parameter index in "
                    f"0-{self.dims - 1} or a parameter name",
                    parameter="periodic", value=periodic)
        return sorted(dims)

    def _wrap_periodic(self, vec, dims):
        """``vec``, holding values of the parameters ``dims`` along its last
        axis, with the periodic ones mapped into their bounds. The input is
        returned as it is if none of them is periodic."""
        if not self._has_periodic:
            return vec
        dims = list(dims)
        mask = self._periodic_mask[dims]
        if not mask.any():
            return vec
        lo = self.bounds[dims, 0][mask]
        span = self.bounds[dims, 1][mask] - lo
        vec = np.array(vec, dtype=float)
        vec[..., mask] = lo + np.mod(vec[..., mask] - lo, span)
        return vec

    def _param_delta(self, a, b, dims):
        """``a - b`` for values of the parameters ``dims`` (last axis). For a
        periodic parameter the difference is taken the short way round."""
        delta = a - b
        if not self._has_periodic:
            return delta
        dims = list(dims)
        mask = self._periodic_mask[dims]
        if not mask.any():
            return delta
        span = self.bounds[dims, 1][mask] - self.bounds[dims, 0][mask]
        delta = np.array(delta, dtype=float)
        delta[..., mask] -= span * np.round(delta[..., mask] / span)
        return delta

    def _deep_update(self, base_dict, update_dict):
        """Recursively merge update_dict into base_dict in place."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    # Everything _reset_for_new_projection sets. With eager_start the master
    # swaps these in before it works on a projection.
    PROJECTION_STATE_ATTRS = (
        'projection_dims', 'grid_points_per_dim', 'optimization_method',
        'patch_coarse_grid', 'patch_refined_grid', 'profiled_dims', 'n_proj_dims',
        'n_prof_dims', 'direct_eval_mode', 'grid_shape', 'grid_axes',
        'profile_likelihood_grid', 'initial_maxima', 'initial_optima_registry',
        '_initial_opt_start_points', '_initial_opt_lhs_idx', 'population',
        'activated_grid_indices', 'pending_activation_indices', 'ahead_cells',
        'current_generation', 'memory_F', 'memory_CR', 'memory_idx',
        '_lbfgsb_cell_scales', 'lbfgsb_hessian_hint', '_neighbor_cache',
        '_proximity_pool_cache', '_profiled_bounds', '_projection_bounds',
        '_patching_jump_thresholds',
    )

    def capture_projection_state(self):
        """The state of the projection the sampler is working on, as a dict."""
        return {name: getattr(self, name) for name in self.PROJECTION_STATE_ATTRS}

    def restore_projection_state(self, state):
        """Make ``state`` (from capture_projection_state) the current projection."""
        for name in self.PROJECTION_STATE_ATTRS:
            setattr(self, name, state[name])

    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        # For projection dicts not seen at construction.
        if 'dims' in projection_config:
            resolved = self._resolve_dims(projection_config['dims'],
                                          context="projection")
            if resolved is not None:
                projection_config['dims'] = resolved

        self.logger.info("=" * 80)
        self.logger.info(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        self.logger.info("=" * 80)

        # The dims are kept in ascending order internally. Each grid size
        # stays with the dim it was given for.
        order = sorted(range(len(projection_config['dims'])),
                       key=lambda i: projection_config['dims'][i])
        self.projection_dims = [projection_config['dims'][i] for i in order]
        # Add +1 to grid points to include both endpoints in linspace
        grid_points = [projection_config['grid_points'][i] + 1 for i in order]
        self.grid_points_per_dim = grid_points

        if len(self.projection_dims) != len(self.grid_points_per_dim):
            raise ValueError("Length of projection_dims must match length of grid_points_per_dim.")
        if any(d >= self.dims for d in self.projection_dims):
            raise ValueError("projection_dims contains an index out of bounds.")

        # Read optimization method
        self.optimization_method = projection_config.get('optimization_method', 'de')

        # Validate optimization method
        valid_methods = ['de', 'lbfgsb']
        if self.optimization_method not in valid_methods:
            raise ConfigurationError(
                f"Invalid optimization_method: '{self.optimization_method}'. "
                f"Must be one of {valid_methods}",
                parameter="optimization_method",
                value=self.optimization_method
            )

        # Read patching configuration
        self.patch_coarse_grid = projection_config.get('patch_coarse_grid', True)
        self.patch_refined_grid = projection_config.get('patch_refined_grid', False)

        # Print configuration
        self.logger.info(f"  Optimization method: {self.optimization_method}")
        if self.optimization_method == 'de':
            self.logger.info(f"  L-BFGS-B polish after DE: {'Enabled' if self.lbfgsb_polish else 'Disabled'}")
        self.logger.info(f"  Patching on coarse grid: {'Enabled' if self.patch_coarse_grid else 'Disabled'}")
        if self.is_refinement_run:
            self.logger.info(f"  Patching on refined grid: {'Enabled' if self.patch_refined_grid else 'Disabled'}")

        self.profiled_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_prof_dims = len(self.profiled_dims)

        # No profiled dimensions: evaluate the grid points directly.
        self.direct_eval_mode = (self.n_prof_dims == 0)

        if self.direct_eval_mode:
            self.logger.info("")
            self.logger.info(f"  NOTE: Grid dimensionality equals function dimensionality ({self.dims}D).")
            self.logger.info("  No profiled dimensions to optimize - will evaluate at grid points directly.")
            self.logger.info("")

        self.grid_shape = tuple(self.grid_points_per_dim)

        self.grid_axes = [np.linspace(self.bounds[d, 0], self.bounds[d, 1], n) for d, n in zip(self.projection_dims, self.grid_points_per_dim)]
        self.profile_likelihood_grid = {} # Use a dict for sparse grid

        # Reset state variables
        self.initial_maxima = []
        self.initial_optima_registry = []
        self._initial_opt_start_points = None
        self._initial_opt_lhs_idx = 0
        self.population = {}
        self.activated_grid_indices = set()
        self.pending_activation_indices = set()
        self.ahead_cells = set()     # activated ahead, not confirmed (activation_depth > 1)
        self._patching_jump_thresholds = None   # set by the first patching wave
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        # Curvature scales of finished cell runs, and the latest inverse
        # Hessian a cell run measured (for a cell without a finished neighbour).
        self._lbfgsb_cell_scales = collections.deque(maxlen=LBFGSB_SCALE_MEMORY)
        self.lbfgsb_hessian_hint = None

        # Performance optimizations: caches
        self._neighbor_cache = {}
        # projection_dims and profiled_dims have changed, so the proximity-pool snapshot is stale.
        self._proximity_pool_cache = None

        # Cache bounds arrays for profiled and projection dimensions
        self._profiled_bounds = self.bounds[self.profiled_dims] if len(self.profiled_dims) > 0 else None
        self._projection_bounds = self.bounds[self.projection_dims] if len(self.projection_dims) > 0 else None

        # --- Handle refinement run: Transfer coarse grid solutions to fine grid ---
        if self.is_refinement_run and self.coarse_grid_solution is not None:
            self.logger.info("--- Transferring coarse grid solutions to fine grid (for visualization) ---")
            coarse_solutions = self.coarse_grid_solution['solutions']
            n_transferred = 0

            for coarse_idx, solution in coarse_solutions.items():
                # Map coarse grid index to fine grid index
                fine_idx = self._map_coarse_to_fine_index(coarse_idx, self.grid_refinement_factor)

                # Verify the fine grid point is valid
                if not all(0 <= i < s for i, s in zip(fine_idx, self.grid_shape)):
                    self.logger.warning(f"Warning: Coarse point {coarse_idx} maps to out-of-bounds fine point {fine_idx}. Skipping.")
                    continue

                likelihood = solution['likelihood']
                profiled_params = solution['profiled_params']

                # Store in profile likelihood grid for visualization
                self.profile_likelihood_grid[fine_idx] = likelihood

                # Population entry, so profiled-parameter plots include the
                # coarse grid.
                self.population[fine_idx] = {
                    'profiled_params': np.array([profiled_params]),  # Shape: (1, n_prof_dims)
                    'fitnesses': np.array([likelihood]),
                    'best_fitness': likelihood,
                    'status': 'optimized',  # Mark as already optimized from coarse run
                    'improvement_history': [],
                    'last_update_gen': 0,
                    'optimizer_state': None,
                    # Needed by export_grid_solution in direct-eval mode.
                    'full_params': solution['full_params'].copy(),
                }

                n_transferred += 1

            self.logger.info(f"Transferred {n_transferred} coarse grid solutions to fine grid (likelihood + profiled params)")
            self.logger.info("=" * 80)


    def export_grid_solution(self):
        """Export the current grid state (axes, dims, solutions, pool) as a dict."""
        solutions = {}
        for grid_idx, state in self.population.items():
            if state['status'] not in ['converged', 'optimized']:
                continue
            if self.direct_eval_mode:
                solutions[grid_idx] = {
                    'profiled_params': np.empty(0),
                    'likelihood': state['best_fitness'],
                    'full_params': state['full_params'].copy()
                }
            else:
                best_ind_idx = np.argmax(state['fitnesses'])
                profiled_params = state['profiled_params'][best_ind_idx]
                likelihood = state['fitnesses'][best_ind_idx]
                full_params = self._construct_params(grid_idx, profiled_params)
                solutions[grid_idx] = {
                    'profiled_params': profiled_params.copy(),
                    'likelihood': likelihood,
                    'full_params': full_params.copy()
                }

        return {
            'grid_axes': [ax.copy() for ax in self.grid_axes],
            'projection_dims': self.projection_dims.copy(),
            'profiled_dims': self.profiled_dims.copy(),
            'solutions': solutions,
            'grid_shape': self.grid_shape,
            'global_solution_pool': [entry.copy() for fitness, count, entry in self.global_solution_pool]
        }


    def setup_refinement_run(self, coarse_solution, refinement_factor, refinement_method='linear'):
        """Configure the sampler for a refined-grid run; only 'linear' interpolation is supported."""
        from .interpolation import GridInterpolator

        self.is_refinement_run = True
        self.grid_refinement_factor = refinement_factor
        self.coarse_grid_solution = coarse_solution
        self.refinement_method = refinement_method

        # Create interpolator from coarse solution
        if refinement_method == 'linear':
            periodic_bounds = {
                i: tuple(self.bounds[dim])
                for i, dim in enumerate(coarse_solution['profiled_dims'])
                if self._periodic_mask[dim]}
            self.refinement_interpolator = GridInterpolator(coarse_solution, periodic_bounds)
        else:
            raise ValueError(f"Unknown refinement_method: '{refinement_method}'. "
                           f"Only 'linear' is supported.")

        # Restore global solution pool from coarse run
        if 'global_solution_pool' in coarse_solution:
            import heapq
            # Heap of (fitness, count, entry); the count breaks ties.
            self.global_solution_pool = [(entry['fitness'], i, entry)
                                         for i, entry in enumerate(coarse_solution['global_solution_pool'])]
            heapq.heapify(self.global_solution_pool)
            # Update counter to continue from where we left off
            self.global_pool_counter = len(self.global_solution_pool)
            self.logger.info(f"Restored {len(self.global_solution_pool)} solutions from coarse run's global pool")

        # Perform clustering if enabled and we have profiled parameters
        if self.use_clustering and len(coarse_solution['profiled_dims']) > 0:
            from .interpolation import cluster_coarse_grid_by_modes
            self.logger.info("=" * 80)
            self.logger.info("--- Clustering Coarse Grid by Modes ---")
            self.logger.info("=" * 80)

            try:
                self.cluster_labels, self.cluster_info, self.boundary_points = cluster_coarse_grid_by_modes(
                    coarse_solution,
                    method=self.clustering_method,
                    eps=self.clustering_eps,
                    min_samples=self.clustering_min_samples,
                    eps_multiplier=self.clustering_eps_multiplier,
                    include_likelihood=False,  # Don't use likelihood for clustering, only for cluster selection
                    include_projection_coords=True,  # Include spatial context
                    projection_weight=self.clustering_projection_weight
                )
            except ImportError as e:
                self.logger.warning(f"Clustering failed: {e}")
                self.logger.warning("Proceeding without clustering-based boundary detection")
                self.cluster_labels = None
                self.cluster_info = None
                self.boundary_points = None
        else:
            self.cluster_labels = None
            self.cluster_info = None
            self.boundary_points = None
            if not self.use_clustering:
                self.logger.info("Clustering disabled by configuration")
            else:
                self.logger.info("No profiled parameters - clustering not applicable")

        self.logger.info("=" * 80)
        self.logger.info("--- Refinement Run Configuration ---")
        self.logger.info(f"Refinement factor: {refinement_factor}x")
        self.logger.info(f"Refinement method: {refinement_method}")
        self.logger.info(f"Coarse grid shape: {coarse_solution['grid_shape']}")
        self.logger.info(f"Coarse grid coverage: {self.refinement_interpolator.get_coverage_fraction():.1%}")
        self.logger.info(f"Number of coarse solutions: {len(coarse_solution['solutions'])}")
        self.logger.info(f"Global solution pool size: {len(self.global_solution_pool)}")
        if self.cluster_labels is not None:
            n_clusters = len(self.cluster_info['cluster_sizes'])
            n_boundaries = len(self.boundary_points)
            self.logger.info(f"Clusters detected: {n_clusters}")
            self.logger.info(f"Boundary points: {n_boundaries}")
        self.logger.info("=" * 80)


    def _get_refined_initialization_candidates(self, grid_coords):
        """Candidate initial params for a fine grid point.

        Near cluster boundaries: multiple candidates, one per nearby cluster.
        Elsewhere: a single interpolated candidate.
        """
        # If clustering is not available or disabled, use standard interpolation
        if self.cluster_labels is None or self.cluster_info is None or self.boundary_points is None:
            profiled_params = self.refinement_interpolator.interpolate(grid_coords)
            return [{
                'profiled_params': profiled_params,
                'cluster_id': None,
                'method': 'interpolated'
            }]

        # Use cluster-based initialization
        from .interpolation import get_cluster_based_initialization

        candidates = get_cluster_based_initialization(
            grid_coords,
            self.coarse_grid_solution,
            self.cluster_labels,
            self.cluster_info,
            self.boundary_points,
            self.refinement_interpolator
        )

        # Log if multiple candidates near boundary
        if len(candidates) > 1:
            cluster_ids = [c['cluster_id'] for c in candidates if c['cluster_id'] is not None]
            self.logger.debug(f"Fine point at {grid_coords}: testing {len(candidates)} candidates "
                            f"from clusters {cluster_ids}")

        return candidates


    def _is_coarse_grid_point(self, fine_idx, refinement_factor):
        """True if a fine-grid index aligns with a coarse-grid point."""
        return all(idx % refinement_factor == 0 for idx in fine_idx)


    def _map_coarse_to_fine_index(self, coarse_idx, refinement_factor):
        """Coarse-grid index -> corresponding fine-grid index."""
        return tuple(idx * refinement_factor for idx in coarse_idx)


    def _map_fine_to_coarse_index(self, fine_idx, refinement_factor):
        """Fine-grid index -> coarse-grid index, or None if not aligned."""
        if not self._is_coarse_grid_point(fine_idx, refinement_factor):
            return None
        return tuple(idx // refinement_factor for idx in fine_idx)


    def _cleanup_refinement_state(self):
        """Reset refinement state so the sampler is ready for the next projection."""
        self.is_refinement_run = False
        self.grid_refinement_factor = None
        self.coarse_grid_solution = None
        self.refinement_interpolator = None
        self.cluster_labels = None
        self.cluster_info = None
        self.boundary_points = None


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Flush buffered samples and close the output file. Idempotent."""
        self._close_sample_file()

    def __del__(self):
        # __del__ is unreliable; use close() or the context manager.
        if getattr(self, '_file_closed', True):
            return
        import warnings
        warnings.warn(
            "ProfileProjector was not explicitly closed. "
            "Use context manager or call close() to ensure data is saved.",
            ResourceWarning,
            stacklevel=2
        )
        try:
            self._close_sample_file()
        except Exception:
            pass

    def _close_sample_file(self):
        """Close the sample output file and flush any remaining data."""
        if self._file_closed:
            return

        try:
            # Flush remaining samples, then release the writer's file.
            self._flush_samples_buffer()
            if self._sample_writer is not None:
                self._sample_writer.close()
            self._file_closed = True
            self.logger.debug(f"Closed sample file: {self.samples_output_file}")
        except Exception as e:
            # print: the logger may already be gone.
            print(f"Warning: Error closing sample file: {e}")


    def _flush_samples_buffer(self):
        """Hand buffered samples to the format writer (crash-safe per batch)."""
        if not self._sample_writer or not self.samples_buffer or self._file_closed:
            return

        try:
            data = np.array([list(params) + [target_val, float(phase)]
                             for params, target_val, phase in self.samples_buffer])
            self._sample_writer.write_batch(data)
            self.samples_buffer = []

        except (OSError, ValueError) as e:
            self.logger.warning(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val, phase=PHASE_UNKNOWN):
        """Record a completed target call (master only). Global max is updated
        by jobs. ``phase`` (see :mod:`paraprof.phases`) tags the sample-log row
        with the algorithm stage that produced the point."""
        self.target_calls += 1
        if self.samples_output_file:
            self.samples_buffer.append((params, target_val, phase))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()

    def _get_grid_indices_from_point(self, point, grid_axes=None):
        """Closest grid indices for a point's projection coords (O(1), assumes linspace axes)."""
        if grid_axes is None:
            grid_axes = self.grid_axes

        grid_coords = point[self.projection_dims]
        indices = []
        for i, coord in enumerate(grid_coords):
            axis = grid_axes[i]
            n_points = len(axis)
            normalized_pos = (coord - axis[0]) / (axis[-1] - axis[0])
            index = int(round(normalized_pos * (n_points - 1)))
            indices.append(int(np.clip(index, 0, n_points - 1)))
        return tuple(indices)


    def _get_grid_coords_from_indices(self, grid_idx, grid_axes=None):
        """Projection-space coordinates at the given grid indices."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _construct_params(self, grid_idx, profiled_params, grid_axes=None):
        """Assemble a full parameter vector from a grid index + profiled-dim values."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes)
        full_params[self.profiled_dims] = profiled_params
        return full_params


    def _ensure_bounds(self, vec, dims_to_check):
        """Bring ``vec`` into the bounds of the given dim subset: periodic
        entries are wrapped, the others clipped. Uses cached bounds arrays."""
        if self._has_periodic and np.shape(vec) == (len(dims_to_check),):
            # A wrapped value lies inside the bounds; the clip below keeps it.
            vec = self._wrap_periodic(vec, dims_to_check)
        if dims_to_check is self.profiled_dims and self._profiled_bounds is not None:
            return np.clip(vec, self._profiled_bounds[:, 0], self._profiled_bounds[:, 1])
        elif dims_to_check is self.projection_dims and self._projection_bounds is not None:
            return np.clip(vec, self._projection_bounds[:, 0], self._projection_bounds[:, 1])

        dims_to_check = np.array(dims_to_check, dtype=int)
        if vec.shape != self.bounds[dims_to_check, 0].shape:
            # Global optimization: vec is full (n_dims,) but dims_to_check is a
            # subset; clip only the listed dims.
            clipped_vec = vec.copy()
            for i, dim_idx in enumerate(dims_to_check):
                clipped_vec[i] = np.clip(vec[i], self.bounds[dim_idx, 0], self.bounds[dim_idx, 1])
            return clipped_vec
        return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])


    def _has_periodic_projection_dim(self):
        return self._has_periodic and bool(self._periodic_mask[self.projection_dims].any())

    def _grid_images(self, idx):
        """The grid points an index tuple stands for; none if it lies outside
        the grid.

        Along a periodic projection dimension the first and the last grid
        point are the same point. An index there is taken modulo the number of
        grid intervals, and the point on the bound is returned as both.
        """
        per_axis = []
        for i, size, dim in zip(idx, self.grid_shape, self.projection_dims):
            i = int(i)
            if self._periodic_mask[dim]:
                i %= size - 1
                per_axis.append((0, size - 1) if i == 0 else (i,))
            elif 0 <= i < size:
                per_axis.append((i,))
            else:
                return []
        return list(itertools.product(*per_axis))

    def _cells_at_offsets(self, grid_idx, offsets, include_center=False):
        """Distinct grid points at the given index offsets from ``grid_idx``,
        with periodic projection dimensions wrapped."""
        center = tuple(int(i) for i in grid_idx)
        cells = {}
        for offset in offsets:
            for image in self._grid_images([i + o for i, o in zip(center, offset)]):
                if include_center or image != center:
                    cells[image] = None
        return list(cells)

    def _get_valid_neighbors(self, grid_idx, include_center=False):
        """Generator yielding in-bounds neighbour indices of ``grid_idx`` (cached).

        With a periodic projection dimension the neighbours include the grid
        points across the bound, and for a point on the bound its copy on the
        other bound.
        """
        cache_key = (grid_idx, include_center)

        if cache_key in self._neighbor_cache:
            yield from self._neighbor_cache[cache_key]
            return

        if self._has_periodic_projection_dim():
            neighbors = self._cells_at_offsets(
                grid_idx, itertools.product([-1, 0, 1], repeat=self.n_proj_dims),
                include_center=include_center)
        else:
            neighbors = []
            for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
                if not include_center and all(o == 0 for o in offset):
                    continue
                neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))
                if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                    neighbors.append(neighbor_idx)

        self._neighbor_cache[cache_key] = neighbors
        yield from neighbors


    def _update_global_pool(self, full_params, fitness, grid_idx):
        """Add a solution to the capped min-heap pool of best results across projections.

        Entries hold full parameter vectors, so any projection can use them.
        The counter in (fitness, counter, entry) breaks fitness ties.
        """
        import heapq

        entry = {
            'full_params': full_params.copy(),
            'fitness': fitness,
            'grid_idx': grid_idx
        }
        if len(self.global_solution_pool) < self.global_pool_size:
            heapq.heappush(self.global_solution_pool, (fitness, self.global_pool_counter, entry))
            self.global_pool_counter += 1
        elif fitness > self.global_solution_pool[0][0]:
            heapq.heapreplace(self.global_solution_pool, (fitness, self.global_pool_counter, entry))
            self.global_pool_counter += 1


    def _sample_from_global_pool(self, n_samples):
        """Random profiled-param samples from the global pool, or None if empty."""
        if not self.global_solution_pool or n_samples == 0:
            return None
        if self.direct_eval_mode:
            return None

        n_available = len(self.global_solution_pool)
        sample_indices = np.random.choice(n_available, size=min(n_samples, n_available), replace=False)
        return np.array([self.global_solution_pool[i][2]['full_params'][self.profiled_dims]
                         for i in sample_indices])


    def _sample_proximity_from_global_pool(self, n_samples, target_proj_coords):
        """Profiled-dim seeds from the past evaluations whose projection-dim
        coordinates are closest to ``target_proj_coords``.

        Distance is Euclidean in the projection dims, normalised by their
        bounds extent. The pool is copied into arrays once per projection and
        again when it has grown by more than 5%; a stale copy still holds
        valid past evaluations.

        Returns ``None`` when the pool is empty, in direct-evaluation mode, or
        when ``n_samples <= 0``.
        """
        if not self.global_solution_pool or n_samples <= 0:
            return None
        if self.direct_eval_mode:
            return None

        cache = self._proximity_pool_cache
        n_pool = len(self.global_solution_pool)
        stale = (cache is None
                 or cache['size'] == 0
                 or n_pool > cache['size'] * 1.05
                 or n_pool < cache['size'])
        if stale:
            full = np.array([entry['full_params']
                             for _f, _c, entry in self.global_solution_pool])
            proj_dims = self.projection_dims
            profiled_dims = self.profiled_dims
            extent = self.bounds[proj_dims, 1] - self.bounds[proj_dims, 0]
            extent = np.where(extent > 0, extent, 1.0)
            cache = {
                'size': n_pool,
                'proj': full[:, proj_dims],
                'profiled': full[:, profiled_dims],
                'extent': extent,
            }
            self._proximity_pool_cache = cache

        target = np.asarray(target_proj_coords, dtype=float)
        deltas = self._param_delta(cache['proj'], target, self.projection_dims) / cache['extent']
        dists = np.einsum('ij,ij->i', deltas, deltas)

        n_cached = cache['size']
        k = min(n_samples, n_cached)
        if k < n_cached:
            nearest = np.argpartition(dists, k - 1)[:k]
        else:
            nearest = np.arange(n_cached)

        return cache['profiled'][nearest].copy()


    def _initialize_from_global_pool(self):
        """Seed ``initial_maxima`` from ``global_solution_pool``, so the global
        L-BFGS-B starts can be skipped. In-memory analogue of
        ``_initialize_from_warm_start_file``.

        The best pool entry per grid cell becomes an initial maximum if it lies
        within ``roi_threshold`` of the global maximum.
        """
        if not self.global_solution_pool:
            return

        best_per_cell = {}
        for _f, _c, entry in self.global_solution_pool:
            params = entry['full_params']
            if not np.all((params >= self.bounds[:, 0])
                          & (params <= self.bounds[:, 1])):
                continue
            grid_idx = self._get_grid_indices_from_point(params)
            cur = best_per_cell.get(grid_idx)
            if cur is None or entry['fitness'] > cur['fitness']:
                best_per_cell[grid_idx] = entry

        if not best_per_cell:
            return

        pool_max = max(e['fitness'] for e in best_per_cell.values())
        self.global_max_target_val = max(self.global_max_target_val, pool_max)
        roi_cutoff = self.global_max_target_val - self.roi_threshold

        seeded = []
        for entry in best_per_cell.values():
            if entry['fitness'] >= roi_cutoff:
                seeded.append({
                    'point': entry['full_params'].copy(),
                    'target_val': entry['fitness'],
                })

        if not seeded:
            return

        seeded.sort(key=lambda x: x['target_val'], reverse=True)
        self.initial_maxima.extend(seeded)
        self.logger.info(
            f"--- Seeded {len(seeded)} initial_maxima from in-memory pool "
            f"(pool size: {len(self.global_solution_pool)}). "
            f"New global max: {self.global_max_target_val:.4e} ---"
        )


    def _initialize_from_warm_start_file(self, warm_start_file):
        """Seed initial_maxima from a previous run's sample file (skips initial opt)."""
        if not warm_start_file or not os.path.exists(warm_start_file):
            self.logger.info("  No warm-start file found or provided. Skipping warm start.")
            return

        self.logger.info(f"--- Initializing from warm-start file: {warm_start_file} ---")
        try:
            samples = read_samples(warm_start_file)
        except Exception as e:
            self.logger.info(f"  Warning: Could not read warm-start file. Error: {e}. Skipping.")
            return
        if samples.size == 0:
            self.logger.info("  Warm-start file contained no samples. Skipping.")
            return
        if samples.shape[1] != self.dims + 2:
            self.logger.info(
                f"  Warm-start file rows have width {samples.shape[1]}, expected "
                f"n_dims + 2 = {self.dims + 2} ([params..., logL, phase]). Skipping."
            )
            return

        # params and logL are read by position; the trailing phase column is
        # ignored, so a run's own sample log round-trips.
        best_candidates = {}
        for sample_row in samples:
            params = sample_row[:self.dims]
            target_val = sample_row[self.dims]
            if not np.all((params >= self.bounds[:, 0]) & (params <= self.bounds[:, 1])):
                continue
            grid_idx = self._get_grid_indices_from_point(params)
            if grid_idx not in best_candidates or target_val > best_candidates[grid_idx]['target_val']:
                best_candidates[grid_idx] = {'params': params, 'target_val': target_val}

        if not best_candidates:
            self.logger.info("  No valid samples found in warm-start file for the current grid.")
            return

        warm_start_max_target_val = max(c['target_val'] for c in best_candidates.values())
        self.global_max_target_val = max(self.global_max_target_val, warm_start_max_target_val)
        roi_cutoff = self.global_max_target_val - self.roi_threshold

        warm_start_maxima = [
            {'point': c['params'], 'target_val': c['target_val']}
            for c in best_candidates.values()
            if c['target_val'] >= roi_cutoff
        ]
        warm_start_maxima.sort(key=lambda x: x['target_val'], reverse=True)
        self.initial_maxima.extend(warm_start_maxima)

        self.logger.info(f"--- Loaded {len(warm_start_maxima)} warm-start maxima from file. New Global Max: {self.global_max_target_val:.4e} ---")


    # --- Job Factory Methods (Master Only) ---

    def create_initial_points_eval_job(self, next_job_id):
        """Create a single job that evaluates every user-provided initial point.

        Returns an empty job list if there are no points or they have
        already been evaluated. As a Job, the evaluations are registered and
        pooled like any other.
        """
        from .jobs.initial_points_job import InitialPointEvalJob

        if (self.initial_points is None
                or self._initial_points_evaluated
                or len(self.initial_points) == 0):
            return [], next_job_id

        self.logger.info(
            f"--- Evaluating {len(self.initial_points)} user-provided initial points ---"
        )
        job = InitialPointEvalJob(
            job_id=next_job_id,
            sampler=self,
            points=self.initial_points,
        )
        return [job], next_job_id + 1

    def _make_initial_opt_job(self, next_job_id, start_point):
        """Build a single global-optimization L-BFGS-B job from a start point."""
        return LBFGSBJob(
            job_id=next_job_id,
            job_type='INITIAL_OPTIMIZATION',
            sampler=self,
            opt_dims=tuple(range(self.dims)),  # Global optimization
            start_params=start_point,          # Full vector
            grid_idx=None,                     # No grid anchor
            start_params_full=start_point,     # Full vector
            seed_history=None,
        )

    def resolve_initial_opt_batch_size(self, n_workers):
        """Number of initial-optimization runs to keep in flight for the rolling
        multistart, given ``n_workers`` available workers.

        ``basin_detection.batch_size`` overrides this. By default it is
        ``n_workers / fd_width``, where ``fd_width`` is the number of parallel
        evaluations in one gradient (one per dim, two for central differences).
        Fewer runs in flight lose less work when the stopping rule aborts them.
        Floored at 2, capped at ``n_workers`` and ``n_initial_optimizations``."""
        cap = self.n_initial_optimizations
        if self.basin_batch_size is not None:
            return max(1, min(int(self.basin_batch_size), cap))
        fd_width = self.dims * (2 if self.lbfgsb_gradient_method == 'central' else 1)
        auto = max(2, math.ceil(n_workers / max(fd_width, 1)))
        return max(1, min(auto, max(n_workers, 1), cap))

    def init_initial_opt_lhs(self, n_points):
        """Pre-generate the LHS start-point pool (sized to the hard cap) for the
        rolling multistart. Jobs consume points from it sequentially, so any
        prefix of the design is itself space-filling."""
        n_points = max(int(n_points), 1)
        lhs = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = lhs.random(n=n_points)
        self._initial_opt_start_points = (
            self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])
        )
        self._initial_opt_lhs_idx = 0

    def start_initial_point_refinement(self):
        """Queue the evaluated ``initial_points`` as the start points of the
        global L-BFGS-B stage, best first. Returns their number, or 0 if there
        is nothing to refine (option off, done already, no finite point)."""
        if (not self.refine_initial_points or self._initial_points_refined
                or not self._initial_points_evaluated):
            return 0
        self._initial_points_refined = True
        values = self._initial_point_values
        order = [i for i in np.argsort(-values) if np.isfinite(values[i])]
        if not order:
            return 0
        self._initial_opt_start_points = np.asarray(self.initial_points, dtype=float)[order]
        self._initial_opt_lhs_idx = 0
        self._refine_best_before = float(values[order[0]])
        return len(order)

    def create_one_initial_optimization_job(self, next_job_id):
        """Single global L-BFGS-B job from the next pooled LHS start point."""
        if (self._initial_opt_start_points is None
                or self._initial_opt_lhs_idx >= len(self._initial_opt_start_points)):
            # Cap normally bounds consumption; refill defensively if exhausted.
            self.init_initial_opt_lhs(max(self.n_initial_optimizations, 1))
        start_point = self._initial_opt_start_points[self._initial_opt_lhs_idx]
        self._initial_opt_lhs_idx += 1
        return self._make_initial_opt_job(next_job_id, start_point), next_job_id + 1

    def register_initial_optimum(self, point, target_val):
        """Online single-linkage clustering of a converged global-opt endpoint.

        Merges into an existing registry entry when within ``basin_merge_tol``
        (RMS bounds-normalized parameter distance), otherwise registers a new
        distinct optimum. Returns True iff a new optimum was registered.
        """
        point = np.array(point, dtype=float)  # copy: stored in the registry
        span = self.bounds[:, 1] - self.bounds[:, 0]
        span = np.where(span > 0, span, 1.0)
        xnorm = (point - self.bounds[:, 0]) / span
        sqrt_d = np.sqrt(self.dims)

        for opt in self.initial_optima_registry:
            diff = xnorm - opt['point_norm']
            if self._has_periodic:
                # One period is 1 in these coordinates.
                diff[self._periodic_mask] -= np.round(diff[self._periodic_mask])
            dist = np.linalg.norm(diff) / sqrt_d
            if dist < self.basin_merge_tol:
                opt['count'] += 1
                if target_val > opt['target_val']:
                    opt['target_val'] = float(target_val)
                    opt['point'] = point
                    opt['point_norm'] = xnorm
                return False

        self.initial_optima_registry.append({
            'point': point,
            'point_norm': xnorm,
            'target_val': float(target_val),
            'count': 1,
        })
        return True

    def basin_detection_roi_stats(self):
        """(W, N_roi): distinct ROI-competitive optima and the number of starts
        that converged into them, using the *current* global max."""
        thresh = self.global_max_target_val - self.roi_threshold
        roi = [o for o in self.initial_optima_registry if o['target_val'] >= thresh]
        W = len(roi)
        n_roi = sum(o['count'] for o in roi)
        return W, n_roi

    def basin_detection_should_stop(self, n_completed):
        """Whether the rolling multistart should stop.

        The ``n_optima`` bounds come first: stop once ``max`` distinct optima
        are found (skipping the ``basin_min_starts`` floor), never before
        ``min`` are found. Otherwise, once ``n_completed`` reaches
        ``basin_min_starts``, apply the Boender-Rinnooy Kan rule: with ``W``
        distinct ROI optima over ``N`` ROI starts, stop once ``W**2/(N-W-1)``
        falls below ``basin_undiscovered_threshold``.
        """
        n_distinct = len(self.initial_optima_registry)

        # n_optima prior takes precedence (see docstring).
        if self.basin_min_optima is not None and n_distinct < self.basin_min_optima:
            return False
        if self.basin_max_optima is not None and n_distinct >= self.basin_max_optima:
            return True

        # Otherwise the Bayesian rule, gated by the min-starts floor.
        if n_completed < self.basin_min_starts:
            return False
        W, n_roi = self.basin_detection_roi_stats()
        if W < 1:
            return False
        denom = n_roi - W - 1
        if denom <= 0:
            return False
        expected_undiscovered = (W * W) / denom
        return expected_undiscovered < self.basin_undiscovered_threshold

    @staticmethod
    def _parse_n_optima(n_optima):
        """Parse the ``n_optima`` prior into ``(min, max)`` int bounds.

        Accepts ``None`` (no prior), an ``int`` (exact: both bounds equal), or
        a dict ``{'min': int, 'max': int}`` with either key optional. Returns
        ``(None, None)`` when no prior is given.
        """
        if n_optima is None:
            return None, None

        def _check(name, val):
            if val is None:
                return None
            if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or val < 1:
                raise ConfigurationError(
                    f"n_optima {name} must be a positive integer",
                    parameter="n_optima", value=n_optima,
                )
            return int(val)

        if isinstance(n_optima, dict):
            unknown = set(n_optima) - {'min', 'max'}
            if unknown:
                raise ConfigurationError(
                    f"n_optima dict has unknown keys {sorted(unknown)}; "
                    f"only 'min' and 'max' are allowed",
                    parameter="n_optima", value=n_optima,
                )
            lo = _check('min', n_optima.get('min'))
            hi = _check('max', n_optima.get('max'))
        else:
            lo = hi = _check('value', n_optima)

        if lo is not None and hi is not None and lo > hi:
            raise ConfigurationError(
                f"n_optima min ({lo}) must not exceed max ({hi})",
                parameter="n_optima", value=n_optima,
            )
        return lo, hi

    def create_activation_jobs(self, next_job_id):
        """Generates ActivationJobs for grid points near found maxima."""
        if not self.initial_maxima:
            self.logger.warning("Warning: No initial maxima found. Cannot create activation jobs.")
            return [], next_job_id

        jobs = []
        activation_job_created_for_grid_points = set()

        # Best maximum first: a cell near two maxima is warm-started from the
        # higher one, not from whichever start finished first.
        ordered = sorted(self.initial_maxima, key=lambda m: -m['target_val'])
        for maximum in ordered:
            point = maximum['point']
            grid_idx = self._get_grid_indices_from_point(point)

            if grid_idx in activation_job_created_for_grid_points:
                continue

            for neighbor_idx in self._get_valid_neighbors(grid_idx, include_center=True):
                if (neighbor_idx in activation_job_created_for_grid_points) or \
                   (neighbor_idx in self.population):
                    continue

                job = ActivationJob(
                    job_id=next_job_id,
                    sampler=self,
                    grid_idx=neighbor_idx,
                    warm_start_params=point[self.profiled_dims] # Warm start
                )
                jobs.append(job)
                activation_job_created_for_grid_points.add(neighbor_idx)
                self.pending_activation_indices.add(neighbor_idx)
                next_job_id += 1

        self.logger.info(f"--- Generating {len(jobs)} activation jobs ---")
        return jobs, next_job_id


    def needs_noise_estimate(self):
        """True if the automatic finite-difference step has not been set yet."""
        return (self.lbfgsb_fd_step == 'auto' and not self._noise_estimated
                and self.grad_func is None)

    def create_noise_estimate_job(self, next_job_id):
        """Job that measures the noise of the target and sets the
        finite-difference step. Base points are the user's initial points if
        there are any, otherwise Latin-hypercube points."""
        from .jobs.noise_job import NoiseEstimateJob

        if self.initial_points is not None and len(self.initial_points) > 0:
            base = np.asarray(self.initial_points, dtype=float)[:NOISE_BASE_POINTS]
        else:
            lhs = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
            unit = lhs.random(n=NOISE_BASE_POINTS)
            base = self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])
        job = NoiseEstimateJob(next_job_id, self, base, LBFGSB_FD_STEP_MIN,
                               LBFGSB_FD_STEP_MAX, LBFGSB_FD_STEP_FALLBACK)
        return [job], next_job_id + 1

    def needs_periodic_check(self):
        """True if the target has not been compared at the bounds of the
        periodic parameters yet."""
        return self._has_periodic and not self._periodic_checked

    def create_periodic_check_job(self, next_job_id):
        """Job that compares the target at the two bounds of each periodic
        parameter. Base points as for the noise estimate."""
        from .jobs.periodic_check_job import PeriodicCheckJob

        if self.initial_points is not None and len(self.initial_points) > 0:
            base = np.asarray(self.initial_points, dtype=float)[:PERIODIC_CHECK_BASE_POINTS]
        else:
            lhs = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
            unit = lhs.random(n=PERIODIC_CHECK_BASE_POINTS)
            base = self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])
        return [PeriodicCheckJob(next_job_id, self, base)], next_job_id + 1

    def lbfgsb_hessian_step_value(self):
        """Step of the second differences of a finite-difference Hessian, as a
        fraction of the bound extent. A second difference divides the noise by
        the step squared, so the step is the fourth root of the noise where
        that is known, and tied to the gradient step otherwise."""
        if self.lbfgsb_noise > 0.0:
            step = self.lbfgsb_noise ** 0.25
        else:
            step = 4.0 * math.sqrt(self.lbfgsb_fd_step_value)
        return min(max(step, LBFGSB_HESSIAN_STEP_MIN), LBFGSB_HESSIAN_STEP_MAX)

    def _lbfgsb_scales(self, job_type):
        return (self._lbfgsb_global_scales if job_type == 'INITIAL_OPTIMIZATION'
                else self._lbfgsb_cell_scales)

    def lbfgsb_scale_hint(self, job_type):
        """Curvature scale for a run that has no curvature pairs of its own:
        the median of the scales accepted by finished runs of the same kind
        (global start or cell), or None if there are none yet."""
        scales = self._lbfgsb_scales(job_type)
        return float(np.median(scales)) if scales else None

    def note_lbfgsb_scale(self, job_type, gamma):
        """Record the last curvature scale of a finished run."""
        self._lbfgsb_scales(job_type).append(gamma)

    def _neighbor_history(self, grid_idx):
        """Curvature pairs of the best finished neighbour, or None.

        One grid step away the Hessian of the profiled block is nearly the
        same, so the pairs let a new cell start with a scaled step.
        """
        best_state = None
        best_fitness = -np.inf
        for neighbor_idx in self._get_valid_neighbors(grid_idx):
            st = self.population.get(neighbor_idx)
            if (st is not None and st['status'] in ('optimized', 'converged')
                    and st.get('optimizer_state')
                    and (st['optimizer_state']['s']
                         or st['optimizer_state'].get('Hinv') is not None)
                    and st['best_fitness'] > best_fitness):
                best_fitness = st['best_fitness']
                best_state = st
        return None if best_state is None else best_state['optimizer_state']

    def create_lbfgsb_job_for_active_cell(self, grid_idx, next_job_id, job_type, terminal_status):
        """L-BFGS-B job for one active grid point, or None if it needs none.

        Returns ``(job_or_None, next_job_id)``.
        """
        state = self.population.get(grid_idx)
        if state is None or state['status'] != 'active':
            return None, next_job_id

        if self.direct_eval_mode or self.n_prof_dims == 0:
            state['status'] = terminal_status
            return None, next_job_id

        state['status'] = 'LBFGSB_queued'
        best_ind_idx = np.argmax(state['fitnesses'])
        start_params_partial = state['profiled_params'][best_ind_idx]
        start_fitness = state['fitnesses'][best_ind_idx]
        start_params_full = self._construct_params(grid_idx, start_params_partial)

        job = LBFGSBJob(
            job_id=next_job_id,
            job_type=job_type,
            sampler=self,
            opt_dims=tuple(self.profiled_dims),
            start_params=start_params_partial,
            grid_idx=grid_idx,
            start_params_full=start_params_full,
            seed_history=self._neighbor_history(grid_idx),
            start_fitness=start_fitness,
        )
        return job, next_job_id + 1

    def _create_lbfgsb_jobs_for_active(self, next_job_id, job_type, terminal_status):
        """Spawn one L-BFGS-B job per active grid point.

        `terminal_status` is the status assigned to direct-eval cells (which
        skip optimization): 'optimized' for the one-shot post-activation
        stage, 'converged' for LBFGSB_LOOP (allows dynamic re-activation).
        """
        jobs = []
        for grid_idx in list(self.population):
            job, next_job_id = self.create_lbfgsb_job_for_active_cell(
                grid_idx, next_job_id, job_type, terminal_status)
            if job is not None:
                jobs.append(job)
        return jobs, next_job_id

    def create_post_activation_lbfgsb_jobs(self, next_job_id):
        """L-BFGS-B jobs for all activated cells (used when optimization_method='lbfgsb')."""
        jobs, next_job_id = self._create_lbfgsb_jobs_for_active(
            next_job_id, 'POST_ACTIVATION_LBFGSB', 'optimized',
        )
        self.logger.info(f"Created {len(jobs)} post-activation L-BFGS-B jobs")
        return jobs, next_job_id

    def create_lbfgsb_loop_jobs(self, next_job_id):
        """L-BFGS-B jobs for active cells in the LBFGSB_LOOP stage."""
        return self._create_lbfgsb_jobs_for_active(
            next_job_id, 'LBFGSB_LOOP', 'converged',
        )


    def create_refinement_activation_jobs(self, next_job_id):
        """ActivationJobs for neighbors of transferred coarse points (refinement only)."""
        if not self.is_refinement_run or self.refinement_interpolator is None:
            self.logger.warning("Warning: create_refinement_activation_jobs called but not in refinement mode.")
            return [], next_job_id

        jobs = []
        activation_job_created_for_grid_points = set()

        # Transferred coarse points inside the region of interest.
        transferred_points = [idx for idx, state in self.population.items()
                             if (state['status'] == 'optimized') and 
                                (state['best_fitness'] >= (self.global_max_target_val - self.roi_threshold))]

        self.logger.info(f"--- Creating refinement activation jobs from {len(transferred_points)} transferred points ---")

        # Activate all neighbors of transferred points
        for grid_idx in transferred_points:
            for neighbor_idx in self._get_valid_neighbors(grid_idx):
                # Skip if already activated or pending
                if (neighbor_idx in activation_job_created_for_grid_points) or \
                   (neighbor_idx in self.population) or \
                   (neighbor_idx in self.pending_activation_indices):
                    continue

                # Get candidate parameters (using clustering if available)
                grid_coords = self._get_grid_coords_from_indices(neighbor_idx)
                candidates = self._get_refined_initialization_candidates(grid_coords)

                # First candidate; NaN or no candidate falls back to the
                # transferred point's parameters.
                if len(candidates) > 0:
                    warm_start_params = candidates[0]['profiled_params']
                    # Handle case where interpolation returns None (no profiled dims)
                    if warm_start_params is None:
                        warm_start_params = None
                    # Check for NaN values
                    elif np.any(np.isnan(warm_start_params)):
                        # Fallback: use nearest transferred point's parameters
                        nearest_state = self.population[grid_idx]
                        warm_start_params = nearest_state['profiled_params'][0]
                else:
                    # No candidates - use fallback
                    nearest_state = self.population[grid_idx]
                    warm_start_params = nearest_state['profiled_params'][0]

                job = ActivationJob(
                    job_id=next_job_id,
                    sampler=self,
                    grid_idx=neighbor_idx,
                    warm_start_params=warm_start_params
                )
                jobs.append(job)
                activation_job_created_for_grid_points.add(neighbor_idx)
                self.pending_activation_indices.add(neighbor_idx)
                next_job_id += 1

        self.logger.info(f"--- Generating {len(jobs)} refinement activation jobs ---")
        return jobs, next_job_id


    def create_refinement_lbfgsb_jobs(self, next_job_id):
        """Refinement jobs for fine-grid points, starting from interpolated coarse params.

        Pre-screens coarse cells via interpolation: a cell is processed if
        any fine point within it is predicted to fall inside the ROI.
        Returns ActivationJobs when ``refinement_direct_eval`` is True,
        otherwise LBFGSBJobs.
        """
        if not self.is_refinement_run or self.refinement_interpolator is None:
            self.logger.warning("Warning: create_refinement_lbfgsb_jobs called but not in refinement mode.")
            return [], next_job_id

        jobs = []
        lbfgsb_job_created_for_grid_points = set()

        # Find all transferred coarse grid points within ROI
        roi_cutoff = self.global_max_target_val - self.roi_threshold
        transferred_points = []

        for coarse_idx, solution in self.coarse_grid_solution['solutions'].items():
            if solution['likelihood'] >= roi_cutoff:
                fine_idx = self._map_coarse_to_fine_index(coarse_idx, self.grid_refinement_factor)
                if all(0 <= i < s for i, s in zip(fine_idx, self.grid_shape)):
                    transferred_points.append(fine_idx)

        self.logger.info(f"--- Creating refinement LBFGSB jobs from {len(transferred_points)} transferred points ---")

        if not transferred_points:
            self.logger.info("--- No transferred points found. No refinement jobs created. ---")
            return jobs, next_job_id

        # Pre-screen coarse cells by the interpolated likelihood of their fine points.
        self.logger.info("--- Performing interpolation-based pre-screening of coarse cells ---")

        coarse_cells_to_process = set()
        n_cells_screened = 0
        n_cells_accepted = 0

        for coarse_idx, solution in self.coarse_grid_solution['solutions'].items():
            n_cells_screened += 1

            # Check if the coarse point itself is in ROI (fast path)
            if solution['likelihood'] >= roi_cutoff:
                coarse_cells_to_process.add(coarse_idx)
                n_cells_accepted += 1
                continue

            # Coarse point outside the ROI: check the fine points of its cell.
            fine_start = tuple(ci * self.grid_refinement_factor for ci in coarse_idx)
            fine_end = tuple((ci + 1) * self.grid_refinement_factor for ci in coarse_idx)
            fine_start = tuple(max(0, fs) for fs in fine_start)
            fine_end = tuple(min(fe, self.grid_shape[i] - 1) for i, fe in enumerate(fine_end))

            # Check all fine points in this cell
            cell_has_roi_point = False
            ranges = [range(fine_start[i], fine_end[i] + 1) for i in range(self.n_proj_dims)]

            for fine_idx in itertools.product(*ranges):
                # Skip coarse grid points (already checked above)
                if self._is_coarse_grid_point(fine_idx, self.grid_refinement_factor):
                    continue

                # Get grid coordinates and interpolate likelihood
                grid_coords = self._get_grid_coords_from_indices(fine_idx)
                try:
                    interpolated_likelihood = self.refinement_interpolator.get_interpolated_likelihood(grid_coords)

                    # Check if interpolated likelihood suggests this point is in ROI
                    if not np.isnan(interpolated_likelihood) and interpolated_likelihood >= roi_cutoff:
                        cell_has_roi_point = True
                        break
                except Exception as e:
                    # Include the cell if interpolation fails.
                    self.logger.debug(f"Interpolation failed for fine point {fine_idx}: {e}")
                    cell_has_roi_point = True
                    break

            if cell_has_roi_point:
                coarse_cells_to_process.add(coarse_idx)
                n_cells_accepted += 1

        self.logger.info(f"--- Pre-screening complete: {n_cells_accepted}/{n_cells_screened} coarse cells selected for processing ---")

        # A job for every fine point of the selected cells. Coarse point
        # (ci, cj) spans the fine cell (ci*RF, cj*RF) to ((ci+1)*RF, (cj+1)*RF).
        for coarse_idx in coarse_cells_to_process:
            fine_start = tuple(ci * self.grid_refinement_factor for ci in coarse_idx)
            fine_end = tuple((ci + 1) * self.grid_refinement_factor for ci in coarse_idx)

            # Ensure cell boundaries are within fine grid bounds
            fine_start = tuple(max(0, fs) for fs in fine_start)
            fine_end = tuple(min(fe, self.grid_shape[i] - 1) for i, fe in enumerate(fine_end))

            # Iterate through all points in this cell
            ranges = [range(fine_start[i], fine_end[i] + 1) for i in range(self.n_proj_dims)]
            for fine_idx in itertools.product(*ranges):
                # Skip coarse grid points (already transferred)
                if self._is_coarse_grid_point(fine_idx, self.grid_refinement_factor):
                    continue

                # Skip if already processed
                if fine_idx in lbfgsb_job_created_for_grid_points:
                    continue

                # No profiled parameters: one ActivationJob evaluation per
                # point. An LBFGSBJob with no opt dims would hang the master
                # (start() returns no tasks but the job stays active).
                if self.direct_eval_mode:
                    job = ActivationJob(
                        job_id=next_job_id,
                        sampler=self,
                        grid_idx=fine_idx,
                        warm_start_params=None,
                    )
                    jobs.append(job)
                    lbfgsb_job_created_for_grid_points.add(fine_idx)
                    next_job_id += 1
                    continue

                # Get candidate parameters (using clustering if available)
                grid_coords = self._get_grid_coords_from_indices(fine_idx)
                candidates = self._get_refined_initialization_candidates(grid_coords)
                if len(candidates) == 0:
                    continue

                # refinement_direct_eval with several candidates: evaluate all,
                # keep the best.
                if self.refinement_direct_eval and len(candidates) > 1:
                    # Create a special multi-candidate evaluation job
                    all_prof_params = []
                    all_full_params = []

                    for candidate in candidates:
                        prof_params = candidate['profiled_params']
                        if prof_params is None:
                            prof_params = np.array([])
                        elif np.any(np.isnan(prof_params)):
                            continue

                        full_params = self._construct_params(fine_idx, prof_params)
                        all_prof_params.append(prof_params)
                        all_full_params.append(full_params)

                    if len(all_prof_params) == 0:
                        continue

                    # Create ActivationJob that evaluates all candidates
                    job = ActivationJob(
                        job_id=next_job_id,
                        sampler=self,
                        grid_idx=fine_idx,
                        warm_start_params=all_prof_params[0],
                        mark_converged=True
                    )
                    # Evaluate the candidates instead of the default population.
                    job.pop_size = len(all_prof_params)
                    job.all_profiled_params = np.array(all_prof_params)
                    job.all_full_params = all_full_params
                    job.fitnesses = np.full(len(all_prof_params), -np.inf)
                    job.evals_remaining = len(all_prof_params)

                    jobs.append(job)
                    lbfgsb_job_created_for_grid_points.add(fine_idx)
                    next_job_id += 1

                else:
                    # Single candidate or optimization mode: use first candidate
                    candidate = candidates[0]
                    start_params_partial = candidate['profiled_params']
                    if start_params_partial is None:
                        start_params_partial = np.array([])
                    elif np.any(np.isnan(start_params_partial)):
                        continue

                    # Construct full parameter vector
                    start_params_full = self._construct_params(fine_idx, start_params_partial)

                    # Create job based on refinement mode
                    if self.refinement_direct_eval:
                        # Direct evaluation mode with single candidate
                        job = ActivationJob(
                            job_id=next_job_id,
                            sampler=self,
                            grid_idx=fine_idx,
                            warm_start_params=start_params_partial,
                            mark_converged=True
                        )
                        # A single evaluation.
                        job.pop_size = 1
                        job.all_profiled_params = np.array([start_params_partial])
                        job.all_full_params = [start_params_full]
                        job.fitnesses = np.full(1, -np.inf)
                        job.evals_remaining = 1
                    else:
                        # Optimization mode: create L-BFGS-B refinement job
                        job = LBFGSBJob(
                            job_id=next_job_id,
                            job_type='REFINEMENT_LBFGSB',
                            sampler=self,
                            opt_dims=tuple(self.profiled_dims),
                            start_params=start_params_partial,
                            grid_idx=fine_idx,
                            start_params_full=start_params_full,
                            seed_history=None,
                            start_fitness=-np.inf
                        )
                    jobs.append(job)
                    lbfgsb_job_created_for_grid_points.add(fine_idx)
                    next_job_id += 1

        # Count multi-candidate jobs
        n_multi_candidate = sum(1 for job in jobs if hasattr(job, 'pop_size') and job.pop_size > 1)
        n_single_candidate = len(jobs) - n_multi_candidate

        if self.direct_eval_mode:
            # In direct_eval_mode the per-point jobs only evaluate the target.
            opt_method = "direct evaluation"
        elif self.refinement_direct_eval:
            opt_method = "direct evaluation"
            if n_multi_candidate > 0:
                self.logger.info(f"--- Multi-candidate evaluations: {n_multi_candidate} grid points with multiple clusters ---")
                self.logger.info(f"--- Single-candidate evaluations: {n_single_candidate} grid points ---")
        else:
            opt_method = "LBFGSB"
        self.logger.info(f"--- Generating {len(jobs)} refinement {opt_method} jobs ---")
        return jobs, next_job_id


    def _is_de_skippable(self, grid_idx):
        """True if ``grid_idx``'s in-population neighbours agree on the profiled
        argmax, so the cell can confirm convergence on a reduced DE window.

        Uses the neighbours' stored best profiled parameters; no target
        evaluations. False with fewer than two neighbours.
        """
        if self.n_prof_dims == 0:
            return False

        # If a random or pool seed beat the neighbour warm start at activation,
        # a better inner mode is nearby: keep DE.
        if not self.population[grid_idx].get('warm_start_best', False):
            return False

        neigh_phis = [
            self._best_profiled_params(n_idx)
            for n_idx in self._get_valid_neighbors(grid_idx)
            if n_idx in self.population
        ]
        if len(neigh_phis) < 2:
            return False

        phis = np.asarray(neigh_phis)
        if self._has_periodic:
            # Values on either side of a periodic bound count as close.
            phis = phis[0] + self._param_delta(phis, phis[0], self.profiled_dims)
        extent = self._profiled_extent()
        # Normalised spread: largest per-dim deviation from the neighbour mean.
        spread = float(np.max(np.abs(phis - phis.mean(axis=0)) / extent))
        return spread < SKIP_DE_PHI_SPREAD

    def create_de_generation_jobs(self, next_job_id, max_num_to_evolve):
        """Generates all DEGridPointJobs for one generation."""

        successful_F = []
        successful_CR = []

        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']

        if not unconverged_indices:
            self.logger.info("All active points have converged. Ending DE phase.")
            return [], next_job_id, successful_F, successful_CR

        # --- Prioritize which grid points to evolve ---
        priority_scores = []
        for idx in unconverged_indices:
            state = self.population[idx]
            fitness_score = max(0, state['best_fitness'] - (self.global_max_target_val - 2 * self.roi_threshold))
            improvement_rate = np.mean(state['improvement_history']) if state['improvement_history'] else 0
            improvement_score = improvement_rate * 10
            pri_score = fitness_score + improvement_score + (1./len(unconverged_indices))
            priority_scores.append(pri_score)

        priority_scores = np.array(priority_scores)
        probabilities = priority_scores / np.sum(priority_scores)

        if max_num_to_evolve is not None:
            num_to_evolve = min(len(unconverged_indices), max_num_to_evolve)
        else:
            num_to_evolve = len(unconverged_indices)

        if num_to_evolve == 0:
             return [], next_job_id, successful_F, successful_CR

        indices_to_process_map = np.random.choice(
            np.arange(len(unconverged_indices)),
            size=num_to_evolve,
            replace=False,
            p=probabilities
        )
        indices_to_process = [unconverged_indices[i] for i in indices_to_process_map]

        active_pop_list = list(self.activated_grid_indices)
        if len(active_pop_list) < 4:
            self.logger.info("Not enough active points (<4) to perform DE. Waiting.")
            return [], next_job_id, successful_F, successful_CR

        # --- Create parent and p-best pools ---
        parent_pool = []
        for idx in active_pop_list:
            state = self.population[idx]
            best_idx = np.argmax(state['fitnesses'])
            parent_pool.append({
                'profiled_params': state['profiled_params'][best_idx],
                'fitness': state['fitnesses'][best_idx]
            })

        pbest_archive = []
        if self.mutation_strategy == 'current-to-pbest/1':
            parent_pool.sort(key=lambda p: p['fitness'], reverse=True)
            pbest_size = max(1, int(len(parent_pool) * self.pbest_fraction))
            pbest_archive = parent_pool[:pbest_size]

        # --- Create a job for each selected grid point ---
        jobs = []
        for grid_idx in indices_to_process:
            state = self.population[grid_idx]
            # allow_early_DE_exit: a fresh, skip-eligible cell takes a one-
            # generation window (set here) then the polish, not the full one.
            if (self.de_allow_early_DE_exit
                    and len(state['improvement_history']) == 0
                    and state.get('conv_window') is None
                    and self._is_de_skippable(grid_idx)):
                state['conv_window'] = SKIP_DE_WINDOW
                self.de_cells_skipped += 1

            job = DEGridPointJob(
                job_id=next_job_id,
                sampler=self,
                grid_idx=grid_idx,
                parent_pool=parent_pool,
                pbest_archive=pbest_archive,
                successful_F_list=successful_F, # Pass shared list
                successful_CR_list=successful_CR # Pass shared list
            )
            jobs.append(job)
            next_job_id += 1

        return jobs, next_job_id, successful_F, successful_CR

    def update_de_memory(self, successful_F, successful_CR):
        """Updates the F and CR memory after a DE generation."""
        if successful_F:
            weights = np.ones(len(successful_F)) # Simple mean
            muF = np.sum(weights * np.array(successful_F)**2) / np.sum(weights * np.array(successful_F))
            muCR = np.sum(weights * np.array(successful_CR)) / np.sum(weights)

            self.memory_F[self.memory_idx] = muF
            self.memory_CR[self.memory_idx] = muCR
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def create_LBFGSB_job_for_point(self, grid_idx, next_job_id):
        """Spawn an L-BFGS-B polish job for a single converged grid point."""
        if self.direct_eval_mode or self.n_prof_dims == 0:
            state = self.population.get(grid_idx)
            if state:
                state['status'] = 'optimized'
            return None

        state = self.population.get(grid_idx)
        if not state or state['status'] == 'LBFGSB_queued':
            return None

        state['status'] = 'LBFGSB_queued'
        best_ind_idx = np.argmax(state['fitnesses'])
        start_params_partial = state['profiled_params'][best_ind_idx]
        start_fitness = state['fitnesses'][best_ind_idx]
        start_params_full = self._construct_params(grid_idx, start_params_partial)
        seed_history = state.get('optimizer_state')

        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='LBFGSB',
            sampler=self,
            opt_dims=tuple(self.profiled_dims),
            start_params=start_params_partial,
            grid_idx=grid_idx,
            start_params_full=start_params_full,
            seed_history=seed_history,
            start_fitness=start_fitness,
        )
        return (job, next_job_id + 1)

    def confirm_ahead_cells(self, grid_idx):
        """Called when ``grid_idx`` has a new value. If it lies inside the
        region of interest, it and its neighbours stop being cells activated
        ahead. Returns the cells this changed."""
        state = self.population.get(grid_idx)
        if state is None or state['best_fitness'] <= self.global_max_target_val - self.roi_threshold:
            return []
        confirmed = [c for c in self._get_valid_neighbors(grid_idx, include_center=True)
                     if c in self.ahead_cells]
        self.ahead_cells.difference_update(confirmed)
        return confirmed

    def drop_ahead_cells(self):
        """Remove the cells activated ahead that were never confirmed.
        Returns how many there were."""
        n = len(self.ahead_cells)
        for idx in self.ahead_cells:
            self.population.pop(idx, None)
            self.profile_likelihood_grid.pop(idx, None)
            self.activated_grid_indices.discard(idx)
            self.pending_activation_indices.discard(idx)
        self.ahead_cells = set()
        return n

    def _cells_within(self, grid_idx, depth):
        """In-bounds cells at Chebyshev distance 1..depth from ``grid_idx``,
        nearest first."""
        if self._has_periodic_projection_dim():
            # The zero offset is the copy of a point on a periodic bound.
            offsets = [(0,) * self.n_proj_dims] + _offsets_within(self.n_proj_dims, depth)
            return self._cells_at_offsets(grid_idx, offsets)
        cells = []
        for offset in _offsets_within(self.n_proj_dims, depth):
            idx = tuple(int(i + o) for i, o in zip(grid_idx, offset))
            if all(0 <= i < s for i, s in zip(idx, self.grid_shape)):
                cells.append(idx)
        return cells

    def create_dynamic_activation_jobs(self, next_job_id, around=None, depth=1):
        """ActivationJobs for un-activated neighbors of high-likelihood grid points.

        `around` limits the search to the given grid points (used when a
        single cell has just finished); by default every cell is considered.
        `depth` is the Chebyshev distance up to which cells are activated: 1
        is the neighbours. A cell further out has no activated neighbour to
        start from and takes the profiled parameters of the cell that
        activates it.
        """
        new_jobs = []
        roi_cutoff = self.global_max_target_val - self.roi_threshold

        # Iterate over a copy as population might be modified
        candidates = self.population.keys() if around is None else [
            idx for idx in around if idx in self.population]
        active_points_in_roi = [
            idx for idx in candidates
            if self.population[idx]['best_fitness'] > roi_cutoff
        ]

        for grid_idx in active_points_in_roi:
            targets = (self._get_valid_neighbors(grid_idx) if depth <= 1
                       else self._cells_within(grid_idx, depth))
            for neighbor_idx in targets:
                if neighbor_idx not in self.population and neighbor_idx not in self.pending_activation_indices:

                    # Warm start from the best activated cell next to the new one.
                    best_warm_start_params = None
                    best_warm_start_fitness = -np.inf

                    for potential_source_idx in self._get_valid_neighbors(neighbor_idx, include_center=True):
                        if potential_source_idx in self.population:
                            source_state = self.population[potential_source_idx]
                            if source_state['best_fitness'] > best_warm_start_fitness:
                                best_warm_start_fitness = source_state['best_fitness']
                                source_best_idx = np.argmax(source_state['fitnesses'])
                                best_warm_start_params = source_state['profiled_params'][source_best_idx]

                    if best_warm_start_params is None:
                        origin = self.population[grid_idx]
                        best_warm_start_params = origin['profiled_params'][np.argmax(origin['fitnesses'])]

                    # Not a neighbour of a cell known to be inside the region
                    # of interest: activated ahead.
                    if best_warm_start_fitness <= roi_cutoff:
                        self.ahead_cells.add(neighbor_idx)

                    job = ActivationJob(
                        job_id=next_job_id,
                        sampler=self,
                        grid_idx=neighbor_idx,
                        warm_start_params=best_warm_start_params
                    )
                    new_jobs.append(job)
                    self.pending_activation_indices.add(neighbor_idx)
                    next_job_id += 1

        return new_jobs, next_job_id

    def _jump_thresholds(self):
        """Per profiled parameter, the difference between two neighbouring
        grid points above which they count as holding different optima, as a
        fraction of the bound extent. Taken from the differences over all
        pairs of neighbours with one inside the region of interest: median
        + ``PATCHING_JUMP_K`` MAD, at least ``PATCHING_JUMP_FLOOR``. None if
        there is no such pair.
        """
        roi_cutoff = self.global_max_target_val - self.roi_threshold
        extent = self._profiled_extent()
        diffs = []
        for idx, state in self.population.items():
            own = self._best_profiled_params(idx)
            for n in self._get_valid_neighbors(idx):
                # each pair once
                if n not in self.population or tuple(n) <= tuple(idx):
                    continue
                if max(state['best_fitness'], self.population[n]['best_fitness']) < roi_cutoff:
                    continue
                delta = self._param_delta(self._best_profiled_params(n), own, self.profiled_dims)
                diffs.append(np.abs(delta) / extent)
        if not diffs:
            return None
        diffs = np.array(diffs)
        med = np.median(diffs, axis=0)
        mad = np.median(np.abs(diffs - med), axis=0)
        return np.maximum(med + PATCHING_JUMP_K * 1.4826 * mad, PATCHING_JUMP_FLOOR)

    def _is_jump(self, idx_a, idx_b, thresholds):
        """True if the profiled parameters of two grid points differ by more
        than ``thresholds`` in some parameter."""
        delta = self._param_delta(self._best_profiled_params(idx_a),
                                  self._best_profiled_params(idx_b), self.profiled_dims)
        return bool(np.any(np.abs(delta) / self._profiled_extent() > thresholds))

    def _patching_sources(self, grid_idx, thresholds, sources=None):
        """Neighbours whose profiled parameters are tested at ``grid_idx``.

        The highest neighbour, if it is higher than the grid point: this
        catches a grid point whose own optimisation ended early. And one
        neighbour for each different optimum held across a jump, whatever its
        value: at the border between two optima the highest neighbour is the
        uphill one, which mostly holds the same optimum as the grid point.

        ``sources`` limits the neighbours considered.
        """
        neighbors = [n for n in self._get_valid_neighbors(grid_idx)
                     if n in self.population and (sources is None or n in sources)]
        if not neighbors:
            return []

        def value(n):
            return self.population[n]['best_fitness']

        chosen = []
        top = max(neighbors, key=value)
        if value(top) > value(grid_idx):
            chosen.append(top)
        if thresholds is not None:
            # Across a jump, nearest first: grid points one step along an
            # axis before the diagonal ones, then by value.
            across = [n for n in neighbors if self._is_jump(n, grid_idx, thresholds)]
            across.sort(key=lambda n: (sum(a != b for a, b in zip(n, grid_idx)), -value(n)))
            for n in across:
                # one neighbour per optimum
                if all(self._is_jump(n, c, thresholds) for c in chosen):
                    chosen.append(n)
        return chosen

    def patching_wave_cap(self):
        """Largest number of patching waves: ``max_patching_waves`` if set,
        else the largest number of grid points along a projection dimension."""
        if self.max_patching_waves is not None:
            return self.max_patching_waves
        return max(self.grid_shape)

    def create_patching_wave_jobs(self, wave_number, updated_points_last_wave, next_job_id):
        """Patching test jobs for one wave.

        Each job evaluates the profiled parameters of one neighbour at a grid
        point (see ``_patching_sources`` for the neighbours chosen).
        Wave 0: every grid point inside the region of interest, and those
        outside it with most of their neighbours inside or with a neighbour
        inside across a jump.
        Wave 1+: the neighbours of the grid points updated in the previous
        wave, tested with the parameters of those grid points.
        """
        from .jobs.patching_test_job import PatchingTestJob

        # The thresholds of the first wave serve the later ones.
        if wave_number == 0 or self._patching_jump_thresholds is None:
            self._patching_jump_thresholds = self._jump_thresholds()
        thresholds = self._patching_jump_thresholds

        if wave_number == 0:
            sources = None
            roi_cutoff = self.global_max_target_val - self.roi_threshold
            candidates = []
            for idx, state in self.population.items():
                if state['best_fitness'] >= roi_cutoff:
                    candidates.append(idx)
                    continue
                neighbors = [n for n in self._get_valid_neighbors(idx) if n in self.population]
                inside = [n for n in neighbors
                          if self.population[n]['best_fitness'] >= roi_cutoff]
                if not inside:
                    continue
                if (len(inside) / len(neighbors) > 0.5
                        or (thresholds is not None
                            and any(self._is_jump(n, idx, thresholds) for n in inside))):
                    candidates.append(idx)
        else:
            sources = set(updated_points_last_wave)
            candidates = list({n for updated_idx in sources
                               for n in self._get_valid_neighbors(updated_idx)
                               if n in self.population})

        jobs = []
        for grid_idx in candidates:
            for neighbor_idx in self._patching_sources(grid_idx, thresholds, sources):
                jobs.append(PatchingTestJob(
                    job_id=next_job_id,
                    sampler=self,
                    grid_idx=grid_idx,
                    test_profiled_params=self._best_profiled_params(neighbor_idx).copy(),
                    wave_number=wave_number,
                ))
                next_job_id += 1

        self.logger.info(f"--- Patching Wave {wave_number}: {len(jobs)} tests at "
                         f"{len(candidates)} grid points ---")
        return jobs, next_job_id


    # ------------------------------------------------------------------
    # Suspect-cell recheck
    # ------------------------------------------------------------------

    def _best_profiled_params(self, grid_idx):
        """Return the best-fitness profiled-params vector at a grid cell."""
        state = self.population[grid_idx]
        return state['profiled_params'][int(np.argmax(state['fitnesses']))]

    def _profiled_extent(self):
        """Per-profiled-dim bounds extent (positive, floored to 1.0)."""
        ext = self.bounds[self.profiled_dims, 1] - self.bounds[self.profiled_dims, 0]
        return np.where(ext > 0, ext, 1.0)

    def _find_suspect_cells(self, wave_number, updated_points_last_wave):
        """Suspect grid indices for a wave. Wave 0 scans all ROI cells and
        flags cells whose profiled-params vector lies far (robust MAD
        threshold) from its neighbour-median. Wave >=1 returns in-population
        neighbours of last wave's winners (boundary propagation).
        """
        if self.n_prof_dims == 0 or not self.population:
            return []

        if wave_number > 0:
            if not updated_points_last_wave:
                return []
            seen = set()
            out = []
            for upd_idx in updated_points_last_wave:
                for n in self._get_valid_neighbors(upd_idx):
                    if n in self.population and n not in seen:
                        seen.add(n)
                        out.append(n)
            return out

        roi_cutoff = self.global_max_target_val - self.roi_threshold
        extent = self._profiled_extent()
        param_d = {}
        for idx, st in self.population.items():
            if st['best_fitness'] < roi_cutoff:
                continue
            neigh = [n for n in self._get_valid_neighbors(idx) if n in self.population]
            if len(neigh) < 2:
                continue
            neigh_params = np.array([self._best_profiled_params(n) for n in neigh])
            own = self._best_profiled_params(idx)
            if self._has_periodic:
                # Place the neighbours' periodic parameters next to the cell's own.
                neigh_params = own + self._param_delta(neigh_params, own, self.profiled_dims)
            median = np.median(neigh_params, axis=0)
            param_d[idx] = float(np.linalg.norm((own - median) / extent))
        if not param_d:
            return []

        # Robust MAD threshold, with a small absolute floor so a
        # perfectly-smooth surface gives no suspects.
        d_vals = np.array(list(param_d.values()))
        d_med = float(np.median(d_vals))
        d_mad = float(np.median(np.abs(d_vals - d_med))) or 1e-12
        d_thresh = max(d_med + self.suspect_param_k * 1.4826 * d_mad, 1e-3)

        suspects = [idx for idx, d in param_d.items() if d > d_thresh]

        max_total = max(1, int(self.suspect_max_fraction * len(param_d)))
        if len(suspects) > max_total:
            suspects.sort(key=param_d.__getitem__, reverse=True)
            suspects = suspects[:max_total]

        self.logger.info(
            f"--- Suspect detection: {len(suspects)}/{len(param_d)} ROI cells flagged "
            f"(d_thresh={d_thresh:.3e}) ---"
        )
        return suspects

    def _get_k_ring(self, grid_idx, k):
        """In-population grid cells at Chebyshev distance 2..k from grid_idx."""
        if k < 2:
            return []
        if self._has_periodic_projection_dim():
            ring = [o for o in itertools.product(range(-k, k + 1), repeat=self.n_proj_dims)
                    if max(abs(v) for v in o) >= 2]
            near = set(self._get_valid_neighbors(grid_idx, include_center=True))
            return [c for c in self._cells_at_offsets(grid_idx, ring)
                    if c not in near and c in self.population]
        cells = []
        center = np.array(grid_idx)
        for offset in itertools.product(range(-k, k + 1), repeat=self.n_proj_dims):
            cheb = max(abs(o) for o in offset)
            if cheb < 2 or cheb > k:
                continue
            nb = tuple(center + np.array(offset))
            if (all(0 <= i < s for i, s in zip(nb, self.grid_shape))
                    and nb in self.population):
                cells.append(nb)
        return cells

    def _gather_suspect_seeds(self, grid_idx, suspect_set):
        """Diverse profiled-params seeds for a suspect cell: own params (as a
        safety baseline), top-fitness non-suspect direct neighbours, top-fitness
        non-suspect cells from an extended Chebyshev ring, and proximity samples
        from the cross-projection global pool. Seeds within 1e-3 (normalised L2)
        of an already-kept seed are dropped.
        """
        extent = self._profiled_extent()
        seeds = []

        def _add(vec):
            v = np.asarray(vec, dtype=float)
            if not np.all(np.isfinite(v)):
                return
            for existing in seeds:
                if np.linalg.norm(self._param_delta(v, existing, self.profiled_dims) / extent) < 1e-3:
                    return
            seeds.append(v.copy())

        _add(self._best_profiled_params(grid_idx))

        def _top_non_suspect(idx_iter, k):
            entries = [(self.population[n]['best_fitness'], n)
                       for n in idx_iter
                       if n in self.population and n not in suspect_set]
            entries.sort(reverse=True)
            for _f, n in entries[:k]:
                _add(self._best_profiled_params(n))

        _top_non_suspect(self._get_valid_neighbors(grid_idx), 3)
        _top_non_suspect(self._get_k_ring(grid_idx, self.suspect_seeds_k_ring), 3)

        n_pool = max(0, int(self.suspect_seeds_from_pool))
        if n_pool > 0:
            target_coords = self._get_grid_coords_from_indices(grid_idx)
            pool_seeds = self._sample_proximity_from_global_pool(n_pool, target_coords)
            if pool_seeds is not None:
                for s in pool_seeds:
                    _add(s)

        return seeds

    def create_suspect_recheck_jobs(self, wave_number, updated_points_last_wave,
                                    next_job_id):
        """Create SuspectRecheckJob instances for the current wave."""
        from .jobs.suspect_recheck_job import SuspectRecheckJob

        candidates = self._find_suspect_cells(wave_number, updated_points_last_wave)
        if not candidates:
            return [], next_job_id

        suspect_set = set(candidates)
        jobs = []
        for grid_idx in candidates:
            seeds = self._gather_suspect_seeds(grid_idx, suspect_set)
            if len(seeds) <= 1:
                # Only own params survived; nothing new to test.
                continue
            jobs.append(SuspectRecheckJob(
                job_id=next_job_id,
                sampler=self,
                grid_idx=grid_idx,
                candidate_seeds=seeds,
                wave_number=wave_number,
            ))
            next_job_id += 1

        self.logger.info(f"--- Suspect Wave {wave_number}: {len(jobs)} jobs created ---")
        return jobs, next_job_id
