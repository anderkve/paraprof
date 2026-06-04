"""Profile Likelihood Projector with Grid-Based Optimization."""
import os
import math
import numpy as np
import itertools
from scipy.stats.qmc import LatinHypercube as LHS
from .logger import get_logger
from .exceptions import (
    InvalidBoundsError, InvalidProjectionError, ConfigurationError,
)
from .sample_io import create_sample_writer, read_samples
from .jobs.lbfgsb_job import LBFGSBJob
from .jobs.activation_job import ActivationJob
from .jobs.de_job import DEGridPointJob


# n_initial_optimizations default: min(MAX, MULTIPLIER * n_dims). It is a safe
# ceiling — the Bayesian stopping rule controls the actual spend — so it is
# generous.
DEFAULT_BASIN_CAP_MULTIPLIER = 50
DEFAULT_BASIN_CAP_MAX = 400

# Basin detection for the initial-optimization stage (always on; see the
# ProfileProjector docstring). Online single-linkage clustering of converged
# optima feeds a Boender-Rinnooy Kan Bayesian stopping rule over ROI optima.
DEFAULT_BASIN_BATCH_SIZE = None             # None -> FD-aware auto (see resolve_initial_opt_batch_size)
# merge_tol is fixed, not a user knob: a sensitivity sweep showed a wide safe
# plateau at/below 0.02, while larger values over-merge distinct optima and bias
# the W count the rule depends on. The right value tracks internal scales
# (bounds-normalization, L-BFGS-B tolerance), not the target.
DEFAULT_BASIN_MERGE_TOL = 0.02              # RMS bounds-normalized param distance to merge optima
DEFAULT_BASIN_UNDISCOVERED_THRESHOLD = 0.5  # stop when E[undiscovered ROI optima] < this
DEFAULT_BASIN_MIN_STARTS_MULTIPLIER = 3     # min starts before the rule applies = mult * n_dims
DEFAULT_BASIN_MIN_STARTS_FLOOR = 10         # ... floored at this many

# Global solution pool size: clip(n_dims * PER_DIM, POOL_SIZE, POOL_MAX).
# Floor keeps the 4-D default at 10 000 entries; ceiling caps master-side
# memory and the per-projection proximity-cache rebuild cost.
DEFAULT_GLOBAL_POOL_SIZE = 10000
DEFAULT_GLOBAL_POOL_PER_DIM = 2500
DEFAULT_GLOBAL_POOL_MAX = 100000

MEMORY_SIZE_MULTIPLIER = 25  # DE memory size = max_grid_size * multiplier

# DE per-cell convergence cutoff (mean logL improvement / generation over
# `de.convergence_window` gens). 1e-6 cuts mean ROI grid error ~2.4x on
# stiff projections (Rosenbrock-style narrow valleys) at negligible cost
# vs the earlier roi_threshold/1000 default.
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6

# DE mutation knobs. Sensitivity benchmarks show all three supported
# strategies give equivalent grid quality; defaults dominate, so these are
# kept hidden from the public API.
DEFAULT_DE_MUTATION_STRATEGY = 'current-to-pbest/1'
DEFAULT_DE_PBEST_FRACTION = 0.1
DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY = 0.5
DEFAULT_DE_CONVERGENCE_WINDOW = 3
DEFAULT_DE_NUM_GENERATIONS = 100000

# allow_skip_DE: let a fresh cell skip DE's global search (idea: reuse the
# already-stored profiled-argmax vectors of its neighbours to decide how much of
# DE's per-cell confirmation budget is actually needed). Every active cell
# normally spends >= `de.convergence_window` generations just proving it has
# converged. When a cell's neighbours agree on the profiled argmax -- i.e. the
# local argmax field is smooth/single-valued -- and the neighbour warm-start
# was the best activation seed, the cell sits on a unimodal patch, so it runs a
# single DE generation and goes straight to the L-BFGS-B polish instead of the
# full window. DE still runs that one generation, so the early exit is
# *measured*, not predicted: a cell that turns out to improve simply keeps
# evolving.
#
# Off by default. A/B benchmarking (examples/run_allow_skip_de_benchmark*.py)
# shows ~10-15% fewer target calls with negligible ROI grid error on targets
# whose profiled (inner) problem is smooth or stiff-but-unimodal -- Himmelblau-
# 4D (-15%, mean |dlogL| ~5e-5) and Rosenbrock-4D (-10%, mean ~7e-3). But on a
# genuinely multimodal inner problem (Rastrigin-4D) one DE generation is too
# little exploration and ROI grid quality degrades, so the feature is opt-in:
# enable it when you know the parameters being profiled out enter smoothly
# (e.g. Gaussian-constrained nuisances), which is the common case.
DEFAULT_DE_ALLOW_SKIP_DE = False
# Reduced per-cell convergence window applied to skip-DE-eligible cells.
SKIP_DE_WINDOW = 1
# Max per-profiled-dim deviation of the neighbours' argmax from their mean (as a
# fraction of the profiled-dim bounds extent) for a cell to be skip-DE-eligible.
# Hidden constant, not a user knob.
SKIP_DE_PHI_SPREAD = 0.05

DEFAULT_LBFGSB_FTOL = 1e-9

DEFAULT_PATCHING_N_NEIGHBORS = 1

# Suspect-cell recheck defaults
DEFAULT_SUSPECT_RECHECK_ENABLED = True
DEFAULT_SUSPECT_MAX_WAVES = 3
DEFAULT_SUSPECT_PARAM_K = 3.0          # MAD multiplier on profiled-param discontinuity
DEFAULT_SUSPECT_MAX_FRACTION = 0.25    # safety cap: max fraction of ROI cells per wave
DEFAULT_SUSPECT_SEEDS_K_RING = 3       # Chebyshev radius for extended-neighbour seeds
DEFAULT_SUSPECT_SEEDS_FROM_POOL = 3
DEFAULT_SUSPECT_POLISH_THRESHOLD = 1e-4  # min improvement over current to trigger LBFGSB

# Activation-population mix ratios (neighbours / global pool / random LHS).
# Hidden — defaults dominate; tuning only degraded benchmarks.
DEFAULT_ACTIVATION_MIX_RATIOS = {
    'neighbors': 0.5,
    'global':    0.25,
    'random':    0.25,
}

DEFAULT_CLUSTERING_EPS_MULTIPLIER = 3.0
DEFAULT_CLUSTERING_PROJECTION_WEIGHT = 1.0


class ProfileProjector:
    """
    Profile Likelihood Projector for computing profile likelihood projections.

    This class primarily holds state and configuration for grid-based profile
    likelihood computation. It supports differential evolution (DE) and L-BFGS-B
    optimization. The execution logic is in the Job classes and master_main.
    """
    def __init__(self,
                 target_func,
                 bounds,
                 projections,
                 # Core tuning parameters (commonly adjusted)
                 roi_threshold=3.0,
                 pop_per_grid_point=3,
                 max_patching_waves=10,
                 lbfgsb_max_iter=50,
                 lbfgsb_polish=True,
                 n_initial_optimizations=None,
                 n_optima=None,
                 initial_points=None,
                 # Optional user-supplied gradient
                 grad_func=None,
                 # Feature toggles
                 use_clustering=True,
                 refinement_direct_eval=False,
                 # I/O
                 samples_output_file=None,
                 warm_start_file=None,
                 # Parameter naming (optional, enables string dims in projections)
                 parameter_names=None,
                 # Advanced configuration (optional)
                 advanced_config=None):
        """
        Initializes the ProfileProjector with simplified interface.

        Parameters
        ----------
        target_func : callable
            The target function to maximize (e.g., log-likelihood)
        bounds : array-like, shape (n_dims, 2)
            Parameter bounds for each dimension
        projections : list of dict
            List of projection configurations, each with 'dims' and 'grid_points'
        grad_func : callable, optional
            User-supplied gradient of ``target_func`` (default: None). Only
            used in the L-BFGS-B paths; ignored by Differential Evolution.
            Sign convention: returns ``∇target_func`` (gradient of the
            function being MAXIMIZED). ParaProf negates internally for the
            minimization objective. Accepts two return formats:

            - length-``n_dims`` array of floats; entries that are ``NaN``,
              ``+inf`` or ``-inf`` are treated as "not provided" and filled
              in by finite differences.
            - ``{dim_index: value}`` dict of known components; any dim not
              in the dict is filled in by finite differences.

            For components the user does provide, paraprof skips the
            corresponding finite-difference target evaluations, cutting
            target-call cost. See ``sampler.target_calls_saved_by_user_gradient``
            for the savings counter.

        Core Tuning Parameters
        ----------------------
        roi_threshold : float, optional
            Region of interest threshold in chi-squared units (default: 3.0)
            Points with likelihood > global_max - roi_threshold are in the ROI
        pop_per_grid_point : int, optional
            Population size per grid point for DE (default: 3)
            Typical values: 1-5. Lower = fewer evaluations per cell but
            more cells need to be activated; higher = more thorough but
            slower per cell.
        max_patching_waves : int, optional
            Maximum number of patching refinement waves (default: 10)
            Typical values: 10-50. Higher = more refinement but more evaluations
        lbfgsb_max_iter : int, optional
            Maximum L-BFGS-B iterations per optimization (default: 50)
            Typical values: 10-50. Higher = more thorough local optimization
        lbfgsb_polish : bool, optional
            Apply L-BFGS-B polishing step after DE optimization (default: True)
            Refines solutions found by DE using gradient-based optimization
        n_initial_optimizations : int, optional
            Number of global L-BFGS-B optimizations to find initial maxima (default: None).
            With basin detection enabled (the default) this is a *cap*, not a fixed
            spend: the Bayesian stopping rule halts early on easy targets and uses
            the full budget only when optima remain, so set it generously. If None,
            auto-configured as min(400, 50 * n_dims) with basin detection on, or the
            modest min(100, 20 * n_dims) when it is off (where it is the fixed number
            of starts). See ``advanced_config['basin_detection']``.
        n_optima : int or dict, optional
            Prior on the number of optima the target has *globally*; use only
            when confident it has one or a few (default: None). A known
            **maximum** stops the initial multistart once that many distinct
            optima are found (the global max is then among them, so the
            ``basin_detection.min_starts`` floor is skipped -- ``n_optima=1``
            stops after the first converged start); a known **minimum** keeps it
            running until that many are found. Pass an ``int`` (exact) or
            ``{'min': int, 'max': int}``. If the true count exceeds the maximum
            the stage may stop before finding the global maximum, so set it only
            when sure.
        initial_points : array-like, shape (n_points, n_dims), optional
            Initial points in full parameter space to activate corresponding grid points (default: None)
            Each point will activate its nearest grid point, independent of optimization
            Useful when you already know good regions of parameter space
            Example: [[3.0, 0.0, -3.0, 0.0]] for a 4D problem

        Feature Toggles
        ---------------
        use_clustering : bool, optional
            Enable mode detection for multi-modal refinement (default: True)
            Helps handle multiple basins during grid refinement
        refinement_direct_eval : bool, optional
            Skip optimization during refinement, just evaluate interpolated points (default: False)
            True = fast, False = thorough

        I/O
        ---
        samples_output_file : str, optional
            Path to save all evaluated points (default: None). Write-only
            during the scan. Format follows the extension: ``.csv`` (text) or
            ``.h5``/``.hdf5`` (HDF5 binary, needs ``h5py``); anything else is
            treated as CSV.
        warm_start_file : str, optional
            Path to a sample file produced by a previous run (any supported
            format; the extension selects the reader). When set and warm-start
            is allowed for the current projection, the master pre-populates
            ``initial_maxima`` from this file and skips the global L-BFGS-B
            starts that would normally seed activation. Default: None (no
            warm start). To round-trip the current run's samples into the
            next one, point ``warm_start_file`` at the same path as
            ``samples_output_file``.

        Parameter Naming
        ----------------
        parameter_names : list of str, optional
            Names for each parameter dimension (default: None).
            When provided, projection ``dims`` entries may be given as parameter
            names (strings) and will be translated to integer indices. The list
            must have length equal to the number of parameters. Mixing names
            and indices within a single projection's ``dims`` is allowed.

        Advanced Configuration
        ----------------------
        advanced_config : dict, optional
            Dictionary for expert-level parameter tuning. Structure:
            {
                'memory_size': int,                # Default: max(grid_sizes) * 25
                'convergence_threshold': float,    # Default: 1e-6

                'de': {
                    'convergence_window': int,     # Default: 3
                    'num_generations': int,        # Default: 100000
                    'max_num_to_evolve': int,      # Default: None (all grid points)
                    'allow_skip_DE': bool,        # Default: False (opt-in)
                },

                'lbfgsb': {
                    'ftol': float,                 # Default: 1e-9
                    'gradient_method': str,        # Default: 'forward'
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
                },

                'suspect_recheck': {
                    'enabled': bool,                # Default: True
                    'max_waves': int,               # Default: 3
                    'param_k': float,               # Default: 3.0
                    'max_fraction': float,          # Default: 0.25
                    'seeds_k_ring': int,            # Default: 3
                    'seeds_from_pool': int,         # Default: 3
                    'polish_threshold': float,      # Default: 1e-4
                },

                'basin_detection': {
                    'batch_size': int,              # Default: None (FD-aware auto)
                    'undiscovered_threshold': float,# Default: 0.5 (0 disables early stop)
                    'min_starts': int,              # Default: None (auto)
                },
            }

        The ``basin_detection`` sub-dict controls the initial-optimization
        stage, where ``n_initial_optimizations`` is treated as a *maximum*:
        paraprof runs a rolling Latin-hypercube multistart, clusters each
        converged optimum online into distinct basins, and applies a
        Boender-Rinnooy Kan Bayesian stopping rule restricted to ROI-competitive
        optima. The stage halts once the expected number of undiscovered ROI
        optima falls below ``undiscovered_threshold`` (after at least
        ``min_starts`` starts), aborting any still-running optimizations; it also
        halts at the ``n_initial_optimizations`` cap, where in-flight runs are
        allowed to finish instead. ``batch_size`` (default ``None``) sets how
        many optimizations run concurrently -- see
        :meth:`resolve_initial_opt_batch_size`. Set ``undiscovered_threshold`` to
        ``0`` to disable early stopping, running the full
        ``n_initial_optimizations`` starts (set that cap explicitly when you do).

        The ``cross_projection`` sub-dict toggles the two cross-projection
        knowledge-transfer hooks. Both default to enabled; set either to
        ``False`` to disable that hook (useful for A/B benchmarking or as a
        safety valve if a target pathology surfaces).

        Several DE knobs that did not show measurable effect on grid quality
        in benchmarking (mutation_strategy, pbest_fraction,
        neighbor_pull_probability, global_pool_size, patching.n_neighbors,
        activation.mix_ratios) are now module-level constants in sampler.py
        and are no longer user-tunable. The basin-clustering ``merge_tol`` is
        likewise a fixed internal constant: a sensitivity sweep showed a wide
        safe plateau at/below its value, with larger values only over-merging
        distinct optima and biasing the stopping statistic.
        """
        # --- Input Validation ---

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

        # Validate projections
        if not isinstance(projections, list) or len(projections) == 0:
            raise InvalidProjectionError("projections must be a non-empty list")

        # Resolve any string dims using parameter_names. We rewrite the
        # projection dicts in place so all downstream code sees integers.
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

        # --- Build configuration with smart defaults ---
        max_grid_size = max(max(proj['grid_points']) for proj in projections)

        config = {
            'memory_size': max_grid_size * MEMORY_SIZE_MULTIPLIER,
            'convergence_threshold': DEFAULT_CONVERGENCE_THRESHOLD,

            'de': {
                'convergence_window': DEFAULT_DE_CONVERGENCE_WINDOW,
                'num_generations': DEFAULT_DE_NUM_GENERATIONS,
                'max_num_to_evolve': None,
                'allow_skip_DE': DEFAULT_DE_ALLOW_SKIP_DE,
            },

            'lbfgsb': {
                'ftol': DEFAULT_LBFGSB_FTOL,
                'gradient_method': 'forward',
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

        # Default cap for the initial-optimization stage: a generous safe
        # ceiling, since the Bayesian stopping rule normally halts the stage
        # well before it. (If early stopping is disabled via
        # undiscovered_threshold=0, set this explicitly.)
        if n_initial_optimizations is None:
            n_initial_optimizations = min(
                DEFAULT_BASIN_CAP_MAX,
                DEFAULT_BASIN_CAP_MULTIPLIER * self.dims,
            )

        # --- Store configuration as instance variables ---
        self.pop_per_grid_point = pop_per_grid_point
        self.roi_threshold = roi_threshold
        self.max_patching_waves = max_patching_waves
        self.lbfgsb_max_iter = lbfgsb_max_iter
        self.n_initial_optimizations = n_initial_optimizations
        self.initial_points = initial_points
        self.refinement_direct_eval = refinement_direct_eval
        self.use_clustering = use_clustering

        # Store advanced config values
        self.memory_size = config['memory_size']
        self.convergence_threshold = config['convergence_threshold']

        # DE configuration
        self.convergence_window = config['de']['convergence_window']
        self.de_num_generations = config['de']['num_generations']
        self.de_max_num_to_evolve = config['de']['max_num_to_evolve']
        self.de_allow_skip_DE = config['de']['allow_skip_DE']

        # L-BFGS-B configuration
        self.lbfgsb_ftol = config['lbfgsb']['ftol']
        self.lbfgsb_gradient_method = config['lbfgsb']['gradient_method']

        # Hidden knobs (kept as instance attributes for read-site compatibility,
        # sourced from module-level constants — see sensitivity benchmarks for
        # rationale)
        self.global_pool_size = min(
            DEFAULT_GLOBAL_POOL_MAX,
            max(DEFAULT_GLOBAL_POOL_SIZE,
                self.dims * DEFAULT_GLOBAL_POOL_PER_DIM),
        )
        self.mutation_strategy = DEFAULT_DE_MUTATION_STRATEGY
        self.pbest_fraction = DEFAULT_DE_PBEST_FRACTION
        self.neighbor_pull_probability = DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY
        self.patching_n_neighbors = DEFAULT_PATCHING_N_NEIGHBORS
        self.activation_mix_ratios = dict(DEFAULT_ACTIVATION_MIX_RATIOS)

        # Clustering configuration
        self.clustering_method = config['clustering']['method']
        self.clustering_eps = config['clustering']['eps']
        self.clustering_min_samples = config['clustering']['min_samples']
        self.clustering_eps_multiplier = config['clustering']['eps_multiplier']
        self.clustering_projection_weight = config['clustering']['projection_weight']

        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        # Warm-start input is now a dedicated path. Set it equal to
        # samples_output_file to round-trip across runs.
        self.warm_start_file = warm_start_file
        # Always initialize buffer (unconditional - makes code robust)
        self.samples_buffer = []
        self.sample_buffer_size = 1000

        # Format is picked from the file extension; the writer opens its file
        # lazily on the first flush (see sample_io).
        if self.samples_output_file:
            # Create the target directory up front so the first flush succeeds
            # even when the user points at a not-yet-existing subdirectory.
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
        # Count of cells that skipped the DE global search via allow_skip_DE
        # (cumulative across projections; for diagnostics).
        self.de_cells_skipped = 0
        # Counters for the grad_func feature (cumulative across projections).
        self.target_calls_saved_by_user_gradient = 0
        self.user_gradient_errors = 0
        self.global_max_target_val = -np.inf
        self.global_solution_pool = []  # Min-heap of (fitness, count, entry) tuples
        self.global_pool_counter = 0  # Unique counter for tiebreaking in heap

        # Cross-projection knowledge transfer (configurable via
        # advanced_config['cross_projection']; default on for both).
        # Both reuse the existing in-memory global_solution_pool, which
        # already accumulates across projections, so no new persistent
        # state is introduced.
        #
        # 1. proximity_warm_start: per-cell activation pop swaps one
        #    random LHS seed for the highest-fitness past evaluation whose
        #    projection-dim coords are closest to the cell.
        # 2. pool_seeded_initial_maxima: at the start of every projection
        #    after the first, seed `initial_maxima` from the pool and skip
        #    the n_initial_optimizations global L-BFGS-B starts that would
        #    otherwise rediscover known maxima.
        self.proximity_warm_start = config['cross_projection']['proximity_warm_start']
        self.pool_seeded_initial_maxima = config['cross_projection']['pool_seeded_initial_maxima']

        # Suspect-cell recheck configuration. Runs after standard patching to
        # catch grid cells (including contiguous strips) that converged to a
        # wrong optimum but slipped past the fitness-only patching filter.
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
        # merge_tol is a fixed internal constant (not a user knob); see the
        # DEFAULT_BASIN_MERGE_TOL sensitivity note above.
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
        # Optional global-optima-count prior, as (min, max) bounds that steer
        # the stopping rule (see basin_detection_should_stop).
        self.basin_min_optima, self.basin_max_optima = (
            self._parse_n_optima(n_optima)
        )
        # Pre-generated LHS start pool for rolling multistart (lazily filled).
        self._initial_opt_start_points = None
        self._initial_opt_lhs_idx = 0
        # Lazy snapshot of (proj_coords, profiled_coords, extent) for the
        # global pool, rebuilt at most once per projection. See
        # _sample_proximity_from_global_pool for the invalidation rule.
        self._proximity_pool_cache = None

        # --- Initial points tracking ---
        self._initial_points_evaluated = False

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
        # Registry of distinct optima found by the initial-optimization stage.
        # Each entry: {'point', 'point_norm', 'target_val', 'count'}. Built up
        # online by register_initial_optimum and consumed by the basin-detection
        # stopping rule. Reset per projection alongside initial_maxima.
        self.initial_optima_registry = []
        self.population = {}  # {grid_idx: state_dict}
        self.activated_grid_indices = set()
        self.pending_activation_indices = set()
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

    def _deep_update(self, base_dict, update_dict):
        """Recursively merge update_dict into base_dict in place."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        # Defensive: resolve string dims if a caller passes a fresh projection
        # dict whose dims weren't normalized at construction time.
        if 'dims' in projection_config:
            resolved = self._resolve_dims(projection_config['dims'],
                                          context="projection")
            if resolved is not None:
                projection_config['dims'] = resolved

        self.logger.info("=" * 80)
        self.logger.info(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        self.logger.info("=" * 80)

        self.projection_dims = sorted(projection_config['dims'])
        # Add +1 to grid points to include both endpoints in linspace
        grid_points = list(projection_config['grid_points']) # Copy
        for i in range(len(grid_points)):
            grid_points[i] += 1
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

        # Detect direct evaluation mode (no profiled dimensions to optimize)
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
        self.current_generation = 0
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0

        # Performance optimizations: caches
        self._neighbor_cache = {}
        # Invalidate the proximity-pool snapshot: projection_dims and
        # profiled_dims have changed, so the cached column slices are stale.
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

                # Create population state for this transferred point
                # This ensures profiled parameter plots include coarse grid data
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
            self.refinement_interpolator = GridInterpolator(coarse_solution)
        else:
            raise ValueError(f"Unknown refinement_method: '{refinement_method}'. "
                           f"Only 'linear' is supported.")

        # Restore global solution pool from coarse run
        if 'global_solution_pool' in coarse_solution:
            import heapq
            # Convert list to heap structure (fitness, count, entry)
            # Use enumeration as the count to maintain uniqueness and ordering
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
            # Use print instead of logger in case logger is already destroyed
            print(f"Warning: Error closing sample file: {e}")


    def _flush_samples_buffer(self):
        """Hand buffered samples to the format writer (crash-safe per batch)."""
        if not self._sample_writer or not self.samples_buffer or self._file_closed:
            return

        try:
            data = np.array([list(params) + [target_val]
                             for params, target_val in self.samples_buffer])
            self._sample_writer.write_batch(data)
            self.samples_buffer = []

        except (OSError, ValueError) as e:
            self.logger.warning(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val):
        """Record a completed target call (master only). Global max is updated by jobs."""
        self.target_calls += 1
        if self.samples_output_file:
            self.samples_buffer.append((params, target_val))
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
        """Clip ``vec`` to the bounds of the given dim subset; uses cached bounds arrays."""
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


    def _get_valid_neighbors(self, grid_idx, include_center=False):
        """Generator yielding in-bounds neighbour indices of ``grid_idx`` (cached)."""
        cache_key = (grid_idx, include_center)

        if cache_key in self._neighbor_cache:
            yield from self._neighbor_cache[cache_key]
            return

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

        Stores FULL parameter vectors so the pool can be reused for any
        future projection. The (fitness, counter, entry) tuple shape keeps
        heapq from comparing dict/array entries when fitnesses tie.
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

        Uses the existing ``global_solution_pool`` (which already accumulates
        across projections), so no new persistent state is introduced. Distance
        is Euclidean in projection-dim space, normalised by the projection-dim
        bounds extent.

        Caches the snapshot matrices per projection so the per-cell call
        becomes a vectorised distance + argpartition; the first call after
        ``_reset_for_new_projection`` rebuilds from the heap, and the cache
        is also rebuilt when the pool has grown by more than 5% since last
        snapshot (the heap reorders on push/replace, but stale snapshots
        are still valid past evaluations).

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
        deltas = (cache['proj'] - target) / cache['extent']
        dists = np.einsum('ij,ij->i', deltas, deltas)

        n_cached = cache['size']
        k = min(n_samples, n_cached)
        if k < n_cached:
            nearest = np.argpartition(dists, k - 1)[:k]
        else:
            nearest = np.arange(n_cached)

        return cache['profiled'][nearest].copy()


    def _initialize_from_global_pool(self):
        """In-memory analogue of ``_initialize_from_warm_start_file``.

        For projections after the first, seed ``initial_maxima`` from the
        accumulated ``global_solution_pool`` so the master can skip the
        ``n_initial_optimizations`` global L-BFGS-B starts that would
        otherwise rediscover known maxima.

        For each pool entry (already a high-fitness full N-D point), map it
        onto the current projection's grid and keep the highest-fitness
        entry per cell. Cells whose best known fitness is within
        ``roi_threshold`` of the running global maximum become candidate
        ``initial_maxima``.
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
        """Seed initial_maxima from a previous run's sample CSV (skips initial opt)."""
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

        best_candidates = {}
        for sample_row in samples:
            params = sample_row[:-1]
            target_val = sample_row[-1]
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

        Returns an empty job list if there are no points to evaluate or they
        have already been processed. Routing the evaluations through a Job
        (rather than a hand-rolled send/recv loop in the master) ensures they
        flow through ``_register_target_call`` and ``_update_global_pool``
        like every other target evaluation.
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

        ``basin_detection.batch_size`` overrides this. The ``None`` default is
        FD-aware: one run's gradient phase already fans out ``fd_width`` parallel
        evaluations (one per dim, two for central differences), so ~``n_workers /
        fd_width`` concurrent runs keep the workers fed -- and fewer runs in
        flight means less partial work discarded when the rule aborts the rest.
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
            dist = np.linalg.norm(xnorm - opt['point_norm']) / sqrt_d
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

        The ``n_optima`` prior (global optima count) takes precedence: a known
        max stops once that many distinct optima have been found -- the global
        maximum is then necessarily among them, so the ``basin_min_starts``
        floor is skipped -- and a known min blocks stopping until that many are
        found. Failing a max prior, the Boender-Rinnooy Kan rule applies once
        ``n_completed`` reaches ``basin_min_starts``: with ``W`` distinct ROI
        optima over ``N`` ROI starts it stops once the expected undiscovered
        ROI optima ``W**2/(N-W-1)`` fall below ``basin_undiscovered_threshold``.
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

        for maximum in self.initial_maxima:
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


    def _create_lbfgsb_jobs_for_active(self, next_job_id, job_type, terminal_status):
        """Spawn one L-BFGS-B job per active grid point.

        `terminal_status` is the status assigned to direct-eval cells (which
        skip optimization): 'optimized' for the one-shot post-activation
        stage, 'converged' for LBFGSB_LOOP (allows dynamic re-activation).
        """
        jobs = []
        for grid_idx, state in self.population.items():
            if state['status'] != 'active':
                continue

            if self.direct_eval_mode or self.n_prof_dims == 0:
                state['status'] = terminal_status
                continue

            state['status'] = 'LBFGSB_queued'
            best_ind_idx = np.argmax(state['fitnesses'])
            start_params_partial = state['profiled_params'][best_ind_idx]
            start_fitness = state['fitnesses'][best_ind_idx]
            start_params_full = self._construct_params(grid_idx, start_params_partial)

            jobs.append(LBFGSBJob(
                job_id=next_job_id,
                job_type=job_type,
                sampler=self,
                opt_dims=tuple(self.profiled_dims),
                start_params=start_params_partial,
                grid_idx=grid_idx,
                start_params_full=start_params_full,
                seed_history=None,
                start_fitness=start_fitness,
            ))
            next_job_id += 1
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

        # Find all transferred coarse grid points (status='refined')
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

                # Use first candidate for activation jobs (optimization will explore from there)
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

        # === INTERPOLATION-BASED PRE-SCREENING ===
        # Determine which coarse cells to process by predicting fine point likelihoods
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

            # Coarse point is outside ROI - check if any fine points might be inside
            # Sample fine grid points within this coarse cell and predict their likelihoods
            fine_start = tuple(ci * self.grid_refinement_factor for ci in coarse_idx)
            fine_end = tuple((ci + 1) * self.grid_refinement_factor for ci in coarse_idx)

            # Ensure cell boundaries are within fine grid bounds
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
                        break  # No need to check more points in this cell
                except Exception as e:
                    # Interpolation failed - be conservative and include the cell
                    self.logger.debug(f"Interpolation failed for fine point {fine_idx}: {e}")
                    cell_has_roi_point = True
                    break

            if cell_has_roi_point:
                coarse_cells_to_process.add(coarse_idx)
                n_cells_accepted += 1

        self.logger.info(f"--- Pre-screening complete: {n_cells_accepted}/{n_cells_screened} coarse cells selected for processing ---")

        # === PROCESS SELECTED COARSE CELLS ===
        # For each selected coarse cell, activate all fine points in its cell
        for coarse_idx in coarse_cells_to_process:

            # Define the cell boundaries in fine grid coordinates
            # Coarse point (ci, cj) maps to fine cell: (ci*RF, cj*RF) to ((ci+1)*RF, (cj+1)*RF)
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

                # Direct-evaluation mode: no profiled parameters to optimize,
                # so refinement is just a single target evaluation at the fine
                # grid point. ActivationJob handles this natively in
                # direct_eval_mode (single eval at grid coords, status set to
                # 'converged'). Bypass interpolation/LBFGSB entirely — an
                # LBFGSBJob with n_opt_dims==0 hangs the master loop because
                # its start() returns no tasks but stays in active_jobs.
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

                # Handle edge cases
                if len(candidates) == 0:
                    continue

                # For direct evaluation mode with multiple candidates, evaluate all and pick best
                if self.refinement_direct_eval and len(candidates) > 1:
                    # Create a special multi-candidate evaluation job
                    all_prof_params = []
                    all_full_params = []

                    for candidate in candidates:
                        prof_params = candidate['profiled_params']
                        if prof_params is None:
                            prof_params = np.array([])
                        elif np.any(np.isnan(prof_params)):
                            continue  # Skip NaN candidates

                        full_params = self._construct_params(fine_idx, prof_params)
                        all_prof_params.append(prof_params)
                        all_full_params.append(full_params)

                    if len(all_prof_params) == 0:
                        continue  # All candidates were invalid

                    # Create ActivationJob that evaluates all candidates
                    job = ActivationJob(
                        job_id=next_job_id,
                        sampler=self,
                        grid_idx=fine_idx,
                        warm_start_params=all_prof_params[0],  # Initial params (will be overridden)
                        mark_converged=True
                    )
                    # Override to evaluate all candidates
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

                    # Handle edge cases
                    if start_params_partial is None:
                        start_params_partial = np.array([])
                    elif np.any(np.isnan(start_params_partial)):
                        continue  # Skip if NaN

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
                        # Override to force single evaluation
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
            # In direct_eval_mode the per-point jobs are ActivationJobs that
            # simply evaluate the target at the fine grid point (no profiled
            # params to optimize).
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
        argmax, so the local argmax field is smooth/single-valued and the cell
        can confirm convergence on a reduced DE window.

        Reuses the neighbours' already-stored best profiled-params vectors --
        no new target evaluations. Returns False whenever the evidence is thin
        (fewer than two neighbours), so the safe default is the full window.
        """
        if self.n_prof_dims == 0:
            return False

        # Multimodality guard: only stand down DE's global search if the
        # neighbour warm-start was the best seed at activation. If a cold
        # random/pool seed beat it, a better inner mode sits nearby and DE's
        # global phase is doing real work here -- keep it.
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
            # allow_skip_DE: a freshly activated cell (no DE generation run yet)
            # whose neighbours agree on the profiled argmax sits on a smooth,
            # single-valued patch, so it skips the DE global search -- one flat
            # generation then the L-BFGS-B polish, instead of the full
            # `convergence_window`. That one generation still runs, so the early
            # exit is self-correcting -- if it improves it keeps going.
            if (self.de_allow_skip_DE
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

    def create_dynamic_activation_jobs(self, next_job_id):
        """ActivationJobs for un-activated neighbors of high-likelihood grid points."""
        new_jobs = []
        roi_cutoff = self.global_max_target_val - self.roi_threshold

        # Iterate over a copy as population might be modified
        active_points_in_roi = [
            idx for idx, state in self.population.items()
            if state['best_fitness'] > roi_cutoff
        ]

        for grid_idx in active_points_in_roi:
            for neighbor_idx in self._get_valid_neighbors(grid_idx):
                if neighbor_idx not in self.population and neighbor_idx not in self.pending_activation_indices:

                    # Find best warm_start_params from neighbor's neighbors (incl. self)
                    best_warm_start_params = None
                    best_warm_start_fitness = -np.inf

                    for potential_source_idx in self._get_valid_neighbors(neighbor_idx, include_center=True):
                        if potential_source_idx in self.population:
                            source_state = self.population[potential_source_idx]
                            if source_state['best_fitness'] > best_warm_start_fitness:
                                best_warm_start_fitness = source_state['best_fitness']
                                source_best_idx = np.argmax(source_state['fitnesses'])
                                best_warm_start_params = source_state['profiled_params'][source_best_idx]

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

    def _get_best_neighbor_profiled_params(self, grid_idx, n_neighbors=None):
        """Top neighbors of ``grid_idx`` as (idx, profiled_params, fitness) tuples, fitness-sorted."""
        if n_neighbors is None:
            n_neighbors = self.patching_n_neighbors

        neighbor_info = []

        for neighbor_idx in self._get_valid_neighbors(grid_idx):
            if neighbor_idx not in self.population:
                continue

            neighbor_state = self.population[neighbor_idx]
            neighbor_fitness = neighbor_state['best_fitness']

            # Get best profiled params from neighbor
            best_idx = np.argmax(neighbor_state['fitnesses'])
            profiled_params = neighbor_state['profiled_params'][best_idx]

            neighbor_info.append((neighbor_idx, profiled_params, neighbor_fitness))

        if not neighbor_info:
            return None

        # Sort by fitness (descending) and take top n_neighbors
        neighbor_info.sort(key=lambda x: x[2], reverse=True)
        return neighbor_info[:n_neighbors]


    def create_patching_wave_jobs(self, wave_number, updated_points_last_wave, next_job_id):
        """Patching test jobs for one wave.

        Wave 0: every ROI cell (+ border cells with majority-ROI neighbors)
        tested with its best neighbor's profiled params.
        Wave 1+: neighbors of cells updated in the previous wave.
        """
        from .jobs.patching_test_job import PatchingTestJob

        # Determine candidate grid points for this wave
        if wave_number == 0:
            # Wave 0: Test all ROI points AND border points
            roi_cutoff = self.global_max_target_val - self.roi_threshold
            candidates = []

            for idx, state in self.population.items():
                if state['best_fitness'] >= roi_cutoff:
                    # Point is within ROI - definitely test it
                    candidates.append(idx)
                else:
                    # Point is below ROI - check if it borders the ROI
                    # (i.e., has neighbors within ROI)
                    neighbor_count = 0
                    roi_neighbor_count = 0
                    for neighbor_idx in self._get_valid_neighbors(idx):
                        if neighbor_idx in self.population:
                            neighbor_count += 1
                            if self.population[neighbor_idx]['best_fitness'] >= roi_cutoff:
                                roi_neighbor_count += 1

                    # Include if >50% of neighbors are in ROI
                    if neighbor_count > 0 and (roi_neighbor_count / neighbor_count) > 0.5:
                        candidates.append(idx)

            self.logger.info(f"--- Patching Wave 0: Testing {len(candidates)} ROI and border points ---")
        else:
            # Wave 1+: Test neighbors of recently updated points
            candidates = set()
            for updated_idx in updated_points_last_wave:
                for neighbor_idx in self._get_valid_neighbors(updated_idx):
                    if neighbor_idx in self.population:
                        candidates.add(neighbor_idx)
            candidates = list(candidates)
            self.logger.info(f"--- Patching Wave {wave_number}: Testing {len(candidates)} neighbor points ---")

        if not candidates:
            return [], next_job_id

        jobs = []
        tests_created = 0

        for grid_idx in candidates:
            current_fitness = self.population[grid_idx]['best_fitness']

            # Get best neighbor(s) profiled parameters
            neighbor_info = self._get_best_neighbor_profiled_params(
                grid_idx, n_neighbors=self.patching_n_neighbors
            )

            if neighbor_info is None:
                continue

            # For each neighbor to test
            for neighbor_idx, test_params, neighbor_fitness in neighbor_info:
                # Optimization: only test if neighbor has better likelihood
                if neighbor_fitness <= current_fitness:
                    continue

                # Create test job
                job = PatchingTestJob(
                    job_id=next_job_id,
                    sampler=self,
                    grid_idx=grid_idx,
                    test_profiled_params=test_params.copy(),
                    wave_number=wave_number
                )
                jobs.append(job)
                tests_created += 1
                next_job_id += 1

        self.logger.info(f"    Created {tests_created} test jobs for wave {wave_number}")
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
            median = np.median(neigh_params, axis=0)
            param_d[idx] = float(np.linalg.norm(
                (self._best_profiled_params(idx) - median) / extent
            ))
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
                if np.linalg.norm((v - existing) / extent) < 1e-3:
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
