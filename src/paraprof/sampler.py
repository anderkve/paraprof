"""
Profile Likelihood Projector with Grid-Based Optimization.
"""
import os
import numpy as np
import itertools
from scipy.stats.qmc import LatinHypercube as LHS
from .logger import get_logger
from .exceptions import (
    InvalidBoundsError, InvalidProjectionError, ConfigurationError,
)
from .jobs.lbfgsb_job import LBFGSBJob
from .jobs.activation_job import ActivationJob
from .jobs.de_job import DEGridPointJob


# ============================================================================
# Configuration Constants - Default Values
# ============================================================================

# Initial optimization defaults
DEFAULT_INITIAL_OPT_MULTIPLIER = 20
"""Multiplier for calculating number of initial optimizations (n_dims * multiplier)"""

DEFAULT_INITIAL_OPT_MAX = 100
"""Maximum number of initial optimizations regardless of dimensionality"""

# Global pool and memory defaults
DEFAULT_GLOBAL_POOL_SIZE = 10000
"""Floor for the global solution pool size; the actual default scales with
the target dimensionality (see ``DEFAULT_GLOBAL_POOL_PER_DIM``)."""

DEFAULT_GLOBAL_POOL_PER_DIM = 2500
"""Per-dimension contribution to the global solution pool size. The default
pool size is
``clip(n_dims * DEFAULT_GLOBAL_POOL_PER_DIM,
       DEFAULT_GLOBAL_POOL_SIZE, DEFAULT_GLOBAL_POOL_MAX)``.
This keeps the 4-D default unchanged at 10 000 entries while preventing
eviction of past-projection knowledge in higher-D scans, where both the
number of 2-D projections (``C(n_dims, 2)``) and the per-projection eval
count grow with ``n_dims``."""

DEFAULT_GLOBAL_POOL_MAX = 100000
"""Hard ceiling on the auto-scaled pool size. Beyond this, additional
samples give diminishing returns for cross-projection knowledge transfer
while substantially inflating master-side memory (each entry holds a full
N-D parameter vector) and the per-projection proximity-cache rebuild cost
(``O(pool_size * n_dims)`` numpy allocation)."""

MEMORY_SIZE_MULTIPLIER = 25
"""Multiplier for calculating DE memory size (max_grid_size * multiplier)"""

DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
"""Default DE per-cell convergence cutoff (logL improvement per generation
averaged over `de.convergence_window` generations). The earlier default of
roi_threshold/1000 was 1000x looser than this and left Rosenbrock-style
narrow valleys insufficiently refined; benchmarking on a gold-standard grid
showed that tightening to 1e-6 cuts mean ROI grid error ~2.4x on stiff
projections at negligible target-call cost."""

# Differential Evolution (DE) defaults
DEFAULT_DE_MUTATION_STRATEGY = 'current-to-pbest/1'
"""DE mutation strategy. Hidden — sensitivity benchmarks show all three
supported strategies (current-to-pbest/1, rand/1, current-to-rand/1) give
equivalent grid quality."""

DEFAULT_DE_PBEST_FRACTION = 0.1
"""Fraction of top performers used in pbest archive for DE mutation. Hidden."""

DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY = 0.5
"""Probability of using neighbor-based mutation in DE. Hidden."""

DEFAULT_DE_CONVERGENCE_WINDOW = 3
"""Number of generations to track for convergence detection."""

DEFAULT_DE_NUM_GENERATIONS = 100000
"""Default maximum number of DE generations."""

# L-BFGS-B defaults
DEFAULT_LBFGSB_FTOL = 1e-9
"""Function tolerance for L-BFGS-B convergence"""

# Patching defaults
DEFAULT_PATCHING_N_NEIGHBORS = 1
"""Number of neighbors to consider during patching refinement. Hidden."""

# Suspect-cell recheck defaults
DEFAULT_SUSPECT_RECHECK_ENABLED = True
DEFAULT_SUSPECT_MAX_WAVES = 3
DEFAULT_SUSPECT_PARAM_K = 3.0          # MAD multiplier on profiled-param discontinuity
DEFAULT_SUSPECT_MAX_FRACTION = 0.25    # safety cap: max fraction of ROI cells per wave
DEFAULT_SUSPECT_SEEDS_K_RING = 3       # Chebyshev radius for extended-neighbour seeds
DEFAULT_SUSPECT_SEEDS_FROM_POOL = 3
DEFAULT_SUSPECT_POLISH_THRESHOLD = 1e-4  # min improvement over current to trigger LBFGSB

# Activation defaults — defaults dominate the algorithm; tuning only made
# results worse in benchmarks, so the mix is fixed.
DEFAULT_ACTIVATION_MIX_RATIOS = {
    'neighbors': 0.5,
    'global':    0.25,
    'random':    0.25,
}
"""Activation-population mix ratios (neighbours / global pool / random LHS).
Hidden — defaults dominate; user-tuning only degraded benchmarks."""


# Clustering defaults
DEFAULT_CLUSTERING_EPS_MULTIPLIER = 3.0
"""Multiplier for automatic DBSCAN epsilon estimation"""

DEFAULT_CLUSTERING_PROJECTION_WEIGHT = 1.0
"""Weight for projection dimensions in clustering distance metric"""


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
                 initial_points=None,
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
            Number of global L-BFGS-B optimizations to find initial maxima (default: None)
            If None, auto-configured as min(100, 20 * n_dims)
            Typical values: 20-100. Higher = better initial coverage but slower startup
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
            Path to save all evaluated points as CSV (default: None).
            Write-only during the scan.
        warm_start_file : str, optional
            Path to a CSV produced by a previous run (same format as
            ``samples_output_file``). When set and warm-start is allowed for
            the current projection, the master pre-populates
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
            }

        The ``cross_projection`` sub-dict toggles the two cross-projection
        knowledge-transfer hooks. Both default to enabled; set either to
        ``False`` to disable that hook (useful for A/B benchmarking or as a
        safety valve if a target pathology surfaces).

        Several DE knobs that did not show measurable effect on grid quality
        in benchmarking (mutation_strategy, pbest_fraction,
        neighbor_pull_probability, global_pool_size, patching.n_neighbors,
        activation.mix_ratios) are now module-level constants in sampler.py
        and are no longer user-tunable.
        """
        # --- Input Validation ---

        # Validate target function
        if not callable(target_func):
            raise ConfigurationError("target_func must be callable", parameter="target_func", value=target_func)

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

        # Set n_initial_optimizations with smart default if not provided
        if n_initial_optimizations is None:
            n_initial_optimizations = min(
                DEFAULT_INITIAL_OPT_MAX,
                DEFAULT_INITIAL_OPT_MULTIPLIER * self.dims
            )

        config = {
            'memory_size': max_grid_size * MEMORY_SIZE_MULTIPLIER,
            'convergence_threshold': DEFAULT_CONVERGENCE_THRESHOLD,

            'de': {
                'convergence_window': DEFAULT_DE_CONVERGENCE_WINDOW,
                'num_generations': DEFAULT_DE_NUM_GENERATIONS,
                'max_num_to_evolve': None,
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
        }

        # Merge with advanced_config if provided
        if advanced_config:
            self._deep_update(config, advanced_config)

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

        # File handle and closed flag depend on whether output file is configured
        if self.samples_output_file:
            # File handle is now None - open/close per flush for crash safety
            self._samples_file_handle = None
            self._file_closed = False
        else:
            self._samples_file_handle = None
            self._file_closed = True

        # --- Persistent State (across projections) ---
        self.target_calls = 0
        self.target_call_errors = 0
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

        # --- Per-Projection State (will be reset) ---
        self.projection_dims = None
        self.grid_points_per_dim = None
        self.initial_maxima = []
        self.population = {} # {grid_idx: state_dict}
        self.active_grid_indices = set()
        self.pending_activation_indices = set() # For dynamic activation
        self.current_generation = 0 # DE generation
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.optimization_method = 'de'  # Default optimization method
        self.lbfgsb_polish = lbfgsb_polish
        self.patch_coarse_grid = True  # Default to True (controlled per-projection)
        self.patch_refined_grid = False  # Default to False (controlled per-projection)

        # --- Per-Projection State (reset) ---
        # --- Logger ---
        self.logger = get_logger()

        self._reset_for_new_projection(self.projections[0])

    def _resolve_dims(self, dims, context="projection"):
        """Translate a ``dims`` specification to integer indices.

        Accepts a list of ints, strings, or a mix. Strings are resolved via
        ``self._name_to_dim``. Returns ``None`` if no translation was needed
        (i.e. all entries were already ints), so callers can avoid mutating
        when nothing changed. Raises ``InvalidProjectionError`` on unknown
        names or wrong types.
        """
        if not isinstance(dims, (list, tuple)):
            return None  # let the main validator emit the type error

        any_string = any(isinstance(d, str) for d in dims)
        if not any_string:
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
        """
        Recursively update base_dict with values from update_dict.

        Parameters
        ----------
        base_dict : dict
            Dictionary to update in-place
        update_dict : dict
            Dictionary with updates to apply
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._deep_update(base_dict[key], value)
            else:
                # Overwrite value
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
        self.population = {}
        self.active_grid_indices = set()
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
        """
        Exports the current grid solution for use in refinement runs.

        Returns
        -------
        dict
            Dictionary containing:
            - 'grid_axes': List of arrays defining the grid coordinates
            - 'projection_dims': List of projection dimension indices
            - 'profiled_dims': List of profiled dimension indices
            - 'solutions': Dict mapping grid_idx -> solution dict
              Each solution dict contains:
              - 'profiled_params': Best profiled parameters at this grid point
              - 'likelihood': Best likelihood value
              - 'full_params': Complete parameter vector
        """
        solutions = {}

        for grid_idx, state in self.population.items():
            if self.direct_eval_mode:
                # Direct evaluation mode: simpler state structure.
                # ActivationJob marks direct-eval points as 'converged', and
                # transferred coarse points are marked 'optimized' on the fine
                # grid (see _reset_for_new_projection); accept both.
                if state['status'] in ['converged', 'optimized']:
                    solutions[grid_idx] = {
                        'profiled_params': np.empty(0),  # Empty array
                        'likelihood': state['best_fitness'],
                        'full_params': state['full_params'].copy()
                    }
            else:
                # Normal mode: only export converged/optimized points
                if state['status'] in ['converged', 'optimized']:
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
            # Convert heap to list of entries for export (extract entry from 3-element tuple)
            'global_solution_pool': [entry.copy() for fitness, count, entry in self.global_solution_pool]
        }


    def setup_refinement_run(self, coarse_solution, refinement_factor, refinement_method='linear'):
        """
        Prepares the sampler for a refined grid run.

        This method stores the coarse grid solution and sets flags to enable
        refinement mode. The actual grid setup happens in _reset_for_new_projection.

        Parameters
        ----------
        coarse_solution : dict
            Dictionary returned by export_grid_solution() containing the
            coarse grid's converged solutions
        refinement_factor : int
            Grid refinement factor (e.g., 2 for 2x finer grid in each dimension)
        refinement_method : str, optional
            Interpolation method to use. Only 'linear' is supported.
            Default: 'linear'
        """
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
        """
        Get candidate initialization parameters for a fine grid point.

        Near cluster boundaries, returns multiple candidates from different clusters
        to be evaluated. Away from boundaries, returns a single interpolated candidate.

        Parameters
        ----------
        grid_coords : np.ndarray
            Projection coordinates of the fine grid point

        Returns
        -------
        list of dict
            List of candidates, each containing:
            - 'profiled_params': np.ndarray
            - 'cluster_id': int or None
            - 'method': str
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
        """
        Checks if a fine grid index corresponds to a coarse grid point.

        Parameters
        ----------
        fine_idx : tuple
            Grid index in the fine grid
        refinement_factor : int
            Grid refinement factor (e.g., 2 for 2x refinement)

        Returns
        -------
        bool
            True if fine_idx aligns with a coarse grid point
        """
        return all(idx % refinement_factor == 0 for idx in fine_idx)


    def _map_coarse_to_fine_index(self, coarse_idx, refinement_factor):
        """
        Maps a coarse grid index to the corresponding fine grid index.

        Parameters
        ----------
        coarse_idx : tuple
            Grid index in the coarse grid
        refinement_factor : int
            Grid refinement factor

        Returns
        -------
        tuple
            Corresponding index in the fine grid
        """
        return tuple(idx * refinement_factor for idx in coarse_idx)


    def _map_fine_to_coarse_index(self, fine_idx, refinement_factor):
        """
        Maps a fine grid index to the corresponding coarse grid index.

        Parameters
        ----------
        fine_idx : tuple
            Grid index in the fine grid
        refinement_factor : int
            Grid refinement factor

        Returns
        -------
        tuple or None
            Corresponding coarse grid index if fine_idx aligns with coarse grid,
            otherwise None
        """
        if not self._is_coarse_grid_point(fine_idx, refinement_factor):
            return None
        return tuple(idx // refinement_factor for idx in fine_idx)


    def _cleanup_refinement_state(self):
        """
        Resets refinement-related state after refinement run completes.

        This ensures the sampler is ready for the next projection without
        carrying over refinement flags and data structures.
        """
        self.is_refinement_run = False
        self.grid_refinement_factor = None
        self.coarse_grid_solution = None
        self.refinement_interpolator = None
        self.cluster_labels = None
        self.cluster_info = None
        self.boundary_points = None


    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """
        Explicitly close the sampler and flush all data to disk.

        This method should be called when done using the sampler.
        Safe to call multiple times.
        """
        self._close_sample_file()

    def __del__(self):
        """
        Destructor: attempt cleanup as last resort.

        Note: __del__ is unreliable and may not be called. Always use
        explicit close() or context manager pattern instead.
        """
        # Guard against partially-initialised objects (e.g. when __init__ raised)
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
            pass  # Best effort in __del__

    def _close_sample_file(self):
        """Close the sample output file and flush any remaining data."""
        if self._file_closed:
            return

        try:
            # Flush any remaining buffered samples
            self._flush_samples_buffer()
            self._file_closed = True
            self.logger.debug(f"Closed sample file: {self.samples_output_file}")
        except Exception as e:
            # Use print instead of logger in case logger is already destroyed
            print(f"Warning: Error closing sample file: {e}")


    def _flush_samples_buffer(self):
        """
        Writes the content of the samples buffer to the output file.
        Opens and closes file per flush for crash safety.
        """
        if not self.samples_output_file or not self.samples_buffer or self._file_closed:
            return

        try:
            # Open file for each flush operation (append mode)
            with open(self.samples_output_file, 'a', buffering=1) as f:
                # Vectorized writing for large buffers (more efficient)
                if len(self.samples_buffer) > 100:
                    # Convert buffer to numpy array
                    data = np.array([(list(params) + [target_val])
                                   for params, target_val in self.samples_buffer])
                    # Use savetxt for efficient vectorized formatting
                    np.savetxt(f, data, fmt='%.10e', delimiter=', ')
                else:
                    # Loop for small buffers (overhead not worth vectorization)
                    for params, target_val in self.samples_buffer:
                        param_str = ", ".join([f"{p:.10e}" for p in params])
                        f.write(f"{param_str}, {target_val:.10e}\n")

                # Explicit flush to OS buffers
                f.flush()

            # Clear buffer only after successful write
            self.samples_buffer = []

        except IOError as e:
            self.logger.warning(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val):
        """Registers a completed target call (only on master)."""
        self.target_calls += 1
        # Buffer samples for output if file is configured
        if self.samples_output_file:
            self.samples_buffer.append((params, target_val))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()

        # Updating global max is now handled by the jobs
        # to ensure it happens at the right time (e.g., after refinement).

    def _get_grid_indices_from_point(self, point, grid_axes=None):
        """Converts a point's projection coordinates to the closest grid indices.

        This function exploits the fact that grid_axes is regularly spaced (created
        via np.linspace) to achieve O(1) lookup instead of O(N) linear search.
        """
        if grid_axes is None:
            grid_axes = self.grid_axes

        grid_coords = point[self.projection_dims]
        indices = []

        for i, coord in enumerate(grid_coords):
            axis = grid_axes[i]
            n_points = len(axis)

            # Direct calculation for regularly-spaced grid (O(1) instead of O(N))
            grid_min = axis[0]
            grid_max = axis[-1]

            # Compute normalized position [0, 1] in the grid
            normalized_pos = (coord - grid_min) / (grid_max - grid_min)

            # Map to grid index and round to nearest
            index = int(round(normalized_pos * (n_points - 1)))

            # Clamp to valid range [0, n_points-1]
            index = np.clip(index, 0, n_points - 1)

            indices.append(index)

        return tuple(indices)


    def _get_grid_coords_from_indices(self, grid_idx, grid_axes=None):
        """Converts grid indices to projection parameter values."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _construct_params(self, grid_idx, profiled_params, grid_axes=None):
        """Constructs a full parameter vector from grid and profiled parts."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes)
        full_params[self.profiled_dims] = profiled_params
        return full_params


    def _ensure_bounds(self, vec, dims_to_check):
        """
        Ensures a vector's components are within the defined bounds.
        Uses cached bounds arrays for performance.
        """
        # Fast path: use cached bounds for common cases
        if dims_to_check is self.profiled_dims and self._profiled_bounds is not None:
            return np.clip(vec, self._profiled_bounds[:, 0], self._profiled_bounds[:, 1])
        elif dims_to_check is self.projection_dims and self._projection_bounds is not None:
            return np.clip(vec, self._projection_bounds[:, 0], self._projection_bounds[:, 1])

        # Slow path: handle custom dims_to_check
        dims_to_check = np.array(dims_to_check, dtype=int)
        if vec.shape != self.bounds[dims_to_check, 0].shape:
            # This happens in global optimization, vec is (N_dims,)
            # but dims_to_check might be smaller.
            # We only want to clip the dimensions specified.
            clipped_vec = vec.copy()
            for i, dim_idx in enumerate(dims_to_check):
                clipped_vec[i] = np.clip(vec[i], self.bounds[dim_idx, 0], self.bounds[dim_idx, 1])
            return clipped_vec
        else:
            return np.clip(vec, self.bounds[dims_to_check, 0], self.bounds[dims_to_check, 1])


    def _get_valid_neighbors(self, grid_idx, include_center=False):
        """
        Generator to yield valid neighbor indices for a given grid point.
        Uses caching for performance.
        """
        cache_key = (grid_idx, include_center)

        # Check cache first
        if cache_key in self._neighbor_cache:
            for neighbor in self._neighbor_cache[cache_key]:
                yield neighbor
            return

        # Compute and cache neighbors
        neighbors = []
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue

            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))

            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                neighbors.append(neighbor_idx)

        # Cache for future use
        self._neighbor_cache[cache_key] = neighbors

        # Yield results
        for neighbor in neighbors:
            yield neighbor


    def _update_global_pool(self, full_params, fitness, grid_idx):
        """
        Adds or updates a solution in the global solution pool.

        Maintains a pool of the best solutions found across all grid points
        using a heap for efficient updates. The pool is capped at global_pool_size.

        Note: Stores FULL parameter vectors to support reuse across different projections.

        Parameters
        ----------
        full_params : np.ndarray
            The full parameter vector (all dimensions)
        fitness : float
            The likelihood/fitness value
        grid_idx : tuple or None
            The grid point index where this solution was found (None for global optim)
        """
        import heapq

        # Create solution entry
        entry = {
            'full_params': full_params.copy(),
            'fitness': fitness,
            'grid_idx': grid_idx
        }

        # Use min-heap (we want to efficiently drop the worst solution)
        # Store as (fitness, count, entry) tuple with count as tiebreaker
        # This prevents heapq from comparing dict/array values when fitness values are equal
        if len(self.global_solution_pool) < self.global_pool_size:
            heapq.heappush(self.global_solution_pool, (fitness, self.global_pool_counter, entry))
            self.global_pool_counter += 1
        elif fitness > self.global_solution_pool[0][0]:  # Better than worst
            heapq.heapreplace(self.global_solution_pool, (fitness, self.global_pool_counter, entry))
            self.global_pool_counter += 1


    def _sample_from_global_pool(self, n_samples):
        """
        Randomly samples profiled parameter values from the global solution pool.

        Extracts profiled dimensions for the current projection from stored full
        parameter vectors.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw

        Returns
        -------
        np.ndarray or None
            Array of shape (n_samples, n_prof_dims) with profiled params for
            current projection, or None if pool is empty
        """
        if not self.global_solution_pool or n_samples == 0:
            return None

        # In direct evaluation mode, there are no profiled dimensions to sample
        if self.direct_eval_mode:
            return None

        # Sample with replacement if needed
        n_available = len(self.global_solution_pool)
        sample_indices = np.random.choice(n_available, size=min(n_samples, n_available), replace=False)

        # Extract profiled dims from full parameter vectors
        # Note: global_solution_pool is a heap of (fitness, count, entry) tuples
        samples = np.array([self.global_solution_pool[i][2]['full_params'][self.profiled_dims]
                           for i in sample_indices])

        return samples


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
        """
        Initializes initial_maxima from a previous sample file.

        This method reads all samples from an accumulated CSV file and uses them
        to populate the initial_maxima list, avoiding expensive global optimization
        for subsequent projection runs.

        Parameters
        ----------
        warm_start_file : str
            Path to CSV file containing previous samples (params, target_val)
        """
        if not warm_start_file or not os.path.exists(warm_start_file):
            self.logger.info("  No warm-start file found or provided. Skipping warm start.")
            return

        self.logger.info(f"--- Initializing from warm-start file: {warm_start_file} ---")
        try:
            samples = np.loadtxt(warm_start_file, delimiter=',')
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
        except Exception as e:
            self.logger.info(f"  Warning: Could not read warm-start file. Error: {e}. Skipping.")
            return

        # Group samples by grid index and keep the best for each
        best_candidates = {}
        for sample_row in samples:
            params = sample_row[:-1]
            target_val = sample_row[-1]

            # Validate sample is within bounds
            if not np.all((params >= self.bounds[:, 0]) & (params <= self.bounds[:, 1])):
                continue

            grid_idx = self._get_grid_indices_from_point(params)

            if grid_idx not in best_candidates or target_val > best_candidates[grid_idx]['target_val']:
                best_candidates[grid_idx] = {'params': params, 'target_val': target_val}

        if not best_candidates:
            self.logger.info("  No valid samples found in warm-start file for the current grid.")
            return

        # Update global maximum from warm start samples
        warm_start_max_target_val = max(c['target_val'] for c in best_candidates.values())
        self.global_max_target_val = max(self.global_max_target_val, warm_start_max_target_val)

        # Define ROI cutoff
        roi_cutoff = self.global_max_target_val - self.roi_threshold

        # Populate initial_maxima with samples above ROI threshold
        warm_start_maxima = []
        for grid_idx, candidate in best_candidates.items():
            if candidate['target_val'] >= roi_cutoff:
                warm_start_maxima.append({
                    'point': candidate['params'],
                    'target_val': candidate['target_val']
                })

        # Sort by target_val (best first)
        warm_start_maxima.sort(key=lambda x: x['target_val'], reverse=True)

        # Add to initial_maxima list
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

    def create_initial_optimization_jobs(self, next_job_id):
        """Generates L-BFGS-B jobs for finding initial maxima."""
        self.logger.info(f"--- Generating {self.n_initial_optimizations} initial optimization jobs ---")
        jobs = []
        sampler = LHS(d=self.dims, seed=np.random.randint(1e6, 1e12))
        unit_samples = sampler.random(n=self.n_initial_optimizations)
        start_points = self.bounds[:, 0] + unit_samples * (self.bounds[:, 1] - self.bounds[:, 0])

        for i, start_point in enumerate(start_points):
            job = LBFGSBJob(
                job_id=next_job_id,
                job_type='INITIAL_OPTIMIZATION',
                sampler=self,
                opt_dims=tuple(range(self.dims)), # Global optimization
                start_params=start_point,          # Full vector
                grid_idx=None,                     # No grid anchor
                start_params_full=start_point,     # Full vector
                seed_history=None
            )
            jobs.append(job)
            next_job_id += 1

        return jobs, next_job_id


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


    def create_post_activation_lbfgsb_jobs(self, next_job_id):
        """
        Create L-BFGS-B jobs for all activated grid points (alternative to DE).

        This is used when optimization_method='lbfgsb' to directly optimize
        activated grid points without running differential evolution.

        Each activated grid point gets a single L-BFGS-B job starting from
        the best point in its initial population.

        Parameters
        ----------
        next_job_id : int
            The next available job ID

        Returns
        -------
        jobs : list
            List of LBFGSBJob instances
        next_job_id : int
            Updated job ID counter
        """
        from .jobs.lbfgsb_job import LBFGSBJob

        jobs = []

        # Find all grid points that need optimization
        # (status 'active' means activated but not yet optimized)
        for grid_idx, state in self.population.items():
            if state['status'] == 'active':
                # Direct evaluation mode: no profiled dimensions to optimize
                if self.direct_eval_mode or self.n_prof_dims == 0:
                    state['status'] = 'optimized'
                    continue

                # Mark as claimed
                state['status'] = 'LBFGSB_queued'

                # Find the best individual to start from
                best_ind_idx = np.argmax(state['fitnesses'])
                start_params_partial = state['profiled_params'][best_ind_idx]
                start_fitness = state['fitnesses'][best_ind_idx]

                # Construct the full parameter vector for the initial task
                start_params_full = self._construct_params(grid_idx, start_params_partial)

                # Create L-BFGS-B job
                job = LBFGSBJob(
                    job_id=next_job_id,
                    job_type='POST_ACTIVATION_LBFGSB',
                    sampler=self,
                    opt_dims=tuple(self.profiled_dims),
                    start_params=start_params_partial,
                    grid_idx=grid_idx,
                    start_params_full=start_params_full,
                    seed_history=None,  # No history seeding for post-activation
                    start_fitness=start_fitness
                )

                jobs.append(job)
                next_job_id += 1

        self.logger.info(f"Created {len(jobs)} post-activation L-BFGS-B jobs")

        return jobs, next_job_id

    def create_lbfgsb_loop_jobs(self, next_job_id):
        """
        Create L-BFGS-B jobs for active (non-converged) grid points in the LBFGSB_LOOP stage.

        This method is used in the iterative LBFGSB_LOOP workflow, similar to how
        DE_LOOP creates jobs. It only processes grid points with status='active',
        skipping those already marked as 'converged' or 'LBFGSB_queued'.

        Parameters
        ----------
        next_job_id : int
            The next available job ID

        Returns
        -------
        jobs : list
            List of LBFGSBJob instances
        next_job_id : int
            Updated job ID counter
        """
        from .jobs.lbfgsb_job import LBFGSBJob

        jobs = []

        # Find all grid points that need optimization
        # (status 'active' means activated but not yet converged)
        for grid_idx, state in self.population.items():
            if state['status'] == 'active':
                # Direct evaluation mode: no profiled dimensions to optimize
                if self.direct_eval_mode or self.n_prof_dims == 0:
                    state['status'] = 'converged'
                    continue

                # Mark as claimed
                state['status'] = 'LBFGSB_queued'

                # Find the best individual to start from
                best_ind_idx = np.argmax(state['fitnesses'])
                start_params_partial = state['profiled_params'][best_ind_idx]
                start_fitness = state['fitnesses'][best_ind_idx]

                # Construct the full parameter vector for the initial task
                start_params_full = self._construct_params(grid_idx, start_params_partial)

                # Create L-BFGS-B job
                job = LBFGSBJob(
                    job_id=next_job_id,
                    job_type='LBFGSB_LOOP',
                    sampler=self,
                    opt_dims=tuple(self.profiled_dims),
                    start_params=start_params_partial,
                    grid_idx=grid_idx,
                    start_params_full=start_params_full,
                    seed_history=None,  # Could potentially seed from neighbors in future
                    start_fitness=start_fitness
                )

                jobs.append(job)
                next_job_id += 1

        return jobs, next_job_id


    def create_refinement_activation_jobs(self, next_job_id):
        """
        Generates ActivationJobs for new fine grid points using interpolation.

        This method is called during refinement runs to activate neighbors of
        transferred coarse grid points, using interpolation to predict good
        starting values for profiled parameters.

        Parameters
        ----------
        next_job_id : int
            Next available job ID

        Returns
        -------
        jobs : list
            List of ActivationJob objects
        next_job_id : int
            Updated job ID counter
        """
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
        """
        Creates jobs for fine grid neighbors during refinement.

        This method bypasses DE entirely and uses interpolated starting points
        from the coarse grid solution. Depending on configuration:
        - If refinement_direct_eval=True: Creates single-evaluation jobs at interpolated points
        - If refinement_direct_eval=False: Creates optimization jobs (CD or L-BFGS-B)

        Uses interpolation-based pre-screening to determine which coarse cells
        should be processed: if any fine grid point within a coarse cell is
        predicted (via interpolation) to be within the ROI, that cell is processed.

        Parameters
        ----------
        next_job_id : int
            Next available job ID

        Returns
        -------
        jobs : list
            List of ActivationJob (if refinement_direct_eval=True) or LBFGSBJob objects
        next_job_id : int
            Updated job ID counter
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

        active_pop_list = list(self.active_grid_indices)
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
        """
        Creates a new L-BFGS-B optimization job for a single converged grid point.
        """
        # Direct evaluation mode: no profiled dimensions to optimize
        if self.direct_eval_mode or self.n_prof_dims == 0:
            # Mark as optimized without L-BFGS-B refinement
            state = self.population.get(grid_idx)
            if state:
                state['status'] = 'optimized'
            return None

        state = self.population.get(grid_idx)

        # Safety check: only optimize active/converged/optimized points
        if not state or state['status'] == 'LBFGSB_queued':
            return None

        # Mark as claimed
        state['status'] = 'LBFGSB_queued'

        # === L-BFGS-B OPTIMIZATION ===
        # Find the best individual to start from
        best_ind_idx = np.argmax(state['fitnesses'])
        start_params_partial = state['profiled_params'][best_ind_idx]
        start_fitness = state['fitnesses'][best_ind_idx]

        # Construct the full parameter vector for the initial task
        start_params_full = self._construct_params(grid_idx, start_params_partial)

        # Get the seed history if it exists
        seed_history = state.get('optimizer_state')

        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='LBFGSB',
            sampler=self,
            opt_dims=tuple(self.profiled_dims), # Optimize profiled dims
            start_params=start_params_partial,     # Partial vector
            grid_idx=grid_idx,                     # Grid anchor
            start_params_full=start_params_full,   # Full vector for first eval
            seed_history=seed_history,
            start_fitness=start_fitness            # Pass current best fitness
        )

        return (job, next_job_id + 1)

    def create_dynamic_activation_jobs(self, next_job_id):
        """
        Creates ActivationJobs for neighbors of high-likelihood grid points
        that have not yet been activated.
        """
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
        """
        Gets profiled parameters from the best neighbor(s) of a grid point.

        Parameters
        ----------
        grid_idx : tuple
            Grid index to find neighbors for
        n_neighbors : int, optional
            Number of best neighbors to return. Defaults to self.patching_n_neighbors

        Returns
        -------
        list of tuples or None
            List of (neighbor_idx, profiled_params, fitness) for best neighbors,
            sorted by fitness (best first). Returns None if no valid neighbors exist.
        """
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
        """
        Creates patching test jobs for a wave.

        Wave 0: Tests all ROI points with their best neighbor's parameters
        Wave 1+: Tests neighbors of points updated in previous wave

        Parameters
        ----------
        wave_number : int
            Current wave number (0-indexed)
        updated_points_last_wave : list or None
            List of grid indices updated in previous wave (None for wave 0)
        next_job_id : int
            Next available job ID

        Returns
        -------
        jobs : list
            List of PatchingTestJob objects
        next_job_id : int
            Updated job ID counter
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
