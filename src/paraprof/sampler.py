"""
Profile Likelihood Projector with Grid-Based Optimization.
"""
import os
import numpy as np
import itertools
from scipy.stats.qmc import LatinHypercube as LHS
from .logger import get_logger
from .jobs.lbfgsb_job import LBFGSBJob
from .jobs.activation_job import ActivationJob
from .jobs.de_job import DEGridPointJob
from .jobs.cd_job import CoordinateDescentJob


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
"""Maximum number of samples kept in global solution pool"""

MEMORY_SIZE_MULTIPLIER = 25
"""Multiplier for calculating DE/CMA-ES memory size (max_grid_size * multiplier)"""

CONVERGENCE_THRESHOLD_DIVISOR = 1000
"""Divisor for calculating convergence threshold from ROI threshold"""

# Differential Evolution (DE) defaults
DEFAULT_DE_PBEST_FRACTION = 0.1
"""Fraction of top performers to use in pbest archive for DE mutation"""

DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY = 0.5
"""Probability of using neighbor-based mutation in DE"""

DEFAULT_DE_CONVERGENCE_WINDOW = 3
"""Number of generations to track for convergence detection"""

DEFAULT_DE_NUM_GENERATIONS = 100000
"""Default maximum number of DE generations"""

# L-BFGS-B defaults
DEFAULT_LBFGSB_FTOL = 1e-9
"""Function tolerance for L-BFGS-B convergence"""

# Patching defaults
DEFAULT_PATCHING_N_NEIGHBORS = 1
"""Number of neighbors to consider during patching refinement"""

# Activation defaults
DEFAULT_ACTIVATION_MIX_NEIGHBOR_FRACTION = 0.5
"""Fraction of population initialized from neighbor samples"""

DEFAULT_ACTIVATION_MIX_GLOBAL_FRACTION = 0.25
"""Fraction of population initialized from global pool"""

DEFAULT_ACTIVATION_MIX_RANDOM_FRACTION = 0.25
"""Fraction of population initialized randomly (LHS)"""

# Emulator defaults
DEFAULT_EMULATOR_CONFIDENCE_THRESHOLD = 2.0
"""Beta parameter for Upper Confidence Bound in emulator screening"""

DEFAULT_EMULATOR_MIN_NEIGHBORS = 10
"""Minimum number of neighbors for building local emulator"""

DEFAULT_EMULATOR_MAX_NEIGHBORS = 100
"""Maximum number of neighbors for building local emulator"""

DEFAULT_EMULATOR_LENGTH_SCALE = 1.0
"""RBF kernel length scale for Gaussian Process emulator"""

DEFAULT_EMULATOR_NOISE_LEVEL = 0.01
"""White noise level for GP emulator regularization"""

# Coordinate Descent defaults
DEFAULT_CD_MAX_CYCLES = 3
"""Maximum number of coordinate descent cycles"""

DEFAULT_CD_STEP_FRACTION = 0.01
"""Step size as fraction of parameter bounds for coordinate descent"""

# CMA-ES defaults
DEFAULT_CMAES_MAX_GENERATIONS = 100
"""Maximum generations per CMA-ES optimization run"""

DEFAULT_CMAES_NUM_GENERATIONS = 100000
"""Total CMA-ES iteration budget across all grid points"""

CMAES_LAMBDA_BASE = 4
"""Base constant for CMA-ES lambda (population size) formula"""

CMAES_LAMBDA_LOG_COEFFICIENT = 3
"""Coefficient for log term in CMA-ES lambda formula: lambda = base + coef * log(n)"""

CMAES_MU_DIVISOR = 2
"""Divisor for calculating CMA-ES mu (parents) from lambda: mu = lambda / divisor"""

# Clustering defaults
DEFAULT_CLUSTERING_EPS_MULTIPLIER = 3.0
"""Multiplier for automatic DBSCAN epsilon estimation"""

DEFAULT_CLUSTERING_PROJECTION_WEIGHT = 1.0
"""Weight for projection dimensions in clustering distance metric"""


class ProfileProjector:
    """
    Profile Likelihood Projector for computing profile likelihood projections.

    This class primarily holds state and configuration for grid-based profile
    likelihood computation. It supports multiple optimization algorithms including
    differential evolution (DE), L-BFGS-B, and CMA-ES. The execution logic
    is in the Job classes and master_main.
    """
    def __init__(self,
                 target_func,
                 bounds,
                 projections,
                 # Core tuning parameters (commonly adjusted)
                 roi_threshold=3.0,
                 pop_per_grid_point=1,
                 max_patching_waves=10,
                 lbfgsb_max_iter=50,
                 lbfgsb_polish=True,
                 n_initial_optimizations=None,
                 initial_points=None,
                 # Feature toggles
                 use_emulator=False,
                 use_clustering=True,
                 use_cd_refinement=True,
                 refinement_direct_eval=False,
                 # I/O
                 samples_output_file=None,
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
            Population size per grid point for DE (default: 1)
            Typical values: 1-5. Higher = more thorough but slower
        max_patching_waves : int, optional
            Maximum number of patching refinement waves (default: 10)
            Typical values: 10-50. Higher = more refinement but more evaluations
        lbfgsb_max_iter : int, optional
            Maximum L-BFGS-B iterations per optimization (default: 50)
            Typical values: 10-50. Higher = more thorough local optimization
        lbfgsb_polish : bool, optional
            Apply L-BFGS-B polishing step after DE/CMA-ES optimization (default: True)
            Refines solutions found by evolutionary algorithms using gradient-based optimization
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
        use_emulator : bool, optional
            Enable GP-based trial pre-screening (default: False)
            Can reduce evaluations by 30-50% but requires scikit-learn
        use_clustering : bool, optional
            Enable mode detection for multi-modal refinement (default: True)
            Helps handle multiple basins during grid refinement
        use_cd_refinement : bool, optional
            Use coordinate descent for refinement (default: True)
            If False, uses L-BFGS-B (slower but potentially more accurate)
        refinement_direct_eval : bool, optional
            Skip optimization during refinement, just evaluate interpolated points (default: False)
            True = fast, False = thorough

        I/O
        ---
        samples_output_file : str, optional
            Path to save all evaluated points as CSV (default: None)

        Advanced Configuration
        ----------------------
        advanced_config : dict, optional
            Dictionary for expert-level parameter tuning. Structure:
            {
                'global_pool_size': int,           # Default: 10000
                'memory_size': int,                # Default: max(grid_sizes) * 25
                'convergence_threshold': float,    # Default: roi_threshold / 1000

                'de': {
                    'mutation_strategy': str,      # Default: 'current-to-pbest/1'
                    'pbest_fraction': float,       # Default: 0.1
                    'neighbor_pull_probability': float,  # Default: 0.5
                    'convergence_window': int,     # Default: 3
                    'num_generations': int,        # Default: 100000
                    'max_num_to_evolve': int,      # Default: None (all grid points)
                },

                'lbfgsb': {
                    'ftol': float,                 # Default: 1e-9
                    'gradient_method': str,        # Default: 'forward'
                },

                'patching': {
                    'n_neighbors': int,            # Default: 1
                },

                'activation': {
                    'mix_ratios': dict,            # Default: {'neighbors': 0.5, 'global': 0.25, 'random': 0.25}
                },

                'emulator': {
                    'confidence_threshold': float,  # Default: 2.0
                    'min_neighbors': int,          # Default: 10
                    'max_neighbors': int,          # Default: 100
                    'length_scale': float,         # Default: 1.0
                    'noise_level': float,          # Default: 0.01
                },

                'cd': {
                    'max_cycles': int,             # Default: 3
                    'step_fraction': float,        # Default: 0.01
                },

                'cmaes': {
                    'lambda': int,                 # Default: 4 + floor(3*log(n_cont_dims))
                    'mu': int,                     # Default: lambda/2
                    'max_generations': int,        # Default: 100
                    'num_generations': int,        # Default: 100000
                    'max_num_to_evolve': int,      # Default: None (all grid points)
                },

                'clustering': {
                    'method': str,                 # Default: 'dbscan'
                    'eps': float,                  # Default: None (auto-estimated)
                    'min_samples': int,            # Default: None (auto: max(2, n_cont_dims))
                    'eps_multiplier': float,       # Default: 3.0
                    'projection_weight': float,    # Default: 1.0
                },
            }
        """
        from .exceptions import InvalidBoundsError, InvalidProjectionError, ConfigurationError

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

        # Validate projections
        if not isinstance(projections, list) or len(projections) == 0:
            raise InvalidProjectionError("projections must be a non-empty list")

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
            # Auto-configured parameters
            'global_pool_size': DEFAULT_GLOBAL_POOL_SIZE,
            'memory_size': max_grid_size * MEMORY_SIZE_MULTIPLIER,
            'convergence_threshold': roi_threshold / CONVERGENCE_THRESHOLD_DIVISOR,

            # DE parameters
            'de': {
                'mutation_strategy': 'current-to-pbest/1',
                'pbest_fraction': DEFAULT_DE_PBEST_FRACTION,
                'neighbor_pull_probability': DEFAULT_DE_NEIGHBOR_PULL_PROBABILITY,
                'convergence_window': DEFAULT_DE_CONVERGENCE_WINDOW,
                'num_generations': DEFAULT_DE_NUM_GENERATIONS,
                'max_num_to_evolve': None,
            },

            # L-BFGS-B parameters
            'lbfgsb': {
                'ftol': DEFAULT_LBFGSB_FTOL,
                'gradient_method': 'forward',
            },

            # Patching parameters
            'patching': {
                'n_neighbors': DEFAULT_PATCHING_N_NEIGHBORS,
            },

            # Activation parameters
            'activation': {
                'mix_ratios': {
                    'neighbors': DEFAULT_ACTIVATION_MIX_NEIGHBOR_FRACTION,
                    'global': DEFAULT_ACTIVATION_MIX_GLOBAL_FRACTION,
                    'random': DEFAULT_ACTIVATION_MIX_RANDOM_FRACTION
                },
            },

            # Emulator parameters
            'emulator': {
                'confidence_threshold': DEFAULT_EMULATOR_CONFIDENCE_THRESHOLD,
                'min_neighbors': DEFAULT_EMULATOR_MIN_NEIGHBORS,
                'max_neighbors': DEFAULT_EMULATOR_MAX_NEIGHBORS,
                'length_scale': DEFAULT_EMULATOR_LENGTH_SCALE,
                'noise_level': DEFAULT_EMULATOR_NOISE_LEVEL,
            },

            # Coordinate Descent parameters
            'cd': {
                'max_cycles': DEFAULT_CD_MAX_CYCLES,
                'step_fraction': DEFAULT_CD_STEP_FRACTION,
            },

            # CMA-ES parameters
            'cmaes': {
                'lambda': None,  # Will be auto-configured per projection
                'mu': None,      # Will be auto-configured per projection
                'max_generations': DEFAULT_CMAES_MAX_GENERATIONS,
                'num_generations': DEFAULT_CMAES_NUM_GENERATIONS,
                'max_num_to_evolve': None,
            },

            # Clustering parameters
            'clustering': {
                'method': 'dbscan',
                'eps': None,  # Auto-estimated
                'min_samples': None,  # Auto-configured
                'eps_multiplier': DEFAULT_CLUSTERING_EPS_MULTIPLIER,
                'projection_weight': DEFAULT_CLUSTERING_PROJECTION_WEIGHT,
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
        self.use_cd_refinement = use_cd_refinement
        self.use_clustering = use_clustering

        # Store advanced config values
        self.global_pool_size = config['global_pool_size']
        self.memory_size = config['memory_size']
        self.convergence_threshold = config['convergence_threshold']

        # DE configuration
        self.mutation_strategy = config['de']['mutation_strategy']
        self.pbest_fraction = config['de']['pbest_fraction']
        self.neighbor_pull_probability = config['de']['neighbor_pull_probability']
        self.convergence_window = config['de']['convergence_window']
        self.de_num_generations = config['de']['num_generations']
        self.de_max_num_to_evolve = config['de']['max_num_to_evolve']

        # L-BFGS-B configuration
        self.lbfgsb_ftol = config['lbfgsb']['ftol']
        self.lbfgsb_gradient_method = config['lbfgsb']['gradient_method']

        # Patching configuration
        self.patching_n_neighbors = config['patching']['n_neighbors']

        # Activation configuration
        self.activation_mix_ratios = config['activation']['mix_ratios']

        # Emulator configuration
        self.use_emulator = use_emulator
        self.emulator_confidence_threshold = config['emulator']['confidence_threshold']
        self.emulator_min_neighbors = config['emulator']['min_neighbors']
        self.emulator_max_neighbors = config['emulator']['max_neighbors']
        self.emulator_length_scale = config['emulator']['length_scale']
        self.emulator_noise_level = config['emulator']['noise_level']

        # Coordinate Descent configuration
        self.cd_max_cycles = config['cd']['max_cycles']
        self.cd_step_fraction = config['cd']['step_fraction']

        # CMA-ES configuration
        cmaes_lambda = config['cmaes']['lambda']
        cmaes_mu = config['cmaes']['mu']
        if cmaes_lambda is None:
            self.cmaes_lambda_base = lambda n: int(
                CMAES_LAMBDA_BASE + CMAES_LAMBDA_LOG_COEFFICIENT * np.log(max(n, 1))
            )
            self.cmaes_lambda = None
        else:
            self.cmaes_lambda_base = None
            self.cmaes_lambda = cmaes_lambda

        if cmaes_mu is None:
            self.cmaes_mu_base = lambda lam: int(lam / CMAES_MU_DIVISOR)
            self.cmaes_mu = None
        else:
            self.cmaes_mu_base = None
            self.cmaes_mu = cmaes_mu
        self.cmaes_max_generations = config['cmaes']['max_generations']
        self.cmaes_num_generations = config['cmaes']['num_generations']
        self.cmaes_max_num_to_evolve = config['cmaes']['max_num_to_evolve']

        # Clustering configuration
        self.clustering_method = config['clustering']['method']
        self.clustering_eps = config['clustering']['eps']
        self.clustering_min_samples = config['clustering']['min_samples']
        self.clustering_eps_multiplier = config['clustering']['eps_multiplier']
        self.clustering_projection_weight = config['clustering']['projection_weight']

        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
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
        self.global_max_target_val = -np.inf
        self.global_solution_pool = []  # Min-heap of (fitness, count, entry) tuples
        self.global_pool_counter = 0  # Unique counter for tiebreaking in heap

        # --- Evaluation cache for emulator training ---
        # Per-grid-point local caches for efficient emulator training
        self.local_eval_caches = {}  # {grid_idx: [{'params': ..., 'fitness': ..., 'call_number': ...}, ...]}
        self.local_cache_max_size = 500 # 5000  # Much smaller per grid point

        # Keep small global cache for initial optimizations (before grid activation)
        self.global_eval_cache = []
        self.global_cache_max_size = 5000

        # --- DE pre-screening statistics ---
        self.de_trials_generated = 0
        self.de_trials_screened_out = 0

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
        valid_methods = ['de', 'lbfgsb', 'cmaes']
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
        elif self.optimization_method == 'cmaes':
            # Note: lambda/mu will be set after computing n_cont_dims below
            self.logger.info(f"  CMA-ES max generations: {self.cmaes_max_generations}")
            self.logger.info(f"  L-BFGS-B polish after CMA-ES: {'Enabled' if self.lbfgsb_polish else 'Disabled'}")
        self.logger.info(f"  Patching on coarse grid: {'Enabled' if self.patch_coarse_grid else 'Disabled'}")
        if self.is_refinement_run:
            self.logger.info(f"  Patching on refined grid: {'Enabled' if self.patch_refined_grid else 'Disabled'}")

        self.continuous_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

        # Initialize CMA-ES population sizes based on continuous dimensions
        if self.optimization_method == 'cmaes':
            # Set lambda and mu based on continuous dimensions if not explicitly set
            if self.cmaes_lambda_base is not None:
                self.cmaes_lambda = max(4, self.cmaes_lambda_base(self.n_cont_dims))
            if self.cmaes_mu_base is not None and self.cmaes_lambda is not None:
                self.cmaes_mu = max(2, self.cmaes_mu_base(self.cmaes_lambda))

            # Log CMA-ES configuration
            self.logger.info(f"  CMA-ES lambda (population size): {self.cmaes_lambda}")
            self.logger.info(f"  CMA-ES mu (parents): {self.cmaes_mu}")

        # Detect direct evaluation mode (no continuous dimensions to optimize)
        self.direct_eval_mode = (self.n_cont_dims == 0)

        if self.direct_eval_mode:
            self.logger.info("")
            self.logger.info(f"  NOTE: Grid dimensionality equals function dimensionality ({self.dims}D).")
            self.logger.info("  No continuous dimensions to optimize - will evaluate at grid points directly.")
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

        # Cache bounds arrays for continuous and projection dimensions
        self._continuous_bounds = self.bounds[self.continuous_dims] if len(self.continuous_dims) > 0 else None
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
                continuous_params = solution['continuous_params']

                # Store in profile likelihood grid for visualization
                self.profile_likelihood_grid[fine_idx] = likelihood

                # Create population state for this transferred point
                # This ensures continuous parameter plots include coarse grid data
                self.population[fine_idx] = {
                    'continuous_params': np.array([continuous_params]),  # Shape: (1, n_cont_dims)
                    'fitnesses': np.array([likelihood]),
                    'best_fitness': likelihood,
                    'status': 'optimized',  # Mark as already optimized from coarse run
                    'improvement_history': [],
                    'last_update_gen': 0,
                    'optimizer_state': None
                }

                n_transferred += 1

            self.logger.info(f"Transferred {n_transferred} coarse grid solutions to fine grid (likelihood + continuous params)")
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
            - 'continuous_dims': List of continuous dimension indices
            - 'solutions': Dict mapping grid_idx -> solution dict
              Each solution dict contains:
              - 'continuous_params': Best continuous parameters at this grid point
              - 'likelihood': Best likelihood value
              - 'full_params': Complete parameter vector
        """
        solutions = {}

        for grid_idx, state in self.population.items():
            if self.direct_eval_mode:
                # Direct evaluation mode: simpler state structure
                if state['status'] == 'evaluated':
                    solutions[grid_idx] = {
                        'continuous_params': np.empty(0),  # Empty array
                        'likelihood': state['fitness'],
                        'full_params': state['full_params'].copy()
                    }
            else:
                # Normal mode: only export converged/optimized points
                if state['status'] in ['converged', 'optimized']:
                    best_ind_idx = np.argmax(state['fitnesses'])
                    continuous_params = state['continuous_params'][best_ind_idx]
                    likelihood = state['fitnesses'][best_ind_idx]
                    full_params = self._construct_params(grid_idx, continuous_params)

                    solutions[grid_idx] = {
                        'continuous_params': continuous_params.copy(),
                        'likelihood': likelihood,
                        'full_params': full_params.copy()
                    }

        return {
            'grid_axes': [ax.copy() for ax in self.grid_axes],
            'projection_dims': self.projection_dims.copy(),
            'continuous_dims': self.continuous_dims.copy(),
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

        # Perform clustering if enabled and we have continuous parameters
        if self.use_clustering and len(coarse_solution['continuous_dims']) > 0:
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
                self.logger.info("No continuous parameters - clustering not applicable")

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
            - 'continuous_params': np.ndarray
            - 'cluster_id': int or None
            - 'method': str
        """
        # If clustering is not available or disabled, use standard interpolation
        if self.cluster_labels is None or self.cluster_info is None or self.boundary_points is None:
            continuous_params = self.refinement_interpolator.interpolate(grid_coords)
            return [{
                'continuous_params': continuous_params,
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
        if not self._file_closed:
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

        # Add to eval cache for emulator training (only if pre-screening enabled)
        if self.use_emulator:
            # Determine which grid point this evaluation belongs to
            # If we can map it to a grid point, add to local cache; otherwise add to global cache
            try:
                grid_idx = self._get_grid_indices_from_point(params)

                # Add to local cache for this grid point
                if grid_idx not in self.local_eval_caches:
                    self.local_eval_caches[grid_idx] = []

                self.local_eval_caches[grid_idx].append({
                    'params': params.copy(),
                    'fitness': target_val,
                    'call_number': self.target_calls
                })

                # Prune local cache if too large
                # Only prune when reaching 2x max size to reduce pruning frequency
                if len(self.local_eval_caches[grid_idx]) > 2 * self.local_cache_max_size:
                    self._prune_local_cache(grid_idx)

            except (AttributeError, KeyError):
                # Grid not yet initialized or point outside grid - add to global cache
                # This handles initial L-BFGS-B optimizations before grid activation
                self.global_eval_cache.append({
                    'params': params.copy(),
                    'fitness': target_val,
                    'call_number': self.target_calls
                })

                # Prune global cache if too large
                if len(self.global_eval_cache) > self.global_cache_max_size:
                    self.global_eval_cache = self._prune_global_cache()

        # Updating global max is now handled by the jobs
        # to ensure it happens at the right time (e.g., after refinement).

    def _prune_local_cache(self, grid_idx):
        """
        Prunes local eval cache for a specific grid point.

        Strategy: Keep top 50% by fitness + most recent 50% by call number.

        Parameters
        ----------
        grid_idx : tuple
            Grid index of the cache to prune
        """
        cache = self.local_eval_caches[grid_idx]
        target_size = self.local_cache_max_size

        # Sort by fitness (keep best)
        sorted_by_fitness = sorted(cache, key=lambda x: x['fitness'], reverse=True)
        keep_best = sorted_by_fitness[:target_size // 2]

        # Sort by recency (keep recent)
        sorted_by_time = sorted(cache, key=lambda x: x['call_number'], reverse=True)
        keep_recent = sorted_by_time[:target_size // 2]

        # Merge and deduplicate by call_number
        combined = {e['call_number']: e for e in (keep_best + keep_recent)}
        pruned = list(combined.values())

        self.local_eval_caches[grid_idx] = pruned

        self.logger.debug(
            f"Pruned local cache for grid {grid_idx} from {len(cache)} to {len(pruned)} entries"
        )

    def _prune_global_cache(self):
        """
        Prunes global eval cache to max size, keeping best and most recent evaluations.

        Strategy: Keep top 50% by fitness + most recent 50% by call number.

        Returns
        -------
        list
            Pruned evaluation cache
        """
        # Sort by fitness (keep best)
        sorted_by_fitness = sorted(
            self.global_eval_cache,
            key=lambda x: x['fitness'],
            reverse=True
        )
        keep_best = sorted_by_fitness[:self.global_cache_max_size // 2]

        # Sort by recency (keep recent)
        sorted_by_time = sorted(
            self.global_eval_cache,
            key=lambda x: x['call_number'],
            reverse=True
        )
        keep_recent = sorted_by_time[:self.global_cache_max_size // 2]

        # Merge and deduplicate by call_number
        combined = {e['call_number']: e for e in (keep_best + keep_recent)}
        pruned = list(combined.values())

        self.logger.debug(
            f"Pruned global cache from {len(self.global_eval_cache)} to {len(pruned)} entries"
        )

        return pruned


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

    def _construct_params(self, grid_idx, continuous_params, grid_axes=None):
        """Constructs a full parameter vector from grid and continuous parts."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes)
        full_params[self.continuous_dims] = continuous_params
        return full_params


    def _ensure_bounds(self, vec, dims_to_check):
        """
        Ensures a vector's components are within the defined bounds.
        Uses cached bounds arrays for performance.
        """
        # Fast path: use cached bounds for common cases
        if dims_to_check is self.continuous_dims and self._continuous_bounds is not None:
            return np.clip(vec, self._continuous_bounds[:, 0], self._continuous_bounds[:, 1])
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
        Randomly samples continuous parameter values from the global solution pool.

        Extracts continuous dimensions for the current projection from stored full
        parameter vectors.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw

        Returns
        -------
        np.ndarray or None
            Array of shape (n_samples, n_cont_dims) with continuous params for
            current projection, or None if pool is empty
        """
        if not self.global_solution_pool or n_samples == 0:
            return None

        # In direct evaluation mode, there are no continuous dimensions to sample
        if self.direct_eval_mode:
            return None

        # Sample with replacement if needed
        n_available = len(self.global_solution_pool)
        sample_indices = np.random.choice(n_available, size=min(n_samples, n_available), replace=False)

        # Extract continuous dims from full parameter vectors
        # Note: global_solution_pool is a heap of (fitness, count, entry) tuples
        samples = np.array([self.global_solution_pool[i][2]['full_params'][self.continuous_dims]
                           for i in sample_indices])

        return samples


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
                    warm_start_params=point[self.continuous_dims] # Warm start
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
                # Direct evaluation mode: no continuous dimensions to optimize
                if self.direct_eval_mode or self.n_cont_dims == 0:
                    state['status'] = 'optimized'
                    continue

                # Mark as claimed
                state['status'] = 'LBFGSB_queued'

                # Find the best individual to start from
                best_ind_idx = np.argmax(state['fitnesses'])
                start_params_partial = state['continuous_params'][best_ind_idx]
                start_fitness = state['fitnesses'][best_ind_idx]

                # Construct the full parameter vector for the initial task
                start_params_full = self._construct_params(grid_idx, start_params_partial)

                # Create L-BFGS-B job
                job = LBFGSBJob(
                    job_id=next_job_id,
                    job_type='POST_ACTIVATION_LBFGSB',
                    sampler=self,
                    opt_dims=tuple(self.continuous_dims),
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
                # Direct evaluation mode: no continuous dimensions to optimize
                if self.direct_eval_mode or self.n_cont_dims == 0:
                    state['status'] = 'converged'
                    continue

                # Mark as claimed
                state['status'] = 'LBFGSB_queued'

                # Find the best individual to start from
                best_ind_idx = np.argmax(state['fitnesses'])
                start_params_partial = state['continuous_params'][best_ind_idx]
                start_fitness = state['fitnesses'][best_ind_idx]

                # Construct the full parameter vector for the initial task
                start_params_full = self._construct_params(grid_idx, start_params_partial)

                # Create L-BFGS-B job
                job = LBFGSBJob(
                    job_id=next_job_id,
                    job_type='LBFGSB_LOOP',
                    sampler=self,
                    opt_dims=tuple(self.continuous_dims),
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
        starting values for continuous parameters.

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
                    warm_start_params = candidates[0]['continuous_params']
                    # Handle case where interpolation returns None (no continuous dims)
                    if warm_start_params is None:
                        warm_start_params = None
                    # Check for NaN values
                    elif np.any(np.isnan(warm_start_params)):
                        # Fallback: use nearest transferred point's parameters
                        nearest_state = self.population[grid_idx]
                        warm_start_params = nearest_state['continuous_params'][0]
                else:
                    # No candidates - use fallback
                    nearest_state = self.population[grid_idx]
                    warm_start_params = nearest_state['continuous_params'][0]

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
            List of ActivationJob (if refinement_direct_eval=True),
            CoordinateDescentJob, or LBFGSBJob objects
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

                # Get candidate parameters (using clustering if available)
                grid_coords = self._get_grid_coords_from_indices(fine_idx)
                candidates = self._get_refined_initialization_candidates(grid_coords)

                # Handle edge cases
                if len(candidates) == 0:
                    continue

                # For direct evaluation mode with multiple candidates, evaluate all and pick best
                if self.refinement_direct_eval and len(candidates) > 1:
                    # Create a special multi-candidate evaluation job
                    all_cont_params = []
                    all_full_params = []

                    for candidate in candidates:
                        cont_params = candidate['continuous_params']
                        if cont_params is None:
                            cont_params = np.array([])
                        elif np.any(np.isnan(cont_params)):
                            continue  # Skip NaN candidates

                        full_params = self._construct_params(fine_idx, cont_params)
                        all_cont_params.append(cont_params)
                        all_full_params.append(full_params)

                    if len(all_cont_params) == 0:
                        continue  # All candidates were invalid

                    # Create ActivationJob that evaluates all candidates
                    job = ActivationJob(
                        job_id=next_job_id,
                        sampler=self,
                        grid_idx=fine_idx,
                        warm_start_params=all_cont_params[0],  # Initial params (will be overridden)
                        mark_converged=True
                    )
                    # Override to evaluate all candidates
                    job.pop_size = len(all_cont_params)
                    job.all_continuous_params = np.array(all_cont_params)
                    job.all_full_params = all_full_params
                    job.fitnesses = np.full(len(all_cont_params), -np.inf)
                    job.evals_remaining = len(all_cont_params)

                    jobs.append(job)
                    lbfgsb_job_created_for_grid_points.add(fine_idx)
                    next_job_id += 1

                else:
                    # Single candidate or optimization mode: use first candidate
                    candidate = candidates[0]
                    start_params_partial = candidate['continuous_params']

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
                        job.all_continuous_params = np.array([start_params_partial])
                        job.all_full_params = [start_params_full]
                        job.fitnesses = np.full(1, -np.inf)
                        job.evals_remaining = 1
                    else:
                        # Optimization mode: create optimization job (CD or L-BFGS-B)
                        if self.use_cd_refinement:
                            job = CoordinateDescentJob(
                                job_id=next_job_id,
                                job_type='REFINEMENT_CD',
                                sampler=self,
                                opt_dims=tuple(self.continuous_dims),
                                start_params=start_params_partial,
                                grid_idx=fine_idx,
                                start_params_full=start_params_full,
                                start_fitness=-np.inf,
                                max_cycles=self.cd_max_cycles,
                                step_fraction=self.cd_step_fraction
                            )
                        else:
                            job = LBFGSBJob(
                                job_id=next_job_id,
                                job_type='REFINEMENT_LBFGSB',
                                sampler=self,
                                opt_dims=tuple(self.continuous_dims),
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

        if self.refinement_direct_eval:
            opt_method = "direct evaluation"
            if n_multi_candidate > 0:
                self.logger.info(f"--- Multi-candidate evaluations: {n_multi_candidate} grid points with multiple clusters ---")
                self.logger.info(f"--- Single-candidate evaluations: {n_single_candidate} grid points ---")
        else:
            opt_method = "CD" if self.use_cd_refinement else "LBFGSB"
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
                'continuous_params': state['continuous_params'][best_idx],
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

    def create_cmaes_generation_jobs(self, next_job_id, max_num_to_evolve):
        """
        Generates all CMAESGridPointJobs for one generation.

        Similar to DE generation, but for CMA-ES optimization.

        Parameters
        ----------
        next_job_id : int
            Next available job ID
        max_num_to_evolve : int or None
            Maximum number of grid points to evolve (None = all)

        Returns
        -------
        jobs : list
            List of CMAESGridPointJob instances
        next_job_id : int
            Updated job ID counter
        """
        from .jobs.cmaes_job import CMAESGridPointJob

        # Find unconverged points (status 'active' means not yet converged)
        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']

        if not unconverged_indices:
            self.logger.info("All active points have converged. Ending CMA-ES phase.")
            # Debug: log status distribution
            status_counts = {}
            for state in self.population.values():
                status = state['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            self.logger.debug(f"Status distribution: {status_counts}")
            return [], next_job_id

        # --- Prioritize which grid points to evolve ---
        # Use same prioritization as DE: fitness + improvement rate
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
             return [], next_job_id

        indices_to_process_map = np.random.choice(
            np.arange(len(unconverged_indices)),
            size=num_to_evolve,
            replace=False,
            p=probabilities
        )
        indices_to_process = [unconverged_indices[i] for i in indices_to_process_map]

        # --- Create CMA-ES jobs for selected grid points ---
        jobs = []
        for grid_idx in indices_to_process:
            job = CMAESGridPointJob(
                job_id=next_job_id,
                sampler=self,
                grid_idx=grid_idx,
                # For first generation, jobs will initialize from neighbors
                # For subsequent generations, they continue with their own state
                initial_mean=None,
                initial_sigma=None,
                initial_C=None
            )
            jobs.append(job)
            next_job_id += 1

        return jobs, next_job_id

    def create_LBFGSB_job_for_point(self, grid_idx, next_job_id):
        """
        Creates a new L-BFGS-B optimization job for a single converged grid point.
        """
        # Direct evaluation mode: no continuous dimensions to optimize
        if self.direct_eval_mode or self.n_cont_dims == 0:
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
        start_params_partial = state['continuous_params'][best_ind_idx]
        start_fitness = state['fitnesses'][best_ind_idx]

        # Construct the full parameter vector for the initial task
        start_params_full = self._construct_params(grid_idx, start_params_partial)

        # Get the seed history if it exists
        seed_history = state.get('optimizer_state')

        job = LBFGSBJob(
            job_id=next_job_id,
            job_type='LBFGSB',
            sampler=self,
            opt_dims=tuple(self.continuous_dims), # Optimize continuous dims
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
                                best_warm_start_params = source_state['continuous_params'][source_best_idx]

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

    def _get_best_neighbor_continuous_params(self, grid_idx, n_neighbors=None):
        """
        Gets continuous parameters from the best neighbor(s) of a grid point.

        Parameters
        ----------
        grid_idx : tuple
            Grid index to find neighbors for
        n_neighbors : int, optional
            Number of best neighbors to return. Defaults to self.patching_n_neighbors

        Returns
        -------
        list of tuples or None
            List of (neighbor_idx, continuous_params, fitness) for best neighbors,
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

            # Get best continuous params from neighbor
            best_idx = np.argmax(neighbor_state['fitnesses'])
            continuous_params = neighbor_state['continuous_params'][best_idx]

            neighbor_info.append((neighbor_idx, continuous_params, neighbor_fitness))

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

            # Get best neighbor(s) continuous parameters
            neighbor_info = self._get_best_neighbor_continuous_params(
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
                    test_continuous_params=test_params.copy(),
                    wave_number=wave_number
                )
                jobs.append(job)
                tests_created += 1
                next_job_id += 1

        self.logger.info(f"    Created {tests_created} test jobs for wave {wave_number}")
        return jobs, next_job_id
