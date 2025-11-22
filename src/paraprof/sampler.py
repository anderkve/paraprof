"""
Main Grid-Anchored Differential Evolution Sampler.
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


class GridAnchoredDESampler:
    """
    Grid-Anchored Differential Evolution Sampler for profile likelihood computation.

    This class primarily holds state and configuration. The execution logic
    is in the Job classes and master_main.
    """
    def __init__(self,
                 target_func,
                 bounds,
                 projections,
                 pop_per_grid_point=1,
                 mutation_strategy='current-to-rand/1',
                 pbest_fraction=0.1,
                 n_initial_optimizations=20,
                 roi_threshold=3.0,
                 convergence_threshold=1e-5,
                 convergence_window=25,
                 neighbor_pull_probability=0.3,
                 LBFGSB_ftol=1e-7,
                 LBFGSB_max_iter=50,
                 LBFGSB_gradient_method="central",
                 max_patching_waves=10,
                 patching_n_neighbors=1,
                 memory_size=100,
                 global_pool_size=200,
                 activation_mix_ratios=None,
                 samples_output_file=None,
                 use_de_prescreening=False,
                 emulator_confidence_threshold=2.0,
                 emulator_min_neighbors=10,
                 emulator_max_neighbors=100,
                 emulator_length_scale=1.0,
                 emulator_noise_level=0.01,
                 use_cd_refinement=True,
                 cd_max_cycles=3,
                 cd_step_fraction=0.01):
        """
        Initializes the Grid-Anchored DE Sampler.

        Parameters
        ----------
        target_func : callable
            The target function to maximize (e.g., log-likelihood)
        bounds : array-like, shape (n_dims, 2)
            Parameter bounds for each dimension
        projections : list of dict
            List of projection configurations, each with 'dims' and 'grid_points'
        pop_per_grid_point : int
            Population size per grid point
        mutation_strategy : str
            DE mutation strategy ('current-to-rand/1', 'rand/1', 'current-to-pbest/1')
        pbest_fraction : float
            Fraction of population to use for p-best archive
        n_initial_optimizations : int
            Number of initial global optimizations
        roi_threshold : float
            Region of interest threshold (chi-squared units)
        convergence_threshold : float
            Convergence threshold for DE
        convergence_window : int
            Number of generations to check for convergence
        neighbor_pull_probability : float
            Probability of using neighbor-pull mutation
        LBFGSB_ftol : float
            Function tolerance for L-BFGS-B optimization
        LBFGSB_max_iter : int
            Maximum iterations for L-BFGS-B
        LBFGSB_gradient_method : str
            Gradient method ('central' or 'forward')
        max_patching_waves : int
            Maximum number of patching waves (default: 10)
        patching_n_neighbors : int
            Number of best neighbors to test per grid point (default: 1)
        memory_size : int
            Size of F/CR memory for adaptive DE
        global_pool_size : int
            Maximum size of global solution pool
        activation_mix_ratios : dict, optional
            Ratios for mixed initialization strategy.
            Keys: 'neighbors', 'global', 'random'
            Defaults to {'neighbors': 0.5, 'global': 0.3, 'random': 0.2}
        samples_output_file : str, optional
            File to save sampled points
        use_de_prescreening : bool, optional
            Enable emulator-based pre-screening of DE trial points (default: False)
        emulator_confidence_threshold : float, optional
            Confidence multiplier for UCB acquisition function (default: 2.0)
            Higher values are more conservative (fewer trials skipped)
        emulator_min_neighbors : int, optional
            Minimum evaluated points required to build emulator (default: 10)
        emulator_max_neighbors : int, optional
            Maximum evaluated points to use for emulator training (default: 100)
            Limits GP training time by capping the dataset size
        emulator_length_scale : float, optional
            Initial RBF kernel length scale (default: 1.0, auto-tuned)
        emulator_noise_level : float, optional
            White noise level for GP (default: 0.01)
        use_cd_refinement : bool, optional
            Use coordinate descent for grid refinement optimization (default: True)
            When enabled, refinement uses fast coordinate descent instead of L-BFGS-B
        cd_max_cycles : int, optional
            Maximum coordinate descent cycles for refinement (default: 3)
        cd_step_fraction : float, optional
            CD step size as fraction of parameter range (default: 0.01)
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

        # Validate numerical parameters
        if not isinstance(pop_per_grid_point, int) or pop_per_grid_point < 1:
            raise ConfigurationError(
                "pop_per_grid_point must be a positive integer",
                parameter="pop_per_grid_point",
                value=pop_per_grid_point
            )

        if not isinstance(n_initial_optimizations, int) or n_initial_optimizations < 0:
            raise ConfigurationError(
                "n_initial_optimizations must be a non-negative integer",
                parameter="n_initial_optimizations",
                value=n_initial_optimizations
            )

        if not isinstance(roi_threshold, (int, float)) or roi_threshold <= 0:
            raise ConfigurationError(
                "roi_threshold must be a positive number",
                parameter="roi_threshold",
                value=roi_threshold
            )

        if not isinstance(convergence_threshold, (int, float)) or convergence_threshold <= 0:
            raise ConfigurationError(
                "convergence_threshold must be a positive number",
                parameter="convergence_threshold",
                value=convergence_threshold
            )

        if not isinstance(convergence_window, int) or convergence_window < 1:
            raise ConfigurationError(
                "convergence_window must be a positive integer",
                parameter="convergence_window",
                value=convergence_window
            )

        if not isinstance(neighbor_pull_probability, (int, float)) or not (0 <= neighbor_pull_probability <= 1):
            raise ConfigurationError(
                "neighbor_pull_probability must be between 0 and 1",
                parameter="neighbor_pull_probability",
                value=neighbor_pull_probability
            )

        if not isinstance(pbest_fraction, (int, float)) or not (0 < pbest_fraction <= 1):
            raise ConfigurationError(
                "pbest_fraction must be between 0 and 1 (exclusive of 0)",
                parameter="pbest_fraction",
                value=pbest_fraction
            )

        # --- Algorithm parameters ---
        self.pop_per_grid_point = pop_per_grid_point
        allowed_strategies = ['current-to-rand/1', 'rand/1', 'current-to-pbest/1']
        if mutation_strategy not in allowed_strategies:
            raise ValueError(f"mutation_strategy must be one of {allowed_strategies}")
        self.mutation_strategy = mutation_strategy
        self.pbest_fraction = pbest_fraction

        self.n_initial_optimizations = n_initial_optimizations
        self.roi_threshold = roi_threshold
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.neighbor_pull_probability = neighbor_pull_probability
        self.LBFGSB_ftol = LBFGSB_ftol
        self.LBFGSB_max_iter = LBFGSB_max_iter
        self.LBFGSB_gradient_method = LBFGSB_gradient_method
        self.max_patching_waves = max_patching_waves
        self.patching_n_neighbors = patching_n_neighbors
        self.memory_size = memory_size

        # --- Global solution pool parameters ---
        self.global_pool_size = global_pool_size
        if activation_mix_ratios is None:
            self.activation_mix_ratios = {'neighbors': 0.5, 'global': 0.25, 'random': 0.25}
        else:
            self.activation_mix_ratios = activation_mix_ratios

        # --- Emulator configuration ---
        self.use_de_prescreening = use_de_prescreening
        self.emulator_confidence_threshold = emulator_confidence_threshold
        self.emulator_min_neighbors = emulator_min_neighbors
        self.emulator_max_neighbors = emulator_max_neighbors
        self.emulator_length_scale = emulator_length_scale
        self.emulator_noise_level = emulator_noise_level

        # --- Coordinate Descent configuration ---
        self.use_cd_refinement = use_cd_refinement
        self.cd_max_cycles = cd_max_cycles
        self.cd_step_fraction = cd_step_fraction

        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        if self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000

        # --- Persistent State (across projections) ---
        self.target_calls = 0
        self.global_max_target_val = -np.inf
        self.global_solution_pool = []  # List of {'continuous_params', 'fitness', 'grid_idx'}

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

        # --- Refinement State ---
        self.is_refinement_run = False
        self.refinement_factor = None
        self.coarse_grid_solution = None
        self.refinement_interpolator = None

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
        self.lbfgsb_refinement = True  # Default to True (controlled per-projection)
        self.patching_coarse = True  # Default to True (controlled per-projection)
        self.patching_refined = False  # Default to False (controlled per-projection)

        # --- Per-Projection State (reset) ---
        # --- Logger ---
        self.logger = get_logger()

        self._reset_for_new_projection(self.projections[0])


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
        valid_methods = ['de', 'lbfgsb']
        if self.optimization_method not in valid_methods:
            raise ConfigurationError(
                f"Invalid optimization_method: '{self.optimization_method}'. "
                f"Must be one of {valid_methods}",
                parameter="optimization_method",
                value=self.optimization_method
            )

        # Read L-BFGS-B refinement flag
        if 'lbfgsb_refinement' in projection_config:
            self.lbfgsb_refinement = projection_config.get('lbfgsb_refinement', True)
        else:
            self.lbfgsb_refinement = True  # Default

        # Read patching configuration
        self.patching_coarse = projection_config.get('patching_coarse', True)
        self.patching_refined = projection_config.get('patching_refined', False)

        # Print configuration
        self.logger.info(f"  Optimization method: {self.optimization_method}")
        if self.optimization_method == 'de':
            self.logger.info(f"  L-BFGS-B refinement after DE: {'Enabled' if self.lbfgsb_refinement else 'Disabled'}")
        self.logger.info(f"  Patching on coarse grid: {'Enabled' if self.patching_coarse else 'Disabled'}")
        if self.is_refinement_run:
            self.logger.info(f"  Patching on refined grid: {'Enabled' if self.patching_refined else 'Disabled'}")

        self.continuous_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

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

        # --- Handle refinement run: Transfer coarse grid solutions to profile grid ---
        if self.is_refinement_run and self.coarse_grid_solution is not None:
            self.logger.info("--- Transferring coarse grid solutions to fine grid (for visualization) ---")
            coarse_solutions = self.coarse_grid_solution['solutions']
            n_transferred = 0

            for coarse_idx, solution in coarse_solutions.items():
                # Map coarse grid index to fine grid index
                fine_idx = self._map_coarse_to_fine_index(coarse_idx, self.refinement_factor)

                # Verify the fine grid point is valid
                if not all(0 <= i < s for i, s in zip(fine_idx, self.grid_shape)):
                    self.logger.warning(f"Warning: Coarse point {coarse_idx} maps to out-of-bounds fine point {fine_idx}. Skipping.")
                    continue

                likelihood = solution['likelihood']

                # Store in profile likelihood grid for visualization
                # Note: No population state is created - LBFGSB jobs will handle neighbors
                self.profile_likelihood_grid[fine_idx] = likelihood

                n_transferred += 1

            self.logger.info(f"Transferred {n_transferred} coarse grid values to profile likelihood grid")
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
            'global_solution_pool': [s.copy() for s in self.global_solution_pool]
        }


    def setup_refinement_run(self, coarse_solution, refinement_factor):
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
        """
        from .interpolation import GridInterpolator

        self.is_refinement_run = True
        self.refinement_factor = refinement_factor
        self.coarse_grid_solution = coarse_solution

        # Create interpolator from coarse solution
        self.refinement_interpolator = GridInterpolator(coarse_solution)

        # Restore global solution pool from coarse run
        if 'global_solution_pool' in coarse_solution:
            self.global_solution_pool = [s.copy() for s in coarse_solution['global_solution_pool']]
            self.logger.info(f"Restored {len(self.global_solution_pool)} solutions from coarse run's global pool")

        self.logger.info("=" * 80)
        self.logger.info("--- Refinement Run Configuration ---")
        self.logger.info(f"Refinement factor: {refinement_factor}x")
        self.logger.info(f"Coarse grid shape: {coarse_solution['grid_shape']}")
        self.logger.info(f"Coarse grid coverage: {self.refinement_interpolator.get_coverage_fraction():.1%}")
        self.logger.info(f"Number of coarse solutions: {len(coarse_solution['solutions'])}")
        self.logger.info(f"Global solution pool size: {len(self.global_solution_pool)}")
        self.logger.info("=" * 80)


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
        self.refinement_factor = None
        self.coarse_grid_solution = None
        self.refinement_interpolator = None


    def _flush_samples_buffer(self):
        """Writes the content of the samples buffer to the output file."""
        if not self.samples_output_file or not self.samples_buffer:
            return

        try:
            with open(self.samples_output_file, 'a') as f:
                for params, target_val in self.samples_buffer:
                    param_str = ", ".join([f"{p:.10e}" for p in params])
                    f.write(f"{param_str}, {target_val:.10e}\n")

            self.samples_buffer = []
        except IOError as e:
            self.logger.warning(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val):
        """Registers a completed target call (only on master)."""
        self.target_calls += 1
        if hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, target_val))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()

        # Add to eval cache for emulator training (only if pre-screening enabled)
        if self.use_de_prescreening:
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

                # Prune local cache if too large (fast operation on small cache)
                if len(self.local_eval_caches[grid_idx]) > self.local_cache_max_size:
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
        """Converts a point's projection coordinates to the closest grid indices."""
        if grid_axes is None:
            grid_axes = self.grid_axes

        grid_coords = point[self.projection_dims]
        indices = []
        for i, coord in enumerate(grid_coords):
            axis = grid_axes[i]
            index = np.argmin(np.abs(axis - coord))
            indices.append(index)
        return tuple(indices)


    def _get_grid_coords_from_indices(self, grid_idx, grid_axes=None):
        """Converts grid indices to projection parameter values."""
        if grid_axes is None:
            grid_axes = self.grid_axes
        return np.array([grid_axes[i][idx] for i, idx in enumerate(grid_idx)])

    def _get_grid_point_coordinates(self, grid_idx, grid_axes=None):
        """
        Get the parameter values at a grid point (alias for _get_grid_coords_from_indices).

        This method returns the actual parameter values (not indices) for the
        projection dimensions at the given grid point.
        """
        return self._get_grid_coords_from_indices(grid_idx, grid_axes)

    def _construct_params(self, grid_idx, continuous_params, grid_axes=None):
        """Constructs a full parameter vector from grid and continuous parts."""
        full_params = np.zeros(self.dims)
        full_params[self.projection_dims] = self._get_grid_coords_from_indices(grid_idx, grid_axes)
        full_params[self.continuous_dims] = continuous_params
        return full_params


    def _ensure_bounds(self, vec, dims_to_check):
        """Ensures a vector's components are within the defined bounds."""
        # Ensure dims_to_check is a list or array of indices
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
        """Generator to yield valid neighbor indices for a given grid point."""
        for offset in itertools.product([-1, 0, 1], repeat=self.n_proj_dims):
            if not include_center and all(o == 0 for o in offset):
                continue

            neighbor_idx = tuple(np.array(grid_idx) + np.array(offset))

            if all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                yield neighbor_idx


    def _update_global_pool(self, full_params, fitness, grid_idx):
        """
        Adds or updates a solution in the global solution pool.

        Maintains a pool of the best solutions found across all grid points,
        sorted by fitness. The pool is capped at global_pool_size.

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
        # Add the new solution
        self.global_solution_pool.append({
            'full_params': full_params.copy(),
            'fitness': fitness,
            'grid_idx': grid_idx
        })

        # Sort by fitness (descending) and keep top N
        self.global_solution_pool.sort(key=lambda x: x['fitness'], reverse=True)
        self.global_solution_pool = self.global_solution_pool[:self.global_pool_size]


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
        samples = np.array([self.global_solution_pool[i]['full_params'][self.continuous_dims]
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

                # Get interpolated starting parameters
                grid_coords = self._get_grid_coords_from_indices(neighbor_idx)
                interpolated_continuous_params = self.refinement_interpolator.interpolate(grid_coords)

                # Handle case where interpolation returns None (no continuous dims)
                if interpolated_continuous_params is None:
                    warm_start_params = None
                # Check for NaN values from interpolation
                elif np.any(np.isnan(interpolated_continuous_params)):
                    # Fallback: use nearest transferred point's parameters
                    nearest_state = self.population[grid_idx]
                    warm_start_params = nearest_state['continuous_params'][0]
                else:
                    warm_start_params = interpolated_continuous_params

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
        Creates optimization jobs directly for fine grid neighbors during refinement.

        This method bypasses DE entirely and uses interpolated starting points
        from the coarse grid solution for fast convergence. Uses either coordinate
        descent (if use_cd_refinement=True) or L-BFGS-B optimization.

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
            List of CoordinateDescentJob or LBFGSBJob objects
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
                fine_idx = self._map_coarse_to_fine_index(coarse_idx, self.refinement_factor)
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
            fine_start = tuple(ci * self.refinement_factor for ci in coarse_idx)
            fine_end = tuple((ci + 1) * self.refinement_factor for ci in coarse_idx)

            # Ensure cell boundaries are within fine grid bounds
            fine_start = tuple(max(0, fs) for fs in fine_start)
            fine_end = tuple(min(fe, self.grid_shape[i] - 1) for i, fe in enumerate(fine_end))

            # Check all fine points in this cell
            cell_has_roi_point = False
            ranges = [range(fine_start[i], fine_end[i] + 1) for i in range(self.n_proj_dims)]

            for fine_idx in itertools.product(*ranges):
                # Skip coarse grid points (already checked above)
                if self._is_coarse_grid_point(fine_idx, self.refinement_factor):
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
            fine_start = tuple(ci * self.refinement_factor for ci in coarse_idx)
            fine_end = tuple((ci + 1) * self.refinement_factor for ci in coarse_idx)

            # Ensure cell boundaries are within fine grid bounds
            fine_start = tuple(max(0, fs) for fs in fine_start)
            fine_end = tuple(min(fe, self.grid_shape[i] - 1) for i, fe in enumerate(fine_end))

            # Iterate through all points in this cell
            ranges = [range(fine_start[i], fine_end[i] + 1) for i in range(self.n_proj_dims)]
            for fine_idx in itertools.product(*ranges):
                # Skip coarse grid points (already transferred)
                if self._is_coarse_grid_point(fine_idx, self.refinement_factor):
                    continue

                # Skip if already processed
                if fine_idx in lbfgsb_job_created_for_grid_points:
                    continue

                # Get interpolated starting parameters
                grid_coords = self._get_grid_coords_from_indices(fine_idx)
                interpolated_continuous_params = self.refinement_interpolator.interpolate(grid_coords)

                # Handle edge cases in interpolation
                if interpolated_continuous_params is None:
                    # No continuous dims - use empty array
                    start_params_partial = np.array([])
                elif np.any(np.isnan(interpolated_continuous_params)):
                    # Interpolation failed - skip this point
                    continue
                else:
                    start_params_partial = interpolated_continuous_params

                # Construct full parameter vector for initial evaluation
                start_params_full = self._construct_params(fine_idx, start_params_partial)

                # Create optimization job (CD or L-BFGS-B based on configuration)
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
