"""
Main Grid-Anchored Differential Evolution Sampler.
"""
import os
import numpy as np
import itertools
from scipy.stats.qmc import LatinHypercube as LHS
from jobs.lbfgsb_job import LBFGSBJob
from jobs.activation_job import ActivationJob
from jobs.de_job import DEGridPointJob


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
                 patching_fraction=0.1,
                 patching_conv_threshold=0.01,
                 max_patching_iterations=100,
                 memory_size=100,
                 samples_output_file=None):
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
        patching_fraction : float
            Fraction of points to patch per iteration
        patching_conv_threshold : float
            Convergence threshold for patching
        max_patching_iterations : int
            Maximum patching iterations
        memory_size : int
            Size of F/CR memory for adaptive DE
        samples_output_file : str, optional
            File to save sampled points
        """
        self.target_func = target_func
        self.bounds = np.array(bounds)
        self.dims = len(self.bounds)
        self.projections = projections

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
        self.patching_fraction = patching_fraction
        self.patching_conv_threshold = patching_conv_threshold
        self.max_patching_iterations = max_patching_iterations
        self.memory_size = memory_size

        # --- File I/O setup ---
        self.samples_output_file = samples_output_file
        if self.samples_output_file:
            self.samples_buffer = []
            self.sample_buffer_size = 1000

        # --- Persistent State (across projections) ---
        self.target_calls = 0
        self.global_max_target_val = -np.inf

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

        # --- Per-Projection State (reset) ---
        self._reset_for_new_projection(self.projections[0])


    def _reset_for_new_projection(self, projection_config):
        """Resets the state for a new projection run."""
        print("\n" + "="*80)
        print(f"--- Configuring for projection on dims: {projection_config['dims']} ---")
        print("="*80 + "\n")

        self.projection_dims = sorted(projection_config['dims'])
        # _Anders: Add +1 to number of grid points, to get nicer coordinates
        grid_points = list(projection_config['grid_points']) # Copy
        for i in range(len(grid_points)):
            grid_points[i] += 1
        self.grid_points_per_dim = grid_points

        if len(self.projection_dims) != len(self.grid_points_per_dim):
            raise ValueError("Length of projection_dims must match length of grid_points_per_dim.")
        if any(d >= self.dims for d in self.projection_dims):
            raise ValueError("projection_dims contains an index out of bounds.")

        self.continuous_dims = [d for d in range(self.dims) if d not in self.projection_dims]
        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

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

        # --- Handle refinement run: Transfer coarse grid solutions to fine grid ---
        if self.is_refinement_run and self.coarse_grid_solution is not None:
            print("--- Transferring coarse grid solutions to fine grid ---")
            coarse_solutions = self.coarse_grid_solution['solutions']
            n_transferred = 0

            for coarse_idx, solution in coarse_solutions.items():
                # Map coarse grid index to fine grid index
                fine_idx = self._map_coarse_to_fine_index(coarse_idx, self.refinement_factor)

                # Verify the fine grid point is valid
                if not all(0 <= i < s for i, s in zip(fine_idx, self.grid_shape)):
                    print(f"Warning: Coarse point {coarse_idx} maps to out-of-bounds fine point {fine_idx}. Skipping.")
                    continue

                # Create population state for this transferred point
                continuous_params = solution['continuous_params']
                likelihood = solution['likelihood']

                # Create a population with a single individual (the transferred solution)
                self.population[fine_idx] = {
                    'continuous_params': np.array([continuous_params]),
                    'fitnesses': np.array([likelihood]),
                    'best_fitness': likelihood,
                    'status': 'optimized',  # Mark as already optimized
                    'improvement_history': [],
                    'optimizer_state': None
                }

                # Add to active grid (even though status is 'optimized')
                self.active_grid_indices.add(fine_idx)

                # Update profile likelihood grid
                self.profile_likelihood_grid[fine_idx] = likelihood

                n_transferred += 1

            print(f"Transferred {n_transferred} coarse grid points to fine grid")
            print(f"Fine grid now has {len(self.population)} populated points")
            print("="*80 + "\n")


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
            # Only export converged/optimized points
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
            'grid_shape': self.grid_shape
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
        from interpolation import GridInterpolator

        self.is_refinement_run = True
        self.refinement_factor = refinement_factor
        self.coarse_grid_solution = coarse_solution

        # Create interpolator from coarse solution
        self.refinement_interpolator = GridInterpolator(coarse_solution)

        print("\n" + "="*80)
        print("--- Refinement Run Configuration ---")
        print(f"Refinement factor: {refinement_factor}x")
        print(f"Coarse grid shape: {coarse_solution['grid_shape']}")
        print(f"Coarse grid coverage: {self.refinement_interpolator.get_coverage_fraction():.1%}")
        print(f"Number of coarse solutions: {len(coarse_solution['solutions'])}")
        print("="*80 + "\n")


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
            print(f"Warning: Could not write to sample file: {e}")


    def _register_target_call(self, params, target_val):
        """Registers a completed target call (only on master)."""
        self.target_calls += 1
        if hasattr(self, 'samples_buffer'):
            self.samples_buffer.append((params, target_val))
            if len(self.samples_buffer) >= self.sample_buffer_size:
                self._flush_samples_buffer()
        # Updating global max is now handled by the jobs
        # to ensure it happens at the right time (e.g., after refinement).


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
            print("  No warm-start file found or provided. Skipping warm start.")
            return

        print(f"--- Initializing from warm-start file: {warm_start_file} ---")
        try:
            samples = np.loadtxt(warm_start_file, delimiter=',')
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
        except Exception as e:
            print(f"  Warning: Could not read warm-start file. Error: {e}. Skipping.")
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
            print("  No valid samples found in warm-start file for the current grid.")
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

        print(f"--- Loaded {len(warm_start_maxima)} warm-start maxima from file. New Global Max: {self.global_max_target_val:.4e} ---")


    # --- Job Factory Methods (Master Only) ---

    def create_initial_optimization_jobs(self, next_job_id):
        """Generates L-BFGS-B jobs for finding initial maxima."""
        print(f"--- Generating {self.n_initial_optimizations} initial optimization jobs ---")
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
            print("Warning: No initial maxima found. Cannot create activation jobs.")
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

        print(f"--- Generating {len(jobs)} activation jobs ---")
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
            print("Warning: create_refinement_activation_jobs called but not in refinement mode.")
            return [], next_job_id

        jobs = []
        activation_job_created_for_grid_points = set()

        # Find all transferred coarse grid points (status='refined')
        transferred_points = [idx for idx, state in self.population.items()
                             if (state['status'] == 'optimized') and 
                                (state['best_fitness'] >= (self.global_max_target_val - self.roi_threshold))]

        print(f"--- Creating refinement activation jobs from {len(transferred_points)} transferred points ---")

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

        print(f"--- Generating {len(jobs)} refinement activation jobs ---")
        return jobs, next_job_id


    def create_de_generation_jobs(self, next_job_id, max_num_to_evolve):
        """Generates all DEGridPointJobs for one generation."""

        successful_F = []
        successful_CR = []

        unconverged_indices = [idx for idx, state in self.population.items() if state['status'] == 'active']

        if not unconverged_indices:
            print("All active points have converged. Ending DE phase.")
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
            print("Not enough active points (<4) to perform DE. Waiting.")
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
        state = self.population.get(grid_idx)

        # Safety check: only optimize active/converged/optimized points
        if not state or state['status'] == 'LBFGSB_queued':
            return None

        # Mark as claimed
        state['status'] = 'LBFGSB_queued'

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

    def create_patching_LBFGSB_jobs(self, next_job_id):
        """
        Identifies candidates for patching and creates LBFGSB jobs for them.
        """
        # 1. Identify all candidates
        roi_cutoff = self.global_max_target_val - self.roi_threshold
        candidate_indices = []
        for grid_idx, state in self.population.items():
            if state['best_fitness'] >= roi_cutoff:
                candidate_indices.append(grid_idx)
            else:
                # Also check points bordering the ROI (from test_38.py)
                neighbor_count = 0
                roi_neighbor_count = 0
                for neighbor_idx in self._get_valid_neighbors(grid_idx):
                    neighbor_count += 1
                    if self.profile_likelihood_grid.get(neighbor_idx, -np.inf) >= roi_cutoff:
                        roi_neighbor_count += 1
                if neighbor_count > 0 and (roi_neighbor_count / neighbor_count) > 0.5:
                    candidate_indices.append(grid_idx)

        if not candidate_indices:
            return [], next_job_id

        # 2. Calculate priority scores for each candidate
        priority_scores = []
        for grid_idx in candidate_indices:
            current_logL = self.profile_likelihood_grid.get(grid_idx, -np.inf)
            if current_logL == -np.inf:
                continue

            scalar_gradient_sum = 0.0
            neighbor_logL_sum = 0.0

            for dim_idx in range(self.n_proj_dims):
                for direction in [-1, 1]:
                    offset = np.zeros(self.n_proj_dims, dtype=int)
                    offset[dim_idx] = direction
                    neighbor_idx = tuple(np.array(grid_idx) + offset)

                    if not all(0 <= i < s for i, s in zip(neighbor_idx, self.grid_shape)):
                        continue

                    neighbor_logL = self.profile_likelihood_grid.get(neighbor_idx, -np.inf)

                    if neighbor_logL > -np.inf and neighbor_logL >= roi_cutoff:
                        gradient = neighbor_logL - current_logL
                        if gradient > 0:
                            scalar_gradient_sum += gradient
                            neighbor_logL_sum += neighbor_logL

            if scalar_gradient_sum > 0:
                likelihood_weight = max(neighbor_logL_sum - roi_cutoff, 0) + 1.0
                priority_score = scalar_gradient_sum * likelihood_weight
                priority_scores.append((priority_score, grid_idx))

        if not priority_scores:
            return [], next_job_id

        # 3. Sort points and select the top fraction
        priority_scores.sort(key=lambda x: x[0], reverse=True)
        num_to_patch = max(1, int(len(priority_scores) * self.patching_fraction))
        points_to_patch = [idx for _, idx in priority_scores[:num_to_patch]]

        # 4. Create LBFGSB jobs for these points
        new_jobs = []
        for grid_idx in points_to_patch:
            state = self.population.get(grid_idx)
            # Only patch if it exists and isn't already being optimized
            if state and state['status'] != 'LBFGSB_queued':
                spawn_result = self.create_LBFGSB_job_for_point(grid_idx, next_job_id)
                if spawn_result:
                    job, next_job_id = spawn_result
                    new_jobs.append(job)

        return new_jobs, next_job_id
