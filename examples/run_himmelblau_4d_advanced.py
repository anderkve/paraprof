"""
Advanced example: Running the ProfileProjector on the Himmelblau 4D test
function, showing every advanced_config option and the underlying lower-level
API (explicit comm.bcast, run_all_projections, terminate_workers).

For a minimal version that uses the higher-level :func:`run_scan` wrapper,
see ``run_himmelblau_4d.py`` in this directory.

Usage:
    mpiexec -n <number_of_cores> python run_himmelblau_4d_advanced.py
"""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from paraprof import ProfileProjector, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function

from paraprof import set_log_level
# set_log_level('DEBUG')
set_log_level('INFO')
# set_log_level('WARNING')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

TEST_FUNCTION = "himmelblau_4d"
# TEST_FUNCTION = "rosenbrock_4d"

PROJECTIONS_TO_RUN = [

    # 1D projections
    {'dims': [0], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 2, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [1], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 2, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [2], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 2, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [3], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 2, 'patch_coarse_grid': True, 'patch_refined_grid': True},

    # 2D projections
    {'dims': [0, 1], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [0, 2], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [0, 3], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [1, 2], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [1, 3], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},
    {'dims': [2, 3], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'grid_refinement_factor': 1, 'patch_coarse_grid': True, 'patch_refined_grid': True},

]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Use a single output file for all projections to enable warm start
    output_file = f"samples_rank_{myrank}.csv"

    # ===== SIMPLIFIED INTERFACE =====
    # Most users only need to specify these core parameters

    # For expert users: demonstrate the advanced_config dict with ALL parameters
    # This shows how to access every configuration option if needed
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    advanced_config = {
        'memory_size': max_grid_points * 25,      # Default: max_grid_size * 25
        'convergence_threshold': 1e-3,            # Default: roi_threshold / 1000

        'de': {
            'convergence_window': 3,              # Default: 3
            'num_generations': 100000,            # Default: 100000
            'max_num_to_evolve': None,            # Default: None (all grid points)
            'allow_early_DE_exit': False,         # Default: False (opt-in). When True, a fresh,
                                                  # skip-eligible cell whose seed population is
                                                  # already converged may exit DE early.
        },

        'lbfgsb': {
            'ftol': 1e-9,                         # Default: 1e-9
            'gradient_method': 'forward',         # Default: 'forward' (use 'central' for higher accuracy at ~50% more calls)
        },

        # Clustering parameters (only used if use_clustering=True during refinement)
        'clustering': {
            'method': 'dbscan',                        # Default: 'dbscan'
            'eps': None,                               # Default: None (auto-estimated)
            'min_samples': None,                       # Default: None (auto = max(2, n_prof))
            'eps_multiplier': 3.0,                     # Default: 3.0
            'projection_weight': 1.0,                  # Default: 1.0
        },

        # Cross-projection knowledge transfer.
        # Both hooks reuse the in-memory global_solution_pool that already
        # accumulates across projections; both default to True and are
        # no-ops on the first projection. Set either to False to disable
        # (e.g. for A/B benchmarking or as a safety valve on a pathological
        # target). The 4D Himmelblau scan in this example sees ~10% fewer
        # target-function calls with both on; Rosenbrock-style narrow-
        # valley targets see 2-3x reductions.
        'cross_projection': {
            'proximity_warm_start': True,              # Default: True. Per-cell
                                                       # activation pop swaps one
                                                       # random LHS seed for the
                                                       # highest-fitness past eval
                                                       # whose projection-dim
                                                       # coordinates are nearest
                                                       # to the cell.
            'pool_seeded_initial_maxima': True,        # Default: True. On every
                                                       # projection after the
                                                       # first, seed initial_maxima
                                                       # from the in-memory pool
                                                       # and skip the global
                                                       # L-BFGS-B starts.
        },

        # Suspect-recheck refinement waves. After the normal patching waves,
        # paraprof re-optimizes ROI cells whose profiled-parameter values look
        # discontinuous relative to their neighbours (a sign of a missed
        # better optimum), seeding the recheck from extended neighbours and the
        # global solution pool.
        'suspect_recheck': {
            'enabled': True,                           # Default: True
            'max_waves': 3,                            # Default: 3. Max recheck waves
            'param_k': 3.0,                            # Default: 3.0. MAD multiplier on the
                                                       # profiled-param discontinuity that
                                                       # flags a cell as suspect
            'max_fraction': 0.25,                      # Default: 0.25. Safety cap on the
                                                       # fraction of ROI cells rechecked per wave
            'seeds_k_ring': 3,                         # Default: 3. Chebyshev radius for
                                                       # extended-neighbour seeds
            'seeds_from_pool': 3,                      # Default: 3. Number of pool seeds added
            'polish_threshold': 1e-4,                  # Default: 1e-4. Min improvement over the
                                                       # current value to trigger an L-BFGS-B polish
        },

        # Initial-optimization basin detection. The initial multistart runs a
        # rolling Latin-hypercube set of L-BFGS-B starts, clusters converged
        # optima into distinct basins online, and applies a Bayesian stopping
        # rule (restricted to ROI-competitive optima). n_initial_optimizations
        # acts as a *cap* on this stage.
        'basin_detection': {
            'batch_size': None,                        # Default: None (FD-aware auto; see
                                                       # resolve_initial_opt_batch_size). Number
                                                       # of optimizations run concurrently.
            'undiscovered_threshold': 0.5,             # Default: 0.5. Stop once the expected number
                                                       # of undiscovered ROI optima falls below this.
                                                       # Set to 0 to disable early stopping.
            'min_starts': None,                        # Default: None (auto = max(floor, mult*n_dims),
                                                       # capped at n_initial_optimizations). Minimum
                                                       # starts before early stopping may trigger.
        },
    }

    # === Optional: Provide initial points for targeted exploration ===
    # If you already know good regions of parameter space, you can provide initial
    # points to activate corresponding grid locations. This is useful when:
    # - You have prior knowledge from previous runs
    # - You want to focus on specific parameter combinations
    # - You want to skip expensive global optimization
    #
    # Example: For Himmelblau 4D, the known optima are at (3,0,-3,0), etc.
    # Uncomment and use with n_initial_optimizations=0 to ONLY use these points:
    # initial_points_example = [
    #     [3.0, 0.0, -3.0, 0.0],    # First known optimum
    #     [-3.0, 0.0, 3.0, 0.0],    # Second known optimum
    # ]

    # Use context manager to ensure proper cleanup of sampler resources
    with ProfileProjector(
        # === Required parameters ===
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,

        # === Core tuning parameters (commonly adjusted) ===
        roi_threshold=4.0,                            # Chi-squared units
        pop_per_grid_point=3,                         # Population size per grid point
        max_patching_waves=50,                        # Refinement iterations
        lbfgsb_max_iter=20,                           # L-BFGS-B iterations per point
        lbfgsb_polish=True,                           # Apply L-BFGS-B polishing after DE
        n_initial_optimizations=100,                  # Global L-BFGS-B runs (default: min(100, 20*n_dims)=80)
        n_optima=None,                                # Default: None. Optional prior on the global number
        #                                             # of optima (int, or {'min': int, 'max': int}). Use only
        #                                             # when confident the target has one/a few optima.
        # initial_points=[[3.0, 0.0, -3.0, 0.0]],       # Optional: User-provided initial points to activate grid
        #                                             # Use with n_initial_optimizations=0 to only use these points

        # === Optional user-supplied gradient ===
        grad_func=None,                               # Default: None. Callable returning grad of target_func
        #                                             # (the function being MAXIMIZED). Only used in the
        #                                             # L-BFGS-B paths; cuts finite-difference target calls.

        # === Feature toggles ===
        use_clustering=True,                          # Mode detection for refinement
        refinement_direct_eval=True,                  # Fast interpolation vs full optimization

        # === Parameter naming (optional) ===
        parameter_names=None,                         # Default: None. List of one name per parameter dimension;
        #                                             # enables string entries in projection 'dims'.

        # === I/O ===
        samples_output_file=output_file,
        warm_start_file=None,                         # Default: None. Path to a sample file from a previous run;
        #                                             # pre-populates initial_maxima and skips global L-BFGS-B
        #                                             # starts. Point at samples_output_file to round-trip runs.

        # === Advanced configuration (optional - for expert users) ===
        advanced_config=advanced_config,
    ) as sampler:

        # Broadcast the target function to all workers (once, before all projections)
        print("Master: Broadcasting target function to workers...")
        comm.bcast(sampler.target_func, root=0)

        # --- Run all projections with automatic refinement handling ---
        results = run_all_projections(
            comm=comm,
            sampler=sampler,
            projections=PROJECTIONS_TO_RUN,
            save_plots=True,
            plot_settings={'dpi': 300, 'filetype': 'png'},
            myrank=myrank
        )

        # Print summary of all projections
        print("\n" + "="*80)
        print("=== Summary of All Projections ===")
        print("="*80)
        for i, res in enumerate(results):
            dims = res['projection_config']['dims']
            calls = res['metrics']['total_target_calls']
            max_ll = res['metrics']['global_max']
            print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
        print("="*80 + "\n")
        # sampler.close() is called automatically on context exit

    # Terminate all workers after all projections complete
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
