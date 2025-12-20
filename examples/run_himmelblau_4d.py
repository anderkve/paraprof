"""
Example script: Running the Grid-Anchored DE Sampler on the Himmelblau 4D test function.

Usage:
    mpiexec -n <number_of_cores> python run_himmelblau_4d.py
"""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
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
    # {'dims': [0], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [1], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [2], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [3], 'grid_points': [100], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},

    # 2D projections
    # {'dims': [0, 2], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': False, 'lbfgsb_refinement': False, 'enable_refinement': True, 'refinement_factor': 3},
    # {'dims': [0, 1], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': False, 'lbfgsb_refinement': False, 'enable_refinement': True, 'refinement_factor': 3},

    {'dims': [0, 1], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},
    {'dims': [0, 2], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},
    {'dims': [0, 3], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},
    {'dims': [1, 2], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},
    {'dims': [1, 3], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},
    {'dims': [2, 3], 'grid_points': [50, 50], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 5},

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
        # Override auto-configured parameters
        'n_initial_optimizations': 100,           # Default: min(100, 20 * n_dims) = 80
        'global_pool_size': 10000,                # Default: 10000 (already good)
        'memory_size': max_grid_points * 25,      # Default: max_grid_size * 25 = 3750
        'convergence_threshold': 1e-3,            # Default: roi_threshold / 1000 = 0.004

        # Fine-tune Differential Evolution
        'de': {
            'mutation_strategy': 'current-to-pbest/1',  # Default (best performer)
            'pbest_fraction': 0.1,                      # Default: 0.1
            'neighbor_pull_probability': 0.5,           # Default: 0.5
            'convergence_window': 3,                    # Default: 3
        },

        # Fine-tune L-BFGS-B
        'lbfgsb': {
            'ftol': 1e-9,                              # Default: 1e-9
            'gradient_method': 'forward',              # Default: 'forward' (faster than 'central')
        },

        # Patching parameters
        'patching': {
            'n_neighbors': 1,                          # Default: 1 (test more neighbors for this problem)
        },

        # Activation mixing ratios
        'activation': {
            'mix_ratios': {
                'neighbors': 0.5,                      # Default: 0.5
                'global': 0.25,                        # Default: 0.25
                'random': 0.25,                        # Default: 0.25
            },
        },

        # Emulator parameters (only used if use_emulator=True)
        'emulator': {
            'confidence_threshold': -1.0,              # Default: 2.0 (negative = accept all)
            'min_neighbors': 10,                       # Default: 10
            'max_neighbors': 100,                      # Default: 100
            'length_scale': 1.0,                       # Default: 1.0
            'noise_level': 0.0001,                     # Default: 0.01
        },

        # Coordinate Descent parameters (only used if use_cd_refinement=True)
        'cd': {
            'max_cycles': 50,                          # Default: 3
            'step_fraction': 0.01,                     # Default: 0.01
        },

        # CMA-ES parameters (only used if optimization_method='cmaes' in projection)
        'cmaes': {
            'lambda': None,                            # Default: auto = 4 + floor(3*log(n_cont))
            'mu': None,                                # Default: auto = lambda/2
            'max_generations': 100,                    # Default: 100
        },

        # Clustering parameters (only used if use_clustering=True during refinement)
        'clustering': {
            'method': 'dbscan',                        # Default: 'dbscan'
            'eps': None,                               # Default: None (auto-estimated)
            'min_samples': None,                       # Default: None (auto = max(2, n_cont))
            'eps_multiplier': 3.0,                     # Default: 3.0
            'projection_weight': 1.0,                  # Default: 1.0
        },
    }

    sampler = GridAnchoredDESampler(
        # === Required parameters ===
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,

        # === Core tuning parameters (commonly adjusted) ===
        roi_threshold=4.0,                            # Chi-squared units
        pop_per_grid_point=3,                         # Population size per grid point
        max_patching_waves=50,                        # Refinement iterations
        lbfgsb_max_iter=20,                           # L-BFGS-B iterations per point

        # === Feature toggles ===
        use_emulator=False,                           # GP-based pre-screening (30-50% speedup)
        use_clustering=True,                          # Mode detection for refinement
        use_cd_refinement=False,                      # Use L-BFGS-B instead of CD for refinement
        refinement_direct_eval=True,                  # Fast interpolation vs full optimization

        # === I/O ===
        samples_output_file=output_file,

        # === Advanced configuration (optional - for expert users) ===
        advanced_config=advanced_config,
    )

    # Broadcast the target function to all workers (once, before all projections)
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # --- Run all projections with automatic refinement handling ---
    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=100000,
        max_num_to_evolve=None,
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

    # Terminate all workers after all projections complete
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
