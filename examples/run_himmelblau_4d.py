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

PROJECTIONS_TO_RUN = [

    # 1D projections
    # {'dims': [0], 'grid_points': [100], 'optimization_method': 'de', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [1], 'grid_points': [100], 'optimization_method': 'de', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [2], 'grid_points': [100], 'optimization_method': 'de', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [3], 'grid_points': [100], 'optimization_method': 'de', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},

    # 2D projections
    # {'dims': [0, 1], 'grid_points': [20, 20], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': False, 'lbfgsb_refinement': True, 'enable_refinement': False, 'refinement_factor': 2},
    # {'dims': [0, 1], 'grid_points': [50, 50], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': False, 'lbfgsb_refinement': True, 'enable_refinement': False, 'refinement_factor': 2},
    {'dims': [0, 2], 'grid_points': [50, 50], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [0, 2], 'grid_points': [100, 100], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': False, 'lbfgsb_refinement': True, 'enable_refinement': False, 'refinement_factor': 2},
    # {'dims': [0, 3], 'grid_points': [75, 75], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [1, 2], 'grid_points': [75, 75], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [1, 3], 'grid_points': [75, 75], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},
    # {'dims': [2, 3], 'grid_points': [75, 75], 'optimization_method': 'de', 'patching_coarse': True, 'patching_refined': True, 'lbfgsb_refinement': True, 'enable_refinement': True, 'refinement_factor': 2},

]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    # Use a single output file for all projections to enable warm start
    output_file = f"samples_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3, # Increased for better DE
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=100, # Increased
        roi_threshold=4.0,
        convergence_threshold=1e-7, # Tighter -> Looser (match serial)
        convergence_window=3, #3,      # Longer window -> Shorter (match serial)
        neighbor_pull_probability=0.5,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=10, #20
        LBFGSB_gradient_method="forward", # "central",
        max_patching_waves=50,  # Maximum number of patching waves
        patching_n_neighbors=1,  # Test only the best neighbor
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,  # Single file for all projections
        use_de_prescreening=True,
        emulator_min_neighbors=10,
        emulator_max_neighbors=100,
        emulator_confidence_threshold=-1.0,
        emulator_length_scale=1.0,  # In standardized units (inputs auto-scaled)
        emulator_noise_level=0.0001,
        # CD settings
        use_cd_refinement=True,
        cd_max_cycles=50,
        cd_step_fraction=0.01,
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
