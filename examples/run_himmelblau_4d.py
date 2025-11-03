"""
Example script: Running the Grid-Anchored DE Sampler on the Himmelblau 4D test function.

Usage:
    mpiexec -n <number_of_cores> python run_himmelblau_4d.py
"""
import sys
import os

# Add parent directory to path to import paraprof
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    sys.exit(1)

from sampler import GridAnchoredDESampler
from master import master_main, terminate_workers
from worker import worker_main
from visualization import plot_profiles
from test_functions import get_test_function

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
TEST_FUNCTION = "himmelblau_4d"

PROJECTIONS_TO_RUN = [
    # Each projection can optionally enable grid refinement
    # 'enable_refinement': True/False - whether to run refinement after coarse grid
    # 'refinement_factor': int - grid refinement factor (e.g., 2 = twice as many points per dim)
    {'dims': [0, 1], 'grid_points': [50, 50], 'patching': True, 'refining': True, 'enable_refinement': True, 'refinement_factor': 2},
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Setup plotting
    try:
        import matplotlib.pyplot as plt
        plt.ioff() # Interactive mode off
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [10, 1]})
    except ImportError:
        fig, axes = None, None
        print("Matplotlib not found. Plotting will be disabled.")

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    # Use a single output file for all projections to enable warm start
    output_file = f"samples_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=1, # Increased for better DE
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=30, # Increased
        roi_threshold=3.2,
        convergence_threshold=1e-3, # Tighter -> Looser (match serial)
        convergence_window=2,      # Longer window -> Shorter (match serial)
        neighbor_pull_probability=0.5,
        refinement_ftol=1e-9,
        refinement_max_iter=20,
        refinement_gradient_method="forward", # "central",
        patching_fraction=0.1,
        patching_conv_threshold=0.01,
        max_patching_iterations=1000, # Limit patching
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,  # Single file for all projections
    )

    def plot_func_wrapper(s, fig, axes):
        plot_profiles(s, fig, axes)

    # Broadcast the target function to all workers (once, before all projections)
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # --- Loop over all projections ---
    for proj_idx, projection_config in enumerate(PROJECTIONS_TO_RUN):
        print("\n" + "="*80)
        print(f"=== Starting Projection {proj_idx + 1}/{len(PROJECTIONS_TO_RUN)} ===")
        print(f"=== Dimensions: {projection_config['dims']} ===")
        print("="*80 + "\n")

        # Reset sampler for new projection
        if proj_idx > 0:
            sampler._reset_for_new_projection(projection_config)

        # Enable warm start for all projections after the first
        skip_init_opt = (proj_idx > 0)

        # --- COARSE GRID RUN ---
        print("\n" + "="*80)
        print("=== Running Coarse Grid ===")
        print("="*80 + "\n")

        # Run the workflow for this projection
        master_main(
            comm=comm,
            sampler=sampler,
            num_generations=100000, # Set a finite number of generations
            max_num_to_evolve=None, # Limit evals per gen -> Evolve all
            plot_callback=plot_func_wrapper,
            plot_interval=100, # Plot every 100 seconds
            skip_init_opt_on_warm_start=skip_init_opt,  # Enable warm start after first projection
            fig=fig,
            axes=axes,
            myrank=myrank
        )

        # Flush samples buffer after coarse grid
        sampler._flush_samples_buffer()

        # Save coarse grid plot
        if fig:
            plot_func_wrapper(sampler, fig, axes)
            dims_str = "_".join(map(str, projection_config['dims']))
            plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}_coarse.png"
            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Saved coarse grid plot to: {plot_filename}")

        # --- REFINEMENT RUN (if enabled) ---
        if projection_config.get('enable_refinement', False):
            print("\n" + "="*80)
            print("=== Starting Grid Refinement ===")
            print("="*80 + "\n")

            # Export coarse grid solution
            coarse_solution = sampler.export_grid_solution()
            refinement_factor = projection_config.get('refinement_factor', 2)

            # Setup refined projection config
            refined_config = projection_config.copy()
            refined_config['grid_points'] = [
                n * refinement_factor for n in projection_config['grid_points']
            ]

            # Configure sampler for refinement
            sampler.setup_refinement_run(coarse_solution, refinement_factor)
            sampler._reset_for_new_projection(refined_config)

            # Run refinement workflow
            master_main(
                comm=comm,
                sampler=sampler,
                num_generations=100000,
                max_num_to_evolve=None,
                plot_callback=plot_func_wrapper,
                plot_interval=100,
                skip_init_opt_on_warm_start=True,  # Always skip for refinement
                fig=fig,
                axes=axes,
                myrank=myrank
            )

            # Flush samples buffer after refinement
            sampler._flush_samples_buffer()

            # Save refined grid plot
            if fig:
                plot_func_wrapper(sampler, fig, axes)
                plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}_refined.png"
                fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"Saved refined grid plot to: {plot_filename}")

            # Reset refinement flags for next projection
            sampler.is_refinement_run = False
            sampler.refinement_factor = None
            sampler.coarse_grid_solution = None
            sampler.refinement_interpolator = None

        print("\n" + "="*80)
        print(f"=== Completed Projection {proj_idx + 1}/{len(PROJECTIONS_TO_RUN)} ===")
        print("="*80 + "\n")

    # Terminate all workers after all projections complete
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
