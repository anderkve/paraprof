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
from master import master_main
from worker import worker_main
from visualization import plot_profiles
from test_functions import get_test_function

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
TEST_FUNCTION = "himmelblau_4d"

PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [100, 100], 'patching': True, 'refining': True},
    # {'dims': [0, 2], 'grid_points': [100, 100], 'patching': True, 'refining': True},
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    # --- Master process ---

    # Setup plotting
    try:
        import matplotlib.pyplot as plt
        plt.ion() # Interactive mode on
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [10, 1]})
    except ImportError:
        fig, axes = None, None
        print("Matplotlib not found. Plotting will be disabled.")

    # Calculate memory size based on max grid points across all projections
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

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
        patching_fraction=0.05,
        patching_conv_threshold=0.01,
        max_patching_iterations=10, # Limit patching
        memory_size=max_grid_points * 25,
        samples_output_file=None,  # Will be set for each projection
    )

    def plot_func_wrapper(s, fig, axes):
        plot_profiles(s, fig, axes)

    # --- Loop over all projections ---
    for proj_idx, projection_config in enumerate(PROJECTIONS_TO_RUN):
        print("\n" + "="*80)
        print(f"=== Starting Projection {proj_idx + 1}/{len(PROJECTIONS_TO_RUN)} ===")
        print(f"=== Dimensions: {projection_config['dims']} ===")
        print("="*80 + "\n")

        # Reset sampler for new projection
        if proj_idx > 0:
            sampler._reset_for_new_projection(projection_config)

        # Set projection-specific output file
        dims_str = "_".join(map(str, projection_config['dims']))
        output_file = f"samples_rank_{myrank}_dims_{dims_str}.csv"
        sampler.samples_output_file = output_file
        sampler.samples_buffer = []
        sampler.sample_buffer_size = 1000  # Initialize buffer size

        # Run the workflow for this projection
        master_main(
            comm=comm,
            sampler=sampler,
            num_generations=100000, # Set a finite number of generations
            max_num_to_evolve=None, # Limit evals per gen -> Evolve all
            plot_callback=plot_func_wrapper,
            plot_interval=100, # Plot every 100 seconds
            skip_init_opt_on_warm_start=False,
            fig=fig,
            axes=axes,
            myrank=myrank
        )

        # Flush samples buffer after each projection
        sampler._flush_samples_buffer()

        # Save projection-specific plot
        if fig:
            plot_func_wrapper(sampler, fig, axes)
            plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}.png"
            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {plot_filename}")

        print("\n" + "="*80)
        print(f"=== Completed Projection {proj_idx + 1}/{len(PROJECTIONS_TO_RUN)} ===")
        print("="*80 + "\n")

    # Show final plot interactively
    if fig:
        print("Master: All projections complete. Final plot displayed. Press Enter to exit.")
        plt.ioff()
        plt.show()

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
