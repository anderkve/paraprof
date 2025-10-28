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
OUTPUT_FILE = f"samples_rank_{myrank}.csv"

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
        memory_size=len(PROJECTIONS_TO_RUN[0]['grid_points']) * 25,
        samples_output_file=OUTPUT_FILE,
    )

    def plot_func_wrapper(s, fig, axes):
        plot_profiles(s, fig, axes)

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

    if fig:
        print("Master: Final plot. Press Enter to exit.")
        plot_func_wrapper(sampler, fig, axes)
        plt.ioff()
        plt.show()

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
