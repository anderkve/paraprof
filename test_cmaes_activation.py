"""
Minimal test to debug CMA-ES dynamic activation issue.
"""
import sys
import numpy as np
from mpi4py import MPI
from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function
from paraprof import set_log_level

# Set logging level
set_log_level('INFO')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# Configuration
np.random.seed(750123)
TEST_FUNCTION = "himmelblau_4d"

# Small 2D projection to test dynamic activation
PROJECTIONS_TO_RUN = [
    {
        'dims': [0, 1],
        'grid_points': [10, 10],  # Small grid
        'optimization_method': 'cmaes',
        'lbfgsb_refinement': False,  # Disable to isolate CMA-ES loop behavior
        'patching_coarse': False,
    },
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    print("="*80)
    print("Testing CMA-ES dynamic activation")
    print("="*80)

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=1,
        n_initial_optimizations=5,
        roi_threshold=4.0,
        convergence_threshold=1e-6,
        convergence_window=3,
        cmaes_max_generations=20,  # Lower for faster testing
        use_de_prescreening=False,
    )

    comm.bcast(sampler.target_func, root=0)

    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=50,
        max_num_to_evolve=None,
        save_plots=True,
        plot_settings={'dpi': 150, 'filetype': 'png'},
        myrank=myrank
    )

    print("\n" + "="*80)
    print("=== Test Summary ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        grid_pts = len(res['coarse_solution']['solutions'])
        print(f"  Projection {i+1} (dims {dims}):")
        print(f"    Function calls: {calls}")
        print(f"    Grid points explored: {grid_pts}")
        print(f"    Max logL: {max_ll:.4e}")
    print("="*80 + "\n")

    terminate_workers(comm, myrank)
else:
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
