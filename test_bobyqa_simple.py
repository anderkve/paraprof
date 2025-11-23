"""Quick test of BOBYQA fixes on a single 2D projection."""
import sys
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed.")
    sys.exit(1)

from paraprof import GridAnchoredDESampler, run_all_projections, terminate_workers, worker_main
from paraprof import get_test_function, set_log_level

set_log_level('INFO')

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

np.random.seed(750123)

TEST_FUNCTION = "himmelblau_4d"

# Single 2D projection to test
PROJECTIONS_TO_RUN = [
    {'dims': [0, 1], 'grid_points': [30, 30], 'optimization_method': 'bobyqa', 'patching_coarse': True},
]

log_likelihood, param_bounds, true_peaks = get_test_function(TEST_FUNCTION)

if myrank == 0:
    output_file = f"samples_bobyqa_test_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=log_likelihood,
        bounds=param_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        n_initial_optimizations=50,
        roi_threshold=4.0,
        bobyqa_initial_trust_radius=0.1,
        bobyqa_max_iterations=50,
        bobyqa_min_trust_radius=1e-6,
        max_patching_waves=50,
        samples_output_file=output_file,
    )

    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=PROJECTIONS_TO_RUN,
        num_generations=100000,
        save_plots=True,
        plot_settings={'dpi': 300, 'filetype': 'png'},
        myrank=myrank
    )

    print("\n" + "="*80)
    print("=== BOBYQA Test Results ===")
    print("="*80)
    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"  Projection {i+1} (dims {dims}): {calls} calls, max logL = {max_ll:.4e}")
    print("="*80 + "\n")

    terminate_workers(comm, myrank)

else:
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
