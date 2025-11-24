"""
Example: Profile likelihood with constrained nuisance parameters.

This example demonstrates how to use the nuisance parameter framework to
test ParaProf on realistic physics-like scenarios where:
- A few parameters of interest (POI) have complex likelihood surfaces
- Many nuisance parameters are tightly constrained
- Nuisance parameters couple to the main likelihood

Usage:
    mpiexec -n <number_of_cores> python run_nuisance_example.py
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
from paraprof import get_test_function, set_log_level
from paraprof.nuisance_wrapper import create_nuisance_wrapped_function

set_log_level('INFO')

# Get MPI info
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

# --- Configuration ---
np.random.seed(750123)

# Base test function (parameters of interest)
BASE_FUNCTION = "himmelblau_4d"
N_POI = 4  # Number of parameters of interest
N_NUISANCE = 20  # Number of nuisance parameters

# Nuisance parameter settings
COUPLING_MODE = 'shift'  # How nuisance params affect POI ('shift', 'scale', 'rotation', 'additive')
CONSTRAINT_SIGMA = 0.2   # Tighter = more constrained (0.2 = tight, 1.0 = loose)

# Projections to run (specified in terms of POI indices only)
# Note: We only project over POI dimensions, not nuisance parameters
PROJECTIONS_TO_RUN = [
    # 1D projection over first POI parameter
    # {'dims': [0], 'grid_points': [100], 'patching_coarse': True, 'lbfgsb': True},

    # 2D projection over first two POI parameters
    # {'dims': [0, 1], 'grid_points': [100, 100], 'optimization_method': 'lbfgsb', 'patching_coarse': True, 'lbfgsb_refinement': True, 'enable_refinement': False, 'patching_refined': True, 'refinement_factor': 2},
    {'dims': [0, 1], 'grid_points': [50, 50], 'optimization_method': 'cmaes',  'patching_coarse': False, 'lbfgsb_refinement': False, 'enable_refinement': False, 'patching_refined': True, 'refinement_factor': 2},
]

if myrank == 0:
    print("="*80)
    print("Nuisance Parameter Test Example")
    print("="*80)
    print(f"Base function: {BASE_FUNCTION}")
    print(f"Parameters of interest: {N_POI}")
    print(f"Nuisance parameters: {N_NUISANCE}")
    print(f"Total dimensions: {N_POI + N_NUISANCE}")
    print(f"Coupling mode: {COUPLING_MODE}")
    print(f"Constraint sigma: {CONSTRAINT_SIGMA}")
    print("="*80)

# Get base test function
base_func, base_bounds, base_peaks = get_test_function(BASE_FUNCTION)

# Wrap with nuisance parameters
wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
    base_func=base_func,
    base_bounds=base_bounds,
    n_poi=N_POI,
    n_nuisance=N_NUISANCE,
    coupling_mode=COUPLING_MODE,
    constraint_sigma=CONSTRAINT_SIGMA,
    nuisance_mean=0.0,
    nuisance_bounds_sigma_multiple=5.0  # Bounds = mean ± 5*sigma
)

if myrank == 0:
    print("\nParameter bounds:")
    for i in range(N_POI):
        print(f"  POI {i}: {wrapped_bounds[i]}")
    for i in range(N_NUISANCE):
        print(f"  Nuisance {i}: {wrapped_bounds[N_POI + i]}")
    print()

    # Demonstrate the effect of nuisance parameters
    print("Effect of nuisance parameters:")
    print("-" * 80)

    # Evaluate at a test point with nuisance at mean (optimal)
    test_poi = np.array([3.0, 2.0, 3.0, 2.0])  # Near a Himmelblau peak
    test_nuis_optimal = np.zeros(N_NUISANCE)  # At constraint mean
    test_full_optimal = np.concatenate([test_poi, test_nuis_optimal])
    ll_optimal = wrapped_func(test_full_optimal)

    print(f"POI = {test_poi}")
    print(f"Nuisance at optimal (all zeros): log L = {ll_optimal:.6f}")

    # Evaluate with nuisance shifted by 1 sigma
    test_nuis_shifted = np.ones(N_NUISANCE) * CONSTRAINT_SIGMA
    test_full_shifted = np.concatenate([test_poi, test_nuis_shifted])
    ll_shifted = wrapped_func(test_full_shifted)

    print(f"Nuisance shifted by +1σ: log L = {ll_shifted:.6f}")
    print(f"Penalty from constraint: {ll_shifted - ll_optimal:.6f} (expect ≈ -4.0)")

    # Show the coupling effect
    if COUPLING_MODE != 'additive':
        print(f"\nCoupling matrix (how nuisance affects POI):")
        print("Rows = POI dimensions, Columns = Nuisance parameters")
        print(wrapper.coupling_matrix)
    print("-" * 80)
    print()

if myrank == 0:
    # --- Master process ---

    # Calculate memory size
    max_grid_points = max(len(proj['grid_points']) for proj in PROJECTIONS_TO_RUN)

    output_file = f"samples_nuisance_rank_{myrank}.csv"

    sampler = GridAnchoredDESampler(
        target_func=wrapped_func,
        bounds=wrapped_bounds,
        projections=PROJECTIONS_TO_RUN,
        pop_per_grid_point=3,
        mutation_strategy='current-to-pbest/1',
        pbest_fraction=0.1,
        n_initial_optimizations=100,
        roi_threshold=4.0,
        convergence_threshold=1e-3,
        convergence_window=3,
        neighbor_pull_probability=0.5,
        LBFGSB_ftol=1e-9,
        LBFGSB_max_iter=10,
        LBFGSB_gradient_method="forward",
        max_patching_waves=50,
        patching_n_neighbors=1,
        memory_size=max_grid_points * 25,
        samples_output_file=output_file,
        use_de_prescreening=False,  # Use emulator to reduce evaluations
        emulator_min_neighbors=10,
        emulator_max_neighbors=100,
        emulator_confidence_threshold=-1.0,
        emulator_length_scale=1.0,
        emulator_noise_level=0.0001,
        # CD settings
        use_cd_refinement=False,
        cd_max_cycles=20,
        cd_step_fraction=0.01,
        # CMA-ES settings
        cmaes_lambda=None,
        cmaes_mu=None,
        cmaes_max_generations=100,
    )

    # Broadcast target function to all workers
    print("Master: Broadcasting target function to workers...")
    comm.bcast(sampler.target_func, root=0)

    # Run all projections
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

    # Print summary
    print("\n" + "="*80)
    print("=== Summary of Nuisance Parameter Test ===")
    print("="*80)
    print(f"Base function: {BASE_FUNCTION} ({N_POI}D)")
    print(f"Nuisance parameters: {N_NUISANCE}")
    print(f"Coupling mode: {COUPLING_MODE}")
    print(f"Constraint sigma: {CONSTRAINT_SIGMA}")
    print("-"*80)

    for i, res in enumerate(results):
        dims = res['projection_config']['dims']
        calls = res['metrics']['total_target_calls']
        max_ll = res['metrics']['global_max']
        print(f"Projection {i+1} (POI dims {dims}): {calls} calls, max logL = {max_ll:.4e}")

    print("="*80)

    # Analyze nuisance parameter behavior at the global optimum
    if results and sampler.global_solution_pool:
        print("\nAnalyzing nuisance parameters at optimum...")
        print("-"*80)

        # Get best point from sampler's global solution pool
        # Pool is sorted by fitness (best first)
        best_solution = sampler.global_solution_pool[0]
        best_params = best_solution['full_params']
        best_ll = sampler.global_max_target_val

        best_poi = best_params[:N_POI]
        best_nuis = best_params[N_POI:]

        print(f"Best log-likelihood found: {best_ll:.6f}")
        print(f"Best POI parameters: {best_poi}")
        print(f"Best nuisance parameters: {best_nuis}")
        print(f"Nuisance parameter deviations from mean (in units of σ):")
        deviations = (best_nuis - wrapper.nuisance_mean) / wrapper.constraint_sigma
        for i, dev in enumerate(deviations):
            print(f"  Nuisance {i}: {dev:.3f}σ")

        # Compare to analytical optimum
        optimal_nuis = wrapper.get_optimal_nuisance(best_poi)
        print(f"\nAnalytical optimal nuisance: {optimal_nuis}")
        print(f"Difference: {np.linalg.norm(best_nuis - optimal_nuis):.6f}")

        print("="*80)

    print("\n")

    # Terminate workers
    print("Master: All projections complete. Terminating workers...")
    terminate_workers(comm, myrank)

else:
    # --- Worker process ---
    worker_main(comm, myrank)

print(f"rank {myrank}: Done.")
