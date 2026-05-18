"""
MPI master process orchestration logic.
"""
import collections
import numpy as np
import sys

from .logger import setup_logger

try:
    from mpi4py import MPI
except ImportError:
    # Can't use logger here since MPI isn't available yet
    print("Error: mpi4py is not installed. This script requires MPI.", file=sys.stderr)
    print("Please install it with: pip install mpi4py", file=sys.stderr)
    sys.exit(1)

TASK_TERMINATE = -1


def _log_worker_error(result, sampler, logger):
    """Log an error reported by a worker, if any. Returns True if an error was reported."""
    error = result.get('error') if isinstance(result, dict) else None
    if not error:
        return False
    sampler.target_call_errors += 1
    worker_rank = result.get('context', {}).get('worker_rank', '?')
    params = result.get('params')
    logger.warning(
        f"Worker {worker_rank} reported failure at params {params}: {error} "
        f"(total errors: {sampler.target_call_errors})"
    )
    return True


def _log_user_gradient_error(result, sampler, logger):
    """Log a grad_func error; paraprof falls back to FD for the affected dims."""
    err = result.get('user_gradient_error') if isinstance(result, dict) else None
    if not err:
        return False
    sampler.user_gradient_errors += 1
    worker_rank = result.get('context', {}).get('worker_rank', '?')
    logger.warning(
        f"Worker {worker_rank} grad_func failure at params {result.get('params')}: "
        f"{err} (falling back to FD; total: {sampler.user_gradient_errors})"
    )
    return True


def terminate_workers(comm, myrank=0):
    """
    Terminates all worker processes.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    myrank : int
        Master rank (usually 0)
    """
    logger = setup_logger(rank=myrank)
    n_workers = comm.Get_size() - 1

    logger.debug("terminate_workers: Sending TASK_TERMINATE to workers.")
    # Use non-blocking sends to terminate all workers simultaneously
    requests = []
    for rank in range(1, n_workers + 1):
        requests.append(comm.isend(TASK_TERMINATE, dest=rank))

    # Wait for all sends to complete
    MPI.Request.Waitall(requests)
    logger.debug("terminate_workers: All workers terminated.")


def run_projection(comm, sampler, projection_config,
                   save_plots=False,
                   plot_settings=None,
                   skip_init_opt_on_warm_start=True,
                   myrank=0):
    """
    Runs a complete projection workflow including optional grid refinement.

    This high-level function encapsulates the entire projection computation,
    including both the coarse grid run and optional refinement. It handles
    all internal state management, cleanup, and optional plotting, providing
    a clean API for users.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    sampler : ProfileProjector
        The sampler instance (already configured with num_generations and max_num_to_evolve
        in advanced_config['de'])
    projection_config : dict
        Projection configuration. Required keys:
        - 'dims': list of int - projection dimension indices
        - 'grid_points': list of int - grid points per dimension
        Optional keys:
        - 'optimization_method': str - optimization algorithm ('de', 'lbfgsb') (default: 'de')
        - 'grid_refinement_factor': int - grid refinement factor (default: 2)
        - 'refinement_method': str - interpolation method (only 'linear' supported) (default: 'linear')
        - 'patch_coarse_grid': bool - enable patching on coarse grid (default: True)
        - 'patch_refined_grid': bool - enable patching on refined grid (default: False)
    save_plots : bool
        Whether to save plots to disk after each stage
    plot_settings : dict, optional
        Plot settings dictionary with keys:
        - 'dpi': int (default: 300)
        - 'filetype': str (default: 'png')
    skip_init_opt_on_warm_start : bool
        Whether to skip initial optimization if warm-start data exists
    myrank : int
        Master rank (usually 0)

    Returns
    -------
    dict
        Results dictionary containing:
        - 'coarse_solution': dict - exported coarse grid state
        - 'refined_solution': dict or None - exported refined grid state (if refinement enabled)
        - 'metrics': dict - performance metrics including:
            - 'coarse_target_calls': int - target function calls for coarse grid
            - 'refined_target_calls': int - total calls after refinement (if enabled)
            - 'total_target_calls': int - cumulative calls
            - 'global_max': float - best likelihood value found

    Examples
    --------
    >>> results = run_projection(
    ...     comm=comm,
    ...     sampler=sampler,
    ...     projection_config={
    ...         'dims': [0, 1],
    ...         'grid_points': [50, 50],
    ...         'grid_refinement_factor': 3
    ...     },
    ...     save_plots=True
    ... )
    >>> print(f"Best likelihood: {results['metrics']['global_max']}")
    """
    # Extract refinement configuration
    refinement_factor = projection_config.get('grid_refinement_factor', None)
    # Auto-determine if refinement should be enabled based on refinement_factor
    use_grid_refinement = refinement_factor is not None and refinement_factor > 1
    dims_str = "_".join(map(str, projection_config['dims']))

    logger = setup_logger(rank=myrank)

    # Initialize results structure
    results = {
        'coarse_solution': None,
        'refined_solution': None,
        'metrics': {}
    }

    # --- COARSE GRID RUN ---
    logger.info("=" * 80)
    logger.info("=== Running Coarse Grid ===")
    logger.info("=" * 80)

    master_main(
        comm=comm,
        sampler=sampler,
        plot_settings=plot_settings,
        skip_init_opt_on_warm_start=skip_init_opt_on_warm_start,
        myrank=myrank
    )

    # Flush samples buffer after coarse grid
    sampler._flush_samples_buffer()

    # Save coarse plot if requested
    if save_plots:
        from .visualization import plot_profiles
        plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}_coarse"
        plot_profiles(sampler, plot_filename, plot_settings)

    # Export coarse solution
    coarse_solution = sampler.export_grid_solution()
    results['coarse_solution'] = coarse_solution
    results['metrics']['coarse_target_calls'] = sampler.target_calls

    # --- REFINEMENT RUN (if enabled) ---
    if use_grid_refinement:
        logger.info("=" * 80)
        logger.info("=== Starting Grid Refinement ===")
        logger.info("=" * 80)

        # Setup refined projection config
        refined_config = projection_config.copy()
        refined_config['grid_points'] = [
            n * refinement_factor for n in projection_config['grid_points']
        ]

        # Get refinement method from config (default to 'linear')
        refinement_method = projection_config.get('refinement_method', 'linear')

        # Configure sampler for refinement
        sampler.setup_refinement_run(coarse_solution, refinement_factor, refinement_method)
        sampler._reset_for_new_projection(refined_config)

        # Run refinement workflow
        master_main(
            comm=comm,
            sampler=sampler,
            plot_settings=plot_settings,
            skip_init_opt_on_warm_start=True,  # Always skip for refinement
            myrank=myrank
        )

        # Flush samples buffer after refinement
        sampler._flush_samples_buffer()

        # Save refined plot if requested
        if save_plots:
            from .visualization import plot_profiles
            plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}_refined"
            plot_profiles(sampler, plot_filename, plot_settings)

        # Export refined solution
        refined_solution = sampler.export_grid_solution()
        results['refined_solution'] = refined_solution
        results['metrics']['refined_target_calls'] = sampler.target_calls

        # Clean up refinement state for next projection
        sampler._cleanup_refinement_state()

    # Record final metrics
    results['metrics']['total_target_calls'] = sampler.target_calls
    results['metrics']['global_max'] = sampler.global_max_target_val

    return results


def run_scan(comm, sampler, projections,
             save_plots=False,
             plot_settings=None,
             broadcast_target_func=True,
             myrank=0):
    """Master-side convenience wrapper that runs a full scan and tears down workers.

    This bundles the three steps a master process always performs into a single call:

    1. Optionally broadcast ``sampler.target_func`` to the workers (so they can
       enter ``worker_main`` with no ``target_func`` argument and pick it up
       from the broadcast).
    2. Run all configured projections via :func:`run_all_projections`.
    3. Send the terminate signal to all workers via :func:`terminate_workers`.

    Use ``broadcast_target_func=False`` when integrating into a host framework
    that already provides the target function on every rank (e.g. as a bound
    method that cannot be pickled). In that case the workers should be started
    with ``worker_main(comm, myrank, target_func=...)`` so they never wait for
    a broadcast.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator. Must be called from the master rank only.
    sampler : ProfileProjector
        Configured sampler instance.
    projections : list of dict
        Projection configurations (see :func:`run_all_projections`).
    save_plots : bool, optional
        Whether to save plots after each projection (default: False).
    plot_settings : dict, optional
        Plot settings forwarded to :func:`run_all_projections`.
    broadcast_target_func : bool, optional
        If True (default), broadcast ``sampler.target_func`` to workers before
        starting the scan. Set to False when workers were started with an
        explicit ``target_func`` argument.
    myrank : int, optional
        Master rank (default: 0).

    Returns
    -------
    list of dict
        Per-projection results, as returned by :func:`run_all_projections`.
    """
    if broadcast_target_func:
        # Tuple form; worker also accepts the legacy bare-callable form.
        comm.bcast((sampler.target_func, sampler.grad_func), root=myrank)

    results = run_all_projections(
        comm=comm,
        sampler=sampler,
        projections=projections,
        save_plots=save_plots,
        plot_settings=plot_settings,
        myrank=myrank,
    )

    terminate_workers(comm, myrank=myrank)
    return results


def run_all_projections(comm, sampler, projections,
                        save_plots=False,
                        plot_settings=None,
                        myrank=0):
    """
    Runs multiple projections sequentially with automatic warm-starting.

    This high-level function orchestrates multiple projection computations,
    automatically managing state transitions and warm-starting between
    projections. Each projection can optionally include grid refinement.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    sampler : ProfileProjector
        The sampler instance (already configured with bounds, algorithm parameters,
        including num_generations and max_num_to_evolve in advanced_config)
    projections : list of dict
        List of projection configurations. Each dict must contain:
        - 'dims': list of int - projection dimension indices
        - 'grid_points': list of int - grid points per dimension
        Optional keys per projection:
        - 'optimization_method': str - optimization algorithm ('de', 'lbfgsb') (default: 'de')
        - 'grid_refinement_factor': int - grid refinement factor (default: 2)
        - 'refinement_method': str - interpolation method (only 'linear' supported) (default: 'linear')
        - 'patch_coarse_grid': bool - enable patching on coarse grid (default: True)
        - 'patch_refined_grid': bool - enable patching on refined grid (default: False)
    save_plots : bool
        Whether to save plots to disk after each stage
    plot_settings : dict, optional
        Plot settings dictionary with keys:
        - 'dpi': int (default: 300)
        - 'filetype': str (default: 'png')
    myrank : int
        Master rank (usually 0)

    Returns
    -------
    list of dict
        List of results dictionaries (one per projection), each containing:
        - 'projection_config': dict - the original projection configuration
        - 'coarse_solution': dict - exported coarse grid state
        - 'refined_solution': dict or None - exported refined grid state (if refinement enabled)
        - 'metrics': dict - performance metrics

    Notes
    -----
    - Warm-starting is automatically enabled after the first projection
    - The global solution pool accumulates knowledge across all projections
    - Total target function calls are cumulative across all projections

    Examples
    --------
    >>> projections = [
    ...     {'dims': [0, 1], 'grid_points': [50, 50], 'grid_refinement_factor': 2},
    ...     {'dims': [0, 2], 'grid_points': [50, 50], 'grid_refinement_factor': 2},
    ... ]
    >>> results = run_all_projections(
    ...     comm=comm,
    ...     sampler=sampler,
    ...     projections=projections,
    ...     save_plots=True
    ... )
    >>> for i, res in enumerate(results):
    ...     print(f"Projection {i}: {res['metrics']['total_target_calls']} calls")
    """
    logger = setup_logger(rank=myrank)
    all_results = []

    for proj_idx, projection_config in enumerate(projections):
        logger.info("=" * 80)
        logger.info(f"=== Starting Projection {proj_idx + 1}/{len(projections)} ===")
        logger.info(f"=== Dimensions: {projection_config['dims']} ===")
        logger.info("=" * 80)

        # Reset sampler for new projection (except first)
        if proj_idx > 0:
            sampler._reset_for_new_projection(projection_config)

        # Enable warm start after first projection
        skip_init_opt = (proj_idx > 0)

        # Run projection (handles coarse + refinement automatically)
        results = run_projection(
            comm=comm,
            sampler=sampler,
            projection_config=projection_config,
            save_plots=save_plots,
            plot_settings=plot_settings,
            skip_init_opt_on_warm_start=skip_init_opt,
            myrank=myrank
        )

        # Add projection config to results for reference
        results['projection_config'] = projection_config
        all_results.append(results)

        logger.info("=" * 80)
        logger.info(f"=== Completed Projection {proj_idx + 1}/{len(projections)} ===")
        logger.info("=" * 80)

    return all_results


def master_main(comm, sampler,
                plot_settings=None, skip_init_opt_on_warm_start=True,
                myrank=0):
    """
    Main control loop for the master process.
    Acts as a state machine, dispatching jobs and processing results.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    sampler : ProfileProjector
        The sampler instance (contains num_generations and max_num_to_evolve
        in de_num_generations, de_max_num_to_evolve)
    plot_settings : dict, optional
        Plot settings dictionary with keys:
        - 'dpi': int (default: 300)
        - 'filetype': str (default: 'png')
        Not used in master_main directly (only passed for compatibility)
    skip_init_opt_on_warm_start : bool
        Whether to skip initial optimization if initial_maxima already exist
    myrank : int
        Master rank (usually 0)
    """
    logger = setup_logger(rank=myrank)
    n_workers = comm.Get_size() - 1
    if n_workers <= 0:
        logger.error("This script requires at least 2 MPI processes (1 master, 1+ workers).")
        return

    logger.debug(f"master_main: STARTING with {n_workers} workers.")

    # --- Master state ---
    free_workers = list(range(1, n_workers + 1))
    pending_sends = []  # Track non-blocking send requests

    # Define the workflow stages (different for refinement runs only)
    if sampler.is_refinement_run:
        stages = ['REFINEMENT_LBFGSB']
        # Disable patching in direct evaluation mode (no profiled params to share)
        if sampler.patch_refined_grid and not sampler.direct_eval_mode:
            stages.append('PATCHING_WAVES')
            if sampler.suspect_recheck_enabled:
                stages.append('SUSPECT_RECHECK_WAVES')
        logger.info("--- Refinement mode: Using direct LBFGSB optimization ---")
    else:
        # Normal mode: workflow depends on optimization method
        stages = []
        # Evaluate user-supplied initial points first (if any). This stage is
        # only added on the first projection that sees them; the job sets
        # ``_initial_points_evaluated`` once complete.
        if (sampler.initial_points is not None
                and not sampler._initial_points_evaluated):
            stages.append('INITIAL_POINTS_EVAL')
        stages += ['INITIAL_OPTIMIZATION', 'ACTIVATION']

        # Add optimization stage based on configured method
        if sampler.optimization_method == 'de':
            stages.append('DE_LOOP')
        elif sampler.optimization_method == 'lbfgsb':
            stages.append('LBFGSB_LOOP')

        # Add patching if enabled (applies to all optimization methods)
        if sampler.patch_coarse_grid and not sampler.direct_eval_mode:
            stages.append('PATCHING_WAVES')
            if sampler.suspect_recheck_enabled:
                stages.append('SUSPECT_RECHECK_WAVES')

        if sampler.direct_eval_mode:
            logger.info("--- Direct Evaluation Mode: No profiled parameters ---")
            logger.info("    Patching automatically disabled")
        else:
            logger.info(f"--- Optimization method: {sampler.optimization_method} ---")

    current_stage = stages.pop(0) if stages else None

    active_jobs = {} # {job_id: Job object}

    # Create priority task queues
    high_prio_tasks = collections.deque()
    low_prio_tasks = collections.deque()

    # Helper function to queue tasks
    def _queue_tasks(tasks, job_type):
        """Add tasks to appropriate priority queue based on job type."""
        if job_type in ['INITIAL_OPTIMIZATION', 'LBFGSB', 'REFINEMENT_LBFGSB', 'POST_ACTIVATION_LBFGSB', 'LBFGSB_LOOP', 'PATCHING_TEST', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK', 'SUSPECT_RECHECK_LBFGSB']:
            high_prio_tasks.extend(tasks)
        else:
            # DE_GRID_POINT, ACTIVATION go to low priority
            low_prio_tasks.extend(tasks)

    next_job_id = 0
    tasks_sent = 0
    tasks_completed = 0

    # DE stage state
    de_generation = 0 # Counter for DE stage
    de_successful_F = [] # Shared list for F/CR
    de_successful_CR = []

    # Wave-based patching state
    patching_wave_number = 0
    patching_updated_last_wave = None
    patching_wave_baseline_fitness = {}  # grid_idx -> fitness at wave start
    patching_wave_test_jobs = set()  # IDs of test jobs in current wave
    patching_wave_lbfgsb_jobs = set()  # IDs of L-BFGS-B jobs spawned by current wave

    # Wave-based suspect-recheck state (mirrors patching state)
    suspect_wave_number = 0
    suspect_updated_last_wave = None
    suspect_wave_baseline_fitness = {}
    suspect_wave_test_jobs = set()
    suspect_wave_lbfgsb_jobs = set()

    # --- Main Event Loop ---
    while current_stage or active_jobs or high_prio_tasks or low_prio_tasks or (tasks_sent > tasks_completed):

        # --- 1. Generate new jobs if a stage is starting or continuing ---
        # This block only runs when no jobs are active and no tasks are queued.
        if not active_jobs and not high_prio_tasks and not low_prio_tasks and (tasks_sent == tasks_completed):

            if not current_stage:
                break # All stages and jobs are complete

            logger.info(f"--- Master: Entering stage: {current_stage} (all jobs complete, creating new jobs) ---")
            new_jobs = []

            if current_stage == 'INITIAL_POINTS_EVAL':
                # Evaluate user-supplied initial_points via the standard Job
                # path so the master result-handler does the bookkeeping
                # (samples_output_file, global solution pool, etc.).
                new_jobs, next_job_id = sampler.create_initial_points_eval_job(next_job_id)
                if not new_jobs:
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'INITIAL_OPTIMIZATION':
                # Try to initialize from a warm-start file (separate from the
                # samples output file; see ProfileProjector.warm_start_file).
                if skip_init_opt_on_warm_start and sampler.warm_start_file:
                    sampler._initialize_from_warm_start_file(sampler.warm_start_file)

                # In-memory cross-projection seeding: skip the global L-BFGS-B
                # starts on projections 2..N when the accumulated pool already
                # covers the relevant peaks. Falls through to the original
                # initial-optimization path when the pool is empty (first
                # projection) or the feature is disabled.
                if (skip_init_opt_on_warm_start
                        and not sampler.initial_maxima
                        and sampler.pool_seeded_initial_maxima
                        and len(sampler.global_solution_pool) > 0):
                    sampler._initialize_from_global_pool()

                # If no maxima found from warm start (or warm start disabled), run global optimization
                if not sampler.initial_maxima:
                    new_jobs, next_job_id = sampler.create_initial_optimization_jobs(next_job_id)
                else:
                    logger.info("Skipping initial optimization - using provided initial points or warm start.")
                    new_jobs = []

                if not new_jobs:
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'ACTIVATION':
                # Use refinement activation for refinement runs
                if sampler.is_refinement_run:
                    new_jobs, next_job_id = sampler.create_refinement_activation_jobs(next_job_id)
                    if not new_jobs:
                        logger.info("No refinement activation jobs created. Moving to next stage.")
                        current_stage = stages.pop(0) if stages else None
                        continue
                else:
                    new_jobs, next_job_id = sampler.create_activation_jobs(next_job_id)
                    if not new_jobs:
                        logger.info("No activation jobs created (no initial maxima?). Moving to next stage.")
                        current_stage = stages.pop(0) if stages else None
                        continue

            elif current_stage == 'DE_LOOP':
                # This stage loops
                if de_generation >= sampler.de_num_generations:
                    logger.info("--- Master: DE generations complete. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                if de_generation > 0:
                    sampler.update_de_memory(de_successful_F, de_successful_CR)

                de_generation += 1
                sampler.current_generation = de_generation # Update sampler state
                logger.info(f"--- Master: Starting DE Generation {de_generation} ---")

                new_de_jobs, next_job_id, de_successful_F, de_successful_CR = sampler.create_de_generation_jobs(
                    next_job_id, sampler.de_max_num_to_evolve
                )

                # --- Add dynamic activation jobs ---
                new_act_jobs, next_job_id = sampler.create_dynamic_activation_jobs(next_job_id)
                new_jobs = new_de_jobs + new_act_jobs

                # --- Print Generation Summary ---
                total_grid_points = len(sampler.population)
                active_count = len([s for s in sampler.population.values() if s['status'] == 'active'])
                converged_count = total_grid_points - active_count
                newly_activated_count = len(new_act_jobs)

                logger.info(f"Gen {sampler.current_generation:4d} | Calls: {sampler.target_calls/1e3:6.1f}k | "
                      f"Grid Pts (act/conv/tot): {active_count:4d}/{converged_count:4d}/{total_grid_points:4d} | "
                      f"Global Max logL: {sampler.global_max_target_val:.4e} | "
                      f"New Activations: {newly_activated_count:3d}")
                # --- End Print ---

                if not new_jobs and not de_successful_F: # DE converged and no new activations
                    logger.info("--- Master: DE converged and no new activations. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'LBFGSB_LOOP':
                # Iterative L-BFGS-B optimization with dynamic activation
                # Similar to DE_LOOP but using gradient-based optimization

                # Create L-BFGS-B jobs for all active (non-converged) grid points
                new_lbfgsb_jobs, next_job_id = sampler.create_lbfgsb_loop_jobs(next_job_id)

                # Create dynamic activation jobs for neighbors of high-likelihood points
                new_act_jobs, next_job_id = sampler.create_dynamic_activation_jobs(next_job_id)
                new_jobs = new_lbfgsb_jobs + new_act_jobs

                # --- Print iteration summary ---
                total_grid_points = len(sampler.population)
                active_count = len([s for s in sampler.population.values() if s['status'] == 'active'])
                converged_count = len([s for s in sampler.population.values() if s['status'] in ['converged', 'optimized']])
                newly_activated_count = len(new_act_jobs)

                logger.info(f"LBFGSB Iter | Calls: {sampler.target_calls/1e3:6.1f}k | "
                      f"Grid Pts (act/conv/tot): {active_count:4d}/{converged_count:4d}/{total_grid_points:4d} | "
                      f"Global Max logL: {sampler.global_max_target_val:.4e} | "
                      f"New Activations: {newly_activated_count:3d} | "
                      f"L-BFGS-B Jobs: {len(new_lbfgsb_jobs):3d}")

                # Check convergence: no active jobs and no new activations
                if not new_jobs:
                    logger.info("--- Master: LBFGSB_LOOP converged (no active points, no new activations). ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'PATCHING_WAVES':
                # Check appropriate flag based on run type
                patching_enabled = sampler.patch_refined_grid if sampler.is_refinement_run else sampler.patch_coarse_grid
                if not patching_enabled:
                    logger.info("--- Master: Patching disabled for this stage. Skipping. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Check if max waves reached
                if patching_wave_number >= sampler.max_patching_waves:
                    logger.info(f"--- Master: Max patching waves ({sampler.max_patching_waves}) reached. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                logger.info(f"--- Master: Starting Patching Wave {patching_wave_number} ---")

                # Create jobs for this wave
                new_jobs, next_job_id = sampler.create_patching_wave_jobs(
                    wave_number=patching_wave_number,
                    updated_points_last_wave=patching_updated_last_wave,
                    next_job_id=next_job_id
                )

                if not new_jobs:
                    logger.info("--- Master: No patching candidates found. Ending patching. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Record baseline fitness for all grid points at wave start
                patching_wave_baseline_fitness = {
                    idx: state['best_fitness']
                    for idx, state in sampler.population.items()
                }

                # Track test job IDs
                patching_wave_test_jobs = {j.id for j in new_jobs}
                patching_wave_lbfgsb_jobs = set()

                # Transition to waiting state
                current_stage = 'WAITING_FOR_PATCHING_WAVE'

            elif current_stage == 'WAITING_FOR_PATCHING_WAVE':
                # This stage just waits for wave jobs to complete
                # Wave is complete when checked in job completion handling
                pass

            elif current_stage == 'SUSPECT_RECHECK_WAVES':
                if not sampler.suspect_recheck_enabled or sampler.direct_eval_mode:
                    logger.info("--- Master: Suspect recheck disabled. Skipping. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                if suspect_wave_number >= sampler.max_suspect_waves:
                    logger.info(f"--- Master: Max suspect waves ({sampler.max_suspect_waves}) reached. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                logger.info(f"--- Master: Starting Suspect Recheck Wave {suspect_wave_number} ---")

                new_jobs, next_job_id = sampler.create_suspect_recheck_jobs(
                    wave_number=suspect_wave_number,
                    updated_points_last_wave=suspect_updated_last_wave,
                    next_job_id=next_job_id,
                )

                if not new_jobs:
                    logger.info("--- Master: No suspect candidates. Ending suspect recheck. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                suspect_wave_baseline_fitness = {
                    idx: state['best_fitness']
                    for idx, state in sampler.population.items()
                }
                suspect_wave_test_jobs = {j.id for j in new_jobs}
                suspect_wave_lbfgsb_jobs = set()
                current_stage = 'WAITING_FOR_SUSPECT_WAVE'

            elif current_stage == 'WAITING_FOR_SUSPECT_WAVE':
                pass

            elif current_stage == 'REFINEMENT_LBFGSB':
                # For refinement runs, directly create LBFGSB jobs from interpolated starts
                new_jobs, next_job_id = sampler.create_refinement_lbfgsb_jobs(next_job_id)

                if not new_jobs:
                    logger.info("--- Master: No refinement LBFGSB jobs created. Ending refinement. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'POST_ACTIVATION_LBFGSB':
                # Direct L-BFGS-B optimization of all activated points
                # (alternative to DE when optimization_method='lbfgsb')
                new_jobs, next_job_id = sampler.create_post_activation_lbfgsb_jobs(next_job_id)

                if not new_jobs:
                    logger.info("--- Master: No post-activation L-BFGS-B jobs created. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue


            # --- Add new jobs to active pool and queue initial tasks ---
            for job in new_jobs:
                active_jobs[job.id] = job
                initial_tasks = job.start()

                # Add to correct priority queue
                _queue_tasks(initial_tasks, job.type)

                # A job may finish during start() without dispatching any tasks
                # (e.g. an LBFGSBJob with zero dims to optimize). Without this
                # cleanup it would stay in active_jobs forever and hang the
                # main loop, since the finalization path below only runs when
                # a result arrives.
                if not initial_tasks and job.is_finished():
                    job.on_finish(next_job_id)
                    del active_jobs[job.id]

            # If we just finished a non-looping stage, advance to the next
            # Looping stages: DE_LOOP, LBFGSB_LOOP (they re-run after jobs complete)
            if current_stage not in ['DE_LOOP', 'LBFGSB_LOOP',
                                     'WAITING_FOR_PATCHING_WAVE',
                                     'WAITING_FOR_SUSPECT_WAVE']:
                 current_stage = stages.pop(0) if stages else None

        # --- 2. Check for and process ALL available results ---
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            # A message is waiting, so now we do a blocking (but instant) receive
            result = comm.recv(source=MPI.ANY_SOURCE)
            worker_rank = result['context']['worker_rank']
            free_workers.append(worker_rank)
            tasks_completed += 1

            _log_worker_error(result, sampler, logger)
            _log_user_gradient_error(result, sampler, logger)
            sampler._register_target_call(result['params'], result['target_val'])

            job_id = result['context'].get('job_id', -1)
            if job_id not in active_jobs:
                logger.warning(f"Received result for unknown/finished job {job_id}. Ignoring.")
                continue

            job = active_jobs[job_id]
            new_tasks = job.process_result(result)

            # Add to correct priority queue
            _queue_tasks(new_tasks, job.type)

            if job.is_finished():
                job_id_finished = job.id

                # Track wave-based patching jobs
                is_patch_test_job = job_id_finished in patching_wave_test_jobs
                is_patch_lbfgsb_job = job_id_finished in patching_wave_lbfgsb_jobs
                is_suspect_test_job = job_id_finished in suspect_wave_test_jobs
                is_suspect_lbfgsb_job = job_id_finished in suspect_wave_lbfgsb_jobs

                if is_patch_test_job:
                    patching_wave_test_jobs.remove(job_id_finished)
                if is_patch_lbfgsb_job:
                    patching_wave_lbfgsb_jobs.remove(job_id_finished)
                if is_suspect_test_job:
                    suspect_wave_test_jobs.remove(job_id_finished)
                if is_suspect_lbfgsb_job:
                    suspect_wave_lbfgsb_jobs.remove(job_id_finished)

                # This updates the sampler state and can spawn a new job
                spawn_result = job.on_finish(next_job_id)
                del active_jobs[job_id_finished]

                if spawn_result:
                    new_job, next_job_id = spawn_result
                    active_jobs[new_job.id] = new_job
                    initial_tasks = new_job.start()

                    # Track child jobs spawned within a wave so the wave's
                    # completion check knows to wait for them.
                    if is_patch_test_job and new_job.type == 'PATCHING_LBFGSB':
                        patching_wave_lbfgsb_jobs.add(new_job.id)
                    if is_suspect_test_job and new_job.type == 'SUSPECT_RECHECK_LBFGSB':
                        suspect_wave_lbfgsb_jobs.add(new_job.id)

                    # Add to correct priority queue
                    _queue_tasks(initial_tasks, new_job.type)

                # Check if the patching wave is now complete
                if current_stage == 'WAITING_FOR_PATCHING_WAVE' and \
                   not patching_wave_test_jobs and not patching_wave_lbfgsb_jobs:

                    # Determine which grid points were updated (fitness improved)
                    updated_points = []
                    for idx, state in sampler.population.items():
                        baseline_fitness = patching_wave_baseline_fitness.get(idx, -np.inf)
                        if state['best_fitness'] > baseline_fitness:
                            updated_points.append(idx)

                    logger.info(f"--- Master: Patching Wave {patching_wave_number} complete. Updated {len(updated_points)} points ---")

                    if updated_points:
                        # Start next wave with updated points as seeds
                        patching_updated_last_wave = updated_points
                        patching_wave_number += 1
                        current_stage = 'PATCHING_WAVES'  # Loop back to start next wave
                    else:
                        # No updates, patching converged
                        logger.info("--- Master: No updates in wave. Patching converged. ---")
                        current_stage = stages.pop(0) if stages else None

                # Check if the suspect-recheck wave is now complete
                if current_stage == 'WAITING_FOR_SUSPECT_WAVE' and \
                   not suspect_wave_test_jobs and not suspect_wave_lbfgsb_jobs:

                    updated_points = []
                    for idx, state in sampler.population.items():
                        baseline_fitness = suspect_wave_baseline_fitness.get(idx, -np.inf)
                        if state['best_fitness'] > baseline_fitness:
                            updated_points.append(idx)

                    logger.info(
                        f"--- Master: Suspect Wave {suspect_wave_number} complete. "
                        f"Updated {len(updated_points)} points ---"
                    )

                    if updated_points:
                        suspect_updated_last_wave = updated_points
                        suspect_wave_number += 1
                        current_stage = 'SUSPECT_RECHECK_WAVES'
                    else:
                        logger.info("--- Master: No updates in wave. Suspect recheck converged. ---")
                        current_stage = stages.pop(0) if stages else None

        # --- End of Iprobe receive loop ---


        # --- 3. Dispatch tasks to free workers ---
        # Prioritize high priority tasks
        while free_workers and (high_prio_tasks or low_prio_tasks):
            worker_rank = free_workers.pop(0)

            if high_prio_tasks:
                task = high_prio_tasks.popleft()
            else:
                task = low_prio_tasks.popleft()

            # Use non-blocking send to overlap communication
            req = comm.isend(task, dest=worker_rank)
            pending_sends.append(req)
            tasks_sent += 1

        # --- 4. No Sleep Needed ---
        # Iprobe() is non-blocking and very lightweight, so no sleep is needed.
        # The loop will naturally wait for results or new tasks to become available.


    # --- End of Main Event Loop ---

    # Wait for all pending non-blocking sends to complete
    if pending_sends:
        logger.debug(f"Waiting for {len(pending_sends)} pending sends to complete...")
        MPI.Request.Waitall(pending_sends)

    logger.debug("master_main: Workflow finished.")

    # Final flush of sample buffer
    sampler._flush_samples_buffer()

    # --- Print Final Summary ---
    logger.info("=" * 80)
    logger.info("--- Master: Workflow Complete ---")
    logger.info(f"  Total Target Function Calls: {sampler.target_calls}")
    if sampler.target_call_errors:
        logger.warning(
            f"  Target Function Errors: {sampler.target_call_errors} "
            f"(out of {sampler.target_calls} calls)"
        )
    if sampler.grad_func is not None:
        logger.info(
            f"  Target Calls Saved by User Gradient: "
            f"{sampler.target_calls_saved_by_user_gradient}"
        )
        if sampler.user_gradient_errors:
            logger.warning(
                f"  User Gradient Errors: {sampler.user_gradient_errors} "
                f"(fell back to finite differences for these)"
            )
    logger.info(f"  Final Global Max logL: {sampler.global_max_target_val:.6e}")
    logger.info(f"  Total Grid Points Explored: {len(sampler.population)}")
    logger.info("=" * 80)
