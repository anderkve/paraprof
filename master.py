"""
MPI master process orchestration logic.
"""
import time
import collections
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    import sys
    sys.exit(1)

from constants import TASK_TERMINATE


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
    n_workers = comm.Get_size() - 1

    print(f"rank {myrank}: DEBUG: terminate_workers: Sending TASK_TERMINATE to workers.", flush=True)
    for rank in range(1, n_workers + 1):
        comm.send(TASK_TERMINATE, dest=rank)

    print(f"rank {myrank}: DEBUG: terminate_workers: All workers terminated.")


def run_projection(comm, sampler, projection_config,
                   num_generations=100000,
                   max_num_to_evolve=None,
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
    sampler : GridAnchoredDESampler
        The sampler instance (already configured)
    projection_config : dict
        Projection configuration. Required keys:
        - 'dims': list of int - projection dimension indices
        - 'grid_points': list of int - grid points per dimension
        Optional keys:
        - 'enable_refinement': bool - enable grid refinement (default: False)
        - 'refinement_factor': int - refinement factor (default: 2)
        - 'lbfgsb': bool - enable L-BFGS-B optimization (default: True)
        - 'patching': bool - enable patching stage (default: True)
    num_generations : int
        Maximum number of DE generations to run
    max_num_to_evolve : int or None
        Maximum number of grid points to evolve per generation (None = all)
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
    ...         'enable_refinement': True,
    ...         'refinement_factor': 3
    ...     },
    ...     save_plots=True
    ... )
    >>> print(f"Best likelihood: {results['metrics']['global_max']}")
    """
    # Extract refinement configuration
    enable_refinement = projection_config.get('enable_refinement', False)
    refinement_factor = projection_config.get('refinement_factor', 2)
    dims_str = "_".join(map(str, projection_config['dims']))

    # Initialize results structure
    results = {
        'coarse_solution': None,
        'refined_solution': None,
        'metrics': {}
    }

    # --- COARSE GRID RUN ---
    print("\n" + "="*80)
    print("=== Running Coarse Grid ===")
    print("="*80 + "\n")

    master_main(
        comm=comm,
        sampler=sampler,
        num_generations=num_generations,
        max_num_to_evolve=max_num_to_evolve,
        plot_settings=plot_settings,
        skip_init_opt_on_warm_start=skip_init_opt_on_warm_start,
        myrank=myrank
    )

    # Flush samples buffer after coarse grid
    sampler._flush_samples_buffer()

    # Save coarse plot if requested
    if save_plots:
        from visualization import plot_profiles
        plot_filename = f"profile_plot_rank_{myrank}_dims_{dims_str}_coarse"
        plot_profiles(sampler, plot_filename, plot_settings)

    # Export coarse solution
    coarse_solution = sampler.export_grid_solution()
    results['coarse_solution'] = coarse_solution
    results['metrics']['coarse_target_calls'] = sampler.target_calls

    # --- REFINEMENT RUN (if enabled) ---
    if enable_refinement:
        print("\n" + "="*80)
        print("=== Starting Grid Refinement ===")
        print("="*80 + "\n")

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
            num_generations=num_generations,
            max_num_to_evolve=max_num_to_evolve,
            plot_settings=plot_settings,
            skip_init_opt_on_warm_start=True,  # Always skip for refinement
            myrank=myrank
        )

        # Flush samples buffer after refinement
        sampler._flush_samples_buffer()

        # Save refined plot if requested
        if save_plots:
            from visualization import plot_profiles
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


def run_all_projections(comm, sampler, projections,
                        num_generations=100000,
                        max_num_to_evolve=None,
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
    sampler : GridAnchoredDESampler
        The sampler instance (already configured with bounds, algorithm parameters)
    projections : list of dict
        List of projection configurations. Each dict must contain:
        - 'dims': list of int - projection dimension indices
        - 'grid_points': list of int - grid points per dimension
        Optional keys per projection:
        - 'enable_refinement': bool - enable grid refinement (default: False)
        - 'refinement_factor': int - refinement factor (default: 2)
        - 'lbfgsb': bool - enable L-BFGS-B optimization (default: True)
        - 'patching': bool - enable patching stage (default: True)
    num_generations : int
        Maximum number of DE generations to run per projection
    max_num_to_evolve : int or None
        Maximum number of grid points to evolve per generation (None = all)
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
    ...     {'dims': [0, 1], 'grid_points': [50, 50], 'enable_refinement': True},
    ...     {'dims': [0, 2], 'grid_points': [50, 50], 'enable_refinement': True},
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
    all_results = []

    for proj_idx, projection_config in enumerate(projections):
        print("\n" + "="*80)
        print(f"=== Starting Projection {proj_idx + 1}/{len(projections)} ===")
        print(f"=== Dimensions: {projection_config['dims']} ===")
        print("="*80 + "\n")

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
            num_generations=num_generations,
            max_num_to_evolve=max_num_to_evolve,
            save_plots=save_plots,
            plot_settings=plot_settings,
            skip_init_opt_on_warm_start=skip_init_opt,
            myrank=myrank
        )

        # Add projection config to results for reference
        results['projection_config'] = projection_config
        all_results.append(results)

        print("\n" + "="*80)
        print(f"=== Completed Projection {proj_idx + 1}/{len(projections)} ===")
        print("="*80 + "\n")

    return all_results


def master_main(comm, sampler, num_generations, max_num_to_evolve,
                plot_settings=None, skip_init_opt_on_warm_start=True,
                myrank=0):
    """
    Main control loop for the master process.
    Acts as a state machine, dispatching jobs and processing results.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    sampler : GridAnchoredDESampler
        The sampler instance
    num_generations : int
        Number of DE generations to run
    max_num_to_evolve : int or None
        Maximum number of grid points to evolve per generation (None = all)
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
    n_workers = comm.Get_size() - 1
    if n_workers <= 0:
        print("Error: This script requires at least 2 MPI processes (1 master, 1+ workers).")
        return

    print(f"rank {myrank}: DEBUG: master_main: STARTING with {n_workers} workers.")

    # --- Master state ---
    free_workers = list(range(1, n_workers + 1))

    # Define the workflow stages (different for refinement runs)
    if sampler.is_refinement_run:
        stages = ['REFINEMENT_LBFGSB']
        if sampler.patching_refined:
            stages.append('PATCHING_WAVES')
        print("--- Refinement mode: Using direct LBFGSB optimization ---")
    else:
        stages = ['INITIAL_OPTIMIZATION', 'ACTIVATION', 'DE_LOOP']
        if sampler.patching_coarse:
            stages.append('PATCHING_WAVES')

    current_stage = stages.pop(0) if stages else None

    active_jobs = {} # {job_id: Job object}

    # --- MODIFICATION: Create priority task queues ---
    high_prio_tasks = collections.deque()
    low_prio_tasks = collections.deque()
    # task_queue = collections.deque() # Old queue
    # --- END MODIFICATION ---

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

    de_gen_start_time = time.time() # Add timer for DE generations

    # --- Main Event Loop ---
    # --- MODIFICATION: Update loop condition to check both queues ---
    while current_stage or active_jobs or high_prio_tasks or low_prio_tasks or (tasks_sent > tasks_completed):
    # --- END MODIFICATION ---

        # --- 1. Generate new jobs if a stage is starting or continuing ---
        # This block only runs when no jobs are active and no tasks are queued.
        # --- MODIFICATION: Update check to include both queues ---
        if not active_jobs and not high_prio_tasks and not low_prio_tasks and (tasks_sent == tasks_completed):
        # --- END MODIFICATION ---

            if not current_stage:
                break # All stages and jobs are complete

            print(f"--- Master: Entering stage: {current_stage} ---")
            new_jobs = []

            if current_stage == 'INITIAL_OPTIMIZATION':
                # Skip initial optimization if this is a refinement run
                if sampler.is_refinement_run:
                    print("Skipping initial optimization - refinement run mode.")
                    new_jobs = []
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Try to initialize from warm start file first
                if skip_init_opt_on_warm_start and sampler.samples_output_file:
                    sampler._initialize_from_warm_start_file(sampler.samples_output_file)

                # If no maxima found from warm start (or warm start disabled), run global optimization
                if not sampler.initial_maxima:
                    new_jobs, next_job_id = sampler.create_initial_optimization_jobs(next_job_id)
                else:
                    print("Skipping initial optimization - using warm start from file.")
                    new_jobs = []

                if not new_jobs:
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'ACTIVATION':
                # Use refinement activation for refinement runs
                if sampler.is_refinement_run:
                    new_jobs, next_job_id = sampler.create_refinement_activation_jobs(next_job_id)
                    if not new_jobs:
                        print("No refinement activation jobs created. Moving to next stage.")
                        current_stage = stages.pop(0) if stages else None
                        continue
                else:
                    new_jobs, next_job_id = sampler.create_activation_jobs(next_job_id)
                    if not new_jobs:
                        print("No activation jobs created (no initial maxima?). Moving to next stage.")
                        current_stage = stages.pop(0) if stages else None
                        continue

            elif current_stage == 'DE_LOOP':
                # This stage loops
                if de_generation >= num_generations:
                    print("--- Master: DE generations complete. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Update F/CR memory from *previous* generation
                if de_generation > 0:
                    sampler.update_de_memory(de_successful_F, de_successful_CR)

                de_generation += 1
                sampler.current_generation = de_generation # Update sampler state
                print(f"--- Master: Starting DE Generation {de_generation} ---")

                new_de_jobs, next_job_id, de_successful_F, de_successful_CR = sampler.create_de_generation_jobs(
                    next_job_id, max_num_to_evolve
                )

                # --- Add dynamic activation jobs ---
                new_act_jobs, next_job_id = sampler.create_dynamic_activation_jobs(next_job_id)
                new_jobs = new_de_jobs + new_act_jobs

                # --- Print Generation Summary ---
                elapsed = time.time() - de_gen_start_time
                de_gen_start_time = time.time() # Reset timer

                total_grid_points = len(sampler.population)
                active_count = len([s for s in sampler.population.values() if s['status'] == 'active'])
                converged_count = total_grid_points - active_count
                newly_activated_count = len(new_act_jobs)

                print(f"Gen {sampler.current_generation:4d} | Calls: {sampler.target_calls/1e3:6.1f}k | "
                      f"Grid Pts (act/conv/tot): {active_count:4d}/{converged_count:4d}/{total_grid_points:4d} | "
                      f"Global Max logL: {sampler.global_max_target_val:.4e} | "
                      f"New Activations: {newly_activated_count:3d} | "
                      f"Elapsed: {elapsed:.1f}s")
                # --- End Print ---

                if not new_jobs and not de_successful_F: # DE converged and no new activations
                    print("--- Master: DE converged and no new activations. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

            elif current_stage == 'PATCHING_WAVES':
                # Check appropriate flag based on run type
                patching_enabled = sampler.patching_refined if sampler.is_refinement_run else sampler.patching_coarse
                if not patching_enabled:
                    print("--- Master: Patching disabled for this stage. Skipping. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Check if max waves reached
                if patching_wave_number >= sampler.max_patching_waves:
                    print(f"--- Master: Max patching waves ({sampler.max_patching_waves}) reached. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue

                print(f"\n--- Master: Starting Patching Wave {patching_wave_number} ---")

                # Create jobs for this wave
                new_jobs, next_job_id = sampler.create_patching_wave_jobs(
                    wave_number=patching_wave_number,
                    updated_points_last_wave=patching_updated_last_wave,
                    next_job_id=next_job_id
                )

                if not new_jobs:
                    print("--- Master: No patching candidates found. Ending patching. ---")
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

            elif current_stage == 'REFINEMENT_LBFGSB':
                # For refinement runs, directly create LBFGSB jobs from interpolated starts
                new_jobs, next_job_id = sampler.create_refinement_lbfgsb_jobs(next_job_id)

                if not new_jobs:
                    print("--- Master: No refinement LBFGSB jobs created. Ending refinement. ---")
                    current_stage = stages.pop(0) if stages else None
                    continue


            # --- Add new jobs to active pool and queue initial tasks ---
            for job in new_jobs:
                active_jobs[job.id] = job
                initial_tasks = job.start()

                # --- MODIFICATION: Add to correct priority queue ---
                if job.type in ['INITIAL_OPTIMIZATION', 'LBFGSB', 'REFINEMENT_LBFGSB', 'PATCHING_TEST', 'PATCHING_LBFGSB']:
                    high_prio_tasks.extend(initial_tasks)
                else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                    low_prio_tasks.extend(initial_tasks)
                # --- END MODIFICATION ---

            # If we just finished a non-looping stage, advance to the next
            if current_stage not in ['DE_LOOP', 'WAITING_FOR_PATCHING_WAVE']:
                 current_stage = stages.pop(0) if stages else None

        # --- 3. (MODIFIED) Check for and process ALL available results ---
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            # A message is waiting, so now we do a blocking (but instant) receive
            result = comm.recv(source=MPI.ANY_SOURCE)
            worker_rank = result['context']['worker_rank']
            free_workers.append(worker_rank)
            tasks_completed += 1

            # Register the call centrally
            sampler._register_target_call(result['params'], result['target_val'])

            job_id = result['context'].get('job_id', -1)
            if job_id not in active_jobs:
                print(f"Warning: Received result for unknown/finished job {job_id}. Ignoring.")
                continue

            job = active_jobs[job_id]
            new_tasks = job.process_result(result)

            # --- MODIFICATION: Add to correct priority queue ---
            if job.type in ['INITIAL_OPTIMIZATION', 'LBFGSB', 'REFINEMENT_LBFGSB', 'PATCHING_TEST', 'PATCHING_LBFGSB']:
                high_prio_tasks.extend(new_tasks)
            else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                low_prio_tasks.extend(new_tasks)
            # --- END MODIFICATION ---

            if job.is_finished():
                job_id_finished = job.id

                # Track wave-based patching jobs
                is_wave_test_job = job_id_finished in patching_wave_test_jobs
                is_wave_lbfgsb_job = job_id_finished in patching_wave_lbfgsb_jobs

                if is_wave_test_job:
                    patching_wave_test_jobs.remove(job_id_finished)

                if is_wave_lbfgsb_job:
                    patching_wave_lbfgsb_jobs.remove(job_id_finished)

                # This updates the sampler state and can spawn a new job
                spawn_result = job.on_finish(next_job_id)
                del active_jobs[job_id_finished]

                if spawn_result:
                    new_job, next_job_id = spawn_result
                    active_jobs[new_job.id] = new_job
                    initial_tasks = new_job.start()

                    # Track if this is an L-BFGS-B job spawned by a patching test
                    if is_wave_test_job and new_job.type == 'PATCHING_LBFGSB':
                        patching_wave_lbfgsb_jobs.add(new_job.id)

                    # --- MODIFICATION: Add to correct priority queue ---
                    if new_job.type in ['INITIAL_OPTIMIZATION', 'LBFGSB', 'REFINEMENT_LBFGSB', 'PATCHING_TEST', 'PATCHING_LBFGSB']:
                        high_prio_tasks.extend(initial_tasks)
                    else: # 'ACTIVATE_GRID_POINT', 'DE_GRID_POINT'
                        low_prio_tasks.extend(initial_tasks)
                    # --- END MODIFICATION ---

                # Check if the patching wave is now complete
                if current_stage == 'WAITING_FOR_PATCHING_WAVE' and \
                   not patching_wave_test_jobs and not patching_wave_lbfgsb_jobs:

                    # Determine which grid points were updated (fitness improved)
                    updated_points = []
                    for idx, state in sampler.population.items():
                        baseline_fitness = patching_wave_baseline_fitness.get(idx, -np.inf)
                        if state['best_fitness'] > baseline_fitness:
                            updated_points.append(idx)

                    print(f"--- Master: Patching Wave {patching_wave_number} complete. Updated {len(updated_points)} points ---")

                    if updated_points:
                        # Start next wave with updated points as seeds
                        patching_updated_last_wave = updated_points
                        patching_wave_number += 1
                        current_stage = 'PATCHING_WAVES'  # Loop back to start next wave
                    else:
                        # No updates, patching converged
                        print("--- Master: No updates in wave. Patching converged. ---")
                        current_stage = stages.pop(0) if stages else None

        # --- End of Iprobe receive loop ---


        # --- 2. (Now Step 3) Dispatch tasks to free workers ---
        # --- MODIFICATION: Prioritize high_prio_tasks ---
        while free_workers and (high_prio_tasks or low_prio_tasks):
            worker_rank = free_workers.pop(0)

            if high_prio_tasks:
                task = high_prio_tasks.popleft()
            else:
                task = low_prio_tasks.popleft()

            comm.send(task, dest=worker_rank)
            tasks_sent += 1
        # --- END MODIFICATION ---

        # --- 3. Wait for and process a result ---
        # Only check for results if we are expecting any
        # --- (MODIFICATION: This block is now removed and replaced by Iprobe loop) ---
        # if tasks_sent > tasks_completed:
        #    ... (old blocking recv logic removed) ...
        # --- (END MODIFICATION) ---

        # --- 4. Polite Sleep ---
        # If no tasks are ready to send and no workers are free,
        # but we are still waiting for results, sleep for a tiny bit
        # to prevent a 100% CPU busy-wait.
        if not free_workers and not high_prio_tasks and not low_prio_tasks and (tasks_sent > tasks_completed):
            time.sleep(0.001) # 1ms sleep


    # --- End of Main Event Loop ---

    print(f"rank {myrank}: DEBUG: master_main: Workflow finished.")

    # Final flush of sample buffer
    sampler._flush_samples_buffer()

    # --- Print Final Summary ---
    print("\n" + "="*80)
    print("--- Master: Workflow Complete ---")
    print(f"  Total Target Function Calls: {sampler.target_calls}")
    print(f"  Final Global Max logL: {sampler.global_max_target_val:.6e}")
    print(f"  Total Grid Points Explored: {len(sampler.population)}")
    print("="*80 + "\n")
