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


def master_main(comm, sampler, num_generations, max_num_to_evolve,
                plot_callback, plot_interval, skip_init_opt_on_warm_start=True,
                fig=None, axes=None, myrank=0):
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
    plot_callback : callable or None
        Function to call for plotting (signature: plot_callback(sampler, fig, axes))
    plot_interval : float
        Seconds between plot updates
    skip_init_opt_on_warm_start : bool
        Whether to skip initial optimization if initial_maxima already exist
    fig : matplotlib.figure.Figure or None
        Figure for plotting
    axes : list of matplotlib.axes.Axes or None
        Axes for plotting
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
        print("--- Refinement mode: Using direct LBFGSB optimization ---")
    else:
        stages = ['INITIAL_OPTIMIZATION', 'ACTIVATION', 'DE_LOOP', 'PATCHING_WAVES']

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

    last_plot_time = time.time()
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
                # Check if patching is enabled for this projection
                if not sampler.enable_patching:
                    print("--- Master: Patching disabled for this projection. Skipping. ---")
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

        # --- 4. Plotting (optional) & Polite Sleep ---
        current_time = time.time()
        if plot_callback and (current_time - last_plot_time > plot_interval):
             # Only plot during DE phase for this example
            if sampler.current_generation > 0:
                print(f"Plotting... (Gen {sampler.current_generation})")
                plot_callback(sampler, fig, axes)
                last_plot_time = current_time

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
