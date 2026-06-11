"""
MPI master process orchestration logic.
"""
import collections
import numpy as np
import sys

from .exceptions import ConfigurationError
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
    """Send TASK_TERMINATE to every worker; master is ``myrank`` (default 0)."""
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
    """Run one projection (coarse grid + optional refinement) end-to-end.

    ``projection_config`` keys: required ``dims`` (list of int) and
    ``grid_points`` (list of int); optional ``optimization_method``
    ('de' or 'lbfgsb'), ``grid_refinement_factor`` (>1 enables a refined
    pass), ``refinement_method`` ('linear' only), ``patch_coarse_grid``,
    ``patch_refined_grid``.

    Returns a dict with ``coarse_solution``, ``refined_solution`` (or
    None), and a ``metrics`` dict (``coarse_target_calls``,
    ``refined_target_calls`` if applicable, ``total_target_calls``,
    ``global_max``).
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
    """Master-side wrapper: broadcast target, run all projections, terminate workers.

    Set ``broadcast_target_func=False`` for hosts that already provide
    the target function on every rank (e.g. bound methods that can't be
    pickled); in that case start workers with an explicit
    ``worker_main(comm, myrank, target_func=...)``.
    """
    if broadcast_target_func:
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
    """Run a list of projections sequentially with automatic cross-projection warm-start.

    See :func:`run_projection` for the per-projection config and result shape.
    Returns a list of per-projection result dicts with an extra
    ``projection_config`` field carrying the original input.
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

    # Optional post-projection volume-sampling stage (see volume.py and
    # docs/volume_sampling_plan.md). Result is stashed on
    # sampler.volume_stage_result; the per-projection result list is
    # returned unchanged.
    if getattr(sampler, 'volume_sampling_config', None):
        run_volume_sampling(comm, sampler, all_results, myrank=myrank)

    return all_results


def _volume_event_loop(comm, sampler, state, initial_jobs, job_source,
                       inflight_cap, logger, myrank=0):
    """Drive volume-stage jobs to completion (a slim master event loop).

    ``job_source`` is None or a callable returning the next job (or None
    when exhausted); it is only consulted while the evaluation budget
    allows and fewer than ``inflight_cap`` jobs are active. Every received
    result goes through the shared bookkeeping path: worker-error logging,
    ``_register_target_call`` (sample file), global-max tracking, and the
    stage state's budget/representative accounting.
    """
    active_jobs = {}
    task_queue = collections.deque()
    free_workers = [r for r in range(comm.Get_size()) if r != myrank]
    pending_sends = []
    tasks_sent = 0
    tasks_completed = 0

    def _finish_job(job):
        job.on_finish(None)
        if job.type == 'VOLUME_SEARCH':
            state.record_search_job(job)
        del active_jobs[job.id]
        # Early termination (a hit mid-gradient) can leave queued tasks
        # behind; drop them so workers aren't spent on a finished job.
        if task_queue:
            kept = [t for t in task_queue if t['context']['job_id'] != job.id]
            if len(kept) != len(task_queue):
                task_queue.clear()
                task_queue.extend(kept)

    def _admit(job):
        active_jobs[job.id] = job
        tasks = job.start()
        task_queue.extend(tasks)
        if not tasks and job.is_finished():
            _finish_job(job)

    for job in initial_jobs:
        _admit(job)

    while True:
        # --- Refill the in-flight job set from the source ---
        while (job_source is not None and len(active_jobs) < inflight_cap
               and state.budget_left()):
            job = job_source()
            if job is None:
                job_source = None
                break
            _admit(job)

        if (not active_jobs and not task_queue
                and tasks_sent == tasks_completed
                and (job_source is None or not state.budget_left())):
            break

        # --- Process all available results ---
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            result = comm.recv(source=MPI.ANY_SOURCE)
            free_workers.append(result['context']['worker_rank'])
            tasks_completed += 1

            _log_worker_error(result, sampler, logger)
            _log_user_gradient_error(result, sampler, logger)
            sampler._register_target_call(result['params'], result['target_val'])

            target_val = result['target_val']
            if target_val > sampler.global_max_target_val:
                logger.warning(
                    f"--- Volume sampling found a NEW GLOBAL MAX: "
                    f"{target_val:.6e} (previous: "
                    f"{sampler.global_max_target_val:.6e}) ---"
                )
                sampler.global_max_target_val = target_val

            # Probe jobs do their own exact per-anchor bookkeeping.
            state.note_eval(result['params'], target_val,
                            offer=result['context'].get('type') != 'VOLUME_PROBE')

            job = active_jobs.get(result['context'].get('job_id'))
            if job is None:
                continue
            task_queue.extend(job.process_result(result))
            if job.is_finished():
                _finish_job(job)

        # --- Dispatch tasks to free workers ---
        while free_workers and task_queue:
            worker_rank = free_workers.pop(0)
            task = task_queue.popleft()
            pending_sends.append(comm.isend(task, dest=worker_rank))
            tasks_sent += 1

    if pending_sends:
        MPI.Request.Waitall(pending_sends)
    return tasks_completed


def run_volume_sampling(comm, sampler, projection_results, myrank=0):
    """Run the post-projection volume-sampling stage (master side).

    Collects a stratified, well-spread set of in-band samples via the
    three-tier funnel (harvest -> probe -> anchored search) described in
    docs/volume_sampling_plan.md, using the projection grids in
    ``projection_results`` (the list returned by ``run_all_projections``)
    as the prefilter. Requires ``sampler.volume_sampling_config`` (the
    ``volume_sampling`` constructor argument) and live workers.

    Returns the stage-result dict (also stored on
    ``sampler.volume_stage_result``); when the stage cannot run, a dict
    with ``skipped=True`` and a ``reason``.
    """
    from .volume import (
        ProjectionEnvelope, VolumeStageState, depth_law_exponent,
        finalize_volume_stage, generate_anchors, harvest_existing_samples,
        resolve_harvest_files, volume_band, write_volume_output,
    )
    from .jobs.volume_jobs import VolumeProbeJob, VolumeSearchJob

    logger = setup_logger(rank=myrank)
    config = sampler.volume_sampling_config
    if not config:
        raise ConfigurationError(
            "run_volume_sampling requires the sampler to be constructed with "
            "a volume_sampling config dict",
            parameter="volume_sampling", value=None,
        )

    logger.info("=" * 80)
    logger.info(f"=== Volume sampling stage (mode: {config['mode']}) ===")
    logger.info("=" * 80)

    def _skip(reason):
        logger.info(f"--- Volume sampling skipped: {reason} ---")
        result = {'skipped': True, 'reason': reason}
        sampler.volume_stage_result = result
        return result

    # The harvest tier reads the run's own sample file; make it complete.
    sampler._flush_samples_buffer()

    global_max_start = sampler.global_max_target_val
    envelope = ProjectionEnvelope.from_projection_results(
        projection_results, global_max_start, sampler.dims)
    if envelope.covers_full_space:
        return _skip(
            "a projection grids the full parameter space, so the grid "
            "already covers the volume (a finer grid adds the same "
            "information)"
        )

    band_lo, band_hi, prefilter_delta = volume_band(
        config, sampler.roi_threshold, global_max_start)
    anchor_set = generate_anchors(
        envelope, sampler.bounds, config['n_points'], prefilter_delta,
        min_spacing=config['min_spacing'])
    if anchor_set.n_anchors == 0:
        return _skip("no anchors found inside the projection envelope")

    # --- Tier 1: harvest existing samples ---
    harvest_existing_samples(
        anchor_set,
        resolve_harvest_files(config, sampler.samples_output_file),
        band_lo, band_hi)

    state = VolumeStageState(anchor_set, band_lo, band_hi,
                             eval_budget=config['eval_budget'])
    n_workers = max(comm.Get_size() - 1, 1)
    inflight_cap = max(2, n_workers)

    # --- Tier 2: direct probes at the anchors ---
    if config['probe_all_anchors']:
        probe_targets = np.arange(anchor_set.n_anchors)
    else:
        probe_targets = np.flatnonzero(~anchor_set.covered)
    if config['eval_budget'] is not None \
            and len(probe_targets) > config['eval_budget']:
        state.unbudgeted[probe_targets[config['eval_budget']:]] = True
        probe_targets = probe_targets[:config['eval_budget']]
        logger.warning(
            f"--- Volume sampling: eval_budget ({config['eval_budget']}) "
            f"truncates the probe stage to {len(probe_targets)} anchors ---"
        )

    if len(probe_targets):
        probe_job = VolumeProbeJob(0, sampler, anchor_set, probe_targets,
                                   band_lo, band_hi)
        _volume_event_loop(comm, sampler, state, [probe_job], None,
                           inflight_cap, logger, myrank)

    # --- Tier 3: anchored searches for anchors whose probe missed ---
    if config['search'] != 'none':
        kappa = sampler.volume_penalty_strength / sampler.roi_threshold ** 2
        depth_exponent = depth_law_exponent(config['depth_law'], sampler.dims)
        # Adaptive depth-target quota (roi mode): walks draw from the
        # law's residual need so depths censored by local reachability
        # get retried at other anchors.
        draw_depth_target = None
        if config['mode'] == 'roi' and config['interior_steps'] > 0:
            state.init_depth_quota(sampler.roi_threshold, depth_exponent)
            draw_depth_target = state.draw_depth_target
        next_job_id = 1
        search_cursor = 0

        def next_search_job():
            nonlocal next_job_id, search_cursor
            while search_cursor < anchor_set.n_anchors:
                k = search_cursor
                search_cursor += 1
                # Skip anchors that were never probed (budget or
                # probe_all_anchors=False on a covered anchor) and anchors
                # covered meanwhile (harvest, probe, or a search
                # byproduct). Passive reps count toward the depth quota so
                # the walks compensate for their (usually band-edge-heavy)
                # depth mix.
                if not anchor_set.probed[k] or anchor_set.covered[k]:
                    if anchor_set.covered[k] and draw_depth_target is not None:
                        state.record_rep_depth(anchor_set.rep_logls[k])
                    continue
                if np.isfinite(anchor_set.rep_dists[k]):
                    warm = anchor_set.rep_points[k]
                else:
                    warm = anchor_set.anchors[k]
                job = VolumeSearchJob(
                    next_job_id, sampler, anchor_set, k, band_lo, band_hi,
                    kappa, warm, max_iter=config['search_max_iter'],
                    interior_steps=config['interior_steps'],
                    depth_exponent=depth_exponent,
                    draw_depth_target=draw_depth_target)
                next_job_id += 1
                return job
            return None

        _volume_event_loop(comm, sampler, state, [], next_search_job,
                           inflight_cap, logger, myrank)

    # --- Finalize: classify anchors against the final global max ---
    sampler._flush_samples_buffer()
    result = finalize_volume_stage(
        state, config, sampler.roi_threshold,
        global_max_start, sampler.global_max_target_val,
        search_enabled=(config['search'] != 'none'))
    stats = result['stats']

    # --- Write the tagged sample file and the JSON summary ---
    try:
        write_volume_output(result, config)
    except (OSError, ValueError) as e:
        logger.warning(f"Volume sampling: could not write output files: {e}")

    logger.info("=" * 80)
    logger.info("--- Volume sampling complete ---")
    logger.info(
        f"  Anchors: {stats['n_anchors']} "
        f"(coverage radius {stats['coverage_radius']:.4g}, scaled units)"
    )
    logger.info(
        f"  Covered: {stats['n_covered']} (harvest "
        f"{stats['n_covered_harvest']}, probe {stats['n_covered_probe']}, "
        f"search {stats['n_covered_search']})"
    )
    logger.info(
        f"  Projected: {stats['n_projected']}, holes: {stats['n_holes']}, "
        f"unbudgeted: {stats['n_unbudgeted']}, "
        f"uncovered: {stats['n_uncovered']}"
    )
    logger.info(f"  Stage evaluations: {stats['evals_used']}")
    logger.info(
        f"  Prefilter acceptance: {stats['prefilter_acceptance']:.3g}; "
        f"probe acceptance: {stats['probe_acceptance']:.3g}"
    )
    if stats['volume_estimate'] is not None:
        logger.info(
            f"  Band volume estimate: {stats['volume_estimate']:.6g} "
            f"+/- {stats['volume_estimate_err']:.2g}"
        )
    if stats['global_max_drift'] > 0:
        logger.warning(
            f"  Global max improved by {stats['global_max_drift']:.6g} "
            f"during volume sampling; band membership was re-derived, but "
            f"the projection results themselves may be stale -- consider "
            f"re-running the projections."
        )
    logger.info("=" * 80)

    sampler.volume_stage_result = result
    return result


def master_main(comm, sampler,
                plot_settings=None, skip_init_opt_on_warm_start=True,
                myrank=0):
    """State-machine main loop: dispatches jobs and processes results until done."""
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

    def _purge_queued_tasks(job_ids):
        """Drop not-yet-dispatched tasks belonging to the given jobs from both
        priority queues (used to abort optimizations). Tasks already running on
        workers return normally and are ignored via the unknown-job path."""
        for q in (high_prio_tasks, low_prio_tasks):
            kept = [t for t in q if t.get('context', {}).get('job_id') not in job_ids]
            q.clear()
            q.extend(kept)

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

    # Rolling initial-optimization (basin detection) state. The stage keeps
    # `initial_opt_batch_size` global L-BFGS-B starts in flight, classifying each
    # converged optimum and refilling until the Bayesian stopping rule trips or
    # the `initial_opt_cap` (= n_initial_optimizations) is reached.
    initial_opt_inflight = set()   # job IDs of in-flight INITIAL_OPTIMIZATION jobs
    initial_opt_started = 0        # starts launched so far
    initial_opt_completed = 0      # starts finished so far (incl. failures)
    initial_opt_batch_size = 0
    initial_opt_cap = 0
    initial_opt_stopped = False    # rule tripped or cap reached: stop refilling

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

                # If maxima already came from warm start / pool seeding, skip the
                # global optimization stage entirely.
                if sampler.initial_maxima:
                    logger.info("Skipping initial optimization - using provided initial points or warm start.")
                    current_stage = stages.pop(0) if stages else None
                    continue

                # Rolling multistart with online basin detection. Launch a
                # first batch and hand control to WAITING_FOR_INITIAL_OPT,
                # which refills and applies the stopping rule as jobs return.
                initial_opt_cap = sampler.n_initial_optimizations
                initial_opt_batch_size = sampler.resolve_initial_opt_batch_size(n_workers)
                initial_opt_started = 0
                initial_opt_completed = 0
                initial_opt_stopped = False
                initial_opt_inflight = set()
                sampler.init_initial_opt_lhs(initial_opt_cap)

                logger.info(
                    f"--- Basin detection: rolling multistart "
                    f"(batch_size={initial_opt_batch_size}, cap={initial_opt_cap}, "
                    f"min_starts={sampler.basin_min_starts}) ---"
                )
                new_jobs = []
                for _ in range(min(initial_opt_batch_size, initial_opt_cap)):
                    job, next_job_id = sampler.create_one_initial_optimization_job(next_job_id)
                    new_jobs.append(job)
                    initial_opt_inflight.add(job.id)
                    initial_opt_started += 1

                if not new_jobs:
                    # Degenerate cap (e.g. n_initial_optimizations == 0):
                    # nothing to optimize, move on.
                    current_stage = stages.pop(0) if stages else None
                    continue
                current_stage = 'WAITING_FOR_INITIAL_OPT'

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
                                     'WAITING_FOR_SUSPECT_WAVE',
                                     'WAITING_FOR_INITIAL_OPT']:
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

                # Rolling initial-optimization with basin detection. The job's
                # on_finish has already clustered its endpoint into the registry;
                # here we apply the stopping rule and refill the in-flight batch.
                if current_stage == 'WAITING_FOR_INITIAL_OPT' and \
                   job_id_finished in initial_opt_inflight:

                    initial_opt_inflight.discard(job_id_finished)
                    initial_opt_completed += 1

                    if not initial_opt_stopped:
                        if initial_opt_started >= initial_opt_cap:
                            initial_opt_stopped = True
                            logger.info(
                                f"--- Basin detection: reached cap of "
                                f"{initial_opt_cap} starts; stopping ---"
                            )
                        elif sampler.basin_detection_should_stop(initial_opt_completed):
                            W, n_roi = sampler.basin_detection_roi_stats()
                            n_distinct = len(sampler.initial_optima_registry)
                            initial_opt_stopped = True
                            # Attribute the stop to the prior vs the Bayesian rule.
                            reason = (
                                "known-optima prior met"
                                if (sampler.basin_max_optima is not None
                                    and n_distinct >= sampler.basin_max_optima)
                                else "stopping rule met"
                            )
                            # Abort the still-running optimizations -- their
                            # remaining evaluations would be pure overshoot.
                            # Dropping them from active_jobs makes their in-flight
                            # results get ignored; queued tasks are purged. (The
                            # cap branch above keeps runs we've already paid for.)
                            aborted = set(initial_opt_inflight)
                            for jid in aborted:
                                active_jobs.pop(jid, None)
                            initial_opt_inflight.clear()
                            _purge_queued_tasks(aborted)
                            logger.info(
                                f"--- Basin detection: {reason} after "
                                f"{initial_opt_completed} optimizations "
                                f"({n_distinct} distinct optima, {W} in ROI); "
                                f"aborted {len(aborted)} in-flight run(s) ---"
                            )

                    # Refill to keep `initial_opt_batch_size` starts in flight.
                    if not initial_opt_stopped:
                        while (len(initial_opt_inflight) < initial_opt_batch_size
                               and initial_opt_started < initial_opt_cap):
                            new_job, next_job_id = sampler.create_one_initial_optimization_job(next_job_id)
                            active_jobs[new_job.id] = new_job
                            initial_opt_inflight.add(new_job.id)
                            initial_opt_started += 1
                            _queue_tasks(new_job.start(), new_job.type)

                    # Stage done once the in-flight batch has fully drained.
                    if not initial_opt_inflight:
                        W, n_roi = sampler.basin_detection_roi_stats()
                        logger.info(
                            f"--- Basin detection complete: {initial_opt_completed} "
                            f"optimizations, {W} distinct ROI optima "
                            f"({len(sampler.initial_optima_registry)} optima total) ---"
                        )
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
    if sampler.de_allow_early_DE_exit and sampler.de_cells_skipped:
        logger.info(
            f"  Cells that skipped the DE global search (allow_early_DE_exit): "
            f"{sampler.de_cells_skipped}"
        )
    logger.info("=" * 80)
