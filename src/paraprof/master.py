"""
MPI master process orchestration logic.
"""
import collections
import logging
import time
import numpy as np
import sys

from .exceptions import ConfigurationError
from .logger import setup_logger
from .phases import phase_for_job_type

try:
    from mpi4py import MPI
except ImportError:
    # Can't use logger here since MPI isn't available yet
    print("Error: mpi4py is not installed. This script requires MPI.", file=sys.stderr)
    print("Please install it with: pip install mpi4py", file=sys.stderr)
    sys.exit(1)

TASK_TERMINATE = -1
# Length of the list of non-blocking sends at which completed ones are dropped.
PENDING_SENDS_PRUNE = 1024

# Fraction of the workers that cells activated ahead (lbfgsb activation_depth
# > 1) leave free. Their evaluations cannot be interrupted, so without a
# reserve ordinary work would wait behind them.
AHEAD_RESERVE_FRACTION = 0.25

# With cross_projection eager_start, the next projection starts when at least
# this fraction of the workers is idle after a dispatch pass.
EAGER_START_IDLE_FRACTION = 0.1
# Seconds between progress lines of the L-BFGS-B loop. With rolling activation
# the loop has no waves to report.
PROGRESS_LOG_INTERVAL = 60.0


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


def _eager_eligible(sampler, projection_config):
    """Whether a projection may overlap with others (cross_projection
    eager_start). Not with grid refinement, whose second pass keeps state
    outside the per-projection state, and not without profiled parameters."""
    factor = projection_config.get('grid_refinement_factor', None)
    if factor is not None and factor > 1:
        return False
    return len(projection_config['dims']) < sampler.dims


def _collect_solution(sampler, projection_config, save_plots, plot_settings, myrank,
                      grid='coarse'):
    """Flush, plot and export the projection the sampler holds."""
    sampler._flush_samples_buffer()
    if save_plots:
        from .visualization import plot_profiles
        dims_str = "_".join(map(str, projection_config['dims']))
        plot_profiles(sampler, f"profile_plot_rank_{myrank}_dims_{dims_str}_{grid}", plot_settings)
    return sampler.export_grid_solution()


class _EagerProjections:
    """Projections that master_main may start while it runs another one.

    master_main calls ``finish`` for each projection when it ends, the one it
    was started for (``host_index``) included.
    """

    def __init__(self, host_index, indices, projections, save_plots, plot_settings, myrank):
        self.host_index = host_index
        self.pending = list(indices)
        self.projections = projections
        self.save_plots = save_plots
        self.plot_settings = plot_settings
        self.myrank = myrank
        self.results = {}       # projection index -> result dict

    def start_next(self, logger):
        index = self.pending.pop(0)
        config = self.projections[index]
        logger.info("=" * 80)
        logger.info(f"=== Starting Projection {index + 1}/{len(self.projections)} "
                    f"on idle workers ===")
        logger.info(f"=== Dimensions: {config['dims']} ===")
        logger.info("=" * 80)
        return index, config

    def finish(self, index, sampler, logger):
        """Called by master_main with the projection's state in the sampler."""
        config = self.projections[index]
        solution = _collect_solution(sampler, config, self.save_plots,
                                     self.plot_settings, self.myrank)
        self.results[index] = {
            'coarse_solution': solution,
            'refined_solution': None,
            'metrics': {'coarse_target_calls': sampler.target_calls,
                        'total_target_calls': sampler.target_calls,
                        'global_max': sampler.global_max_target_val},
            'projection_config': config,
        }
        logger.info("=" * 80)
        logger.info(f"=== Completed Projection {index + 1}/{len(self.projections)} ===")
        logger.info(f"  Total Grid Points Explored: {len(sampler.population)}")
        logger.info("=" * 80)


def run_projection(comm, sampler, projection_config,
                   save_plots=False,
                   plot_settings=None,
                   skip_init_opt_on_warm_start=True,
                   myrank=0, eager=None):
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
        myrank=myrank,
        eager=eager,
    )

    # With eager start master_main collected the result when the projection ended.
    if eager is not None and eager.results:
        return eager.results.pop(eager.host_index)

    coarse_solution = _collect_solution(sampler, projection_config, save_plots,
                                        plot_settings, myrank)
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

        results['refined_solution'] = _collect_solution(
            sampler, projection_config, save_plots, plot_settings, myrank, grid='refined')
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
    """Run a list of projections with automatic cross-projection warm-start.

    With ``cross_projection`` ``eager_start`` a projection may start while an
    earlier one is running (see ``master_main``).

    See :func:`run_projection` for the per-projection config and result shape.
    Returns a list of per-projection result dicts, in the order of
    ``projections``, with an extra ``projection_config`` field carrying the
    original input.
    """
    logger = setup_logger(rank=myrank)
    all_results = []
    done_eagerly = {}   # projection index -> result, for those started on idle workers

    for proj_idx, projection_config in enumerate(projections):
        if proj_idx in done_eagerly:
            all_results.append(done_eagerly.pop(proj_idx))
            continue

        logger.info("=" * 80)
        logger.info(f"=== Starting Projection {proj_idx + 1}/{len(projections)} ===")
        logger.info(f"=== Dimensions: {projection_config['dims']} ===")
        logger.info("=" * 80)

        # Reset sampler for new projection (except first)
        if proj_idx > 0:
            sampler._reset_for_new_projection(projection_config)

        # Enable warm start after first projection
        skip_init_opt = (proj_idx > 0)

        # eager_start: the eligible projections that follow may be started
        # while this one runs.
        eager = None
        if (sampler.eager_projection_start and sampler.pool_seeded_initial_maxima
                and _eager_eligible(sampler, projection_config)):
            following = []
            for j in range(proj_idx + 1, len(projections)):
                if not _eager_eligible(sampler, projections[j]):
                    break
                following.append(j)
            if following:
                eager = _EagerProjections(proj_idx, following, projections,
                                          save_plots, plot_settings, myrank)

        # Run projection (handles coarse + refinement automatically)
        results = run_projection(
            comm=comm,
            sampler=sampler,
            projection_config=projection_config,
            save_plots=save_plots,
            plot_settings=plot_settings,
            skip_init_opt_on_warm_start=skip_init_opt,
            myrank=myrank,
            eager=eager,
        )
        if eager is not None:
            done_eagerly.update(eager.results)

        # Add projection config to results for reference
        results['projection_config'] = projection_config
        all_results.append(results)

        # With eager start the line was logged when the projection ended.
        if eager is None:
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


def _volume_eval_batch(comm, sampler, logger, items, myrank):
    """Evaluate a batch of ``(idx, params)`` on the worker pool.

    Each result is logged to the sample file under ``PHASE_VOLUME`` and the
    global max is tracked. Returns ``{idx: (logL, params)}``. Items in a batch
    are independent (the red/black split guarantees it), so the whole batch can
    be in flight at once.
    """
    out = {}
    free = [r for r in range(comm.Get_size()) if r != myrank]
    queue = collections.deque(items)
    pending = []
    received = 0
    while received < len(items):
        while free and queue:
            idx, params = queue.popleft()
            wr = free.pop()
            pending.append(comm.isend(
                {'params': params, 'context': {'type': 'VOLUME', 'idx': idx}},
                dest=wr))
        result = comm.recv(source=MPI.ANY_SOURCE)
        free.append(result['context']['worker_rank'])
        received += 1
        _log_worker_error(result, sampler, logger)
        target_val = result['target_val']
        sampler._register_target_call(
            result['params'], target_val, phase=phase_for_job_type('VOLUME'))
        if target_val > sampler.global_max_target_val:
            logger.warning(
                f"--- Volume sampling found a NEW GLOBAL MAX: "
                f"{target_val:.6e} (previous: "
                f"{sampler.global_max_target_val:.6e}) ---")
            sampler.global_max_target_val = target_val
        out[result['context']['idx']] = (
            target_val, np.asarray(result['params'], dtype=float))
    if pending:
        MPI.Request.Waitall(pending)
    return out


def _read_scan_samples(sampler):
    """Read this run's logged samples as ``(M, n_dims + 1)`` ``[params, logL]``.

    Returns None when there is no readable sample file (warm start then falls
    back to envelope seeds).
    """
    import os

    from .sample_io import read_samples

    path = getattr(sampler, 'samples_output_file', None)
    if not path or not os.path.exists(path):
        return None
    try:
        rows = read_samples(path)
    except (OSError, ValueError):
        return None
    if rows.size == 0 or rows.shape[1] < sampler.dims + 1:
        return None
    return rows[:, :sampler.dims + 1]


def run_volume_sampling(comm, sampler, projection_results, myrank=0):
    """Run the post-projection volume-sampling stage (master side).

    Populates the ROI ``{logL > global_max - roi_threshold}`` with an
    affine-invariant ensemble of umbrella walkers (see
    docs/volume_sampling_plan.md), seeded inside the ProjectionEnvelope built
    from ``projection_results``. Requires ``sampler.volume_sampling_config``
    and live workers. Returns the stage-result dict (also stored on
    ``sampler.volume_stage_result``); a dict with ``skipped=True`` and a
    ``reason`` when it cannot run.
    """
    from .volume import (
        ProjectionEnvelope, assign_levels, draw_envelope_seeds, draw_stretch,
        lnl_histogram, umbrella_logpi, warm_start_positions,
        write_volume_output,
    )

    logger = setup_logger(rank=myrank)
    config = sampler.volume_sampling_config
    if not config:
        raise ConfigurationError(
            "run_volume_sampling requires the sampler to be constructed with "
            "a volume_sampling config dict",
            parameter="volume_sampling", value=None,
        )

    logger.info("=" * 80)
    logger.info(
        f"=== Volume sampling stage (roi_threshold: {config['roi_threshold']}) ===")
    logger.info("=" * 80)

    def _skip(reason):
        logger.info(f"--- Volume sampling skipped: {reason} ---")
        result = {'skipped': True, 'reason': reason}
        sampler.volume_stage_result = result
        return result

    # Make the run's own sample file complete (warm start reads it).
    sampler._flush_samples_buffer()

    global_max_start = sampler.global_max_target_val
    envelope = ProjectionEnvelope.from_projection_results(
        projection_results, global_max_start, sampler.dims)
    if envelope.covers_full_space:
        return _skip(
            "a projection grids the full parameter space, so the grid "
            "already covers the volume (a finer grid adds the same "
            "information)")

    roi = config['roi_threshold']
    sigma = config['sigma_frac'] * roi
    window = config['partner_level_window']
    rng = np.random.default_rng()
    n_dims = sampler.dims
    all_dims = list(range(n_dims))

    # Seed walkers inside the envelope (shrink the ensemble if the prefilter
    # region yields fewer seeds than requested).
    seeds = draw_envelope_seeds(envelope, sampler.bounds, config['n_walkers'],
                                roi, seed=int(rng.integers(2**31)))
    if len(seeds) < 2:
        return _skip("fewer than 2 seeds found inside the projection envelope")
    n_walkers = len(seeds)
    levels = assign_levels(n_walkers, global_max_start, roi)

    # Warm-start from scan samples near each level (reuses the scan's work).
    theta = seeds.copy()
    if config['warm_start']:
        scan = _read_scan_samples(sampler)
        if scan is not None and len(scan):
            theta = warm_start_positions(
                levels, sigma, scan[:, :n_dims], scan[:, n_dims], seeds, rng)

    n_steps = config['n_steps']
    eval_budget = config['eval_budget'] or n_walkers * (n_steps + 1)
    half = [np.flatnonzero(np.arange(n_walkers) % 2 == p) for p in (0, 1)]

    logger.info(
        f"--- Volume sampling: {n_walkers} walkers, sigma={sigma:.3g}, "
        f"budget {eval_budget} evals "
        f"({'full pool' if window is None else f'level window {window:g}'}) ---")

    # Seed evaluation for every walker.
    lnl_cur = np.empty(n_walkers)
    all_evals = []   # (params, logL) for every evaluation
    seed_res = _volume_eval_batch(
        comm, sampler, logger,
        [(i, theta[i].copy()) for i in range(n_walkers)], myrank)
    for i, (lnl, p) in seed_res.items():
        lnl_cur[i] = lnl
        theta[i] = p
        all_evals.append((p, lnl))
    n_evals = n_walkers
    n_acc = n_prop = 0

    n_sweeps = max(eval_budget // n_walkers - 1, 1)
    for _ in range(n_sweeps):
        if n_evals >= eval_budget:
            break
        for p_a in (0, 1):
            active, frozen = half[p_a], half[1 - p_a]
            items, meta = [], {}
            for k in active:
                k = int(k)
                if window is None:
                    pool = frozen
                else:
                    pool = frozen[np.abs(levels[frozen] - levels[k]) <= window]
                    if len(pool) == 0:
                        pool = frozen
                j = int(pool[rng.integers(len(pool))])
                z = float(draw_stretch(rng, 1)[0])
                meta[k] = z
                # Periodic parameters: stretch along the shorter arc, and
                # bring a proposal that leaves the bounds back in.
                step = sampler._param_delta(theta[k], theta[j], all_dims)
                items.append((k, sampler._wrap_periodic(theta[j] + z * step, all_dims)))
            res = _volume_eval_batch(comm, sampler, logger, items, myrank)
            n_evals += len(items)
            n_prop += len(items)
            for k, (lnl, p) in res.items():
                all_evals.append((p, lnl))
                log_acc = ((n_dims - 1) * np.log(meta[k])
                           + umbrella_logpi(lnl, levels[k], sigma)
                           - umbrella_logpi(lnl_cur[k], levels[k], sigma))
                if np.log(rng.uniform()) < log_acc:
                    theta[k] = p
                    lnl_cur[k] = lnl
                    n_acc += 1

    sampler._flush_samples_buffer()

    # Band membership against the (possibly improved) final global max.
    global_max_final = sampler.global_max_target_val
    band_lo_final = global_max_final - roi
    params = np.array([p for p, _ in all_evals])
    logls = np.array([lv for _, lv in all_evals])
    in_band = np.isfinite(logls) & (logls >= band_lo_final)
    samples = (np.column_stack([params[in_band], logls[in_band]])
               if in_band.any() else np.empty((0, n_dims + 1)))

    stats = {
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'n_sweeps': int(n_sweeps),
        'n_evals': int(n_evals),
        'n_in_band': int(in_band.sum()),
        'in_band_fraction': float(in_band.mean()) if len(logls) else 0.0,
        'mean_acceptance': float(n_acc / n_prop) if n_prop else float('nan'),
        'sigma': sigma,
        'global_max_drift': global_max_final - global_max_start,
        'lnl_histogram': lnl_histogram(logls[in_band], band_lo_final,
                                       global_max_final),
    }
    result = {
        'skipped': False, 'reason': None,
        'samples': samples,
        'band_lo_final': band_lo_final,
        'global_max': global_max_final,
        'stats': stats,
    }

    try:
        write_volume_output(result, config)
    except (OSError, ValueError) as e:
        logger.warning(f"Volume sampling: could not write output files: {e}")

    logger.info("=" * 80)
    logger.info("--- Volume sampling complete ---")
    logger.info(f"  Walkers: {n_walkers}, evaluations: {n_evals}")
    logger.info(
        f"  In-band samples: {stats['n_in_band']} "
        f"(fraction {stats['in_band_fraction']:.3g}); "
        f"mean acceptance {stats['mean_acceptance']:.3g}")
    if stats['global_max_drift'] > 0:
        logger.warning(
            f"  Global max improved by {stats['global_max_drift']:.6g} "
            f"during volume sampling; band membership was re-derived, but the "
            f"projection results themselves may be stale -- consider "
            f"re-running the projections.")
    logger.info("=" * 80)

    sampler.volume_stage_result = result
    return result


class _ProjectionRun:
    """Stage machine and queues of one projection inside master_main."""

    def __init__(self, index, stages, state, skip_init_opt_on_warm_start):
        self.index = index                  # position in the list of projections
        self.state = state                  # sampler.capture_projection_state()
        self.skip_init_opt_on_warm_start = skip_init_opt_on_warm_start
        self.stages = stages
        self.next_stage()
        self.active_jobs = {}               # {job_id: Job object}
        self.high_prio_tasks = collections.deque()
        self.low_prio_tasks = collections.deque()
        # Tasks of cells activated ahead (lbfgsb activation_depth > 1), served last.
        self.ahead_tasks = collections.deque()
        # Ids of jobs that offer speculative tasks (LBFGSBJob.speculate), served
        # round-robin: ordinary cells, and cells activated ahead. The dicts are
        # used as ordered sets.
        self.spec_ring = {}
        self.ahead_spec_ring = {}
        self.inflight = 0                   # evaluations in flight
        self.ahead_inflight = 0             # of which for cells activated ahead
        self.idle_after_dispatch = 0        # workers ordinary work of this run left over
        self.optimisations_finished = 0     # cell optimisations or DE generation jobs done
        self.last_progress_log = time.time()

        # DE stage state
        self.de_generation = 0
        self.de_successful_F = []
        self.de_successful_CR = []

        # Wave-based patching state
        self.patching_wave_number = 0
        self.patching_updated_last_wave = None
        self.patching_wave_baseline_fitness = {}   # grid_idx -> fitness at wave start
        self.patching_wave_test_jobs = set()       # IDs of test jobs in current wave
        self.patching_wave_lbfgsb_jobs = set()     # IDs of L-BFGS-B jobs spawned by current wave

        # Wave-based suspect-recheck state (mirrors patching state)
        self.suspect_wave_number = 0
        self.suspect_updated_last_wave = None
        self.suspect_wave_baseline_fitness = {}
        self.suspect_wave_test_jobs = set()
        self.suspect_wave_lbfgsb_jobs = set()

        # Rolling initial-optimization (basin detection) state. The stage
        # keeps `initial_opt_batch_size` global L-BFGS-B starts in flight,
        # classifying each converged optimum and refilling until the Bayesian
        # stopping rule trips or the `initial_opt_cap` (=
        # n_initial_optimizations) is reached.
        self.initial_opt_inflight = set()   # job IDs of in-flight INITIAL_OPTIMIZATION jobs
        self.initial_opt_started = 0        # starts launched so far
        self.initial_opt_completed = 0      # starts finished so far (incl. failures)
        self.initial_opt_batch_size = 0
        self.initial_opt_cap = 0
        self.initial_opt_stopped = False    # rule tripped or cap reached: stop refilling
        self.initial_opt_refining = False   # the starts are the user's initial_points

    def next_stage(self):
        self.current_stage = self.stages.pop(0) if self.stages else None


class _ProjectionLabel(logging.Filter):
    """Prefixes log lines with the projection being worked on (eager start)."""

    def __init__(self):
        super().__init__()
        self.prefix = ''

    def set(self, index):
        self.prefix = f"[P{index + 1}] "

    def filter(self, record):
        record.msg = self.prefix + str(record.msg)
        return True


def master_main(comm, sampler,
                plot_settings=None, skip_init_opt_on_warm_start=True,
                myrank=0, eager=None):
    """State-machine main loop: dispatches jobs and processes results until done.

    ``eager`` (an ``_EagerProjections``) lists projections that may be started
    on workers this one leaves idle. Each projection being worked on has its
    own ``_ProjectionRun``; the sampler state of the one being handled is
    swapped in. The sampler is left holding the projection started last.
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

    def _build_stages():
        """Workflow stages of the projection the sampler is set up for."""
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
            # fd_step 'auto': measure the noise before anything is optimised.
            if sampler.needs_noise_estimate():
                stages.append('NOISE_ESTIMATE')
            # Periodic parameters: compare the target at their two bounds.
            if sampler.needs_periodic_check():
                stages.append('PERIODIC_CHECK')
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

        return stages

    host = _ProjectionRun(index=(eager.host_index if eager is not None else 0),
                          stages=_build_stages(),
                          state=sampler.capture_projection_state(),
                          skip_init_opt_on_warm_start=skip_init_opt_on_warm_start)
    R = host                           # the run being worked on
    loaded = host                      # the run whose state the sampler holds
    runs = [host]                      # unfinished runs, in the order they were started
    run_by_index = {host.index: host}
    label = _ProjectionLabel() if eager is not None else None
    if label is not None:
        label.set(host.index)
        logger.addFilter(label)

    def _switch(run):
        """Make ``run`` the projection the sampler works on."""
        nonlocal loaded
        if run is loaded:
            return
        loaded.state.update(sampler.capture_projection_state())
        sampler.restore_projection_state(run.state)
        loaded = run
        if label is not None:
            label.set(run.index)

    def _is_ahead(job):
        return getattr(job, 'grid_idx', None) in sampler.ahead_cells

    # Helper function to queue tasks
    def _queue_tasks(tasks, job_type, job=None):
        """Add tasks to appropriate priority queue based on job type."""
        if job is not None and _is_ahead(job):
            R.ahead_tasks.extend(tasks)
        elif job_type in ['NOISE_ESTIMATE', 'PERIODIC_CHECK', 'INITIAL_OPTIMIZATION', 'LBFGSB', 'REFINEMENT_LBFGSB', 'POST_ACTIVATION_LBFGSB', 'LBFGSB_LOOP', 'PATCHING_TEST', 'PATCHING_LBFGSB', 'SUSPECT_RECHECK', 'SUSPECT_RECHECK_LBFGSB']:
            R.high_prio_tasks.extend(tasks)
        else:
            # DE_GRID_POINT, ACTIVATION go to low priority
            R.low_prio_tasks.extend(tasks)

    def _purge_queued_tasks(job_ids):
        """Drop not-yet-dispatched tasks belonging to the given jobs from both
        priority queues (used to abort optimizations). Tasks already running on
        workers return normally and are ignored via the unknown-job path."""
        for q in (R.high_prio_tasks, R.low_prio_tasks, R.ahead_tasks):
            kept = [t for t in q if t.get('context', {}).get('job_id') not in job_ids]
            q.clear()
            q.extend(kept)

    def _offer_speculation(job):
        if getattr(job, 'wants_speculation', False):
            ring = R.ahead_spec_ring if _is_ahead(job) else R.spec_ring
            ring.setdefault(job.id)

    def _next_speculative_task(run, ring):
        """A task from the first job in the ring that has one; that job moves
        to the end, jobs with nothing to offer leave the ring."""
        nonlocal speculative_sent
        for jid in list(ring):
            del ring[jid]
            job = run.active_jobs.get(jid)
            task = job.speculate() if job is not None else None
            if task is not None:
                ring[jid] = None
                speculative_sent += 1
                return task
        return None

    def _next_task(ahead):
        """The next task and its run, in the order the runs were started:
        queued tasks, then speculative ones. ``ahead`` selects the work for
        cells activated ahead."""
        for run in runs:
            queues = (run.ahead_tasks,) if ahead else (run.high_prio_tasks, run.low_prio_tasks)
            task = next((q.popleft() for q in queues if q), None)
            ring = run.ahead_spec_ring if ahead else run.spec_ring
            if task is None and ring:
                _switch(run)
                task = _next_speculative_task(run, ring)
            if task is not None:
                return task, run
        return None, None

    def _start_jobs(jobs):
        """Add jobs to the active pool and queue their first tasks."""
        for job in jobs:
            R.active_jobs[job.id] = job
            initial_tasks = job.start()
            _queue_tasks(initial_tasks, job.type, job)
            _offer_speculation(job)
            # A job may finish during start() without dispatching any tasks
            # (e.g. an LBFGSBJob with zero dims to optimize). Without this
            # cleanup it would stay in active_jobs forever and hang the
            # main loop, since the finalization path below only runs when
            # a result arrives.
            if not initial_tasks and job.is_finished():
                job.on_finish(next_job_id)
                del R.active_jobs[job.id]

    next_job_id = 0
    tasks_sent = 0
    tasks_completed = 0
    speculative_sent = 0
    ahead_reserve = int(AHEAD_RESERVE_FRACTION * n_workers)
    eager_min_idle = max(1, int(EAGER_START_IDLE_FRACTION * n_workers))

    def _drained(run):
        """Nothing of this run is active, queued or in flight. Evaluations
        still out for dropped cells activated ahead do not count: their
        results are ignored when they return."""
        return (not run.active_jobs and not run.high_prio_tasks and not run.low_prio_tasks
                and run.inflight == run.ahead_inflight)

    def _advance():
        """Create the jobs of the current stage of run ``R``. Returns True if
        the stage had nothing to do and the next one should be tried at once."""
        nonlocal next_job_id
        logger.info(f"--- Master: Entering stage: {R.current_stage} (all jobs complete, creating new jobs) ---")
        new_jobs = []

        if R.current_stage == 'NOISE_ESTIMATE':
            new_jobs, next_job_id = sampler.create_noise_estimate_job(next_job_id)

        elif R.current_stage == 'PERIODIC_CHECK':
            new_jobs, next_job_id = sampler.create_periodic_check_job(next_job_id)

        elif R.current_stage == 'INITIAL_POINTS_EVAL':
            # Evaluate user-supplied initial_points via the standard Job
            # path so the master result-handler does the bookkeeping
            # (samples_output_file, global solution pool, etc.).
            new_jobs, next_job_id = sampler.create_initial_points_eval_job(next_job_id)
            if not new_jobs:
                R.next_stage()
                return True

        elif R.current_stage == 'INITIAL_OPTIMIZATION':
            # Try to initialize from a warm-start file (separate from the
            # samples output file; see ProfileProjector.warm_start_file).
            if R.skip_init_opt_on_warm_start and sampler.warm_start_file:
                sampler._initialize_from_warm_start_file(sampler.warm_start_file)

            # In-memory cross-projection seeding: skip the global L-BFGS-B
            # starts on projections 2..N when the accumulated pool already
            # covers the relevant peaks. Falls through to the original
            # initial-optimization path when the pool is empty (first
            # projection) or the feature is disabled.
            if (R.skip_init_opt_on_warm_start
                    and not sampler.initial_maxima
                    and sampler.pool_seeded_initial_maxima
                    and len(sampler.global_solution_pool) > 0):
                sampler._initialize_from_global_pool()

            # refine_initial_points: one global L-BFGS-B run per user-supplied
            # point, without the basin stopping rule.
            n_refine = sampler.start_initial_point_refinement()
            R.initial_opt_refining = n_refine > 0

            # If maxima already came from warm start / pool seeding, skip the
            # global optimization stage entirely.
            if sampler.initial_maxima and not R.initial_opt_refining:
                logger.info("Skipping initial optimization - using provided initial points or warm start.")
                R.next_stage()
                return True

            # Rolling multistart with online basin detection. Launch a
            # first batch and hand control to WAITING_FOR_INITIAL_OPT,
            # which refills and applies the stopping rule as jobs return.
            R.initial_opt_started = 0
            R.initial_opt_completed = 0
            R.initial_opt_stopped = False
            R.initial_opt_inflight = set()
            if R.initial_opt_refining:
                R.initial_opt_cap = n_refine
                R.initial_opt_batch_size = min(n_refine, n_workers)
                logger.info(
                    f"--- Refining {n_refine} user-provided initial point(s) "
                    f"with L-BFGS-B over all parameters ---"
                )
            else:
                R.initial_opt_cap = sampler.n_initial_optimizations
                R.initial_opt_batch_size = sampler.resolve_initial_opt_batch_size(n_workers)
                sampler.init_initial_opt_lhs(R.initial_opt_cap)
                logger.info(
                    f"--- Basin detection: rolling multistart "
                    f"(batch_size={R.initial_opt_batch_size}, cap={R.initial_opt_cap}, "
                    f"min_starts={sampler.basin_min_starts}) ---"
                )
            new_jobs = []
            for _ in range(min(R.initial_opt_batch_size, R.initial_opt_cap)):
                job, next_job_id = sampler.create_one_initial_optimization_job(next_job_id)
                new_jobs.append(job)
                R.initial_opt_inflight.add(job.id)
                R.initial_opt_started += 1

            if not new_jobs:
                # Degenerate cap (e.g. n_initial_optimizations == 0):
                # nothing to optimize, move on.
                R.next_stage()
                return True
            R.current_stage = 'WAITING_FOR_INITIAL_OPT'

        elif R.current_stage == 'ACTIVATION':
            # Use refinement activation for refinement runs
            if sampler.is_refinement_run:
                new_jobs, next_job_id = sampler.create_refinement_activation_jobs(next_job_id)
                if not new_jobs:
                    logger.info("No refinement activation jobs created. Moving to next stage.")
                    R.next_stage()
                    return True
            else:
                new_jobs, next_job_id = sampler.create_activation_jobs(next_job_id)
                if not new_jobs:
                    logger.info("No activation jobs created (no initial maxima?). Moving to next stage.")
                    R.next_stage()
                    return True

        elif R.current_stage == 'DE_LOOP':
            # This stage loops
            if R.de_generation >= sampler.de_num_generations:
                logger.info("--- Master: DE generations complete. ---")
                R.next_stage()
                return True

            if R.de_generation > 0:
                sampler.update_de_memory(R.de_successful_F, R.de_successful_CR)

            R.de_generation += 1
            sampler.current_generation = R.de_generation # Update sampler state
            logger.info(f"--- Master: Starting DE Generation {R.de_generation} ---")

            new_de_jobs, next_job_id, R.de_successful_F, R.de_successful_CR = sampler.create_de_generation_jobs(
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

            if not new_jobs and not R.de_successful_F: # DE converged and no new activations
                logger.info("--- Master: DE converged and no new activations. ---")
                R.next_stage()
                return True

        elif R.current_stage == 'LBFGSB_LOOP':
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
                R.next_stage()
                return True

        elif R.current_stage == 'PATCHING_WAVES':
            # Check appropriate flag based on run type
            patching_enabled = sampler.patch_refined_grid if sampler.is_refinement_run else sampler.patch_coarse_grid
            if not patching_enabled:
                logger.info("--- Master: Patching disabled for this stage. Skipping. ---")
                R.next_stage()
                return True

            # Check if max waves reached
            if R.patching_wave_number >= sampler.max_patching_waves:
                logger.info(f"--- Master: Max patching waves ({sampler.max_patching_waves}) reached. ---")
                R.next_stage()
                return True

            logger.info(f"--- Master: Starting Patching Wave {R.patching_wave_number} ---")

            # Create jobs for this wave
            new_jobs, next_job_id = sampler.create_patching_wave_jobs(
                wave_number=R.patching_wave_number,
                updated_points_last_wave=R.patching_updated_last_wave,
                next_job_id=next_job_id
            )

            if not new_jobs:
                logger.info("--- Master: No patching candidates found. Ending patching. ---")
                R.next_stage()
                return True

            # Record baseline fitness for all grid points at wave start
            R.patching_wave_baseline_fitness = {
                idx: state['best_fitness']
                for idx, state in sampler.population.items()
            }

            # Track test job IDs
            R.patching_wave_test_jobs = {j.id for j in new_jobs}
            R.patching_wave_lbfgsb_jobs = set()

            # Transition to waiting state
            R.current_stage = 'WAITING_FOR_PATCHING_WAVE'

        elif R.current_stage == 'WAITING_FOR_PATCHING_WAVE':
            # This stage just waits for wave jobs to complete
            # Wave is complete when checked in job completion handling
            pass

        elif R.current_stage == 'SUSPECT_RECHECK_WAVES':
            if not sampler.suspect_recheck_enabled or sampler.direct_eval_mode:
                logger.info("--- Master: Suspect recheck disabled. Skipping. ---")
                R.next_stage()
                return True

            if R.suspect_wave_number >= sampler.max_suspect_waves:
                logger.info(f"--- Master: Max suspect waves ({sampler.max_suspect_waves}) reached. ---")
                R.next_stage()
                return True

            logger.info(f"--- Master: Starting Suspect Recheck Wave {R.suspect_wave_number} ---")

            new_jobs, next_job_id = sampler.create_suspect_recheck_jobs(
                wave_number=R.suspect_wave_number,
                updated_points_last_wave=R.suspect_updated_last_wave,
                next_job_id=next_job_id,
            )

            if not new_jobs:
                logger.info("--- Master: No suspect candidates. Ending suspect recheck. ---")
                R.next_stage()
                return True

            R.suspect_wave_baseline_fitness = {
                idx: state['best_fitness']
                for idx, state in sampler.population.items()
            }
            R.suspect_wave_test_jobs = {j.id for j in new_jobs}
            R.suspect_wave_lbfgsb_jobs = set()
            R.current_stage = 'WAITING_FOR_SUSPECT_WAVE'

        elif R.current_stage == 'WAITING_FOR_SUSPECT_WAVE':
            pass

        elif R.current_stage == 'REFINEMENT_LBFGSB':
            # For refinement runs, directly create LBFGSB jobs from interpolated starts
            new_jobs, next_job_id = sampler.create_refinement_lbfgsb_jobs(next_job_id)

            if not new_jobs:
                logger.info("--- Master: No refinement LBFGSB jobs created. Ending refinement. ---")
                R.next_stage()
                return True

        elif R.current_stage == 'POST_ACTIVATION_LBFGSB':
            # Direct L-BFGS-B optimization of all activated points
            # (alternative to DE when optimization_method='lbfgsb')
            new_jobs, next_job_id = sampler.create_post_activation_lbfgsb_jobs(next_job_id)

            if not new_jobs:
                logger.info("--- Master: No post-activation L-BFGS-B jobs created. ---")
                R.next_stage()
                return True


        # --- Add new jobs to active pool and queue initial tasks ---
        _start_jobs(new_jobs)

        # If we just finished a non-looping stage, advance to the next
        # Looping stages: DE_LOOP, LBFGSB_LOOP (they re-run after jobs complete)
        if R.current_stage not in ['DE_LOOP', 'LBFGSB_LOOP',
                                 'WAITING_FOR_PATCHING_WAVE',
                                 'WAITING_FOR_SUSPECT_WAVE',
                                 'WAITING_FOR_INITIAL_OPT']:
             R.next_stage()
        return False

    # --- Main Event Loop ---
    while runs or tasks_sent > tasks_completed:

        # --- 1. Generate new jobs if a stage is starting or continuing ---
        for R in list(runs):
            while _drained(R):
                _switch(R)
                if not R.current_stage:
                    # All stages of this projection are complete.
                    runs.remove(R)
                    if eager is not None:
                        eager.finish(R.index, sampler, logger)
                    break
                if not _advance():
                    break

        # --- 2. Check for and process ALL available results ---
        while comm.Iprobe(source=MPI.ANY_SOURCE):
            # A message is waiting, so now we do a blocking (but instant) receive
            result = comm.recv(source=MPI.ANY_SOURCE)
            worker_rank = result['context']['worker_rank']
            free_workers.append(worker_rank)
            tasks_completed += 1
            R = run_by_index[result['context']['proj']]
            R.inflight -= 1
            if result['context'].get('ahead_task'):
                R.ahead_inflight -= 1

            _log_worker_error(result, sampler, logger)
            _log_user_gradient_error(result, sampler, logger)
            sampler._register_target_call(
                result['params'], result['target_val'],
                phase=phase_for_job_type(result['context'].get('type')))

            job_id = result['context'].get('job_id', -1)
            if job_id not in R.active_jobs:
                # Expected for evaluations sent ahead of time for a job that
                # has since finished, and for aborted multistart runs.
                logger.debug(f"Received result for unknown/finished job {job_id}. Ignoring.")
                continue

            _switch(R)
            job = R.active_jobs[job_id]
            new_tasks = job.process_result(result)

            # Add to correct priority queue
            _queue_tasks(new_tasks, job.type, job)
            _offer_speculation(job)

            if job.is_finished():
                job_id_finished = job.id

                # Track wave-based patching jobs
                is_patch_test_job = job_id_finished in R.patching_wave_test_jobs
                is_patch_lbfgsb_job = job_id_finished in R.patching_wave_lbfgsb_jobs
                is_suspect_test_job = job_id_finished in R.suspect_wave_test_jobs
                is_suspect_lbfgsb_job = job_id_finished in R.suspect_wave_lbfgsb_jobs

                if is_patch_test_job:
                    R.patching_wave_test_jobs.remove(job_id_finished)
                if is_patch_lbfgsb_job:
                    R.patching_wave_lbfgsb_jobs.remove(job_id_finished)
                if is_suspect_test_job:
                    R.suspect_wave_test_jobs.remove(job_id_finished)
                if is_suspect_lbfgsb_job:
                    R.suspect_wave_lbfgsb_jobs.remove(job_id_finished)

                # This updates the sampler state and can spawn a new job
                spawn_result = job.on_finish(next_job_id)
                del R.active_jobs[job_id_finished]

                if spawn_result:
                    new_job, next_job_id = spawn_result
                    R.active_jobs[new_job.id] = new_job
                    initial_tasks = new_job.start()

                    # Track child jobs spawned within a wave so the wave's
                    # completion check knows to wait for them.
                    if is_patch_test_job and new_job.type == 'PATCHING_LBFGSB':
                        R.patching_wave_lbfgsb_jobs.add(new_job.id)
                    if is_suspect_test_job and new_job.type == 'SUSPECT_RECHECK_LBFGSB':
                        R.suspect_wave_lbfgsb_jobs.add(new_job.id)

                    # Add to correct priority queue
                    _queue_tasks(initial_tasks, new_job.type)
                    _offer_speculation(new_job)

                if job.type in ('LBFGSB_LOOP', 'DE_GRID_POINT'):
                    R.optimisations_finished += 1

                # Rolling activation: a cell is optimised as soon as its
                # activation returns, and a finished cell inside the region
                # of interest activates its neighbours at once. _advance
                # still ends the stage when everything is idle.
                if R.current_stage == 'LBFGSB_LOOP':
                    rolling = []
                    idle = R.idle_after_dispatch
                    if job.type == 'ACTIVATE_GRID_POINT':
                        new_job, next_job_id = sampler.create_lbfgsb_job_for_active_cell(
                            job.grid_idx, next_job_id, 'LBFGSB_LOOP', 'converged')
                        if new_job is not None:
                            rolling.append(new_job)
                    # Eager activation: with workers idle, a cell whose
                    # activation samples put it inside the region of interest
                    # activates its neighbours now. Its optimised value can only
                    # be higher, so no extra cell is activated. With more than
                    # the reserve idle it activates activation_depth cells ahead.
                    if job.type == 'LBFGSB_LOOP' or (job.type == 'ACTIVATE_GRID_POINT' and idle > 0):
                        more, next_job_id = sampler.create_dynamic_activation_jobs(
                            next_job_id, around=[job.grid_idx],
                            depth=sampler.lbfgsb_activation_depth if idle > ahead_reserve else 1)
                        rolling += more
                    _start_jobs(rolling)

                    if sampler.ahead_cells and job.type in ('ACTIVATE_GRID_POINT', 'LBFGSB_LOOP'):
                        # A cell found inside the region of interest makes
                        # itself and its neighbours ordinary cells: their
                        # queued tasks move up.
                        confirmed = sampler.confirm_ahead_cells(job.grid_idx)
                        if confirmed:
                            moved = {j.id: j for j in R.active_jobs.values()
                                     if j.grid_idx in confirmed}
                            kept = []
                            for t in R.ahead_tasks:
                                owner = moved.get(t['context']['job_id'])
                                if owner is None:
                                    kept.append(t)
                                else:
                                    _queue_tasks([t], owner.type)
                            R.ahead_tasks.clear()
                            R.ahead_tasks.extend(kept)
                            for j in moved.values():
                                _offer_speculation(j)
                        # Only cells activated ahead are still being worked
                        # on: every cell the result needs is done. Drop them.
                        if (sampler.ahead_cells and not R.high_prio_tasks and not R.low_prio_tasks
                                and all(_is_ahead(j) for j in R.active_jobs.values())):
                            R.active_jobs.clear()
                            R.ahead_tasks.clear()
                            R.ahead_spec_ring.clear()
                            n_dropped = sampler.drop_ahead_cells()
                            logger.info(f"--- Dropped {n_dropped} cells activated ahead "
                                        f"that stayed outside the border ---")

                # Check if the patching wave is now complete
                if R.current_stage == 'WAITING_FOR_PATCHING_WAVE' and \
                   not R.patching_wave_test_jobs and not R.patching_wave_lbfgsb_jobs:

                    # Determine which grid points were updated (fitness improved)
                    updated_points = []
                    for idx, state in sampler.population.items():
                        baseline_fitness = R.patching_wave_baseline_fitness.get(idx, -np.inf)
                        if state['best_fitness'] > baseline_fitness:
                            updated_points.append(idx)

                    logger.info(f"--- Master: Patching Wave {R.patching_wave_number} complete. Updated {len(updated_points)} points ---")

                    if updated_points:
                        # Start next wave with updated points as seeds
                        R.patching_updated_last_wave = updated_points
                        R.patching_wave_number += 1
                        R.current_stage = 'PATCHING_WAVES'  # Loop back to start next wave
                    else:
                        # No updates, patching converged
                        logger.info("--- Master: No updates in wave. Patching converged. ---")
                        R.next_stage()

                # Check if the suspect-recheck wave is now complete
                if R.current_stage == 'WAITING_FOR_SUSPECT_WAVE' and \
                   not R.suspect_wave_test_jobs and not R.suspect_wave_lbfgsb_jobs:

                    updated_points = []
                    for idx, state in sampler.population.items():
                        baseline_fitness = R.suspect_wave_baseline_fitness.get(idx, -np.inf)
                        if state['best_fitness'] > baseline_fitness:
                            updated_points.append(idx)

                    logger.info(
                        f"--- Master: Suspect Wave {R.suspect_wave_number} complete. "
                        f"Updated {len(updated_points)} points ---"
                    )

                    if updated_points:
                        R.suspect_updated_last_wave = updated_points
                        R.suspect_wave_number += 1
                        R.current_stage = 'SUSPECT_RECHECK_WAVES'
                    else:
                        logger.info("--- Master: No updates in wave. Suspect recheck converged. ---")
                        R.next_stage()

                # Rolling initial-optimization with basin detection. The job's
                # on_finish has already clustered its endpoint into the registry;
                # here we apply the stopping rule and refill the in-flight batch.
                if R.current_stage == 'WAITING_FOR_INITIAL_OPT' and \
                   job_id_finished in R.initial_opt_inflight:

                    R.initial_opt_inflight.discard(job_id_finished)
                    R.initial_opt_completed += 1

                    if not R.initial_opt_stopped:
                        if R.initial_opt_started >= R.initial_opt_cap:
                            R.initial_opt_stopped = True
                            if not R.initial_opt_refining:
                                logger.info(
                                    f"--- Basin detection: reached cap of "
                                    f"{R.initial_opt_cap} starts; stopping ---"
                                )
                        elif (not R.initial_opt_refining
                              and sampler.basin_detection_should_stop(R.initial_opt_completed)):
                            W, n_roi = sampler.basin_detection_roi_stats()
                            n_distinct = len(sampler.initial_optima_registry)
                            R.initial_opt_stopped = True
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
                            aborted = set(R.initial_opt_inflight)
                            for jid in aborted:
                                R.active_jobs.pop(jid, None)
                            R.initial_opt_inflight.clear()
                            _purge_queued_tasks(aborted)
                            logger.info(
                                f"--- Basin detection: {reason} after "
                                f"{R.initial_opt_completed} optimizations "
                                f"({n_distinct} distinct optima, {W} in ROI); "
                                f"aborted {len(aborted)} in-flight run(s) ---"
                            )

                    # Refill to keep `initial_opt_batch_size` starts in flight.
                    if not R.initial_opt_stopped:
                        while (len(R.initial_opt_inflight) < R.initial_opt_batch_size
                               and R.initial_opt_started < R.initial_opt_cap):
                            new_job, next_job_id = sampler.create_one_initial_optimization_job(next_job_id)
                            R.active_jobs[new_job.id] = new_job
                            R.initial_opt_inflight.add(new_job.id)
                            R.initial_opt_started += 1
                            _queue_tasks(new_job.start(), new_job.type)

                    # Stage done once the in-flight batch has fully drained.
                    if not R.initial_opt_inflight:
                        if R.initial_opt_refining:
                            logger.info(
                                f"--- Initial points refined: best target "
                                f"{sampler._refine_best_before:.6f} -> "
                                f"{sampler.global_max_target_val:.6f} ---"
                            )
                        else:
                            W, n_roi = sampler.basin_detection_roi_stats()
                            logger.info(
                                f"--- Basin detection complete: {R.initial_opt_completed} "
                                f"optimizations, {W} distinct ROI optima "
                                f"({len(sampler.initial_optima_registry)} optima total) ---"
                            )
                        R.next_stage()

        # --- End of Iprobe receive loop ---


        # --- 3. Dispatch tasks to free workers ---
        # Runs are served in the order they were started: queued tasks, then
        # evaluations a job may need next. What is left goes to cells
        # activated ahead, which leave a reserve of workers free.
        while free_workers:
            task, owner = _next_task(ahead=False)
            if task is None and len(free_workers) > ahead_reserve:
                task, owner = _next_task(ahead=True)
                if task is not None:
                    task['context']['ahead_task'] = True
                    owner.ahead_inflight += 1
            if task is None:
                break
            task['context']['proj'] = owner.index
            owner.inflight += 1
            worker_rank = free_workers.pop(0)

            # Use non-blocking send to overlap communication
            req = comm.isend(task, dest=worker_rank)
            pending_sends.append(req)
            tasks_sent += 1

        # Workers left without anything to do after this pass. For each run,
        # the workers that it and the runs before it left over: idle, busy
        # with a later run, or busy with a cell activated ahead.
        free_after_dispatch = len(free_workers)
        spare = free_after_dispatch + sum(run.ahead_inflight for run in runs)
        for run in reversed(runs):
            run.idle_after_dispatch = spare
            spare += run.inflight - run.ahead_inflight

        # Eager start: with workers idle and the newest run under way, the
        # next projection starts. It takes its initial maxima from the pool.
        if (eager is not None and runs and eager.pending
                and free_after_dispatch >= eager_min_idle
                and runs[-1].optimisations_finished > 0):
            loaded.state.update(sampler.capture_projection_state())
            label.set(eager.pending[0])
            new_index, new_config = eager.start_next(logger)
            sampler._reset_for_new_projection(new_config)
            R = _ProjectionRun(index=new_index, stages=_build_stages(),
                               state=sampler.capture_projection_state(),
                               skip_init_opt_on_warm_start=True)
            loaded = R
            runs.append(R)
            run_by_index[R.index] = R

        now = time.time()
        for run in runs:
            if run.current_stage == 'LBFGSB_LOOP' and now - run.last_progress_log > PROGRESS_LOG_INTERVAL:
                run.last_progress_log = now
                _switch(run)
                done = sum(1 for st in sampler.population.values()
                           if st['status'] in ('converged', 'optimized'))
                logger.info(f"LBFGSB Loop | Calls: {sampler.target_calls/1e3:6.1f}k | "
                            f"Grid Pts (done/tot): {done:4d}/{len(sampler.population):4d} | "
                            f"Global Max logL: {sampler.global_max_target_val:.4e} | "
                            f"Busy workers: {n_workers - len(free_workers):4d}/{n_workers}")

        # Completed sends are dropped as the scan goes, so the list does not
        # grow with the number of evaluations.
        if len(pending_sends) >= PENDING_SENDS_PRUNE:
            pending_sends = [r for r in pending_sends if not r.Test()]

        # --- 4. No Sleep Needed ---
        # Iprobe() is non-blocking and very lightweight, so no sleep is needed.
        # The loop will naturally wait for results or new tasks to become available.


    # --- End of Main Event Loop ---

    # Wait for all pending non-blocking sends to complete
    if pending_sends:
        logger.debug(f"Waiting for {len(pending_sends)} pending sends to complete...")
        MPI.Request.Waitall(pending_sends)

    logger.debug("master_main: Workflow finished.")

    # The summary below is that of the projection this was started for.
    _switch(host)
    if label is not None:
        logger.removeFilter(label)

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
    if speculative_sent:
        logger.info(f"  Evaluations sent ahead of time on idle workers: {speculative_sent}")
    logger.info(f"  Final Global Max logL: {sampler.global_max_target_val:.6e}")
    logger.info(f"  Total Grid Points Explored: {len(sampler.population)}")
    if sampler.de_allow_early_DE_exit and sampler.de_cells_skipped:
        logger.info(
            f"  Cells that skipped the DE global search (allow_early_DE_exit): "
            f"{sampler.de_cells_skipped}"
        )
    logger.info("=" * 80)
    _switch(run_by_index[max(run_by_index)])
