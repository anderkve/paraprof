"""
MPI worker process logic.
"""
import sys
import numpy as np
from .logger import setup_logger

try:
    from mpi4py import MPI
except ImportError:
    # Can't use logger here since MPI isn't available yet
    print("Error: mpi4py is not installed. This script requires MPI.", file=sys.stderr)
    print("Please install it with: pip install mpi4py", file=sys.stderr)
    sys.exit(1)

# Try to import emulator utilities for worker-side pre-screening
try:
    from .emulator_utils import build_emulator_from_cache
    EMULATOR_AVAILABLE = True
except ImportError:
    EMULATOR_AVAILABLE = False

TASK_TERMINATE = -1


def screen_trial_with_emulator(params, target_fitness, emulator_cache, logger):
    """
    Worker-side emulator pre-screening of trial points.

    Uses a GP emulator built from cached evaluations to predict if a trial
    point is worth evaluating. This runs on the worker to avoid bottlenecking
    the master process.

    Decision criterion: Upper Confidence Bound (UCB)
    - Evaluate if UCB = μ + β*σ > current_fitness
    - β controlled by emulator_confidence_threshold (in cache)

    Parameters
    ----------
    params : np.ndarray
        Trial point to evaluate
    target_fitness : float
        Current fitness of target individual (for comparison)
    emulator_cache : dict or None
        Pre-gathered evaluation cache from master, or None if disabled
    logger : Logger
        Logger instance

    Returns
    -------
    should_evaluate : bool
        True if trial should be evaluated, False to skip
    """
    # No emulator data provided - must evaluate
    if emulator_cache is None:
        return True

    # Emulator utilities not available - must evaluate
    if not EMULATOR_AVAILABLE:
        return True

    # Build GP emulator from cached data
    emulator = build_emulator_from_cache(emulator_cache)

    if emulator is None:
        # GP fit failed - must evaluate
        return True

    # Predict trial fitness with uncertainty
    pred_fitness, pred_std = emulator.predict(
        params.reshape(1, -1),
        return_std=True
    )
    pred_fitness = float(pred_fitness[0])
    pred_std = float(pred_std[0])

    # Upper Confidence Bound acquisition function
    beta = emulator_cache.get('confidence_threshold', 2.0)
    ucb = pred_fitness + beta * pred_std

    # Decision: evaluate if UCB suggests potential improvement
    should_evaluate = ucb > target_fitness

    if not should_evaluate:
        logger.debug(
            f"Worker pre-screen: Skipping trial, "
            f"predicted {pred_fitness:.3e} (±{pred_std:.2e}) "
            f"vs target {target_fitness:.3e}, UCB={ucb:.3e}"
        )

    return should_evaluate

def worker_main(comm, myrank):
    """
    Main loop for a worker process.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    myrank : int
        Worker rank
    """
    logger = setup_logger(rank=myrank)

    # First, receive the target function from the master.
    target_func = comm.bcast(None, root=0)
    logger.info("Received target function. Ready for tasks.")

    while True:
        # Wait for a task from the master
        task = comm.recv(source=0, tag=MPI.ANY_TAG)

        if task == TASK_TERMINATE:
            logger.info("Received terminate signal. Exiting.")
            break

        # Extract task components
        params = task['params']
        context = task['context']
        emulator_cache = task.get('emulator_cache', None)

        # === WORKER-SIDE EMULATOR PRE-SCREENING ===
        emulator_screened = False
        if emulator_cache is not None:
            target_fitness = context.get('target_fitness', -np.inf)
            should_evaluate = screen_trial_with_emulator(
                params, target_fitness, emulator_cache, logger
            )
            if not should_evaluate:
                # Emulator predicts no improvement - skip expensive evaluation
                emulator_screened = True
                target_val = -np.inf  # Placeholder value (won't be used)
                logger.debug(f"Worker {myrank}: Trial screened out by emulator")
        # === END PRE-SCREENING ===

        # Execute the task (only if not screened out)
        if not emulator_screened:
            try:
                target_val = target_func(params)
            except Exception as e:
                logger.error(f"Error evaluating target function at params {params}: {e}")
                target_val = -np.inf  # Return -inf for failed evaluations

        # Send the result back to the master
        context['worker_rank'] = myrank
        result = {
            'target_val': target_val,
            'params': params,
            'context': context,
            'emulator_screened': emulator_screened
        }
        comm.send(result, dest=0)
