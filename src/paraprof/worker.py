"""
MPI worker process logic.
"""
import sys
import numpy as np
from .logger import setup_logger

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.", file=sys.stderr)
    print("Please install it with: pip install mpi4py", file=sys.stderr)
    sys.exit(1)

TASK_TERMINATE = -1


def worker_main(comm, myrank, target_func=None):
    """
    Main loop for a worker process.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator
    myrank : int
        Worker rank
    target_func : callable, optional
        Pre-supplied target function to evaluate. If ``None`` (the default),
        the worker waits for the master to broadcast the target function via
        ``comm.bcast(..., root=0)``. Pass a callable here when the target
        function is already available on every rank (e.g. when integrating
        into a host framework whose evaluation entry point cannot be pickled).
    """
    logger = setup_logger(rank=myrank)

    if target_func is None:
        # Receive the target function from the master.
        target_func = comm.bcast(None, root=0)
        logger.info("Received target function. Ready for tasks.")
    else:
        logger.info("Using pre-supplied target function. Ready for tasks.")

    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG)

        if task == TASK_TERMINATE:
            logger.info("Received terminate signal. Exiting.")
            break

        params = task['params']
        context = task['context']

        try:
            target_val = target_func(params)
        except Exception as e:
            logger.error(f"Error evaluating target function at params {params}: {e}")
            target_val = -np.inf

        context['worker_rank'] = myrank
        result = {
            'target_val': target_val,
            'params': params,
            'context': context,
        }
        comm.send(result, dest=0)
