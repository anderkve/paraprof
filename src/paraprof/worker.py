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

TASK_TERMINATE = -1

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

        # Execute the task (a single target evaluation)
        params = task['params']
        try:
            target_val = target_func(params)
        except Exception as e:
            logger.error(f"Error evaluating target function at params {params}: {e}")
            target_val = -np.inf  # Return -inf for failed evaluations

        # Send the result back to the master
        context = task['context']
        context['worker_rank'] = myrank
        result = {'target_val': target_val, 'params': params, 'context': context}
        comm.send(result, dest=0)
