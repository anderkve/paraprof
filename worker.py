"""
MPI worker process logic.
"""
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py is not installed. This script requires MPI.")
    print("Please install it with: pip install mpi4py")
    import sys
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
    # First, receive the target function from the master.
    target_func = comm.bcast(None, root=0)
    print(f"Worker {myrank}: Received target function. Ready for tasks.")

    while True:
        # Wait for a task from the master
        task = comm.recv(source=0, tag=MPI.ANY_TAG)

        if task == TASK_TERMINATE:
            print(f"Worker {myrank}: Received terminate signal. Exiting.")
            break

        # Execute the task (a single target evaluation)
        params = task['params']
        try:
            target_val = target_func(params)
        except Exception as e:
            print(f"Worker {myrank}: Error evaluating target function at params {params}: {e}")
            target_val = -np.inf  # Return -inf for failed evaluations

        # Send the result back to the master
        context = task['context']
        context['worker_rank'] = myrank
        result = {'target_val': target_val, 'params': params, 'context': context}
        comm.send(result, dest=0)
