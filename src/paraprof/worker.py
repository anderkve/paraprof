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


def _normalize_user_gradient(raw, n_dims):
    """Normalize a user grad_func return value to a length-``n_dims`` numpy
    array, with ``np.nan`` for missing or non-finite components.

    Accepts either a length-``n_dims`` array-like (with NaN/inf entries
    treated as missing) or a ``{dim_index: value}`` dict for partial
    gradients.

    Returns
    -------
    (array, error_msg) tuple. ``array`` is None on a hard shape/type error;
    ``error_msg`` is a string when the worker should report a problem and
    fall back to FD for all dims.
    """
    if raw is None:
        return None, "grad_func returned None"

    if isinstance(raw, dict):
        out = np.full(n_dims, np.nan, dtype=float)
        for k, v in raw.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                return None, f"grad_func dict has non-integer key {k!r}"
            if not (0 <= i < n_dims):
                return None, f"grad_func dict has out-of-range key {k!r}"
            try:
                val = float(v)
            except (TypeError, ValueError):
                return None, f"grad_func dict has non-numeric value at key {k!r}"
            if np.isfinite(val):
                out[i] = val
        return out, None

    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError) as e:
        return None, f"grad_func return is not array-like: {e}"

    if arr.size != n_dims:
        return None, (
            f"grad_func returned array of size {arr.size}, "
            f"expected {n_dims}"
        )

    # Non-finite entries (NaN/inf) become NaN sentinels for "missing".
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, None


def worker_main(comm, myrank, target_func=None, grad_func=None):
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
    grad_func : callable, optional
        Pre-supplied user gradient function. If ``None`` and the master's
        broadcast payload is a ``(target_func, grad_func)`` tuple, the worker
        picks ``grad_func`` up from the broadcast. Workers only call
        ``grad_func(params)`` when the master sets ``context['compute_gradient']``
        on a task. ``grad_func(params)`` must return either a length-``n_dims``
        array (NaN allowed for unknown components) or a
        ``{dim_index: value}`` dict of known components. Sign convention:
        gradient of the function being **maximized** (i.e. ``∇target_func``).
    """
    logger = setup_logger(rank=myrank)

    if target_func is None:
        # Receive the target function (and optionally grad_func) from the master.
        # Two wire formats are accepted for backwards compatibility:
        #   - bare callable: legacy, no user gradient
        #   - (target_func, grad_func) tuple: new, grad_func may be None
        payload = comm.bcast(None, root=0)
        if (isinstance(payload, tuple) and len(payload) == 2
                and (payload[0] is None or callable(payload[0]))):
            target_func, grad_bcast = payload
            if grad_func is None:
                grad_func = grad_bcast
        else:
            target_func = payload
        logger.info(
            "Received target function%s. Ready for tasks."
            % (" and user gradient" if grad_func is not None else "")
        )
    else:
        logger.info("Using pre-supplied target function. Ready for tasks.")

    while True:
        task = comm.recv(source=0, tag=MPI.ANY_TAG)

        if task == TASK_TERMINATE:
            logger.info("Received terminate signal. Exiting.")
            break

        params = task['params']
        context = task['context']

        error_message = None
        try:
            target_val = target_func(params)
        except Exception as e:
            logger.error(f"Error evaluating target function at params {params}: {e}")
            target_val = -np.inf
            error_message = f"{type(e).__name__}: {e}"

        if target_val is None or not np.isfinite(target_val):
            # Non-finite results (NaN, +inf) are coerced to -inf so the master
            # never feeds NaN into comparisons. Report them as errors so the
            # master can warn the user.
            if not (target_val == -np.inf):
                error_message = error_message or (
                    f"Non-finite target value {target_val!r}"
                )
                target_val = -np.inf

        # Optionally evaluate the user gradient at the same point. Only run
        # when the master explicitly asks (context['compute_gradient']) and
        # the target evaluation itself did not fail — a failed target_val
        # tells us nothing useful about the gradient.
        user_gradient = None
        user_gradient_error = None
        compute_gradient = bool(context.get('compute_gradient', False))
        if compute_gradient and grad_func is not None and error_message is None:
            try:
                raw = grad_func(params)
                user_gradient, user_gradient_error = _normalize_user_gradient(
                    raw, len(params)
                )
            except Exception as e:
                user_gradient = None
                user_gradient_error = f"{type(e).__name__}: {e}"
            if user_gradient_error:
                logger.warning(
                    f"grad_func at params {params}: {user_gradient_error}"
                )

        context['worker_rank'] = myrank
        result = {
            'target_val': target_val,
            'params': params,
            'context': context,
            'error': error_message,
            'user_gradient': user_gradient,
            'user_gradient_error': user_gradient_error,
        }
        comm.send(result, dest=0)
