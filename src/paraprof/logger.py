"""
Logging utilities for ParaProf with MPI rank support.

This module provides a configured logger that includes MPI rank information
in log messages for easier debugging of parallel execution.
"""

import logging
import sys
from typing import Optional, Union


# Global log level storage - allows set_log_level() to persist across setup_logger() calls
_GLOBAL_LOG_LEVEL: Optional[int] = None


def setup_logger(
    name: str = "paraprof",
    level: Optional[Union[int, str]] = None,
    rank: Optional[int] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with MPI rank prefix and configurable output.

    Parameters
    ----------
    name : str, optional
        Logger name (default: "paraprof")
    level : int, str, or None, optional
        Logging level. Can be an integer constant (e.g., logging.DEBUG),
        a string (e.g., 'DEBUG', 'INFO'), or None to use the globally
        set level from set_log_level() (or logging.INFO if none set).
        Default: None (uses global level or logging.INFO)
    rank : int, optional
        MPI rank for this process. If None, defaults to 0.
        When using MPI, this should be explicitly provided to avoid
        accidentally using MPI_COMM_WORLD
    log_file : str, optional
        Path to log file. If None, logs only to stderr

    Returns
    -------
    logger : logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger(level=logging.DEBUG)
    >>> logger.info("Starting optimization")
    [Rank 0] INFO - Starting optimization

    >>> logger = setup_logger(name="myapp", rank=2)
    >>> logger.warning("Convergence slow")
    [Rank 2] WARNING - Convergence slow
    """
    global _GLOBAL_LOG_LEVEL

    # Determine effective log level
    if level is None:
        # Use global level if set, otherwise default to INFO
        level = _GLOBAL_LOG_LEVEL if _GLOBAL_LOG_LEVEL is not None else logging.INFO
    elif isinstance(level, str):
        # Convert string to int
        level = logging.getLevelName(level.upper())

    # Default to rank 0 if not provided
    if rank is None:
        rank = 0

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with rank prefix
    formatter = logging.Formatter(
        f'[Rank {rank}] %(levelname)s - %(message)s'
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str = "paraprof") -> logging.Logger:
    """
    Get an existing logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name (default: "paraprof")

    Returns
    -------
    logger : logging.Logger
        Logger instance (creates default if doesn't exist)

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.debug("Debug message")
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up a default one
    if not logger.handlers:
        return setup_logger(name=name)

    return logger


def set_log_level(level: Union[int, str], name: str = "paraprof") -> None:
    """
    Change log level for existing logger and store globally.

    This function sets the log level for the current logger and stores it
    globally so that future calls to setup_logger() will use this level
    as the default. This ensures log level settings persist across MPI
    process initialization.

    Parameters
    ----------
    level : int or str
        New logging level. Can be an integer constant (e.g., logging.DEBUG,
        logging.INFO) or a string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR')
    name : str, optional
        Logger name (default: "paraprof")

    Examples
    --------
    >>> set_log_level(logging.DEBUG)  # Show debug messages
    >>> set_log_level('WARNING')  # Only warnings and errors (string form)
    >>> set_log_level(logging.ERROR)  # Only errors
    """
    global _GLOBAL_LOG_LEVEL

    # Convert string to int if needed
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    # Store globally so setup_logger() can use it
    _GLOBAL_LOG_LEVEL = level

    # Update existing logger if it exists
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
