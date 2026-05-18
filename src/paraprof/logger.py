"""Logging utilities for ParaProf with MPI rank support."""

import logging
import sys
from typing import Optional, Union


_GLOBAL_LOG_LEVEL: Optional[int] = None


def setup_logger(
    name: str = "paraprof",
    level: Optional[Union[int, str]] = None,
    rank: Optional[int] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with MPI rank prefix and configurable output.

    If ``level`` is None, falls back to the most recent ``set_log_level()``
    value (or INFO if never set). ``rank`` defaults to 0 — pass it explicitly
    to avoid touching MPI_COMM_WORLD.
    """
    global _GLOBAL_LOG_LEVEL

    if level is None:
        level = _GLOBAL_LOG_LEVEL if _GLOBAL_LOG_LEVEL is not None else logging.INFO
    elif isinstance(level, str):
        level = logging.getLevelName(level.upper())

    if rank is None:
        rank = 0

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(f'[Rank {rank}] %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(name: str = "paraprof") -> logging.Logger:
    """Get an existing logger, creating a default one if missing."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name=name)
    return logger


def set_log_level(level: Union[int, str], name: str = "paraprof") -> None:
    """Change the log level globally and on the named logger.

    Stored globally so later setup_logger() calls (e.g. MPI worker init)
    use the same level by default.
    """
    global _GLOBAL_LOG_LEVEL

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    _GLOBAL_LOG_LEVEL = level

    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
