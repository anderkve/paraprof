"""
ParaProf: Parallel Profile Likelihood Computation using Grid-Anchored Differential Evolution.
"""
from .sampler import GridAnchoredDESampler
from .master import master_main, run_projection, run_all_projections, terminate_workers
from .worker import worker_main
from .visualization import plot_profiles
from .test_functions import get_test_function
from .logger import setup_logger, get_logger, set_log_level
from .exceptions import (
    ParaProfError,
    InvalidProjectionError,
    InvalidBoundsError,
    ConvergenceError,
    MPIError,
    ConfigurationError,
    JobError,
    ValidationError,
)

__all__ = [
    'GridAnchoredDESampler',
    'master_main',
    'run_projection',
    'run_all_projections',
    'terminate_workers',
    'worker_main',
    'plot_profiles',
    'get_test_function',
    'setup_logger',
    'get_logger',
    'set_log_level',
    'ParaProfError',
    'InvalidProjectionError',
    'InvalidBoundsError',
    'ConvergenceError',
    'MPIError',
    'ConfigurationError',
    'JobError',
    'ValidationError',
]

__version__ = '1.0.0'
