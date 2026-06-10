"""
ParaProf: Parallel Profile Likelihood Computation using Grid-Based Optimization.
"""
from .sampler import ProfileProjector
from .master import (
    master_main,
    run_projection,
    run_all_projections,
    run_scan,
    run_volume_sampling,
    terminate_workers,
)
from .worker import worker_main
from .visualization import plot_profiles, plot_profiled_parameters
from .test_functions import get_test_function
from .logger import setup_logger, get_logger, set_log_level
from .exceptions import (
    ParaProfError,
    InvalidProjectionError,
    InvalidBoundsError,
    ConfigurationError,
)
from .nuisance_wrapper import (
    NuisanceParameterWrapper,
    create_nuisance_wrapped_function,
    register_nuisance_wrapped_test_functions,
)
from .sample_io import (
    read_samples,
    write_samples,
    combine_samples,
    create_sample_writer,
)

__all__ = [
    'ProfileProjector',
    'master_main',
    'run_projection',
    'run_all_projections',
    'run_scan',
    'run_volume_sampling',
    'terminate_workers',
    'worker_main',
    'plot_profiles',
    'plot_profiled_parameters',
    'get_test_function',
    'setup_logger',
    'get_logger',
    'set_log_level',
    'ParaProfError',
    'InvalidProjectionError',
    'InvalidBoundsError',
    'ConfigurationError',
    'NuisanceParameterWrapper',
    'create_nuisance_wrapped_function',
    'register_nuisance_wrapped_test_functions',
    'read_samples',
    'write_samples',
    'combine_samples',
    'create_sample_writer',
]

__version__ = '1.0.0'
