"""
ParaProf: Parallel Profile Likelihood Computation using Grid-Anchored Differential Evolution.
"""
from .sampler import GridAnchoredDESampler
from .master import master_main
from .worker import worker_main
from .visualization import plot_profiles
from .test_functions import get_test_function

__all__ = [
    'GridAnchoredDESampler',
    'master_main',
    'worker_main',
    'plot_profiles',
    'get_test_function',
]

__version__ = '1.0.0'
