"""
Test functions for benchmarking the optimization algorithm.
"""
import numpy as np


def rosenbrock_4d(params):
    """
    4D Rosenbrock function (negated for maximization).

    Global minimum at (1, 1, 1, 1) with value 0.
    """
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)


def himmelblau_4d(params):
    """
    4D Himmelblau function (negated and scaled for maximization).

    Has 4 known maxima in the 4D space.
    """
    x1, x2, x3, x4 = params
    term1 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    term2 = (x3**2 + x4 - 11)**2 + (x3 + x4**2 - 7)**2
    scale = 0.05
    return -1 * scale * (term1 + term2)


def get_test_function(name):
    """
    Factory function to get a test likelihood, its bounds, and true peaks.

    Parameters
    ----------
    name : str
        Name of the test function ('rosenbrock_4d' or 'himmelblau_4d')

    Returns
    -------
    func : callable
        The test function
    bounds : list of [min, max] pairs
        Parameter bounds for each dimension
    peaks : list of numpy arrays
        Known peak locations
    """
    if name == "rosenbrock_4d":
        bounds = [[-6, 6], [-6, 6], [-6, 6], [-6, 6]]
        peaks = [np.array([1.0, 1.0, 1.0, 1.0])]
        return rosenbrock_4d, bounds, peaks

    elif name == "himmelblau_4d":
        bounds = [[-6, 6], [-6, 6], [-6, 6], [-6, 6]]
        peaks = [
            np.array([3.0, 2.0, 3.0, 2.0]),
            np.array([-2.805118, 3.131312, -2.805118, 3.131312]),
            np.array([-3.779310, -3.283186, -3.779310, -3.283186]),
            np.array([3.584428, -1.848126, 3.584428, -1.848126])
        ]
        return himmelblau_4d, bounds, peaks
    else:
        raise ValueError(f"Unknown test function: {name}")
