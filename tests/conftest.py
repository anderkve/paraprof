"""
Pytest configuration and shared fixtures for ParaProf tests.
"""
import numpy as np
import pytest


@pytest.fixture
def simple_2d_function():
    """Simple 2D quadratic function for testing."""
    def func(params):
        return -(params[0]**2 + params[1]**2)
    return func


@pytest.fixture
def simple_bounds_2d():
    """Simple 2D bounds for testing."""
    return np.array([[-5.0, 5.0], [-5.0, 5.0]])


@pytest.fixture
def simple_bounds_4d():
    """Simple 4D bounds for testing."""
    return np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])


@pytest.fixture
def basic_projection_1d():
    """Basic 1D projection configuration."""
    return {
        'dims': [0],
        'grid_points': [10],
        'patch_coarse_grid': False,
    }


@pytest.fixture
def basic_projection_2d():
    """Basic 2D projection configuration."""
    return {
        'dims': [0, 1],
        'grid_points': [5, 5],
        'patch_coarse_grid': False,
    }
