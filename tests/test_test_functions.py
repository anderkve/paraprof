"""
Tests for benchmark test functions.
"""
import numpy as np
import pytest
from paraprof import get_test_function


class TestTestFunctions:
    """Test suite for benchmark test functions."""

    def test_get_test_function_himmelblau_4d(self):
        """Test retrieving Himmelblau 4D function."""
        func, bounds, peaks = get_test_function("himmelblau_4d")

        assert callable(func)
        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (4, 2)
        assert len(peaks) > 0

    def test_get_test_function_rosenbrock_4d(self):
        """Test retrieving Rosenbrock 4D function."""
        func, bounds, peaks = get_test_function("rosenbrock_4d")

        assert callable(func)
        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (4, 2)
        assert len(peaks) == 1  # Rosenbrock has single global optimum

    def test_invalid_function_name(self):
        """Test that invalid function names raise errors."""
        with pytest.raises(ValueError, match="Unknown test function"):
            get_test_function("nonexistent_function")

    def test_function_evaluates_at_optimum(self):
        """Test that functions return expected value at known optimum."""
        func, bounds, peaks = get_test_function("sphere_4d")

        # Sphere optimum is at origin with value 0.0
        optimum = np.zeros(4)
        value = func(optimum)

        np.testing.assert_allclose(value, 0.0, atol=1e-10)

    def test_function_accepts_correct_dimensionality(self):
        """Test that functions accept parameters of correct dimension."""
        func_2d, _, _ = get_test_function("sphere_2d")
        func_4d, _, _ = get_test_function("sphere_4d")

        # Should work
        value_2d = func_2d(np.array([1.0, 2.0]))
        value_4d = func_4d(np.array([1.0, 2.0, 3.0, 4.0]))

        assert np.isfinite(value_2d)
        assert np.isfinite(value_4d)

    def test_bounds_are_reasonable(self):
        """Test that returned bounds are finite and properly ordered."""
        func, bounds, peaks = get_test_function("himmelblau_4d")

        bounds_arr = np.array(bounds)
        assert np.all(np.isfinite(bounds_arr))
        assert np.all(bounds_arr[:, 0] < bounds_arr[:, 1])  # lower < upper
