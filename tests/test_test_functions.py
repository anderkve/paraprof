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
