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

    def test_octant_6d_has_two_octant_maxima(self):
        """The registered peaks are the best point of each s23 optimum."""
        func, bounds, peaks = get_test_function("octant_6d")

        assert np.array(bounds).shape == (6, 2)
        upper, lower = peaks
        assert upper[5] > 0.6 and lower[5] < 0.45
        np.testing.assert_allclose(func(upper), 0.0, atol=1e-6)
        np.testing.assert_allclose(func(lower), -0.1272, atol=1e-3)
        # Both are local maxima: a small step in any parameter lowers the value.
        for peak in peaks:
            for i in range(6):
                for step in (-1e-3, 1e-3):
                    moved = peak.copy()
                    moved[i] += step
                    assert func(moved) < func(peak)

