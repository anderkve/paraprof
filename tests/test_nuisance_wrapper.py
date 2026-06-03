"""
Tests for the nuisance parameter wrapper functionality.
"""
import pytest
import numpy as np
from paraprof.nuisance_wrapper import (
    NuisanceParameterWrapper,
    create_nuisance_wrapped_function,
    register_nuisance_wrapped_test_functions
)
from paraprof.test_functions import get_test_function


class TestNuisanceParameterWrapper:
    """Test suite for NuisanceParameterWrapper class."""

    def test_basic_initialization(self):
        """Test basic wrapper initialization."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=4,
            coupling_mode='shift'
        )

        assert wrapper.n_poi == 2
        assert wrapper.n_nuisance == 4
        assert wrapper.n_total == 6
        assert wrapper.coupling_mode == 'shift'

    def test_parameter_splitting(self):
        """Test that parameters are correctly split into POI and nuisance."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=3,
            n_nuisance=2,
            coupling_mode='additive'
        )

        params = np.array([1.0, 2.0, 3.0, 0.1, 0.2])
        result = wrapper(params)

        # With additive coupling, nuisance doesn't affect POI
        # Base function sees [1, 2, 3], constraint penalty on [0.1, 0.2]
        base_val = base_func(np.array([1.0, 2.0, 3.0]))
        constraint_penalty = -0.5 * (0.1**2 + 0.2**2)  # Gaussian penalty

        assert np.isclose(result, base_val + constraint_penalty)

    def test_shift_coupling(self):
        """Test shift coupling mode."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='shift',
            constraint_sigma=1.0
        )

        # POI = [1, 2], nuisance = [0, 0] → base sees [1, 2]
        params_no_shift = np.array([1.0, 2.0, 0.0, 0.0])
        result_no_shift = wrapper(params_no_shift)
        expected_no_shift = base_func(np.array([1.0, 2.0]))
        assert np.isclose(result_no_shift, expected_no_shift)

        # POI = [1, 2], nuisance = [0.5, 0.3] → base sees shifted values
        params_with_shift = np.array([1.0, 2.0, 0.5, 0.3])
        result_with_shift = wrapper(params_with_shift)

        # With default coupling matrix, each POI gets one nuisance
        # POI 0 gets nuis 0, POI 1 gets nuis 1
        transformed = np.array([1.0 + 0.5, 2.0 + 0.3])
        expected_base = base_func(transformed)
        expected_constraint = -0.5 * (0.5**2 + 0.3**2)
        expected_total = expected_base + expected_constraint

        assert np.isclose(result_with_shift, expected_total, atol=1e-6)

    def test_constraint_penalty(self):
        """Test Gaussian constraint penalty calculation."""
        base_func = lambda x: -np.sum(x**2)
        sigma = 0.5
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='additive',
            constraint_sigma=sigma,
            nuisance_mean=0.0
        )

        # Nuisance at mean → no penalty
        params_at_mean = np.array([1.0, 2.0, 0.0, 0.0])
        result_at_mean = wrapper(params_at_mean)
        expected_at_mean = base_func(np.array([1.0, 2.0]))
        assert np.isclose(result_at_mean, expected_at_mean)

        # Nuisance 1σ away → penalty of -0.5
        params_1sig = np.array([1.0, 2.0, sigma, 0.0])
        result_1sig = wrapper(params_1sig)
        expected_penalty = -0.5 * 1.0**2  # (sigma/sigma)^2 = 1
        expected_1sig = base_func(np.array([1.0, 2.0])) + expected_penalty
        assert np.isclose(result_1sig, expected_1sig)

        # Both nuisance 1σ away → penalty of -1.0
        params_both_1sig = np.array([1.0, 2.0, sigma, sigma])
        result_both_1sig = wrapper(params_both_1sig)
        expected_both = base_func(np.array([1.0, 2.0])) - 1.0
        assert np.isclose(result_both_1sig, expected_both)

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        base_func = lambda x: -np.sum(x**2)

        # Wrong number of parameters
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='shift'
        )

        with pytest.raises(ValueError, match="Expected 4 parameters"):
            wrapper(np.array([1.0, 2.0]))  # Too few

        # Invalid coupling mode
        with pytest.raises(ValueError, match="Unknown coupling_mode"):
            NuisanceParameterWrapper(
                base_func=base_func,
                n_poi=2,
                n_nuisance=2,
                coupling_mode='invalid'
            )

        # Mismatched constraint_sigma length
        with pytest.raises(ValueError, match="constraint_sigma length"):
            NuisanceParameterWrapper(
                base_func=base_func,
                n_poi=2,
                n_nuisance=3,
                coupling_mode='shift',
                constraint_sigma=np.array([0.5, 1.0])  # Length 2, need 3
            )

        # Invalid coupling matrix shape
        with pytest.raises(ValueError, match="coupling_matrix shape"):
            NuisanceParameterWrapper(
                base_func=base_func,
                n_poi=2,
                n_nuisance=3,
                coupling_mode='shift',
                coupling_matrix=np.ones((3, 2))  # Wrong shape
            )


class TestCreateNuisanceWrappedFunction:
    """Test suite for create_nuisance_wrapped_function helper."""

    def test_basic_creation(self):
        """Test basic wrapped function creation."""
        base_func, base_bounds, _ = get_test_function("sphere_4d")

        wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
            base_func=base_func,
            base_bounds=base_bounds,
            n_poi=4,
            n_nuisance=4,
            coupling_mode='shift',
            constraint_sigma=0.5
        )

        # Check dimensions
        assert len(wrapped_bounds) == 8
        assert wrapper.n_poi == 4
        assert wrapper.n_nuisance == 4

        # POI bounds should match base bounds
        assert wrapped_bounds[:4] == base_bounds

        # Nuisance bounds should be symmetric around mean
        for i in range(4):
            nuis_bounds = wrapped_bounds[4 + i]
            center = (nuis_bounds[0] + nuis_bounds[1]) / 2
            width = (nuis_bounds[1] - nuis_bounds[0]) / 2
            assert np.isclose(center, 0.0)  # Mean = 0
            assert np.isclose(width, 5.0 * 0.5)  # 5*sigma

        # Test evaluation
        params = np.zeros(8)
        result = wrapped_func(params)
        assert np.isfinite(result)

    def test_different_coupling_modes(self):
        """Test different coupling modes produce different results."""
        base_func, base_bounds, _ = get_test_function("rosenbrock_4d")

        # Create wrappers with different coupling modes
        func_shift, _, _ = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=4, n_nuisance=4,
            coupling_mode='shift', constraint_sigma=0.5
        )

        func_scale, _, _ = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=4, n_nuisance=4,
            coupling_mode='scale', constraint_sigma=0.1
        )

        func_additive, _, _ = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=4, n_nuisance=4,
            coupling_mode='additive', constraint_sigma=0.5
        )

        # Same parameters, different results
        params = np.ones(8) * 0.5
        result_shift = func_shift(params)
        result_scale = func_scale(params)
        result_additive = func_additive(params)

        # Results should differ (except by chance)
        assert not np.isclose(result_shift, result_scale)
        assert not np.isclose(result_shift, result_additive)


class TestRegisterNuisanceWrappedTestFunctions:
    """Test suite for pre-registered test function configurations."""

    def test_registry_specific_cases(self):
        """Test specific registry entries."""
        registry = register_nuisance_wrapped_test_functions()

        # Test Himmelblau with 8 nuisance, shift, sigma=0.5
        name = "himmelblau_4d_shift_8nuis_sigma0.5"
        assert name in registry

        func, bounds, wrapper, peaks = registry[name]
        assert len(bounds) == 12  # 4 POI + 8 nuisance
        assert wrapper.n_poi == 4
        assert wrapper.n_nuisance == 8
        assert wrapper.coupling_mode == 'shift'
        assert np.allclose(wrapper.constraint_sigma, 0.5)

        # Evaluate at a peak
        peak_poi = peaks[0]
        params = np.concatenate([peak_poi, np.zeros(8)])
        result = func(params)
        assert np.isclose(result, 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
