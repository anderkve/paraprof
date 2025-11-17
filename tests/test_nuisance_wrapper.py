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

    def test_scale_coupling(self):
        """Test scale coupling mode."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='scale',
            constraint_sigma=1.0
        )

        # POI = [2, 3], nuisance = [0.1, 0.2]
        # → base sees [2*(1+0.1), 3*(1+0.2)] = [2.2, 3.6]
        params = np.array([2.0, 3.0, 0.1, 0.2])
        result = wrapper(params)

        transformed = np.array([2.0 * 1.1, 3.0 * 1.2])
        expected_base = base_func(transformed)
        expected_constraint = -0.5 * (0.1**2 + 0.2**2)
        expected_total = expected_base + expected_constraint

        assert np.isclose(result, expected_total, atol=1e-6)

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

    def test_custom_coupling_matrix(self):
        """Test custom coupling matrix."""
        base_func = lambda x: np.sum(x)  # Simple sum

        # Custom matrix: first nuisance affects both POI
        coupling_matrix = np.array([
            [1.0, 0.0],  # POI 0 gets nuisance 0
            [1.0, 0.5]   # POI 1 gets nuisance 0 and 0.5*nuisance 1
        ])

        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='shift',
            coupling_matrix=coupling_matrix,
            constraint_sigma=1.0
        )

        # POI = [1, 2], nuisance = [0.5, 0.4]
        # Transformed POI = [1 + 0.5, 2 + 0.5 + 0.5*0.4] = [1.5, 2.7]
        params = np.array([1.0, 2.0, 0.5, 0.4])
        result = wrapper(params)

        transformed = np.array([1.0 + 0.5, 2.0 + 0.5 + 0.5*0.4])
        expected_base = base_func(transformed)  # Sum = 1.5 + 2.7 = 4.2
        expected_constraint = -0.5 * (0.5**2 + 0.4**2)
        expected_total = expected_base + expected_constraint

        assert np.isclose(result, expected_total, atol=1e-6)

    def test_vector_constraint_parameters(self):
        """Test per-parameter constraint sigmas and means."""
        base_func = lambda x: 0.0  # Constant function

        sigmas = np.array([0.5, 1.0, 0.2])
        means = np.array([0.0, 0.5, -0.3])

        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=3,
            coupling_mode='additive',
            constraint_sigma=sigmas,
            nuisance_mean=means
        )

        # Set nuisance to their means → no penalty
        params_at_means = np.array([1.0, 2.0, 0.0, 0.5, -0.3])
        result = wrapper(params_at_means)
        assert np.isclose(result, 0.0)

        # Shift each by 1σ → penalty = -0.5 * 3 = -1.5
        params_1sig = np.array([1.0, 2.0, 0.0 + 0.5, 0.5 + 1.0, -0.3 + 0.2])
        result_1sig = wrapper(params_1sig)
        assert np.isclose(result_1sig, -1.5)

    def test_optimal_nuisance(self):
        """Test analytical optimal nuisance calculation."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=3,
            coupling_mode='shift',
            constraint_sigma=0.5,
            nuisance_mean=0.1
        )

        poi_values = np.array([1.0, 2.0])
        optimal_nuis = wrapper.get_optimal_nuisance(poi_values)

        # For most coupling modes, optimal = constraint mean
        assert np.allclose(optimal_nuis, 0.1)

    def test_profile_over_nuisance(self):
        """Test profile likelihood computation."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=2,
            coupling_mode='shift',
            constraint_sigma=0.5,
            nuisance_mean=0.0
        )

        poi_values = np.array([1.0, 2.0])
        profile_ll, optimal_nuis = wrapper.profile_over_nuisance(
            poi_values, method='analytical'
        )

        # Optimal nuisance should be at mean
        assert np.allclose(optimal_nuis, 0.0)

        # Profile likelihood = base function at POI (with optimal nuisance)
        expected_ll = base_func(poi_values)
        assert np.isclose(profile_ll, expected_ll)

    def test_rotation_coupling(self):
        """Test rotation coupling mode."""
        base_func = lambda x: -np.sum(x**2)
        wrapper = NuisanceParameterWrapper(
            base_func=base_func,
            n_poi=2,
            n_nuisance=1,
            coupling_mode='rotation',
            rotation_scale=0.1,
            constraint_sigma=1.0
        )

        # Small rotation should give nearly same result
        params = np.array([1.0, 0.0, 0.1])  # Rotate by 0.1 * 0.1 = 0.01 rad
        result = wrapper(params)

        # Rotated point should have same norm (for quadratic function)
        angle = 0.1 * 0.1
        rotated = np.array([np.cos(angle) * 1.0, np.sin(angle) * 1.0])
        expected_base = base_func(rotated)
        expected_constraint = -0.5 * 0.1**2
        expected_total = expected_base + expected_constraint

        assert np.isclose(result, expected_total, atol=1e-6)

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

    def test_himmelblau_wrapping(self):
        """Test wrapping Himmelblau function."""
        base_func, base_bounds, base_peaks = get_test_function("himmelblau_4d")

        wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
            base_func=base_func,
            base_bounds=base_bounds,
            n_poi=4,
            n_nuisance=8,
            coupling_mode='shift',
            constraint_sigma=0.3
        )

        assert len(wrapped_bounds) == 12

        # Evaluate at a known peak with optimal nuisance
        peak = base_peaks[0]  # [3.0, 2.0, 3.0, 2.0]
        params = np.concatenate([peak, np.zeros(8)])
        result = wrapped_func(params)

        # Should be at optimum (0.0 for negated function)
        expected = base_func(peak)  # Should be 0.0
        assert np.isclose(result, expected)

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

    def test_registry_creation(self):
        """Test that registry is created successfully."""
        registry = register_nuisance_wrapped_test_functions()

        assert isinstance(registry, dict)
        assert len(registry) > 0

        # Check that all entries have correct structure
        for name, (func, bounds, wrapper, peaks) in registry.items():
            assert callable(func)
            assert isinstance(bounds, list)
            assert isinstance(wrapper, NuisanceParameterWrapper)
            assert isinstance(peaks, list)

            # Name should be descriptive
            assert 'nuis' in name
            assert 'sigma' in name

    def test_registry_functions_callable(self):
        """Test that all registered functions are callable."""
        registry = register_nuisance_wrapped_test_functions()

        for name, (func, bounds, wrapper, peaks) in registry.items():
            # Create random params within bounds
            params = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])

            # Should evaluate without error
            result = func(params)
            assert np.isfinite(result)

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


class TestPhysicsRealisticScenarios:
    """Test scenarios mimicking realistic physics use cases."""

    def test_profile_with_many_nuisance(self):
        """Test computing profile with many constrained nuisance parameters."""
        base_func, base_bounds, peaks = get_test_function("himmelblau_4d")

        # 20 nuisance parameters (more than POI)
        wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
            base_func=base_func,
            base_bounds=base_bounds,
            n_poi=4,
            n_nuisance=20,
            coupling_mode='shift',
            constraint_sigma=0.3
        )

        # Evaluate at a peak with all nuisance at optimal
        peak_poi = peaks[0]
        params_optimal = np.concatenate([peak_poi, np.zeros(20)])
        ll_optimal = wrapped_func(params_optimal)

        # Evaluate with random nuisance fluctuations within 1σ
        np.random.seed(42)
        nuis_fluct = np.random.randn(20) * 0.3
        params_fluct = np.concatenate([peak_poi, nuis_fluct])
        ll_fluct = wrapped_func(params_fluct)

        # Likelihood should be lower with fluctuations
        assert ll_fluct < ll_optimal

        # Penalty should be roughly -0.5 * n_nuisance (for 1σ fluctuations)
        # (This is statistical, not exact)
        expected_penalty_magnitude = 0.5 * 20
        actual_penalty = ll_optimal - ll_fluct
        assert 0.2 * expected_penalty_magnitude < actual_penalty < 2.0 * expected_penalty_magnitude

    def test_tight_vs_loose_constraints(self):
        """Test effect of tight vs loose constraints."""
        base_func, base_bounds, _ = get_test_function("sphere_4d")

        # Tight constraints (sigma = 0.1)
        func_tight, _, wrapper_tight = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=4, n_nuisance=4,
            coupling_mode='shift', constraint_sigma=0.1
        )

        # Loose constraints (sigma = 2.0)
        func_loose, _, wrapper_loose = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=4, n_nuisance=4,
            coupling_mode='shift', constraint_sigma=2.0
        )

        # Same POI, same nuisance deviation from mean
        poi = np.ones(4)
        nuis_deviation = np.ones(4) * 0.5  # 0.5 units from mean

        params_tight = np.concatenate([poi, nuis_deviation])
        params_loose = np.concatenate([poi, nuis_deviation])

        ll_tight = func_tight(params_tight)
        ll_loose = func_loose(params_loose)

        # Tight constraints should penalize more heavily
        # Penalty ~ -0.5 * (0.5/sigma)^2 * n_nuis
        penalty_tight = -0.5 * (0.5/0.1)**2 * 4  # = -50
        penalty_loose = -0.5 * (0.5/2.0)**2 * 4  # = -0.125

        base_ll = base_func(poi + nuis_deviation)  # Shift coupling

        # Note: base_ll will differ due to different transformations
        # Focus on relative penalty
        assert ll_loose > ll_tight  # Loose is less penalized

    def test_calibration_shift_scenario(self):
        """
        Test a physics-like calibration scenario.

        Scenario: Measuring properties of a system with 4 observables,
        but detector calibration introduces systematic shifts in each measurement.
        """
        # True underlying function
        def physics_model(observables):
            # Some multimodal function representing physics
            return -np.sum((observables - 2.0)**2)

        base_bounds = [[-5, 5]] * 4

        # 8 calibration uncertainties (2 per observable)
        wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
            base_func=physics_model,
            base_bounds=base_bounds,
            n_poi=4,
            n_nuisance=8,
            coupling_mode='shift',
            constraint_sigma=0.2,  # Tight calibration
            nuisance_mean=0.0
        )

        # Best fit should have observables ≈ 2.0 and calibrations ≈ 0.0
        best_guess = np.concatenate([
            np.array([2.0, 2.0, 2.0, 2.0]),  # Observables at true value
            np.zeros(8)  # Calibrations at nominal
        ])

        ll_best = wrapped_func(best_guess)

        # Small miscalibration should reduce likelihood
        miscalibrated = np.concatenate([
            np.array([2.0, 2.0, 2.0, 2.0]),
            np.ones(8) * 0.1  # Small calibration shifts
        ])

        ll_miscalib = wrapped_func(miscalibrated)
        assert ll_miscalib < ll_best

        # Wrong observables with perfect calibration
        wrong_obs = np.concatenate([
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.zeros(8)
        ])
        ll_wrong = wrapped_func(wrong_obs)

        # Wrong observables should be worse than small miscalibration
        assert ll_wrong < ll_miscalib


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
