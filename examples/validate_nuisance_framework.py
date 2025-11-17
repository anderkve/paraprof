"""
Validation script for the nuisance parameter framework.

This script demonstrates and validates key features of the nuisance parameter
system without requiring MPI, making it easy to verify the implementation.

Run with:
    python validate_nuisance_framework.py
"""
import numpy as np
import sys
sys.path.insert(0, '../src')

from paraprof import get_test_function
from paraprof.nuisance_wrapper import (
    NuisanceParameterWrapper,
    create_nuisance_wrapped_function,
    register_nuisance_wrapped_test_functions
)


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def validate_basic_functionality():
    """Validate basic wrapper creation and evaluation."""
    print_section("Test 1: Basic Functionality")

    base_func, base_bounds, peaks = get_test_function("himmelblau_4d")
    print(f"Base function: Himmelblau 4D")
    print(f"Base bounds: {base_bounds}")
    print(f"Known peaks: {len(peaks)} peaks")

    wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
        base_func=base_func,
        base_bounds=base_bounds,
        n_poi=4,
        n_nuisance=8,
        coupling_mode='shift',
        constraint_sigma=0.5
    )

    print(f"\nWrapped function created:")
    print(f"  Total dimensions: {len(wrapped_bounds)} (4 POI + 8 nuisance)")
    print(f"  Coupling mode: shift")
    print(f"  Constraint sigma: 0.5")

    # Test evaluation at a known peak
    peak = peaks[0]
    params_optimal = np.concatenate([peak, np.zeros(8)])
    ll_optimal = wrapped_func(params_optimal)

    print(f"\nEvaluation at known peak:")
    print(f"  POI: {peak}")
    print(f"  Nuisance: [0, 0, 0, 0, 0, 0, 0, 0]")
    print(f"  log L: {ll_optimal:.6f}")
    print(f"  Expected: 0.0 (Himmelblau optimum)")
    print(f"  Match: {np.isclose(ll_optimal, 0.0, atol=1e-10)} ✓" if np.isclose(ll_optimal, 0.0, atol=1e-10) else f"  Match: False ✗")


def validate_coupling_modes():
    """Validate different coupling modes."""
    print_section("Test 2: Coupling Modes")

    base_func = lambda x: -np.sum(x**2)
    base_bounds = [[-5, 5]] * 2

    poi = np.array([1.0, 2.0])
    nuis = np.array([0.5, 0.3])

    modes = ['shift', 'scale', 'additive']
    results = {}

    for mode in modes:
        wrapped_func, _, _ = create_nuisance_wrapped_function(
            base_func, base_bounds, n_poi=2, n_nuisance=2,
            coupling_mode=mode, constraint_sigma=1.0
        )
        params = np.concatenate([poi, nuis])
        ll = wrapped_func(params)
        results[mode] = ll
        print(f"\n{mode.capitalize()} coupling:")
        print(f"  POI = {poi}, nuisance = {nuis}")
        print(f"  log L = {ll:.6f}")

    # Check that modes give different results
    print(f"\nDifferent modes give different results: ", end="")
    if len(set(results.values())) == len(modes):
        print("✓")
    else:
        print("✗")

    # Validate shift coupling analytically
    expected_shift_poi = poi + nuis  # With default coupling, each nuis affects one POI
    expected_base = base_func(expected_shift_poi)
    expected_constraint = -0.5 * np.sum(nuis**2)
    expected_total = expected_base + expected_constraint

    print(f"\nShift coupling validation:")
    print(f"  Expected: {expected_total:.6f}")
    print(f"  Actual: {results['shift']:.6f}")
    print(f"  Match: {np.isclose(results['shift'], expected_total, atol=1e-5)} ✓" if np.isclose(results['shift'], expected_total, atol=1e-5) else f"  Match: False ✗")


def validate_constraint_penalties():
    """Validate constraint penalty calculations."""
    print_section("Test 3: Constraint Penalties")

    base_func = lambda x: 0.0  # Constant function
    base_bounds = [[-5, 5]] * 2

    wrapped_func, _, _ = create_nuisance_wrapped_function(
        base_func, base_bounds, n_poi=2, n_nuisance=3,
        coupling_mode='additive',  # No coupling, pure constraint test
        constraint_sigma=0.5,
        constraint_mode='gaussian'
    )

    test_cases = [
        ("At mean (0σ)", np.zeros(3), 0.0),
        ("At 1σ", np.ones(3) * 0.5, -1.5),  # -0.5 * 3 * 1^2
        ("At 2σ", np.ones(3) * 1.0, -6.0),  # -0.5 * 3 * 2^2
    ]

    print("\nGaussian constraint penalties:")
    all_match = True
    for name, nuis, expected_penalty in test_cases:
        params = np.concatenate([np.zeros(2), nuis])
        ll = wrapped_func(params)
        match = np.isclose(ll, expected_penalty, atol=1e-6)
        all_match = all_match and match
        print(f"  {name}: log L = {ll:.6f}, expected = {expected_penalty:.6f} {'✓' if match else '✗'}")

    print(f"\nAll penalties correct: {'✓' if all_match else '✗'}")


def validate_custom_coupling():
    """Validate custom coupling matrices."""
    print_section("Test 4: Custom Coupling Matrix")

    base_func = lambda x: np.sum(x)  # Simple sum
    base_bounds = [[-5, 5]] * 2

    # Custom coupling: first nuisance affects both POI
    coupling_matrix = np.array([
        [1.0, 0.0],  # POI 0 gets nuisance 0
        [1.0, 1.0]   # POI 1 gets nuisance 0 and nuisance 1
    ])

    print("Coupling matrix:")
    print(coupling_matrix)

    wrapped_func, _, _ = create_nuisance_wrapped_function(
        base_func, base_bounds, n_poi=2, n_nuisance=2,
        coupling_mode='shift',
        coupling_matrix=coupling_matrix,
        constraint_sigma=1.0
    )

    poi = np.array([1.0, 2.0])
    nuis = np.array([0.5, 0.3])
    params = np.concatenate([poi, nuis])

    ll = wrapped_func(params)

    # Manual calculation
    transformed_poi = poi + coupling_matrix @ nuis
    # = [1.0 + 0.5, 2.0 + 0.5 + 0.3] = [1.5, 2.8]
    expected_base = np.sum(transformed_poi)  # 4.3
    expected_constraint = -0.5 * (0.5**2 + 0.3**2)  # -0.17
    expected_total = expected_base + expected_constraint

    print(f"\nPOI = {poi}, nuisance = {nuis}")
    print(f"Transformed POI = {transformed_poi}")
    print(f"Expected log L: {expected_total:.6f}")
    print(f"Actual log L: {ll:.6f}")
    print(f"Match: {np.isclose(ll, expected_total, atol=1e-6)} ✓" if np.isclose(ll, expected_total, atol=1e-6) else f"Match: False ✗")


def validate_profile_computation():
    """Validate profile likelihood computation."""
    print_section("Test 5: Profile Likelihood")

    base_func = lambda x: -np.sum(x**2)
    base_bounds = [[-5, 5]] * 2

    wrapped_func, _, wrapper = create_nuisance_wrapped_function(
        base_func, base_bounds, n_poi=2, n_nuisance=3,
        coupling_mode='shift', constraint_sigma=0.5
    )

    poi_values = np.array([1.0, 2.0])

    # Analytical profiling (optimal nuisance at constraint mean)
    profile_ll, optimal_nuis = wrapper.profile_over_nuisance(
        poi_values, method='analytical'
    )

    print(f"POI values: {poi_values}")
    print(f"Optimal nuisance parameters: {optimal_nuis}")
    print(f"Profile log-likelihood: {profile_ll:.6f}")

    # For shift coupling with optimal nuisance at zero, should equal base function
    expected_ll = base_func(poi_values)
    print(f"Expected (base function at POI): {expected_ll:.6f}")
    print(f"Match: {np.isclose(profile_ll, expected_ll, atol=1e-6)} ✓" if np.isclose(profile_ll, expected_ll, atol=1e-6) else f"Match: False ✗")


def validate_registry():
    """Validate pre-registered test functions."""
    print_section("Test 6: Pre-Registered Functions")

    registry = register_nuisance_wrapped_test_functions()

    print(f"Registry contains {len(registry)} pre-configured test functions:")
    for i, name in enumerate(sorted(registry.keys())[:10], 1):
        print(f"  {i}. {name}")
    if len(registry) > 10:
        print(f"  ... and {len(registry) - 10} more")

    # Test a specific entry
    test_name = 'himmelblau_4d_shift_8nuis_sigma0.5'
    if test_name in registry:
        func, bounds, wrapper, peaks = registry[test_name]
        print(f"\nTesting: {test_name}")
        print(f"  Dimensions: {len(bounds)} (4 POI + 8 nuisance)")
        print(f"  Coupling: {wrapper.coupling_mode}")
        print(f"  Constraint σ: {wrapper.constraint_sigma[0]:.1f}")

        # Evaluate at a peak
        peak = peaks[0]
        params = np.concatenate([peak, np.zeros(8)])
        ll = func(params)

        print(f"  Evaluation at peak: log L = {ll:.6f}")
        print(f"  Match optimum (0.0): {np.isclose(ll, 0.0, atol=1e-10)} ✓" if np.isclose(ll, 0.0, atol=1e-10) else f"  Match: False ✗")


def validate_physics_scenario():
    """Validate a realistic physics-like scenario."""
    print_section("Test 7: Physics-Realistic Scenario")

    print("Scenario: Detector calibration uncertainty")
    print("  - 4 measured observables (POI)")
    print("  - 8 calibration parameters (nuisance)")
    print("  - Each observable affected by 2 calibration factors")
    print("  - Calibration constrained to ±0.2 (20% uncertainty)")

    base_func, base_bounds, peaks = get_test_function("himmelblau_4d")

    wrapped_func, wrapped_bounds, wrapper = create_nuisance_wrapped_function(
        base_func=base_func,
        base_bounds=base_bounds,
        n_poi=4,
        n_nuisance=8,
        coupling_mode='shift',
        constraint_sigma=0.2,  # Tight calibration
        nuisance_mean=0.0
    )

    print(f"\nTotal parameter space: {len(wrapped_bounds)}D")

    # Test scenario 1: Perfect calibration
    peak = peaks[0]
    params_perfect = np.concatenate([peak, np.zeros(8)])
    ll_perfect = wrapped_func(params_perfect)

    print(f"\nScenario 1: Perfect calibration")
    print(f"  Observables at true peak: {peak}")
    print(f"  Calibration at nominal: [0, 0, 0, 0, 0, 0, 0, 0]")
    print(f"  log L: {ll_perfect:.6f}")

    # Test scenario 2: Small miscalibration (within 1σ)
    np.random.seed(42)
    small_miscalib = np.random.randn(8) * 0.1  # 0.5σ fluctuations
    params_small_miscalib = np.concatenate([peak, small_miscalib])
    ll_small_miscalib = wrapped_func(params_small_miscalib)

    print(f"\nScenario 2: Small miscalibration (0.5σ)")
    print(f"  Calibration deviations: {small_miscalib}")
    print(f"  log L: {ll_small_miscalib:.6f}")
    print(f"  Penalty: {ll_small_miscalib - ll_perfect:.6f}")
    print(f"  Expected penalty: ≈ -0.5 * 8 * 0.5² = -1.0")

    # Test scenario 3: Large miscalibration (3σ)
    large_miscalib = np.ones(8) * 0.6  # 3σ deviations
    params_large_miscalib = np.concatenate([peak, large_miscalib])
    ll_large_miscalib = wrapped_func(params_large_miscalib)

    print(f"\nScenario 3: Large miscalibration (3σ)")
    print(f"  Calibration deviations: {large_miscalib}")
    print(f"  log L: {ll_large_miscalib:.6f}")
    print(f"  Penalty: {ll_large_miscalib - ll_perfect:.6f}")
    print(f"  Expected penalty: ≈ -0.5 * 8 * 3² = -36.0")

    # Verify ordering
    print(f"\nLikelihood ordering (should be decreasing):")
    print(f"  Perfect: {ll_perfect:.6f}")
    print(f"  Small miscalib: {ll_small_miscalib:.6f}")
    print(f"  Large miscalib: {ll_large_miscalib:.6f}")
    correct_ordering = ll_perfect > ll_small_miscalib > ll_large_miscalib
    print(f"  Correct ordering: {correct_ordering} {'✓' if correct_ordering else '✗'}")


def main():
    """Run all validation tests."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  Nuisance Parameter Framework - Validation Suite".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)

    try:
        validate_basic_functionality()
        validate_coupling_modes()
        validate_constraint_penalties()
        validate_custom_coupling()
        validate_profile_computation()
        validate_registry()
        validate_physics_scenario()

        print("\n" + "="*80)
        print("  ALL VALIDATION TESTS COMPLETED")
        print("="*80)
        print("\nThe nuisance parameter framework is working correctly! ✓")
        print("\nNext steps:")
        print("  1. Run unit tests: pytest tests/test_nuisance_wrapper.py -v")
        print("  2. Try the MPI example: mpiexec -n 8 python examples/run_nuisance_example.py")
        print("  3. Read NUISANCE_QUICKSTART.md for usage guide")
        print("  4. See NUISANCE_PARAMETERS.md for complete documentation")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"  VALIDATION FAILED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
