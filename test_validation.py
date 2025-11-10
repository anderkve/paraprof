"""
Validation script for all test functions in test_functions.py

This script verifies that:
1. All functions can be called without errors
2. Known optima produce values close to 0.0
3. Functions have appropriate bounds
4. get_test_function factory works correctly
"""
import numpy as np
from test_functions import get_test_function


def test_function_at_optimum(name, tolerance=2e-4):
    """
    Test that a function returns a value close to 0.0 at its known optimum.

    Parameters
    ----------
    name : str
        Name of the test function
    tolerance : float
        Maximum allowed deviation from 0.0
        Note: Higher-dimensional functions may have small accumulated
        floating-point errors, so tolerance is set to 2e-4

    Returns
    -------
    tuple : (bool, str, float)
        (success, message, value_at_optimum)
    """
    try:
        func, bounds, peaks = get_test_function(name)

        if not peaks:
            return True, f"  {name:25s} - No known peaks (OK)", None

        # Test the first peak
        peak = peaks[0]
        value = func(peak)

        if np.abs(value) < tolerance:
            return True, f"  {name:25s} - Value at optimum: {value:+.6e} ✓", value
        else:
            return False, f"  {name:25s} - Value at optimum: {value:+.6e} ✗ (expected ≈ 0.0)", value

    except Exception as e:
        return False, f"  {name:25s} - ERROR: {str(e)}", None


def test_function_bounds(name):
    """
    Test that bounds are correctly defined.

    Parameters
    ----------
    name : str
        Name of the test function

    Returns
    -------
    tuple : (bool, str)
        (success, message)
    """
    try:
        func, bounds, peaks = get_test_function(name)

        # Check bounds structure
        if not isinstance(bounds, list):
            return False, f"  {name:25s} - Bounds not a list"

        n_dims = len(bounds)

        # Check that bounds are valid
        for i, (low, high) in enumerate(bounds):
            if low >= high:
                return False, f"  {name:25s} - Invalid bounds for dim {i}: [{low}, {high}]"

        # Check that peaks are within bounds (if peaks exist)
        if peaks:
            for peak in peaks:
                if len(peak) != n_dims:
                    return False, f"  {name:25s} - Peak dimension mismatch"

                for i, (val, (low, high)) in enumerate(zip(peak, bounds)):
                    if not (low <= val <= high):
                        return False, f"  {name:25s} - Peak outside bounds in dim {i}"

        return True, f"  {name:25s} - Bounds OK ({n_dims}D)"

    except Exception as e:
        return False, f"  {name:25s} - ERROR: {str(e)}"


def test_function_evaluation(name):
    """
    Test that function can be evaluated at random points.

    Parameters
    ----------
    name : str
        Name of the test function

    Returns
    -------
    tuple : (bool, str)
        (success, message)
    """
    try:
        func, bounds, peaks = get_test_function(name)
        n_dims = len(bounds)

        # Generate random point within bounds
        point = np.array([np.random.uniform(low, high) for low, high in bounds])

        # Evaluate function
        value = func(point)

        # Check that value is finite
        if not np.isfinite(value):
            return False, f"  {name:25s} - Non-finite value: {value}"

        return True, f"  {name:25s} - Evaluation OK"

    except Exception as e:
        return False, f"  {name:25s} - ERROR: {str(e)}"


def main():
    """Run all validation tests."""

    print("="*70)
    print("VALIDATION OF TEST FUNCTIONS")
    print("="*70)

    # List of all test functions
    all_functions = [
        # Sphere
        "sphere_2d", "sphere_4d", "sphere_6d", "sphere_10d",
        # Rosenbrock
        "rosenbrock_2d", "rosenbrock_4d", "rosenbrock_6d", "rosenbrock_10d",
        # Himmelblau
        "himmelblau_4d",
        # Beale
        "beale_2d",
        # Eggholder
        "eggholder_2d", "eggholder_4d", "eggholder_6d",
        # Rastrigin
        "rastrigin_2d", "rastrigin_4d", "rastrigin_6d", "rastrigin_10d",
        # Ackley
        "ackley_2d", "ackley_4d", "ackley_6d", "ackley_10d",
        # Griewank
        "griewank_2d", "griewank_4d", "griewank_6d", "griewank_10d",
        # Michalewicz
        "michalewicz_2d", "michalewicz_4d", "michalewicz_6d", "michalewicz_10d",
        # Styblinski-Tang
        "styblinski_tang_2d", "styblinski_tang_4d", "styblinski_tang_6d", "styblinski_tang_10d",
        # Levy
        "levy_2d", "levy_4d", "levy_6d", "levy_10d",
        # Schwefel
        "schwefel_2d", "schwefel_4d", "schwefel_6d", "schwefel_10d",
    ]

    print(f"\nTotal functions to test: {len(all_functions)}\n")

    # Test 1: Function evaluation
    print("-"*70)
    print("TEST 1: Function Evaluation at Random Points")
    print("-"*70)
    eval_results = [test_function_evaluation(name) for name in all_functions]
    eval_passed = sum(1 for success, _ in eval_results if success)
    for success, msg in eval_results:
        print(msg)
    print(f"\nPassed: {eval_passed}/{len(all_functions)}")

    # Test 2: Bounds validation
    print("\n" + "-"*70)
    print("TEST 2: Bounds Validation")
    print("-"*70)
    bounds_results = [test_function_bounds(name) for name in all_functions]
    bounds_passed = sum(1 for success, _ in bounds_results if success)
    for success, msg in bounds_results:
        print(msg)
    print(f"\nPassed: {bounds_passed}/{len(all_functions)}")

    # Test 3: Value at known optima
    print("\n" + "-"*70)
    print("TEST 3: Value at Known Optima (should be ≈ 0.0)")
    print("-"*70)
    optima_results = [test_function_at_optimum(name) for name in all_functions]
    optima_passed = sum(1 for success, _, _ in optima_results if success)
    for success, msg, _ in optima_results:
        print(msg)
    print(f"\nPassed: {optima_passed}/{len(all_functions)}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_tests = len(all_functions) * 3
    total_passed = eval_passed + bounds_passed + optima_passed
    print(f"Total tests passed: {total_passed}/{total_tests}")
    print(f"Success rate: {100*total_passed/total_tests:.1f}%")

    if total_passed == total_tests:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total_tests - total_passed} tests failed")

    print("="*70)

    # Detailed statistics
    print("\nDETAILED STATISTICS")
    print("-"*70)

    function_families = {
        "Sphere": [f"sphere_{d}d" for d in [2, 4, 6, 10]],
        "Rosenbrock": [f"rosenbrock_{d}d" for d in [2, 4, 6, 10]],
        "Himmelblau": ["himmelblau_4d"],
        "Beale": ["beale_2d"],
        "Eggholder": [f"eggholder_{d}d" for d in [2, 4, 6]],
        "Rastrigin": [f"rastrigin_{d}d" for d in [2, 4, 6, 10]],
        "Ackley": [f"ackley_{d}d" for d in [2, 4, 6, 10]],
        "Griewank": [f"griewank_{d}d" for d in [2, 4, 6, 10]],
        "Michalewicz": [f"michalewicz_{d}d" for d in [2, 4, 6, 10]],
        "Styblinski-Tang": [f"styblinski_tang_{d}d" for d in [2, 4, 6, 10]],
        "Levy": [f"levy_{d}d" for d in [2, 4, 6, 10]],
        "Schwefel": [f"schwefel_{d}d" for d in [2, 4, 6, 10]],
    }

    print(f"{'Function Family':<20s} {'Count':<10s} {'Dimensions':<30s}")
    print("-"*70)
    for family, funcs in function_families.items():
        dims = sorted(set([int(f.split('_')[-1].replace('d', '')) for f in funcs]))
        dims_str = ', '.join([f"{d}D" for d in dims])
        print(f"{family:<20s} {len(funcs):<10d} {dims_str:<30s}")

    print("-"*70)
    print(f"{'TOTAL':<20s} {len(all_functions):<10d}")
    print("="*70)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()
