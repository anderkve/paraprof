#!/usr/bin/env python
"""
Verification script to test that the optimized _get_grid_indices_from_point
produces the same results as the original implementation.
"""
import numpy as np

def old_implementation(point, projection_dims, grid_axes):
    """Original O(N) implementation using argmin."""
    grid_coords = point[projection_dims]
    indices = []
    for i, coord in enumerate(grid_coords):
        axis = grid_axes[i]
        index = np.argmin(np.abs(axis - coord))
        indices.append(index)
    return tuple(indices)

def new_implementation(point, projection_dims, grid_axes):
    """New O(1) implementation using direct calculation."""
    grid_coords = point[projection_dims]
    indices = []

    for i, coord in enumerate(grid_coords):
        axis = grid_axes[i]
        n_points = len(axis)

        # Direct calculation for regularly-spaced grid (O(1) instead of O(N))
        grid_min = axis[0]
        grid_max = axis[-1]

        # Compute normalized position [0, 1] in the grid
        normalized_pos = (coord - grid_min) / (grid_max - grid_min)

        # Map to grid index and round to nearest
        index = int(round(normalized_pos * (n_points - 1)))

        # Clamp to valid range [0, n_points-1]
        index = np.clip(index, 0, n_points - 1)

        indices.append(index)

    return tuple(indices)

def run_verification_tests():
    """Run a comprehensive set of verification tests."""
    print("Running verification tests...")
    print("=" * 60)

    all_passed = True

    # Test 1: 1D projection with various grid sizes
    print("\nTest 1: 1D projections")
    for n_points in [10, 50, 100, 200]:
        bounds = np.array([[-5.0, 5.0]])
        projection_dims = [0]
        grid_axes = [np.linspace(bounds[0, 0], bounds[0, 1], n_points)]

        # Test with random points
        np.random.seed(42)
        test_points = np.random.uniform(bounds[0, 0], bounds[0, 1], 100).reshape(-1, 1)

        mismatches = 0
        for point in test_points:
            old = old_implementation(point, projection_dims, grid_axes)
            new = new_implementation(point, projection_dims, grid_axes)
            if old != new:
                mismatches += 1

        status = "✓ PASS" if mismatches == 0 else f"✗ FAIL ({mismatches} mismatches)"
        print(f"  Grid size {n_points:3d}: {status}")
        if mismatches > 0:
            all_passed = False

    # Test 2: 2D projection
    print("\nTest 2: 2D projections")
    for n_points in [10, 50, 100]:
        bounds = np.array([[-5.0, 5.0], [-3.0, 3.0]])
        projection_dims = [0, 1]
        grid_axes = [
            np.linspace(bounds[0, 0], bounds[0, 1], n_points),
            np.linspace(bounds[1, 0], bounds[1, 1], n_points)
        ]

        # Test with random points
        np.random.seed(123)
        test_points = np.random.uniform([bounds[0, 0], bounds[1, 0]],
                                       [bounds[0, 1], bounds[1, 1]],
                                       (100, 2))

        mismatches = 0
        for point in test_points:
            old = old_implementation(point, projection_dims, grid_axes)
            new = new_implementation(point, projection_dims, grid_axes)
            if old != new:
                mismatches += 1

        status = "✓ PASS" if mismatches == 0 else f"✗ FAIL ({mismatches} mismatches)"
        print(f"  Grid size {n_points}x{n_points}: {status}")
        if mismatches > 0:
            all_passed = False

    # Test 3: 4D projection (like Himmelblau example)
    print("\nTest 3: 4D projection")
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    projection_dims = [0, 1]
    grid_axes = [
        np.linspace(bounds[0, 0], bounds[0, 1], 100),
        np.linspace(bounds[1, 0], bounds[1, 1], 100)
    ]

    # Test with random points (only first 2 dims matter for projection)
    np.random.seed(456)
    test_points = np.random.uniform([bounds[i, 0] for i in range(4)],
                                   [bounds[i, 1] for i in range(4)],
                                   (200, 4))

    mismatches = 0
    for point in test_points:
        old = old_implementation(point, projection_dims, grid_axes)
        new = new_implementation(point, projection_dims, grid_axes)
        if old != new:
            mismatches += 1

    status = "✓ PASS" if mismatches == 0 else f"✗ FAIL ({mismatches} mismatches)"
    print(f"  4D space, 2D projection (100x100 grid): {status}")
    if mismatches > 0:
        all_passed = False

    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    bounds = np.array([[-10.0, 10.0]])
    projection_dims = [0]
    grid_axes = [np.linspace(bounds[0, 0], bounds[0, 1], 50)]

    edge_cases = [
        (np.array([bounds[0, 0]]), "Lower boundary"),
        (np.array([bounds[0, 1]]), "Upper boundary"),
        (np.array([0.0]), "Midpoint"),
        (np.array([bounds[0, 0] - 1.0]), "Below bounds"),
        (np.array([bounds[0, 1] + 1.0]), "Above bounds"),
    ]

    for point, desc in edge_cases:
        old = old_implementation(point, projection_dims, grid_axes)
        new = new_implementation(point, projection_dims, grid_axes)
        match = old == new
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  {desc:20s}: {status}")
        if not match:
            print(f"    Old: {old}, New: {new}")
            all_passed = False

    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance comparison:")
    print("=" * 60)

    import time

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    projection_dims = [0, 1]
    n_points = 100
    grid_axes = [
        np.linspace(bounds[0, 0], bounds[0, 1], n_points),
        np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    ]

    # Generate many test points (simulating warm-start scenario)
    np.random.seed(789)
    test_points = np.random.uniform([bounds[0, 0], bounds[1, 0]],
                                   [bounds[0, 1], bounds[1, 1]],
                                   (10000, 2))

    # Time old implementation
    start = time.time()
    for point in test_points:
        old_implementation(point, projection_dims, grid_axes)
    old_time = time.time() - start

    # Time new implementation
    start = time.time()
    for point in test_points:
        new_implementation(point, projection_dims, grid_axes)
    new_time = time.time() - start

    speedup = old_time / new_time

    print(f"Old implementation: {old_time:.4f}s (10,000 points, 100x100 grid)")
    print(f"New implementation: {new_time:.4f}s (10,000 points, 100x100 grid)")
    print(f"Speedup: {speedup:.1f}x")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All verification tests PASSED!")
        print("The optimized implementation is correct and compatible.")
    else:
        print("✗ Some verification tests FAILED!")
        print("Please review the implementation.")
    print("=" * 60)

    return all_passed

if __name__ == "__main__":
    success = run_verification_tests()
    exit(0 if success else 1)
