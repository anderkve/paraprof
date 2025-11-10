"""
Diagnostic test to reproduce the 2D function crash issue.
"""
import numpy as np
import sys

# Test what happens with 0 continuous dimensions
print("="*70)
print("DIAGNOSING 2D FUNCTION CRASH")
print("="*70)

print("\n1. Problem: Projecting onto ALL dimensions of a 2D function")
print("-" * 70)

dims = 2
projection_dims = [0, 1]
continuous_dims = [d for d in range(dims) if d not in projection_dims]

print(f"Function dimensions: {dims}")
print(f"Projection dimensions: {projection_dims}")
print(f"Continuous dimensions: {continuous_dims}")
print(f"Number of continuous dims: {len(continuous_dims)}")

print("\n2. What breaks with 0 continuous dimensions:")
print("-" * 70)

# Latin Hypercube Sampling
try:
    from scipy.stats.qmc import LatinHypercube as LHS
    n_cont_dims = len(continuous_dims)
    print(f"   - LHS(d={n_cont_dims}): ", end="")
    lhs_sampler = LHS(d=n_cont_dims)
    print("Creates empty sampler (may fail or give unexpected behavior)")
    samples = lhs_sampler.random(n=10)
    print(f"     Samples shape: {samples.shape}")
except Exception as e:
    print(f"FAILS with error: {e}")

# Continuous parameter arrays
print(f"   - continuous_params arrays would have shape (n, 0)")
print(f"   - No dimensions to optimize at each grid point!")

# Construct params
print(f"   - _construct_params would assign empty array to continuous_dims")

print("\n3. Root Cause:")
print("-" * 70)
print("   ParaProf is designed for PROFILE LIKELIHOOD projections:")
print("   - Fix some dimensions on a grid (projection_dims)")
print("   - Optimize over remaining dimensions (continuous_dims)")
print("   - For 2D functions, you CANNOT project onto BOTH dimensions")
print("   - You must leave at least 1 dimension as 'continuous' to optimize")

print("\n4. Solution:")
print("-" * 70)
print("   For 2D functions:")
print("   - Use 1D projections: dims=[0] or dims=[1]")
print("   - This creates a 1D grid, optimizing the other dimension at each point")
print("   ")
print("   Example configurations:")
print("   - 2D function: dims=[0] (1D projection on dimension 0)")
print("   - 2D function: dims=[1] (1D projection on dimension 1)")
print("   - 4D function: dims=[0, 1] (2D projection on dimensions 0 and 1)")
print("   - 4D function: dims=[0, 2, 3] (3D projection)")

print("\n5. Valid projection configurations:")
print("-" * 70)

test_cases = [
    (2, [0], "✓ Valid: 1D projection in 2D space"),
    (2, [1], "✓ Valid: 1D projection in 2D space"),
    (2, [0, 1], "✗ Invalid: No continuous dims left!"),
    (4, [0], "✓ Valid: 1D projection in 4D space"),
    (4, [0, 1], "✓ Valid: 2D projection in 4D space"),
    (4, [0, 1, 2], "✓ Valid: 3D projection in 4D space"),
    (4, [0, 1, 2, 3], "✗ Invalid: No continuous dims left!"),
    (6, [0, 1], "✓ Valid: 2D projection in 6D space"),
    (6, [0, 1, 2, 3, 4], "✓ Valid: 5D projection in 6D space"),
    (6, [0, 1, 2, 3, 4, 5], "✗ Invalid: No continuous dims left!"),
]

print(f"{'Dims':<6s} {'Projection':<20s} {'Continuous':<15s} {'Status':<40s}")
print("-" * 70)
for total_dims, proj_dims, status in test_cases:
    cont_dims = [d for d in range(total_dims) if d not in proj_dims]
    print(f"{total_dims:<6d} {str(proj_dims):<20s} {str(cont_dims):<15s} {status:<40s}")

print("\n" + "="*70)
print("CONSTRAINT: len(projection_dims) < total_dimensions")
print("           (must have at least 1 continuous dimension)")
print("="*70)
