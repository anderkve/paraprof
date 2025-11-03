"""
Simple unit test for global pool methods (no MPI required).
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler import GridAnchoredDESampler
from test_functions import get_test_function

print("="*80)
print("Unit Test: Global Solution Pool Methods")
print("="*80)

# Get test function
log_likelihood, param_bounds, _ = get_test_function("himmelblau_4d")

# Create sampler
PROJECTIONS = [{'dims': [0, 1], 'grid_points': [5, 5], 'patching': False, 'lbfgsb': True}]

sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=PROJECTIONS,
    pop_per_grid_point=10,
    global_pool_size=20,
    activation_mix_ratios={'neighbors': 0.5, 'global': 0.3, 'random': 0.2}
)

print("\n1. Testing initialization...")
assert len(sampler.global_solution_pool) == 0, "Pool should start empty"
assert sampler.global_pool_size == 20, "Pool size should be 20"
assert sampler.activation_mix_ratios['neighbors'] == 0.5, "Neighbor ratio should be 0.5"
print("   ✓ Initialization successful")

print("\n2. Testing _update_global_pool...")
# Add some test solutions
for i in range(25):  # Add more than pool_size to test truncation
    full_params = np.random.randn(4)  # 4 total dims
    fitness = float(i)  # Increasing fitness
    grid_idx = (i % 5, i % 5)
    sampler._update_global_pool(full_params, fitness, grid_idx)

assert len(sampler.global_solution_pool) == 20, f"Pool should be capped at 20, got {len(sampler.global_solution_pool)}"
# Check that pool is sorted by fitness (descending)
for i in range(len(sampler.global_solution_pool) - 1):
    assert sampler.global_solution_pool[i]['fitness'] >= sampler.global_solution_pool[i+1]['fitness'], \
        "Pool should be sorted by fitness (descending)"
# Check that best solutions are kept
assert sampler.global_solution_pool[0]['fitness'] == 24.0, "Best solution should have fitness 24"
assert sampler.global_solution_pool[-1]['fitness'] == 5.0, "Worst kept solution should have fitness 5"
print("   ✓ Pool update and truncation working correctly")

print("\n3. Testing _sample_from_global_pool...")
# Test sampling (should extract continuous dims from full params)
# For projection [0, 1], continuous dims are [2, 3]
samples = sampler._sample_from_global_pool(5)
assert samples is not None, "Should return samples when pool is not empty"
assert samples.shape == (5, 2), f"Should return 5 samples of 2 continuous dims each, got shape {samples.shape}"
# Verify that we're extracting the right dimensions
for sample in samples:
    assert len(sample) == 2, "Each sample should have 2 continuous dims"
print("   ✓ Sampling and continuous dim extraction working correctly")

# Test sampling from empty pool
empty_sampler = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=PROJECTIONS,
    pop_per_grid_point=10
)
samples = empty_sampler._sample_from_global_pool(5)
assert samples is None, "Should return None when pool is empty"
print("   ✓ Empty pool handling working correctly")

print("\n4. Testing export/import with refinement...")
# Add solution to pool (full params)
sampler._update_global_pool(np.array([1.0, 2.0, 3.0, 4.0]), 100.0, (0, 0))

# Export
exported = sampler.export_grid_solution()
assert 'global_solution_pool' in exported, "Export should include global_solution_pool"
assert len(exported['global_solution_pool']) > 0, "Exported pool should not be empty"
print("   ✓ Export working correctly")

# Setup refinement (this should restore the pool)
sampler2 = GridAnchoredDESampler(
    target_func=log_likelihood,
    bounds=param_bounds,
    projections=[{'dims': [0, 1], 'grid_points': [10, 10], 'patching': False, 'lbfgsb': True}],
    pop_per_grid_point=10
)
assert len(sampler2.global_solution_pool) == 0, "New sampler should start with empty pool"

sampler2.setup_refinement_run(exported, refinement_factor=2)
assert len(sampler2.global_solution_pool) > 0, "Pool should be restored from coarse run"
assert len(sampler2.global_solution_pool) == len(exported['global_solution_pool']), \
    "Restored pool should have same size as exported pool"
print("   ✓ Refinement import working correctly")

print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
