"""
Test script to verify patching flag configuration behavior.
"""
import numpy as np
from sampler import GridAnchoredDESampler

def dummy_func(x):
    return -np.sum(x**2)

bounds = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]]

print("Testing patching flag configurations...")
print("="*80)

# Test 1: New flags - patching_coarse only
print("\nTest 1: New flags with patching_coarse=True, patching_refined=False")
proj1 = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching_coarse': True, 'patching_refined': False}]
sampler1 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj1)
assert sampler1.patching_coarse == True, "patching_coarse should be True"
assert sampler1.patching_refined == False, "patching_refined should be False"
print(f"  PASS: patching_coarse={sampler1.patching_coarse}, patching_refined={sampler1.patching_refined}")

# Test 2: New flags - patching_refined only
print("\nTest 2: New flags with patching_coarse=False, patching_refined=True")
proj2 = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching_coarse': False, 'patching_refined': True}]
sampler2 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj2)
assert sampler2.patching_coarse == False, "patching_coarse should be False"
assert sampler2.patching_refined == True, "patching_refined should be True"
print(f"  PASS: patching_coarse={sampler2.patching_coarse}, patching_refined={sampler2.patching_refined}")

# Test 3: New flags - both enabled
print("\nTest 3: New flags with both patching_coarse=True and patching_refined=True")
proj3 = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching_coarse': True, 'patching_refined': True}]
sampler3 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj3)
assert sampler3.patching_coarse == True, "patching_coarse should be True"
assert sampler3.patching_refined == True, "patching_refined should be True"
print(f"  PASS: patching_coarse={sampler3.patching_coarse}, patching_refined={sampler3.patching_refined}")

# Test 4: Legacy flag - patching=True (backward compatibility)
print("\nTest 4: Legacy flag with patching=True (should set patching_coarse=True, patching_refined=False)")
proj4 = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching': True}]
sampler4 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj4)
assert sampler4.patching_coarse == True, "patching_coarse should be True for backward compatibility"
assert sampler4.patching_refined == False, "patching_refined should be False for backward compatibility"
print(f"  PASS: patching_coarse={sampler4.patching_coarse}, patching_refined={sampler4.patching_refined}")

# Test 5: Legacy flag - patching=False (backward compatibility)
print("\nTest 5: Legacy flag with patching=False (should set patching_coarse=False, patching_refined=False)")
proj5 = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching': False}]
sampler5 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj5)
assert sampler5.patching_coarse == False, "patching_coarse should be False for backward compatibility"
assert sampler5.patching_refined == False, "patching_refined should be False for backward compatibility"
print(f"  PASS: patching_coarse={sampler5.patching_coarse}, patching_refined={sampler5.patching_refined}")

# Test 6: Default behavior (no flags specified)
print("\nTest 6: No flags specified (should use defaults: patching_coarse=True, patching_refined=False)")
proj6 = [{'dims': [0, 1], 'grid_points': [10, 10]}]
sampler6 = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj6)
assert sampler6.patching_coarse == True, "patching_coarse should default to True"
assert sampler6.patching_refined == False, "patching_refined should default to False"
print(f"  PASS: patching_coarse={sampler6.patching_coarse}, patching_refined={sampler6.patching_refined}")

print("\n" + "="*80)
print("All tests passed!")
