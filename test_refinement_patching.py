"""
Test script to verify patching flag behavior during refinement runs.
"""
import numpy as np
from sampler import GridAnchoredDESampler

def dummy_func(x):
    return -np.sum(x**2)

bounds = [[-5, 5], [-5, 5], [-5, 5], [-5, 5]]

print("Testing patching flags during refinement runs...")
print("="*80)

# Test refinement mode with patching_refined=True
print("\nTest: Refinement mode with patching_refined=True")
proj = [{'dims': [0, 1], 'grid_points': [10, 10], 'patching_coarse': False, 'patching_refined': True}]
sampler = GridAnchoredDESampler(target_func=dummy_func, bounds=bounds, projections=proj)

# Create a mock coarse solution
coarse_solution = {
    'grid_axes': [np.linspace(-5, 5, 11), np.linspace(-5, 5, 11)],
    'projection_dims': [0, 1],
    'continuous_dims': [2, 3],
    'solutions': {},
    'grid_shape': (11, 11),
    'global_solution_pool': []
}

# Setup refinement run
sampler.setup_refinement_run(coarse_solution, refinement_factor=2)

# Create refined projection config
refined_config = {
    'dims': [0, 1],
    'grid_points': [20, 20],
    'patching_coarse': False,
    'patching_refined': True
}

# Reset for refinement projection (this should print the patching_refined status)
print("\nResetting for refinement projection (should show patching_refined status):")
sampler._reset_for_new_projection(refined_config)

print(f"\nFinal state: is_refinement_run={sampler.is_refinement_run}")
print(f"             patching_coarse={sampler.patching_coarse}")
print(f"             patching_refined={sampler.patching_refined}")

assert sampler.is_refinement_run == True, "Should be in refinement run mode"
assert sampler.patching_coarse == False, "patching_coarse should be False"
assert sampler.patching_refined == True, "patching_refined should be True"

print("\n" + "="*80)
print("Refinement patching test passed!")
