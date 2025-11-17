"""
Nuisance parameter wrapper for test functions.

This module provides functionality to augment existing test functions with
constrained nuisance parameters, mimicking realistic physics scenarios where:
- A few parameters of interest (POI) have complex likelihood surfaces
- Many nuisance parameters are tightly constrained by Gaussian penalties
- Nuisance parameters couple to the main likelihood (not just additive)

The wrapper supports multiple coupling modes representing different types of
systematic uncertainties common in physics analyses.
"""
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union


class NuisanceParameterWrapper:
    """
    Wraps a base test function with constrained nuisance parameters.

    The augmented likelihood has the form:
        log L_total = log L_base(transformed_POI) + Σ log L_constraint(nuisance_i)

    where the transformation of POI depends on nuisance parameters according
    to the specified coupling mode.

    Parameter Ordering
    ------------------
    The full parameter vector is: [poi_0, ..., poi_n, nuis_0, ..., nuis_m]
    - First n_poi parameters: parameters of interest
    - Last n_nuisance parameters: nuisance parameters

    Examples
    --------
    >>> # Wrap Himmelblau 4D with 8 shift-type nuisance parameters
    >>> from paraprof import get_test_function
    >>> base_func, base_bounds, peaks = get_test_function("himmelblau_4d")
    >>>
    >>> wrapped_func, wrapped_bounds = create_nuisance_wrapped_function(
    ...     base_func=base_func,
    ...     base_bounds=base_bounds,
    ...     n_poi=4,
    ...     n_nuisance=8,
    ...     coupling_mode='shift',
    ...     constraint_sigma=0.5,
    ...     nuisance_mean=0.0
    ... )
    >>>
    >>> # Now wrapped_func accepts 12D input: [x0, x1, x2, x3, d0, d1, d2, d3, d4, d5, d6, d7]
    >>> # The base function sees: [x0+d0+d1, x1+d2+d3, x2+d4+d5, x3+d6+d7]
    >>> result = wrapped_func(np.zeros(12))
    """

    def __init__(
        self,
        base_func: Callable,
        n_poi: int,
        n_nuisance: int,
        coupling_mode: str = 'shift',
        coupling_matrix: Optional[np.ndarray] = None,
        constraint_sigma: Union[float, np.ndarray] = 1.0,
        nuisance_mean: Union[float, np.ndarray] = 0.0,
        constraint_mode: str = 'gaussian',
        rotation_scale: float = 0.1
    ):
        """
        Initialize the nuisance parameter wrapper.

        Parameters
        ----------
        base_func : callable
            Base test function accepting n_poi-dimensional input
        n_poi : int
            Number of parameters of interest (base function dimensionality)
        n_nuisance : int
            Number of nuisance parameters to add
        coupling_mode : str, optional
            How nuisance parameters couple to POI. Options:
            - 'shift': Additive shifts (x_i → x_i + Σ d_j)
            - 'scale': Multiplicative scaling (x_i → x_i * (1 + Σ s_j))
            - 'rotation': Small rotations in POI space
            - 'additive': No coupling, just additive penalty term
            - 'mixed': Use coupling_matrix for custom linear combinations
            Default: 'shift'
        coupling_matrix : np.ndarray, optional
            Custom (n_poi, n_nuisance) matrix defining how nuisance parameters
            affect POI. If None, uses default patterns based on coupling_mode.
            Element (i,j) determines how nuisance_j affects poi_i.
        constraint_sigma : float or np.ndarray, optional
            Standard deviation(s) for Gaussian constraints on nuisance parameters.
            Single float applies to all, array specifies per-parameter.
            Smaller values = tighter constraints.
            Default: 1.0
        nuisance_mean : float or np.ndarray, optional
            Central value(s) for nuisance parameter constraints.
            Default: 0.0 (constrains nuisance params to be near zero)
        constraint_mode : str, optional
            Type of constraint. Options:
            - 'gaussian': Standard Gaussian penalty -0.5*((x-μ)/σ)²
            - 'uniform': Flat within ±σ, -inf outside
            - 'soft_uniform': Flat within ±σ, quadratic penalty outside
            Default: 'gaussian'
        rotation_scale : float, optional
            Angular scale for rotation coupling mode (in radians).
            Only used if coupling_mode='rotation'.
            Default: 0.1
        """
        self.base_func = base_func
        self.n_poi = n_poi
        self.n_nuisance = n_nuisance
        self.n_total = n_poi + n_nuisance
        self.coupling_mode = coupling_mode
        self.constraint_mode = constraint_mode
        self.rotation_scale = rotation_scale

        # Handle scalar or array constraint parameters
        if np.isscalar(constraint_sigma):
            self.constraint_sigma = np.full(n_nuisance, float(constraint_sigma))
        else:
            self.constraint_sigma = np.asarray(constraint_sigma)
            if len(self.constraint_sigma) != n_nuisance:
                raise ValueError(
                    f"constraint_sigma length {len(self.constraint_sigma)} "
                    f"doesn't match n_nuisance {n_nuisance}"
                )

        if np.isscalar(nuisance_mean):
            self.nuisance_mean = np.full(n_nuisance, float(nuisance_mean))
        else:
            self.nuisance_mean = np.asarray(nuisance_mean)
            if len(self.nuisance_mean) != n_nuisance:
                raise ValueError(
                    f"nuisance_mean length {len(self.nuisance_mean)} "
                    f"doesn't match n_nuisance {n_nuisance}"
                )

        # Set up coupling matrix
        if coupling_matrix is not None:
            self.coupling_matrix = np.asarray(coupling_matrix)
            if self.coupling_matrix.shape != (n_poi, n_nuisance):
                raise ValueError(
                    f"coupling_matrix shape {self.coupling_matrix.shape} "
                    f"doesn't match (n_poi={n_poi}, n_nuisance={n_nuisance})"
                )
        else:
            self.coupling_matrix = self._create_default_coupling_matrix()

    def _create_default_coupling_matrix(self) -> np.ndarray:
        """
        Create default coupling matrix based on coupling_mode.

        Returns
        -------
        coupling_matrix : np.ndarray, shape (n_poi, n_nuisance)
            Default coupling pattern
        """
        if self.coupling_mode == 'additive':
            # No coupling
            return np.zeros((self.n_poi, self.n_nuisance))

        elif self.coupling_mode in ['shift', 'scale']:
            # Distribute nuisance parameters across POI dimensions
            # Each POI gets approximately n_nuisance/n_poi nuisance params
            matrix = np.zeros((self.n_poi, self.n_nuisance))

            # Strategy: assign nuisance parameters cyclically or in blocks
            # For example, if n_poi=4, n_nuisance=8:
            # POI 0 gets nuisance 0, 1
            # POI 1 gets nuisance 2, 3
            # POI 2 gets nuisance 4, 5
            # POI 3 gets nuisance 6, 7

            nuis_per_poi = self.n_nuisance // self.n_poi
            remainder = self.n_nuisance % self.n_poi

            nuis_idx = 0
            for poi_idx in range(self.n_poi):
                # Assign base number of nuisance params
                n_assign = nuis_per_poi
                # Distribute remainder to first few POI
                if poi_idx < remainder:
                    n_assign += 1

                for _ in range(n_assign):
                    if nuis_idx < self.n_nuisance:
                        matrix[poi_idx, nuis_idx] = 1.0
                        nuis_idx += 1

            return matrix

        elif self.coupling_mode == 'rotation':
            # For rotation, we need at least n_poi*(n_poi-1)/2 nuisance params
            # for independent rotation angles in all planes
            # For simplicity, cycle through POI pairs
            matrix = np.zeros((self.n_poi, self.n_nuisance))

            # Simple approach: each nuisance parameter rotates in a specific plane
            # This is a placeholder - actual rotation is handled in _transform_poi
            nuis_idx = 0
            for i in range(self.n_poi):
                for j in range(i+1, self.n_poi):
                    if nuis_idx < self.n_nuisance:
                        # Mark which POI dimensions are involved in this rotation
                        matrix[i, nuis_idx] = 1.0
                        matrix[j, nuis_idx] = 1.0
                        nuis_idx += 1

            return matrix

        elif self.coupling_mode == 'mixed':
            # Random sparse coupling pattern
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            matrix = rng.randn(self.n_poi, self.n_nuisance) * 0.3
            # Sparsify: keep only ~30% of connections
            mask = rng.rand(self.n_poi, self.n_nuisance) > 0.7
            matrix[mask] = 0.0
            return matrix

        else:
            raise ValueError(f"Unknown coupling_mode: {self.coupling_mode}")

    def _transform_poi(self, poi_values: np.ndarray, nuisance_values: np.ndarray) -> np.ndarray:
        """
        Transform POI values based on nuisance parameters.

        Parameters
        ----------
        poi_values : np.ndarray, shape (n_poi,)
            Original parameter of interest values
        nuisance_values : np.ndarray, shape (n_nuisance,)
            Nuisance parameter values

        Returns
        -------
        transformed_poi : np.ndarray, shape (n_poi,)
            Transformed POI values to pass to base function
        """
        if self.coupling_mode == 'additive':
            # No transformation
            return poi_values

        elif self.coupling_mode == 'shift':
            # Additive shifts: x_i → x_i + Σ_j M_ij * d_j
            shifts = self.coupling_matrix @ nuisance_values
            return poi_values + shifts

        elif self.coupling_mode == 'scale':
            # Multiplicative scaling: x_i → x_i * (1 + Σ_j M_ij * s_j)
            scale_factors = 1.0 + self.coupling_matrix @ nuisance_values
            return poi_values * scale_factors

        elif self.coupling_mode == 'rotation':
            # Apply small rotations in parameter space
            # Each nuisance parameter controls a rotation in a specific plane

            transformed = poi_values.copy()
            nuis_idx = 0

            # Apply rotations plane by plane
            for i in range(self.n_poi):
                for j in range(i+1, self.n_poi):
                    if nuis_idx < self.n_nuisance:
                        # Rotation angle scaled by nuisance parameter
                        angle = self.rotation_scale * nuisance_values[nuis_idx]

                        # Rotate in plane (i,j)
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        x_i = transformed[i]
                        x_j = transformed[j]
                        transformed[i] = cos_a * x_i - sin_a * x_j
                        transformed[j] = sin_a * x_i + cos_a * x_j

                        nuis_idx += 1

            return transformed

        elif self.coupling_mode == 'mixed':
            # Linear combination: x_i → x_i + Σ_j M_ij * d_j
            return poi_values + self.coupling_matrix @ nuisance_values

        else:
            raise ValueError(f"Unknown coupling_mode: {self.coupling_mode}")

    def _compute_constraint_penalty(self, nuisance_values: np.ndarray) -> float:
        """
        Compute log-likelihood penalty for nuisance parameter constraints.

        Parameters
        ----------
        nuisance_values : np.ndarray, shape (n_nuisance,)
            Nuisance parameter values

        Returns
        -------
        log_penalty : float
            Log-likelihood contribution from nuisance constraints
        """
        deviations = (nuisance_values - self.nuisance_mean) / self.constraint_sigma

        if self.constraint_mode == 'gaussian':
            # Standard Gaussian: -0.5 * Σ ((x - μ) / σ)²
            return -0.5 * np.sum(deviations**2)

        elif self.constraint_mode == 'uniform':
            # Uniform within ±1σ, -inf outside
            if np.all(np.abs(deviations) <= 1.0):
                return 0.0
            else:
                return -np.inf

        elif self.constraint_mode == 'soft_uniform':
            # Flat within ±1σ, quadratic penalty outside
            penalty = 0.0
            for dev in deviations:
                if abs(dev) <= 1.0:
                    penalty += 0.0
                else:
                    penalty += -0.5 * (abs(dev) - 1.0)**2
            return penalty

        else:
            raise ValueError(f"Unknown constraint_mode: {self.constraint_mode}")

    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate the augmented likelihood function.

        Parameters
        ----------
        params : np.ndarray, shape (n_total,)
            Full parameter vector: [poi_0, ..., poi_n, nuis_0, ..., nuis_m]

        Returns
        -------
        log_likelihood : float
            Total log-likelihood including base function and constraints
        """
        if len(params) != self.n_total:
            raise ValueError(
                f"Expected {self.n_total} parameters, got {len(params)}"
            )

        # Split parameter vector
        poi_values = params[:self.n_poi]
        nuisance_values = params[self.n_poi:]

        # Transform POI based on nuisance parameters
        transformed_poi = self._transform_poi(poi_values, nuisance_values)

        # Evaluate base likelihood
        base_log_likelihood = self.base_func(transformed_poi)

        # Add constraint penalty
        constraint_penalty = self._compute_constraint_penalty(nuisance_values)

        return base_log_likelihood + constraint_penalty

    def get_optimal_nuisance(self, poi_values: np.ndarray) -> np.ndarray:
        """
        Get analytically optimal nuisance parameter values for given POI.

        For most coupling modes, the optimal nuisance values are simply
        the constraint means (typically zero), since moving away from them
        incurs a penalty without improving the base likelihood.

        This is useful for validation and understanding the profile structure.

        Parameters
        ----------
        poi_values : np.ndarray, shape (n_poi,)
            Parameter of interest values

        Returns
        -------
        optimal_nuisance : np.ndarray, shape (n_nuisance,)
            Optimal nuisance parameter values for this POI configuration
        """
        # For most coupling modes, optimal nuisance = constraint mean
        # (no benefit to moving away from the constrained value)
        return self.nuisance_mean.copy()

    def profile_over_nuisance(
        self,
        poi_values: np.ndarray,
        method: str = 'analytical'
    ) -> Tuple[float, np.ndarray]:
        """
        Compute the profile likelihood at given POI by optimizing nuisance params.

        Parameters
        ----------
        poi_values : np.ndarray, shape (n_poi,)
            Fixed parameter of interest values
        method : str, optional
            Optimization method: 'analytical' (use get_optimal_nuisance) or
            'numerical' (use scipy.optimize). Default: 'analytical'

        Returns
        -------
        profile_log_likelihood : float
            Maximum log-likelihood with nuisance params optimized
        optimal_nuisance : np.ndarray, shape (n_nuisance,)
            Nuisance parameter values at the optimum
        """
        if method == 'analytical':
            optimal_nuis = self.get_optimal_nuisance(poi_values)
            full_params = np.concatenate([poi_values, optimal_nuis])
            profile_ll = self(full_params)
            return profile_ll, optimal_nuis

        elif method == 'numerical':
            from scipy.optimize import minimize

            def objective(nuisance_vals):
                full_params = np.concatenate([poi_values, nuisance_vals])
                return -self(full_params)  # Minimize negative log-likelihood

            # Start from constraint means
            x0 = self.nuisance_mean.copy()

            # Optimize
            result = minimize(objective, x0, method='L-BFGS-B')

            if result.success:
                optimal_nuis = result.x
                profile_ll = -result.fun
                return profile_ll, optimal_nuis
            else:
                # Fallback to analytical
                return self.profile_over_nuisance(poi_values, method='analytical')

        else:
            raise ValueError(f"Unknown method: {method}")


def create_nuisance_wrapped_function(
    base_func: Callable,
    base_bounds: List[List[float]],
    n_poi: int,
    n_nuisance: int,
    coupling_mode: str = 'shift',
    coupling_matrix: Optional[np.ndarray] = None,
    constraint_sigma: Union[float, np.ndarray] = 1.0,
    nuisance_mean: Union[float, np.ndarray] = 0.0,
    nuisance_bounds_sigma_multiple: float = 5.0,
    **wrapper_kwargs
) -> Tuple[Callable, List[List[float]], NuisanceParameterWrapper]:
    """
    Convenience function to create a nuisance-wrapped function with bounds.

    Parameters
    ----------
    base_func : callable
        Base test function
    base_bounds : list of [min, max]
        Bounds for POI parameters
    n_poi : int
        Number of parameters of interest
    n_nuisance : int
        Number of nuisance parameters
    coupling_mode : str, optional
        Coupling mode (see NuisanceParameterWrapper)
    coupling_matrix : np.ndarray, optional
        Custom coupling matrix
    constraint_sigma : float or array, optional
        Constraint width(s)
    nuisance_mean : float or array, optional
        Constraint center(s)
    nuisance_bounds_sigma_multiple : float, optional
        Set nuisance parameter bounds to mean ± this_factor * sigma
        Default: 5.0 (covers 99.999% for Gaussian)
    **wrapper_kwargs
        Additional arguments passed to NuisanceParameterWrapper

    Returns
    -------
    wrapped_func : callable
        Wrapped function accepting (n_poi + n_nuisance) parameters
    wrapped_bounds : list of [min, max]
        Bounds for full parameter vector
    wrapper : NuisanceParameterWrapper
        The wrapper object (useful for accessing methods like profile_over_nuisance)

    Examples
    --------
    >>> base_func, base_bounds, _ = get_test_function("himmelblau_4d")
    >>> func, bounds, wrapper = create_nuisance_wrapped_function(
    ...     base_func, base_bounds, n_poi=4, n_nuisance=8,
    ...     coupling_mode='shift', constraint_sigma=0.5
    ... )
    >>> print(f"Function accepts {len(bounds)} parameters")
    >>> result = func(np.random.randn(12))
    """
    # Create wrapper
    wrapper = NuisanceParameterWrapper(
        base_func=base_func,
        n_poi=n_poi,
        n_nuisance=n_nuisance,
        coupling_mode=coupling_mode,
        coupling_matrix=coupling_matrix,
        constraint_sigma=constraint_sigma,
        nuisance_mean=nuisance_mean,
        **wrapper_kwargs
    )

    # Construct bounds for full parameter vector
    wrapped_bounds = list(base_bounds[:n_poi])  # POI bounds

    # Add nuisance parameter bounds
    sigma_array = wrapper.constraint_sigma
    mean_array = wrapper.nuisance_mean

    for i in range(n_nuisance):
        nuis_range = nuisance_bounds_sigma_multiple * sigma_array[i]
        nuis_min = mean_array[i] - nuis_range
        nuis_max = mean_array[i] + nuis_range
        wrapped_bounds.append([nuis_min, nuis_max])

    return wrapper, wrapped_bounds, wrapper


def register_nuisance_wrapped_test_functions():
    """
    Register common nuisance-wrapped test functions for easy access.

    This creates a set of standard test cases with varying numbers of
    nuisance parameters and coupling modes.

    Returns
    -------
    registry : dict
        Dictionary mapping names to (func, bounds, wrapper, base_peaks)

    Examples
    --------
    >>> registry = register_nuisance_wrapped_test_functions()
    >>> func, bounds, wrapper, peaks = registry['himmelblau_4d_shift_8nuis']
    """
    from .test_functions import get_test_function

    registry = {}

    # Configuration: (base_name, n_nuisance, coupling_mode, constraint_sigma)
    configs = [
        # Himmelblau 4D variations
        ("himmelblau_4d", 4, "shift", 0.5),
        ("himmelblau_4d", 8, "shift", 0.5),
        ("himmelblau_4d", 16, "shift", 0.5),
        ("himmelblau_4d", 8, "shift", 0.2),  # Tighter constraint
        ("himmelblau_4d", 8, "shift", 1.0),  # Looser constraint
        ("himmelblau_4d", 8, "scale", 0.1),

        # Rosenbrock 4D variations
        ("rosenbrock_4d", 8, "shift", 0.5),
        ("rosenbrock_4d", 16, "shift", 0.5),
        ("rosenbrock_4d", 8, "scale", 0.1),

        # Rastrigin 4D variations (multimodal)
        ("rastrigin_4d", 8, "shift", 0.5),
        ("rastrigin_4d", 16, "shift", 0.5),

        # Higher dimensional cases
        ("himmelblau_4d", 32, "shift", 0.5),  # Many nuisance params
        ("rosenbrock_6d", 12, "shift", 0.5),
        ("sphere_10d", 20, "shift", 0.5),
    ]

    for base_name, n_nuis, coupling, sigma in configs:
        base_func, base_bounds, base_peaks = get_test_function(base_name)
        n_poi = len(base_bounds)

        func, bounds, wrapper = create_nuisance_wrapped_function(
            base_func=base_func,
            base_bounds=base_bounds,
            n_poi=n_poi,
            n_nuisance=n_nuis,
            coupling_mode=coupling,
            constraint_sigma=sigma
        )

        # Create descriptive name
        registry_name = f"{base_name}_{coupling}_{n_nuis}nuis_sigma{sigma}"
        registry[registry_name] = (func, bounds, wrapper, base_peaks)

    return registry
