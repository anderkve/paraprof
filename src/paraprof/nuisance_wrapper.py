"""Nuisance parameter wrapper for test functions.

Augments an existing test function with constrained nuisance parameters
to mimic realistic physics scenarios where a few parameters of interest
(POI) drive a complex likelihood and many nuisance parameters are tightly
constrained and couple to the POIs.
"""
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple, Union


class NuisanceParameterWrapper:
    """Wrap a base test function with constrained nuisance parameters.

    Augmented likelihood:
        log L_total = log L_base(transformed_POI) + Σ log L_constraint(nuisance_i)

    Parameter vector is ordered as
    ``[poi_0, ..., poi_{n_poi-1}, nuis_0, ..., nuis_{n_nuisance-1}]``.

    ``coupling_mode`` selects how nuisance parameters transform the POIs:
    ``'shift'`` (additive), ``'scale'`` (multiplicative), ``'rotation'``
    (small rotations in POI space, controlled by ``rotation_scale``),
    ``'additive'`` (no coupling — penalty term only), or ``'mixed'``
    (custom ``coupling_matrix``). ``constraint_sigma`` and
    ``nuisance_mean`` accept a scalar or per-parameter array.
    ``constraint_mode`` is ``'gaussian'``, ``'uniform'`` (flat within ±σ,
    -inf outside), or ``'soft_uniform'`` (flat within ±σ, quadratic
    outside).
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
        self.base_func = base_func
        self.n_poi = n_poi
        self.n_nuisance = n_nuisance
        self.n_total = n_poi + n_nuisance
        self.coupling_mode = coupling_mode
        self.constraint_mode = constraint_mode
        self.rotation_scale = rotation_scale

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
        """Default (n_poi, n_nuisance) coupling matrix for the active mode."""
        if self.coupling_mode == 'additive':
            return np.zeros((self.n_poi, self.n_nuisance))

        elif self.coupling_mode in ['shift', 'scale']:
            # Distribute nuisance params across POIs in roughly equal blocks
            # (POI 0 gets the first chunk, POI 1 the next, etc.; leftover
            # nuisance params from non-divisible counts go to the first POIs).
            matrix = np.zeros((self.n_poi, self.n_nuisance))
            nuis_per_poi = self.n_nuisance // self.n_poi
            remainder = self.n_nuisance % self.n_poi
            nuis_idx = 0
            for poi_idx in range(self.n_poi):
                n_assign = nuis_per_poi + (1 if poi_idx < remainder else 0)
                for _ in range(n_assign):
                    if nuis_idx < self.n_nuisance:
                        matrix[poi_idx, nuis_idx] = 1.0
                        nuis_idx += 1
            return matrix

        elif self.coupling_mode == 'rotation':
            # One nuisance param per (i, j) rotation plane (cycled across pairs).
            # Marks the two POI dims each plane involves; the actual rotation
            # happens in _transform_poi.
            matrix = np.zeros((self.n_poi, self.n_nuisance))
            nuis_idx = 0
            for i in range(self.n_poi):
                for j in range(i+1, self.n_poi):
                    if nuis_idx < self.n_nuisance:
                        matrix[i, nuis_idx] = 1.0
                        matrix[j, nuis_idx] = 1.0
                        nuis_idx += 1
            return matrix

        elif self.coupling_mode == 'mixed':
            # ~30% sparse random Gaussian. Fixed seed so the wrapper is
            # reproducible from a constructor call.
            rng = np.random.RandomState(42)
            matrix = rng.randn(self.n_poi, self.n_nuisance) * 0.3
            mask = rng.rand(self.n_poi, self.n_nuisance) > 0.7
            matrix[mask] = 0.0
            return matrix

        else:
            raise ValueError(f"Unknown coupling_mode: {self.coupling_mode}")

    def _transform_poi(self, poi_values: np.ndarray, nuisance_values: np.ndarray) -> np.ndarray:
        """POI values seen by ``base_func`` after applying the active coupling."""
        if self.coupling_mode == 'additive':
            return poi_values

        elif self.coupling_mode == 'shift':
            return poi_values + self.coupling_matrix @ nuisance_values

        elif self.coupling_mode == 'scale':
            return poi_values * (1.0 + self.coupling_matrix @ nuisance_values)

        elif self.coupling_mode == 'rotation':
            transformed = poi_values.copy()
            nuis_idx = 0
            for i in range(self.n_poi):
                for j in range(i+1, self.n_poi):
                    if nuis_idx < self.n_nuisance:
                        angle = self.rotation_scale * nuisance_values[nuis_idx]
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        x_i, x_j = transformed[i], transformed[j]
                        transformed[i] = cos_a * x_i - sin_a * x_j
                        transformed[j] = sin_a * x_i + cos_a * x_j
                        nuis_idx += 1
            return transformed

        elif self.coupling_mode == 'mixed':
            return poi_values + self.coupling_matrix @ nuisance_values

        else:
            raise ValueError(f"Unknown coupling_mode: {self.coupling_mode}")

    def _compute_constraint_penalty(self, nuisance_values: np.ndarray) -> float:
        """Log-likelihood penalty contribution from the nuisance constraints."""
        deviations = (nuisance_values - self.nuisance_mean) / self.constraint_sigma

        if self.constraint_mode == 'gaussian':
            return -0.5 * np.sum(deviations**2)

        elif self.constraint_mode == 'uniform':
            return 0.0 if np.all(np.abs(deviations) <= 1.0) else -np.inf

        elif self.constraint_mode == 'soft_uniform':
            penalty = 0.0
            for dev in deviations:
                if abs(dev) > 1.0:
                    penalty += -0.5 * (abs(dev) - 1.0)**2
            return penalty

        else:
            raise ValueError(f"Unknown constraint_mode: {self.constraint_mode}")

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate the augmented log-likelihood at a full parameter vector."""
        if len(params) != self.n_total:
            raise ValueError(
                f"Expected {self.n_total} parameters, got {len(params)}"
            )
        poi_values = params[:self.n_poi]
        nuisance_values = params[self.n_poi:]
        transformed_poi = self._transform_poi(poi_values, nuisance_values)
        return self.base_func(transformed_poi) + self._compute_constraint_penalty(nuisance_values)

    def get_optimal_nuisance(self, poi_values: np.ndarray) -> np.ndarray:
        """Analytically optimal nuisance values for given POIs.

        For every supported coupling mode the optimum equals the constraint
        mean: moving off it incurs a penalty without changing the base
        likelihood.
        """
        return self.nuisance_mean.copy()

    def profile_over_nuisance(
        self,
        poi_values: np.ndarray,
        method: str = 'analytical'
    ) -> Tuple[float, np.ndarray]:
        """Profile likelihood at ``poi_values`` (optimizing nuisance params).

        Returns ``(profile_log_likelihood, optimal_nuisance)``. ``method``
        is ``'analytical'`` (use :meth:`get_optimal_nuisance`) or
        ``'numerical'`` (scipy.optimize); numerical falls back to analytical
        on optimizer failure.
        """
        if method == 'analytical':
            optimal_nuis = self.get_optimal_nuisance(poi_values)
            full_params = np.concatenate([poi_values, optimal_nuis])
            return self(full_params), optimal_nuis

        elif method == 'numerical':
            from scipy.optimize import minimize

            def objective(nuisance_vals):
                return -self(np.concatenate([poi_values, nuisance_vals]))

            result = minimize(objective, self.nuisance_mean.copy(), method='L-BFGS-B')
            if result.success:
                return -result.fun, result.x
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
    """Build a NuisanceParameterWrapper plus full-vector bounds.

    Nuisance bounds are set to ``mean ± nuisance_bounds_sigma_multiple * sigma``
    (default 5σ covers ~99.999% for Gaussian constraints).
    Returns ``(wrapper, wrapped_bounds, wrapper)`` — the wrapper is
    returned twice for callers that expect a ``(func, bounds, wrapper)``
    triple.
    """
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

    wrapped_bounds = list(base_bounds[:n_poi])
    sigma_array = wrapper.constraint_sigma
    mean_array = wrapper.nuisance_mean
    for i in range(n_nuisance):
        nuis_range = nuisance_bounds_sigma_multiple * sigma_array[i]
        wrapped_bounds.append([mean_array[i] - nuis_range, mean_array[i] + nuis_range])

    return wrapper, wrapped_bounds, wrapper


def register_nuisance_wrapped_test_functions():
    """Build a registry ``{name: (func, bounds, wrapper, base_peaks)}`` of common test cases."""
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
