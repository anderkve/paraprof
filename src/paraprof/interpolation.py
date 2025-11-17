"""
Grid interpolation utilities for refinement runs.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .logger import get_logger

logger = get_logger()


class GridInterpolator:
    """
    Interpolates continuous parameter values from a coarse grid solution
    to predict good starting points for a refined grid.

    This class creates separate interpolators for each continuous parameter
    dimension, allowing prediction of optimal continuous parameters at any
    point in the projection space.
    """

    def __init__(self, coarse_grid_solution):
        """
        Initializes the interpolator from a coarse grid solution.

        Parameters
        ----------
        coarse_grid_solution : dict
            Dictionary returned by GridAnchoredDESampler.export_grid_solution()
            containing:
            - 'grid_axes': List of arrays defining grid coordinates
            - 'projection_dims': List of projection dimension indices
            - 'continuous_dims': List of continuous dimension indices
            - 'solutions': Dict mapping grid_idx -> solution dict
            - 'grid_shape': Tuple of grid dimensions
        """
        self.grid_axes = coarse_grid_solution['grid_axes']
        self.projection_dims = coarse_grid_solution['projection_dims']
        self.continuous_dims = coarse_grid_solution['continuous_dims']
        self.grid_shape = coarse_grid_solution['grid_shape']
        self.solutions = coarse_grid_solution['solutions']

        self.n_proj_dims = len(self.projection_dims)
        self.n_cont_dims = len(self.continuous_dims)

        # Build interpolators for each continuous parameter
        self._build_interpolators()

        # Cache for likelihood interpolator (built on first use)
        self._likelihood_interpolator = None


    def _build_interpolators(self):
        """
        Constructs RegularGridInterpolator objects for each continuous parameter.

        For each continuous dimension, creates a dense grid of values from the
        sparse coarse solution, then builds an interpolator.
        """
        self.interpolators = []

        if self.n_cont_dims == 0:
            # No continuous dimensions to interpolate
            return

        # For each continuous parameter dimension
        for cont_dim_idx in range(self.n_cont_dims):
            # Create a dense grid filled with NaN
            values_grid = np.full(self.grid_shape, np.nan)

            # Fill in the known values from coarse solutions
            for grid_idx, solution in self.solutions.items():
                continuous_params = solution['continuous_params']
                values_grid[grid_idx] = continuous_params[cont_dim_idx]

            # Check if we have enough data for interpolation
            valid_mask = ~np.isnan(values_grid)
            n_valid = np.sum(valid_mask)

            if n_valid == 0:
                logger.warning(f" No valid data for continuous dim {cont_dim_idx}. "
                      f"Interpolator will return NaN.")
                # Create dummy interpolator that returns NaN
                interpolator = None
            elif n_valid < 2**self.n_proj_dims:
                logger.warning(f" Sparse data ({n_valid} points) for continuous dim {cont_dim_idx}. "
                      f"Using nearest neighbor interpolation.")
                # Use nearest neighbor for sparse data
                interpolator = RegularGridInterpolator(
                    self.grid_axes,
                    values_grid,
                    method='nearest',
                    bounds_error=False,
                    fill_value=None  # Will extrapolate using nearest
                )
            else:
                # Use linear interpolation with nearest neighbor extrapolation
                interpolator = RegularGridInterpolator(
                    self.grid_axes,
                    values_grid,
                    method='linear',
                    bounds_error=False,
                    fill_value=None  # Will extrapolate using nearest
                )

            self.interpolators.append(interpolator)


    def interpolate(self, projection_coords):
        """
        Interpolates continuous parameter values at given projection coordinates.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        np.ndarray or None
            Interpolated continuous parameter values, shape (n_cont_dims,)
            Returns None if no continuous dimensions exist
        """
        if self.n_cont_dims == 0:
            return None

        projection_coords = np.asarray(projection_coords)

        # Ensure projection_coords is in the right shape for interpolation
        # RegularGridInterpolator expects shape (n_points, n_dims)
        coords_for_interp = projection_coords.reshape(1, -1)

        interpolated_params = np.zeros(self.n_cont_dims)

        for i, interpolator in enumerate(self.interpolators):
            if interpolator is None:
                # No valid data, return NaN
                interpolated_params[i] = np.nan
            else:
                try:
                    interpolated_params[i] = interpolator(coords_for_interp)[0]
                except Exception as e:
                    logger.warning(f"Warning: Interpolation failed for continuous dim {i}: {e}")
                    interpolated_params[i] = np.nan

        return interpolated_params


    def get_interpolated_likelihood(self, projection_coords):
        """
        Interpolates the likelihood value at given projection coordinates.

        Uses a cached interpolator for efficiency when called multiple times.

        Parameters
        ----------
        projection_coords : array-like
            Coordinates in projection space, shape (n_proj_dims,)

        Returns
        -------
        float
            Interpolated likelihood value
        """
        # Build likelihood interpolator on first use and cache it
        if self._likelihood_interpolator is None:
            likelihood_grid = np.full(self.grid_shape, np.nan)

            for grid_idx, solution in self.solutions.items():
                likelihood_grid[grid_idx] = solution['likelihood']

            # Create interpolator for likelihood
            self._likelihood_interpolator = RegularGridInterpolator(
                self.grid_axes,
                likelihood_grid,
                method='linear',
                bounds_error=False,
                fill_value=None
            )

        projection_coords = np.asarray(projection_coords).reshape(1, -1)
        return self._likelihood_interpolator(projection_coords)[0]


    def get_coverage_fraction(self):
        """
        Returns the fraction of grid points that have valid solutions.

        Returns
        -------
        float
            Fraction of grid points with solutions (0 to 1)
        """
        total_grid_points = np.prod(self.grid_shape)
        valid_points = len(self.solutions)
        return valid_points / total_grid_points if total_grid_points > 0 else 0.0


    def __repr__(self):
        """String representation of the interpolator."""
        coverage = self.get_coverage_fraction()
        return (f"GridInterpolator(grid_shape={self.grid_shape}, "
                f"n_continuous_dims={self.n_cont_dims}, "
                f"coverage={coverage:.1%})")
