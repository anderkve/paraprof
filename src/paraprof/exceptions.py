"""
Custom exception classes for ParaProf.

This module defines all custom exceptions used throughout ParaProf,
providing clear error messages and helpful suggestions for users.
"""


class ParaProfError(Exception):
    """
    Base exception class for all ParaProf errors.

    All ParaProf-specific exceptions inherit from this class,
    allowing users to catch all ParaProf errors with a single except clause.
    """
    pass


class InvalidProjectionError(ParaProfError):
    """
    Raised when projection configuration is invalid.

    Examples
    --------
    - Projection dimensions out of bounds
    - Grid points not positive
    - Conflicting projection settings

    Attributes
    ----------
    projection : dict
        The invalid projection configuration
    """
    def __init__(self, message: str, projection: dict = None):
        self.projection = projection
        super().__init__(message)


class InvalidBoundsError(ParaProfError):
    """
    Raised when parameter bounds are invalid.

    Examples
    --------
    - Lower bound >= upper bound
    - Bounds array has wrong shape
    - Non-numeric bounds

    Attributes
    ----------
    bounds : array-like
        The invalid bounds array
    """
    def __init__(self, message: str, bounds=None):
        self.bounds = bounds
        super().__init__(message)


class ConfigurationError(ParaProfError):
    """
    Raised when sampler configuration is invalid.

    Examples
    --------
    - Negative population size
    - Invalid mutation strategy
    - Conflicting parameter settings

    Attributes
    ----------
    parameter : str, optional
        Name of the problematic parameter
    value : any, optional
        The invalid value
    """
    def __init__(self, message: str, parameter: str = None, value=None):
        self.parameter = parameter
        self.value = value
        super().__init__(message)
