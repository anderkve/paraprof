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
    message : str
        Error message with details
    """
    def __init__(self, message: str, projection: dict = None):
        self.projection = projection
        self.message = message
        super().__init__(self.message)


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
    message : str
        Error message with details
    """
    def __init__(self, message: str, bounds=None):
        self.bounds = bounds
        self.message = message
        super().__init__(self.message)


class ConvergenceError(ParaProfError):
    """
    Raised when optimization fails to converge properly.

    This exception indicates that an optimization routine (e.g., L-BFGS-B)
    did not converge within the specified criteria.

    Attributes
    ----------
    grid_idx : tuple, optional
        Grid index where convergence failed
    message : str
        Error message with details
    """
    def __init__(self, message: str, grid_idx: tuple = None):
        self.grid_idx = grid_idx
        self.message = message
        super().__init__(self.message)


class MPIError(ParaProfError):
    """
    Raised when MPI-related operations fail.

    Examples
    --------
    - MPI not initialized
    - Invalid rank or communicator
    - Communication failure

    Attributes
    ----------
    rank : int, optional
        MPI rank where error occurred
    message : str
        Error message with details
    """
    def __init__(self, message: str, rank: int = None):
        self.rank = rank
        self.message = message
        super().__init__(self.message)


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
    message : str
        Error message with details
    """
    def __init__(self, message: str, parameter: str = None, value=None):
        self.parameter = parameter
        self.value = value
        self.message = message
        super().__init__(self.message)


class JobError(ParaProfError):
    """
    Raised when a job fails to execute properly.

    This exception indicates problems during job execution,
    such as invalid task results or job state errors.

    Attributes
    ----------
    job_id : int, optional
        ID of the failed job
    job_type : str, optional
        Type of job that failed
    message : str
        Error message with details
    """
    def __init__(self, message: str, job_id: int = None, job_type: str = None):
        self.job_id = job_id
        self.job_type = job_type
        self.message = message
        super().__init__(self.message)


class ValidationError(ParaProfError):
    """
    Raised when input validation fails.

    This is a general validation error for various input types
    that don't fit into more specific categories.

    Attributes
    ----------
    field : str, optional
        Name of the field that failed validation
    message : str
        Error message with details
    """
    def __init__(self, message: str, field: str = None):
        self.field = field
        self.message = message
        super().__init__(self.message)
