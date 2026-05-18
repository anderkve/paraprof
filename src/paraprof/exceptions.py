"""Custom exception classes for ParaProf."""


class ParaProfError(Exception):
    """Base class for all ParaProf errors."""
    pass


class InvalidProjectionError(ParaProfError):
    """Raised when projection configuration is invalid."""
    def __init__(self, message: str, projection: dict = None):
        self.projection = projection
        super().__init__(message)


class InvalidBoundsError(ParaProfError):
    """Raised when parameter bounds are invalid."""
    def __init__(self, message: str, bounds=None):
        self.bounds = bounds
        super().__init__(message)


class ConfigurationError(ParaProfError):
    """Raised when sampler configuration is invalid."""
    def __init__(self, message: str, parameter: str = None, value=None):
        self.parameter = parameter
        self.value = value
        super().__init__(message)
