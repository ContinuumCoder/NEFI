"""Exception hierarchy for nefi. All library errors derive from :class:`NefiError`."""

from __future__ import annotations


class NefiError(Exception):
    """Base class for all nefi errors."""


class ShapeError(NefiError, ValueError):
    """Raised when tensor / grid shapes are inconsistent."""


class ConfigError(NefiError, ValueError):
    """Raised for invalid or incomplete configuration."""


class RegistryError(NefiError, KeyError):
    """Raised when a registry lookup fails."""


class SolverError(NefiError, RuntimeError):
    """Raised when optimization cannot proceed (e.g. persistent NaNs)."""


class OperatorError(NefiError, RuntimeError):
    """Raised by forward operators for invalid inputs."""
