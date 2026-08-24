"""Exception hierarchy for IBVAP.

Every failure mode the platform can surface derives from :class:`IbvapError`
so that supervisory code can distinguish *recoverable* faults (a camera that
dropped its stream) from *fatal* ones (a corrupt model artefact) without
resorting to string matching.
"""

from __future__ import annotations


class IbvapError(Exception):
    """Base class for all IBVAP errors."""

    #: Whether a supervisor should retry the failed operation.
    recoverable: bool = False


class ConfigError(IbvapError):
    """Raised when configuration is missing, malformed or self-contradictory."""


class ModelError(IbvapError):
    """Raised when a model artefact cannot be loaded, verified or executed."""


class ModelNotFoundError(ModelError):
    """Raised when a model referenced by the registry is absent on disk."""


class ChecksumMismatchError(ModelError):
    """Raised when a model artefact fails integrity verification.

    Treated as fatal: a surveillance platform must never run inference with an
    artefact whose provenance cannot be established.
    """


class StreamError(IbvapError):
    """Raised when a video source cannot be opened or read."""

    recoverable = True


class StreamClosedError(StreamError):
    """Raised when a previously healthy stream ends (EOF or peer disconnect)."""

    recoverable = True


class AnalyticsError(IbvapError):
    """Raised when an analytics rule is misconfigured."""


class IntegrationError(IbvapError):
    """Raised when an outbound C2 integration fails."""

    recoverable = True


class AuthError(IbvapError):
    """Raised on authentication or authorisation failure."""


class StorageError(IbvapError):
    """Raised when persistence or evidence storage fails."""
