"""HTTP status constants with forward/backward compatibility.

Starlette renamed two constants to match the current RFC 9110 wording
(``422 Unprocessable Content``, ``413 Content Too Large``) and deprecated the
older spellings. Resolving them once here keeps a deprecation warning from
being emitted on every validation failure, while still working on the older
Starlette that ships in existing deployments.
"""

from __future__ import annotations

from starlette import status

def _resolve(current: str, legacy: str, fallback: int) -> int:
    """Prefer the current constant name, falling back to the legacy spelling.

    ``getattr(obj, name, default)`` is not usable here: Python evaluates the
    default eagerly, so naming the deprecated attribute as the fallback emits
    the very warning this module exists to avoid - at import time, on every
    process start.
    """
    if hasattr(status, current):
        return int(getattr(status, current))
    if hasattr(status, legacy):
        return int(getattr(status, legacy))
    return fallback


#: 422 - the request was well-formed but semantically invalid.
HTTP_422_UNPROCESSABLE = _resolve(
    "HTTP_422_UNPROCESSABLE_CONTENT", "HTTP_422_UNPROCESSABLE_ENTITY", 422
)

#: 413 - the request body is larger than the server will accept.
HTTP_413_TOO_LARGE = _resolve(
    "HTTP_413_CONTENT_TOO_LARGE", "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413
)

__all__ = ["HTTP_422_UNPROCESSABLE", "HTTP_413_TOO_LARGE"]
