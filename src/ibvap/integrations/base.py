"""Sink abstraction and severity filtering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ibvap.core.types import Event, Severity

#: Severity ordering used for ``min_severity`` filtering.
_ORDER: dict[str, int] = {
    Severity.INFO.value: 0,
    Severity.LOW.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.HIGH.value: 3,
    Severity.CRITICAL.value: 4,
}


class Sink(ABC):
    """An outbound destination for events."""

    #: Stable identifier used in the outbox, metrics and logs.
    name: str = "sink"

    def __init__(
        self,
        *,
        min_severity: str = "info",
        event_types: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.min_severity = min_severity
        self.event_types = set(event_types or [])
        self.enabled = enabled

    def accepts(self, event: Event) -> bool:
        """Whether this sink should receive ``event``.

        Filtering here rather than centrally lets one node feed a SIEM every
        informational detection while sending a sector control room only what
        needs a decision.
        """
        if not self.enabled:
            return False
        if _ORDER.get(event.severity.value, 0) < _ORDER.get(self.min_severity, 0):
            return False
        if self.event_types and event.event_type.value not in self.event_types:
            return False
        return True

    @abstractmethod
    async def deliver(self, payload: dict[str, Any]) -> None:
        """Deliver one payload, raising on failure so the caller can retry."""

    async def start(self) -> None:
        """Optional connection setup."""

    async def close(self) -> None:
        """Optional teardown."""

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "min_severity": self.min_severity,
            "event_types": sorted(self.event_types) or "all",
        }
