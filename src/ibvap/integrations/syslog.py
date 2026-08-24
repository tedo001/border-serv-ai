"""Syslog sink emitting ArcSight CEF.

CEF is what SIEM platforms in this space ingest without a custom parser, so
speaking it natively means an existing security operations centre can correlate
IBVAP alerts with everything else it already collects on day one.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

from ibvap.core.config import SyslogConfig
from ibvap.core.errors import IntegrationError
from ibvap.core.logging import get_logger
from ibvap.integrations.base import Sink

log = get_logger(__name__)

CEF_VERSION = 0
CEF_VENDOR = "IBVAP"
CEF_PRODUCT = "Border Video Analytics"
CEF_DEVICE_VERSION = "1.0"

#: IBVAP severity to the CEF 0-10 scale.
_CEF_SEVERITY: dict[str, int] = {
    "info": 2, "low": 3, "medium": 5, "high": 8, "critical": 10,
}

#: Syslog PRI severity levels (RFC 5424).
_SYSLOG_SEVERITY: dict[str, int] = {
    "info": 6, "low": 5, "medium": 4, "high": 3, "critical": 2,
}


def escape_cef_value(value: Any) -> str:
    """Escape a CEF extension value.

    Backslash first, then the delimiters. Doing it in the other order would
    double-escape the backslashes this function itself inserts, producing a
    payload the SIEM parses incorrectly.
    """
    text = str(value)
    return text.replace("\\", "\\\\").replace("=", "\\=").replace("\n", "\\n").replace("\r", "")


def escape_cef_header(value: Any) -> str:
    """Escape a CEF header field, where the pipe is the delimiter."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|")


def format_cef(payload: dict[str, Any]) -> str:
    """Render a canonical event payload as a CEF message."""
    severity = str(payload.get("severity", "info"))
    event_type = str(payload.get("event_type", "unknown"))
    camera = payload.get("camera", {}) or {}
    site = payload.get("site", {}) or {}

    header = "|".join([
        f"CEF:{CEF_VERSION}",
        escape_cef_header(CEF_VENDOR),
        escape_cef_header(CEF_PRODUCT),
        escape_cef_header(CEF_DEVICE_VERSION),
        escape_cef_header(event_type),
        escape_cef_header(payload.get("message", event_type)),
        str(_CEF_SEVERITY.get(severity, 5)),
    ])

    # Standard CEF keys where one exists, custom-string slots otherwise. Using
    # the standard keys is what lets a SIEM correlate these without mapping.
    extensions: dict[str, Any] = {
        "rt": int(float(payload.get("timestamp", time.time())) * 1000),
        "externalId": payload.get("event_id", ""),
        "cat": event_type,
        "deviceExternalId": camera.get("id", ""),
        "dvchost": site.get("id", ""),
        "sourceServiceName": camera.get("name", ""),
        "cs1Label": "site", "cs1": site.get("name", ""),
        "cs2Label": "rule", "cs2": payload.get("rule_id", ""),
        "cs3Label": "zone", "cs3": payload.get("zone_id") or "",
        "cn1Label": "confidence", "cn1": payload.get("confidence", 0),
    }
    if camera.get("latitude") is not None:
        extensions["deviceLatitude"] = camera["latitude"]
        extensions["deviceLongitude"] = camera.get("longitude", "")

    # Surface the operationally interesting attributes in the remaining custom
    # string slots. CEF defines exactly cs1..cs6 and cs1-cs3 are already taken,
    # so at most three attributes are promoted - in priority order, because a
    # plate or an identity matters more to a SIEM rule than an object class.
    # Anything beyond that would have to invent non-standard keys, which is
    # precisely what makes a CEF feed need a custom parser.
    attributes = payload.get("attributes", {}) or {}
    slot = 4
    for key in ("plate", "person_id", "name", "category", "object_class"):
        if slot > 6:
            break
        if key in attributes and attributes[key] not in ("", None):
            extensions[f"cs{slot}Label"] = key
            extensions[f"cs{slot}"] = attributes[key]
            slot += 1

    body = " ".join(
        f"{k}={escape_cef_value(v)}" for k, v in extensions.items() if v not in ("", None)
    )
    return f"{header}|{body}"


class SyslogSink(Sink):
    """Sends CEF messages over UDP or TCP syslog."""

    def __init__(self, config: SyslogConfig, *, name: str = "syslog") -> None:
        super().__init__(min_severity=config.min_severity, enabled=config.enabled)
        self.name = name
        self.config = config
        self._socket: socket.socket | None = None

    async def start(self) -> None:
        if not self.enabled or self._socket is not None:
            return
        try:
            if self.config.protocol == "tcp":
                sock = socket.create_connection(
                    (self.config.host, self.config.port), timeout=10.0
                )
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket = sock
        except OSError as exc:
            raise IntegrationError(
                f"syslog sink {self.name} cannot reach "
                f"{self.config.host}:{self.config.port}: {exc}"
            ) from exc

    async def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    async def deliver(self, payload: dict[str, Any]) -> None:
        if self._socket is None:
            await self.start()
        if self._socket is None:
            raise IntegrationError(f"syslog sink {self.name} is not connected")

        severity = _SYSLOG_SEVERITY.get(str(payload.get("severity", "info")), 6)
        priority = self.config.facility * 8 + severity
        stamp = time.strftime("%b %d %H:%M:%S", time.localtime(payload.get("timestamp", time.time())))
        message = f"<{priority}>{stamp} ibvap {format_cef(payload)}"
        data = message.encode("utf-8", errors="replace")

        try:
            if self.config.protocol == "tcp":
                # Octet-counting framing (RFC 6587) - without a frame delimiter
                # a TCP collector cannot tell where one message ends.
                await asyncio.to_thread(self._socket.sendall, data + b"\n")
            else:
                await asyncio.to_thread(
                    self._socket.sendto, data, (self.config.host, self.config.port)
                )
        except OSError as exc:
            # Force a reconnect on the next attempt: a broken TCP socket stays
            # broken, and silently retrying on it fails forever.
            await self.close()
            raise IntegrationError(f"syslog sink {self.name} send failed: {exc}") from exc

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "kind": "syslog",
            "target": f"{self.config.host}:{self.config.port}/{self.config.protocol}",
            "format": "CEF",
        }
