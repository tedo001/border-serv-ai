"""HTTP webhook sink.

The most common way to reach an existing command-and-control system: a signed
JSON POST. Signing is HMAC-SHA256 over the exact request body, which lets the
receiver verify both origin and integrity without mutual TLS - frequently the
only practical option when the C2 system is a legacy application that cannot be
issued client certificates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import httpx

from ibvap.core.config import WebhookConfig
from ibvap.core.errors import IntegrationError
from ibvap.core.logging import get_logger
from ibvap.integrations.base import Sink

log = get_logger(__name__)

#: Header names a receiver checks to authenticate a delivery.
SIGNATURE_HEADER = "X-IBVAP-Signature"
TIMESTAMP_HEADER = "X-IBVAP-Timestamp"
EVENT_ID_HEADER = "X-IBVAP-Event-Id"


class WebhookSink(Sink):
    """Delivers events to an HTTP endpoint."""

    def __init__(self, config: WebhookConfig) -> None:
        super().__init__(
            min_severity=config.min_severity,
            event_types=config.event_types,
            enabled=config.enabled,
        )
        self.name = config.name
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            # A modest pool: a BOP node talks to one C2 endpoint, and holding
            # many idle sockets open over a metered link is pure waste.
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            follow_redirects=False,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def deliver(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            await self.start()
        assert self._client is not None

        # Serialise once and sign those exact bytes. Re-serialising for the
        # signature would let key ordering differ from the transmitted body and
        # the receiver's verification would fail intermittently.
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        timestamp = str(int(time.time()))

        headers = {
            "Content-Type": "application/json",
            TIMESTAMP_HEADER: timestamp,
            EVENT_ID_HEADER: str(payload.get("event_id", "")),
            **self.config.headers,
        }
        if self.config.hmac_secret:
            # The timestamp is inside the signed material so a captured
            # delivery cannot be replayed later against the same endpoint.
            signed = f"{timestamp}.".encode() + body
            digest = hmac.new(
                self.config.hmac_secret.encode(), signed, hashlib.sha256
            ).hexdigest()
            headers[SIGNATURE_HEADER] = f"sha256={digest}"

        try:
            response = await self._client.post(self.config.url, content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise IntegrationError(f"webhook {self.name} transport error: {exc}") from exc

        if response.status_code >= 400:
            raise IntegrationError(
                f"webhook {self.name} rejected delivery: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "kind": "webhook",
            "url": self.config.url,
            "signed": bool(self.config.hmac_secret),
        }


def verify_signature(
    body: bytes, signature: str, timestamp: str, secret: str, *, max_age_seconds: float = 300.0
) -> bool:
    """Verify a webhook signature. Provided for receiver implementations.

    Included in the platform so integrators have a reference that matches the
    sender exactly, rather than reimplementing the scheme from prose and
    getting the signed material subtly wrong.
    """
    try:
        age = abs(time.time() - float(timestamp))
    except (TypeError, ValueError):
        return False
    if age > max_age_seconds:
        return False

    expected = hmac.new(
        secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    presented = signature.removeprefix("sha256=")
    # Constant-time comparison: a timing side channel here would leak the
    # signature byte by byte.
    return hmac.compare_digest(expected, presented)
