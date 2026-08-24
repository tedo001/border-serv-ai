"""Canonical event payload.

One serialisation shared by the REST API, the WebSocket feed, every C2 sink and
the desktop console. A single definition means an integrator writes one parser,
and a field added for the console automatically reaches the control room.

The schema is versioned. Downstream C2 systems in this domain are long-lived
and are not redeployed when a BOP node is upgraded, so a consumer must be able
to detect a format change rather than silently mis-parse one.
"""

from __future__ import annotations

from typing import Any

from ibvap.core.timeutils import to_iso
from ibvap.core.types import Event

#: Increment on any breaking change to the payload structure.
PAYLOAD_SCHEMA_VERSION = "1.0"


def event_to_payload(
    event: Event,
    *,
    site_id: str = "",
    site_name: str = "",
    camera_name: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    include_geometry: bool = True,
) -> dict[str, Any]:
    """Render an event as the canonical outbound payload."""
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "severity": event.severity.value,
        "message": event.message,
        "confidence": round(event.confidence, 4),
        "timestamp": event.timestamp,
        "timestamp_iso": to_iso(event.timestamp),
        "site": {"id": site_id, "name": site_name},
        "camera": {
            "id": event.camera_id,
            "name": camera_name or event.camera_id,
            "latitude": latitude,
            "longitude": longitude,
        },
        "rule_id": event.rule_id,
        "zone_id": event.zone_id,
        "track_ids": list(event.track_ids),
        "attributes": dict(event.attributes),
        "evidence": {
            # Paths are node-local; a C2 system fetches artefacts through the
            # API rather than by path, so the endpoints are given explicitly.
            "snapshot_url": f"/api/v1/events/{event.event_id}/snapshot"
            if event.snapshot_path else None,
            "clip_url": f"/api/v1/events/{event.event_id}/clip"
            if event.clip_path else None,
            "sha256": event.evidence_hash,
        },
    }
    if include_geometry and event.boxes:
        payload["boxes"] = [
            {"x1": round(b.x1, 1), "y1": round(b.y1, 1),
             "x2": round(b.x2, 1), "y2": round(b.y2, 1)}
            for b in event.boxes
        ]
    return payload


def payload_summary(payload: dict[str, Any]) -> str:
    """One-line rendering for logs and terminal output."""
    return (
        f"[{payload.get('severity', '?').upper()}] "
        f"{payload.get('event_type', '?')} "
        f"@{payload.get('camera', {}).get('id', '?')} "
        f"- {payload.get('message', '')}"
    )
