"""API request and response models.

Separate from the internal domain types on purpose. The wire format is a
contract with a C2 system that will not be redeployed when this node is
upgraded, so it must be able to evolve independently of internal refactoring.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ibvap.core.timeutils import to_iso
from ibvap.storage.models import AuditRecord, EventRecord

# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: float
    role: str
    username: str
    must_change_password: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=12, max_length=256)


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    password: str = Field(min_length=12, max_length=256)
    role: Literal["viewer", "operator", "supervisor", "admin"] = "viewer"
    full_name: str = ""


class UserResponse(BaseModel):
    username: str
    role: str
    full_name: str
    active: bool
    created_at: float
    last_login: float | None = None


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=96)
    role: Literal["viewer", "operator", "supervisor", "admin"] = "viewer"
    ttl_days: int = Field(default=0, ge=0, le=3650)


class ApiKeyResponse(BaseModel):
    name: str
    role: str
    key_prefix: str
    active: bool
    created_at: float
    last_used: float | None = None
    expires_at: float | None = None
    #: Populated only in the creation response - never retrievable afterwards.
    key: str | None = None


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


class EventResponse(BaseModel):
    """One analytics event as the API returns it."""

    event_id: str
    camera_id: str
    event_type: str
    severity: str
    timestamp: float
    timestamp_iso: str
    confidence: float
    message: str
    rule_id: str
    zone_id: str | None
    track_ids: list[int]
    attributes: dict[str, Any]
    boxes: list[list[float]]
    frame_index: int
    has_snapshot: bool
    has_clip: bool
    evidence_hash: str | None
    acknowledged: bool
    acknowledged_by: str | None
    acknowledged_at: float | None
    disposition: str | None

    @classmethod
    def from_record(cls, record: EventRecord) -> EventResponse:
        return cls(
            event_id=record.event_id,
            camera_id=record.camera_id,
            event_type=record.event_type,
            severity=record.severity,
            timestamp=record.timestamp,
            timestamp_iso=to_iso(record.timestamp),
            confidence=record.confidence,
            message=record.message,
            rule_id=record.rule_id,
            zone_id=record.zone_id,
            track_ids=list(record.track_ids or []),
            attributes=dict(record.attributes or {}),
            boxes=[list(b) for b in (record.boxes or [])],
            frame_index=record.frame_index,
            # Expose presence, not the filesystem path: an artefact is fetched
            # through the API so access is authorised and audited, and node
            # storage layout never becomes part of the published contract.
            has_snapshot=bool(record.snapshot_path),
            has_clip=bool(record.clip_path),
            evidence_hash=record.evidence_hash,
            acknowledged=record.acknowledged,
            acknowledged_by=record.acknowledged_by,
            acknowledged_at=record.acknowledged_at,
            disposition=record.disposition,
        )


class EventPage(BaseModel):
    """A page of events plus enough context to drive a console's paging."""

    events: list[EventResponse]
    total: int
    limit: int
    offset: int


class AcknowledgeRequest(BaseModel):
    disposition: str = Field(default="", max_length=1000)


# --------------------------------------------------------------------------- #
# Cameras, zones and rules
# --------------------------------------------------------------------------- #


class ZonePayload(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = ""
    points: list[tuple[float, float]] = Field(min_length=3)
    kind: Literal["restricted", "buffer", "mask", "counting"] = "restricted"
    enabled: bool = True

    @field_validator("points")
    @classmethod
    def _normalised(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in v:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("zone points must be normalised to [0, 1]")
        return v


class TripwirePayload(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = ""
    start: tuple[float, float]
    end: tuple[float, float]
    direction: Literal["any", "left", "right"] = "any"
    left_label: str = "left"
    right_label: str = "right"
    enabled: bool = True


class RulePayload(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    type: str
    enabled: bool = True
    zones: list[str] = Field(default_factory=list)
    tripwires: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    severity: Literal["info", "low", "medium", "high", "critical"] | None = None
    cooldown_seconds: float = 30.0
    active_hours: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class CameraPayload(BaseModel):
    """A camera as created or updated through the API."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9._-]+$")
    name: str = ""
    url: str = Field(min_length=1)
    enabled: bool = True
    location: str = ""
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    bearing: float | None = Field(default=None, ge=0, le=360)
    target_fps: float = Field(default=8.0, gt=0, le=60)
    detect_interval: int = Field(default=1, ge=1, le=30)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    process_width: int = Field(default=960, ge=160, le=3840)
    zones: list[ZonePayload] = Field(default_factory=list)
    tripwires: list[TripwirePayload] = Field(default_factory=list)
    rules: list[RulePayload] = Field(default_factory=list)
    anpr_enabled: bool = False
    face_enabled: bool = False
    night_enhancement: bool = True
    tags: list[str] = Field(default_factory=list)


class CameraStatusResponse(BaseModel):
    """Live status of one camera."""

    camera: dict[str, Any]
    running: bool
    stream: dict[str, Any]
    stats: dict[str, Any]
    analytics: dict[str, Any]
    detector_mode: str
    anpr_enabled: bool
    face_enabled: bool


# --------------------------------------------------------------------------- #
# Watchlists
# --------------------------------------------------------------------------- #


class PlateWatchRequest(BaseModel):
    plate: str = Field(min_length=4, max_length=24)
    category: Literal["wanted", "stolen", "suspect", "permitted", "banned"] = "wanted"
    reason: str = Field(default="", max_length=500)
    reference: str = Field(default="", max_length=128)
    expires_at: float | None = None


class PlateWatchResponse(BaseModel):
    plate: str
    category: str
    reason: str
    reference: str
    active: bool
    created_at: float
    created_by: str
    expires_at: float | None = None


class FaceEnrolRequest(BaseModel):
    """Enrol an identity from a base64 face image or a raw embedding."""

    person_id: str = Field(min_length=1, max_length=64)
    name: str = Field(default="", max_length=128)
    category: Literal["wanted", "suspect", "staff", "permitted", "banned"] = "wanted"
    notes: str = Field(default="", max_length=1000)
    reference: str = Field(default="", max_length=128)
    #: Base64-encoded JPEG/PNG containing exactly one clear face.
    image_base64: str | None = None
    #: Alternatively, a pre-computed embedding from another enrolment station.
    embedding: list[float] | None = None


class FaceWatchResponse(BaseModel):
    person_id: str
    name: str
    category: str
    notes: str
    reference: str
    embedding_count: int
    active: bool
    created_at: float
    created_by: str


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str
    version: str
    site_id: str
    site_name: str
    tier: str
    uptime_seconds: float
    cameras_total: int
    cameras_online: int
    models: dict[str, Any]
    watchlists: dict[str, int]
    storage: dict[str, Any] = Field(default_factory=dict)
    integrations: dict[str, Any] = Field(default_factory=dict)


class AuditResponse(BaseModel):
    timestamp: float
    timestamp_iso: str
    actor: str
    action: str
    target: str
    detail: dict[str, Any]
    source_ip: str
    success: bool

    @classmethod
    def from_record(cls, record: AuditRecord) -> AuditResponse:
        return cls(
            timestamp=record.timestamp,
            timestamp_iso=to_iso(record.timestamp),
            actor=record.actor,
            action=record.action,
            target=record.target,
            detail=dict(record.detail or {}),
            source_ip=record.source_ip,
            success=record.success,
        )


class MessageResponse(BaseModel):
    """Generic acknowledgement."""

    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
