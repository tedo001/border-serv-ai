"""SQLAlchemy ORM models.

SQLite is the default so a BOP node needs no database server, no DBA and no
extra failure domain; the same schema runs on PostgreSQL at the sector tier
where many nodes aggregate.

Indexing is driven by how a control room actually queries: "what happened on
this camera in the last hour", "show me every critical alert today", "find that
vehicle". Those are the composite indexes below - a bare timestamp index would
force a scan for the first and third.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all IBVAP tables."""


class EventRecord(Base):
    """A persisted analytics event."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    site_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    #: Unix seconds of the triggering frame.
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    message: Mapped[str] = mapped_column(Text, default="")
    rule_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    track_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    boxes: Mapped[list[list[float]]] = mapped_column(JSON, default=list)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)

    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Operator acknowledgement - the record of who acted on this alert.
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    acknowledged_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Free-text disposition recorded by the operator on acknowledgement.
    disposition: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[float] = mapped_column(Float, default=time.time)

    __table_args__ = (
        # "What happened on this camera recently" - the console's main query.
        Index("ix_events_camera_time", "camera_id", "timestamp"),
        # "Show me today's critical alerts" - the triage query.
        Index("ix_events_severity_time", "severity", "timestamp"),
        # "Which alerts are still unhandled" - the supervisor's query.
        Index("ix_events_ack_time", "acknowledged", "timestamp"),
        Index("ix_events_type_time", "event_type", "timestamp"),
    )


class UserRecord(Base):
    """An operator account."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #: bcrypt hash. Plaintext passwords are never stored or logged.
    password_hash: Mapped[str] = mapped_column(String(255))
    #: ``admin`` | ``supervisor`` | ``operator`` | ``viewer``
    role: Mapped[str] = mapped_column(String(24), default="viewer")
    full_name: Mapped[str] = mapped_column(String(128), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_login: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Forces a password change on next login - used for bootstrap accounts.
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)


class ApiKeyRecord(Base):
    """A machine credential for command-and-control integration."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(96), unique=True)
    #: SHA-256 of the key. The key itself is shown once, at creation, and
    #: never recoverable afterwards - a stored key is a key that leaks.
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #: First characters of the key, so an operator can identify it in a list.
    key_prefix: Mapped[str] = mapped_column(String(12), default="")
    role: Mapped[str] = mapped_column(String(24), default="viewer")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    created_by: Mapped[str] = mapped_column(String(64), default="")
    last_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class AuditRecord(Base):
    """An append-only record of security-relevant actions.

    Everything that changes what the platform watches, who can see it, or who
    is on a watchlist is recorded here. In a system that can identify people by
    face, the ability to answer "who added this person to the watchlist, and
    when" is not optional.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(128), default="")
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_ip: Mapped[str] = mapped_column(String(64), default="")
    success: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (Index("ix_audit_actor_time", "actor", "timestamp"),)


class FaceWatchRecord(Base):
    """A watchlist identity and its enrolled face embeddings."""

    __tablename__ = "watchlist_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    category: Mapped[str] = mapped_column(String(32), default="watchlist", index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    reference: Mapped[str] = mapped_column(String(128), default="")
    #: float32 embeddings, shape (n, dim), stored as raw bytes.
    embeddings: Mapped[bytes] = mapped_column(LargeBinary, default=b"")
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0)
    embedding_count: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    created_by: Mapped[str] = mapped_column(String(64), default="")
    #: Enrolments must be reviewed periodically; this drives that report.
    review_due: Mapped[float | None] = mapped_column(Float, nullable=True)


class PlateWatchRecord(Base):
    """A vehicle registration of interest."""

    __tablename__ = "watchlist_plates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate: Mapped[str] = mapped_column(String(24), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(32), default="wanted", index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    reference: Mapped[str] = mapped_column(String(128), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    created_by: Mapped[str] = mapped_column(String(64), default="")
    expires_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class CameraRecord(Base):
    """A camera configured at runtime through the API or console.

    Cameras declared in the site YAML are not duplicated here; this table holds
    only those added operationally, so the YAML remains the source of truth for
    the provisioned estate.
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    #: The full CameraConfig, serialised.
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)
    created_by: Mapped[str] = mapped_column(String(64), default="")


class OutboxRecord(Base):
    """An event awaiting delivery to a command-and-control sink.

    The heart of store-and-forward. At a remote BOP the uplink is intermittent
    by default rather than by exception, so undelivered alerts are persisted
    and replayed when the link returns instead of being lost to a failed POST.
    """

    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    sink: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    #: Unix seconds before which no delivery attempt should be made.
    next_attempt_at: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    #: Set when the sink finally accepted the event; rows are pruned after.
    delivered_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("event_id", "sink", name="uq_outbox_event_sink"),
        Index("ix_outbox_pending", "sink", "delivered_at", "next_attempt_at"),
    )
