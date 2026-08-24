"""Repositories: all database access lives behind these classes.

Keeping queries here rather than scattered through API handlers means the
retention sweep, the console's filters and the C2 replay path all share one
definition of what "recent critical events for this camera" means.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Event, EventType, Severity
from ibvap.storage.models import (
    ApiKeyRecord,
    AuditRecord,
    CameraRecord,
    EventRecord,
    FaceWatchRecord,
    OutboxRecord,
    PlateWatchRecord,
    UserRecord,
)

log = get_logger(__name__)

#: Cap on any single page of results. A console filter that accidentally
#: matches a month of events must not try to serialise all of it.
MAX_PAGE_SIZE = 500


class EventRepository:
    """Reads and writes analytics events."""

    def __init__(self, session: AsyncSession, site_id: str = "") -> None:
        self.session = session
        self.site_id = site_id

    async def add(self, event: Event) -> EventRecord:
        record = EventRecord(
            event_id=event.event_id,
            site_id=self.site_id,
            camera_id=event.camera_id,
            event_type=event.event_type.value,
            severity=event.severity.value,
            timestamp=event.timestamp,
            confidence=event.confidence,
            message=event.message,
            rule_id=event.rule_id,
            zone_id=event.zone_id,
            track_ids=list(event.track_ids),
            attributes=_json_safe(event.attributes),
            boxes=[list(b.as_tuple()) for b in event.boxes],
            frame_index=event.frame_index,
            snapshot_path=event.snapshot_path,
            clip_path=event.clip_path,
            evidence_hash=event.evidence_hash,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def add_many(self, events: Sequence[Event]) -> int:
        for event in events:
            await self.add(event)
        return len(events)

    async def get(self, event_id: str) -> EventRecord | None:
        result = await self.session.execute(
            select(EventRecord).where(EventRecord.event_id == event_id)
        )
        return result.scalar_one_or_none()

    async def query(
        self,
        *,
        camera_ids: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        severities: Sequence[str] | None = None,
        min_severity: str | None = None,
        since: float | None = None,
        until: float | None = None,
        acknowledged: bool | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order: str = "desc",
    ) -> list[EventRecord]:
        """Filtered event search, newest first by default."""
        statement = select(EventRecord)

        if camera_ids:
            statement = statement.where(EventRecord.camera_id.in_(list(camera_ids)))
        if event_types:
            statement = statement.where(EventRecord.event_type.in_(list(event_types)))
        if severities:
            statement = statement.where(EventRecord.severity.in_(list(severities)))
        if min_severity:
            # Severity is stored as text, so "at least MEDIUM" has to be
            # expanded into the explicit set rather than compared with >=.
            allowed = _severities_at_least(min_severity)
            statement = statement.where(EventRecord.severity.in_(allowed))
        if since is not None:
            statement = statement.where(EventRecord.timestamp >= since)
        if until is not None:
            statement = statement.where(EventRecord.timestamp <= until)
        if acknowledged is not None:
            statement = statement.where(EventRecord.acknowledged.is_(acknowledged))
        if search:
            pattern = f"%{search}%"
            statement = statement.where(EventRecord.message.ilike(pattern))

        column = EventRecord.timestamp
        statement = statement.order_by(column.asc() if order == "asc" else column.desc())
        statement = statement.limit(min(max(1, limit), MAX_PAGE_SIZE)).offset(max(0, offset))

        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def count(
        self, *, since: float | None = None, acknowledged: bool | None = None
    ) -> int:
        statement = select(func.count(EventRecord.id))
        if since is not None:
            statement = statement.where(EventRecord.timestamp >= since)
        if acknowledged is not None:
            statement = statement.where(EventRecord.acknowledged.is_(acknowledged))
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def acknowledge(
        self, event_id: str, actor: str, disposition: str = ""
    ) -> bool:
        """Record that an operator has actioned an alert."""
        result = await self.session.execute(
            update(EventRecord)
            .where(EventRecord.event_id == event_id, EventRecord.acknowledged.is_(False))
            .values(
                acknowledged=True,
                acknowledged_by=actor,
                acknowledged_at=time.time(),
                disposition=disposition,
            )
        )
        return bool(result.rowcount)

    async def statistics(self, since: float) -> dict[str, Any]:
        """Aggregates for the dashboard: counts by type, severity and camera."""
        async def grouped(column: Any) -> dict[str, int]:
            result = await self.session.execute(
                select(column, func.count(EventRecord.id))
                .where(EventRecord.timestamp >= since)
                .group_by(column)
            )
            return {str(key): int(count) for key, count in result.all()}

        total = await self.count(since=since)
        return {
            "since": since,
            "total": total,
            "unacknowledged": await self.count(since=since, acknowledged=False),
            "by_type": await grouped(EventRecord.event_type),
            "by_severity": await grouped(EventRecord.severity),
            "by_camera": await grouped(EventRecord.camera_id),
        }

    async def prune(self, older_than_seconds: float) -> int:
        """Delete events older than the retention window."""
        cutoff = time.time() - older_than_seconds
        result = await self.session.execute(
            delete(EventRecord).where(EventRecord.timestamp < cutoff)
        )
        return int(result.rowcount or 0)


class OutboxRepository:
    """The store-and-forward queue for C2 delivery."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def enqueue(self, event_id: str, sink: str, payload: dict[str, Any]) -> None:
        """Queue an event for a sink, ignoring an existing duplicate.

        The (event, sink) uniqueness constraint means a retry that races with
        the original enqueue cannot produce two deliveries of one alert.
        """
        existing = await self.session.execute(
            select(OutboxRecord.id).where(
                OutboxRecord.event_id == event_id, OutboxRecord.sink == sink
            )
        )
        if existing.scalar_one_or_none() is not None:
            return
        self.session.add(
            OutboxRecord(event_id=event_id, sink=sink, payload=payload, next_attempt_at=0.0)
        )

    async def pending(self, sink: str, limit: int = 50) -> list[OutboxRecord]:
        """Undelivered rows for a sink whose backoff has elapsed, oldest first."""
        result = await self.session.execute(
            select(OutboxRecord)
            .where(
                OutboxRecord.sink == sink,
                OutboxRecord.delivered_at.is_(None),
                OutboxRecord.next_attempt_at <= time.time(),
            )
            .order_by(OutboxRecord.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_delivered(self, record_id: int) -> None:
        await self.session.execute(
            update(OutboxRecord)
            .where(OutboxRecord.id == record_id)
            .values(delivered_at=time.time(), last_error="")
        )

    async def mark_failed(self, record_id: int, error: str, backoff_seconds: float) -> None:
        await self.session.execute(
            update(OutboxRecord)
            .where(OutboxRecord.id == record_id)
            .values(
                attempts=OutboxRecord.attempts + 1,
                next_attempt_at=time.time() + backoff_seconds,
                last_error=error[:500],
            )
        )

    async def depth(self, sink: str | None = None) -> int:
        statement = select(func.count(OutboxRecord.id)).where(
            OutboxRecord.delivered_at.is_(None)
        )
        if sink:
            statement = statement.where(OutboxRecord.sink == sink)
        result = await self.session.execute(statement)
        return int(result.scalar_one())

    async def prune(self, *, keep_delivered_seconds: float = 3600.0, max_pending: int = 50_000) -> int:
        """Drop delivered rows, and the oldest pending rows past the cap.

        The cap exists because an uplink that has been down for a week would
        otherwise grow the outbox until the node's disk fills - at which point
        it stops recording *new* alerts, which is far worse than losing the
        oldest undelivered ones.
        """
        cutoff = time.time() - keep_delivered_seconds
        result = await self.session.execute(
            delete(OutboxRecord).where(
                OutboxRecord.delivered_at.is_not(None),
                OutboxRecord.delivered_at < cutoff,
            )
        )
        removed = int(result.rowcount or 0)

        pending = await self.depth()
        if pending > max_pending:
            excess = pending - max_pending
            oldest = await self.session.execute(
                select(OutboxRecord.id)
                .where(OutboxRecord.delivered_at.is_(None))
                .order_by(OutboxRecord.created_at.asc())
                .limit(excess)
            )
            ids = list(oldest.scalars().all())
            if ids:
                await self.session.execute(
                    delete(OutboxRecord).where(OutboxRecord.id.in_(ids))
                )
                removed += len(ids)
                log.warning("outbox_overflow_pruned", dropped=len(ids), cap=max_pending)
        return removed


class UserRepository:
    """Operator accounts."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, username: str) -> UserRecord | None:
        result = await self.session.execute(
            select(UserRecord).where(UserRecord.username == username)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        username: str,
        password_hash: str,
        *,
        role: str = "viewer",
        full_name: str = "",
        must_change_password: bool = False,
    ) -> UserRecord:
        record = UserRecord(
            username=username,
            password_hash=password_hash,
            role=role,
            full_name=full_name,
            must_change_password=must_change_password,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def list_all(self) -> list[UserRecord]:
        result = await self.session.execute(select(UserRecord).order_by(UserRecord.username))
        return list(result.scalars().all())

    async def count(self) -> int:
        result = await self.session.execute(select(func.count(UserRecord.id)))
        return int(result.scalar_one())

    async def touch_login(self, username: str) -> None:
        await self.session.execute(
            update(UserRecord)
            .where(UserRecord.username == username)
            .values(last_login=time.time())
        )

    async def set_password(self, username: str, password_hash: str) -> bool:
        result = await self.session.execute(
            update(UserRecord)
            .where(UserRecord.username == username)
            .values(password_hash=password_hash, must_change_password=False)
        )
        return bool(result.rowcount)

    async def set_active(self, username: str, active: bool) -> bool:
        result = await self.session.execute(
            update(UserRecord).where(UserRecord.username == username).values(active=active)
        )
        return bool(result.rowcount)


class ApiKeyRepository:
    """Machine credentials for C2 integration."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    async def create(
        self, name: str, *, role: str = "viewer", created_by: str = "", ttl_days: int = 0
    ) -> tuple[ApiKeyRecord, str]:
        """Mint a key. The plaintext is returned once and never stored."""
        key = f"ibv_{secrets.token_urlsafe(32)}"
        record = ApiKeyRecord(
            name=name,
            key_hash=self.hash_key(key),
            key_prefix=key[:12],
            role=role,
            created_by=created_by,
            expires_at=time.time() + ttl_days * 86400 if ttl_days > 0 else None,
        )
        self.session.add(record)
        await self.session.flush()
        return record, key

    async def verify(self, key: str) -> ApiKeyRecord | None:
        """Resolve a presented key to its record, if valid and unexpired."""
        result = await self.session.execute(
            select(ApiKeyRecord).where(
                ApiKeyRecord.key_hash == self.hash_key(key),
                ApiKeyRecord.active.is_(True),
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        if record.expires_at is not None and record.expires_at < time.time():
            return None
        record.last_used = time.time()
        return record

    async def list_all(self) -> list[ApiKeyRecord]:
        result = await self.session.execute(select(ApiKeyRecord).order_by(ApiKeyRecord.name))
        return list(result.scalars().all())

    async def revoke(self, name: str) -> bool:
        result = await self.session.execute(
            update(ApiKeyRecord).where(ApiKeyRecord.name == name).values(active=False)
        )
        return bool(result.rowcount)


class AuditRepository:
    """Append-only security audit trail."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        actor: str,
        action: str,
        *,
        target: str = "",
        detail: dict[str, Any] | None = None,
        source_ip: str = "",
        success: bool = True,
    ) -> None:
        self.session.add(
            AuditRecord(
                actor=actor,
                action=action,
                target=target,
                detail=_json_safe(detail or {}),
                source_ip=source_ip,
                success=success,
            )
        )

    async def query(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        statement = select(AuditRecord)
        if actor:
            statement = statement.where(AuditRecord.actor == actor)
        if action:
            statement = statement.where(AuditRecord.action == action)
        if since is not None:
            statement = statement.where(AuditRecord.timestamp >= since)
        statement = statement.order_by(AuditRecord.timestamp.desc()).limit(
            min(max(1, limit), MAX_PAGE_SIZE)
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())


class WatchlistRepository:
    """Persistence for the face and plate watchlists."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # -- faces ------------------------------------------------------------- #

    async def upsert_face(
        self,
        person_id: str,
        embeddings: np.ndarray,
        *,
        name: str = "",
        category: str = "watchlist",
        notes: str = "",
        reference: str = "",
        created_by: str = "",
    ) -> FaceWatchRecord:
        """Store an identity's embeddings as a contiguous float32 blob."""
        matrix = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        result = await self.session.execute(
            select(FaceWatchRecord).where(FaceWatchRecord.person_id == person_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = FaceWatchRecord(person_id=person_id, created_by=created_by)
            self.session.add(record)

        record.name = name or record.name or person_id
        record.category = category or record.category
        record.notes = notes or record.notes
        record.reference = reference or record.reference
        record.embeddings = matrix.tobytes()
        record.embedding_dim = int(matrix.shape[1])
        record.embedding_count = int(matrix.shape[0])
        await self.session.flush()
        return record

    async def list_faces(self, *, active_only: bool = True) -> list[FaceWatchRecord]:
        statement = select(FaceWatchRecord)
        if active_only:
            statement = statement.where(FaceWatchRecord.active.is_(True))
        result = await self.session.execute(statement.order_by(FaceWatchRecord.name))
        return list(result.scalars().all())

    @staticmethod
    def decode_embeddings(record: FaceWatchRecord) -> np.ndarray:
        """Recover the ``(n, dim)`` embedding matrix from stored bytes."""
        if not record.embeddings or not record.embedding_dim:
            return np.empty((0, 0), dtype=np.float32)
        return np.frombuffer(record.embeddings, dtype=np.float32).reshape(
            record.embedding_count, record.embedding_dim
        )

    async def remove_face(self, person_id: str) -> bool:
        result = await self.session.execute(
            delete(FaceWatchRecord).where(FaceWatchRecord.person_id == person_id)
        )
        return bool(result.rowcount)

    # -- plates ------------------------------------------------------------ #

    async def upsert_plate(
        self,
        plate: str,
        *,
        category: str = "wanted",
        reason: str = "",
        reference: str = "",
        created_by: str = "",
        expires_at: float | None = None,
    ) -> PlateWatchRecord:
        result = await self.session.execute(
            select(PlateWatchRecord).where(PlateWatchRecord.plate == plate)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = PlateWatchRecord(plate=plate, created_by=created_by)
            self.session.add(record)
        record.category = category
        record.reason = reason
        record.reference = reference
        record.expires_at = expires_at
        await self.session.flush()
        return record

    async def list_plates(self, *, active_only: bool = True) -> list[PlateWatchRecord]:
        statement = select(PlateWatchRecord)
        if active_only:
            statement = statement.where(PlateWatchRecord.active.is_(True))
        result = await self.session.execute(statement.order_by(PlateWatchRecord.plate))
        return list(result.scalars().all())

    async def remove_plate(self, plate: str) -> bool:
        result = await self.session.execute(
            delete(PlateWatchRecord).where(PlateWatchRecord.plate == plate)
        )
        return bool(result.rowcount)


class CameraRepository:
    """Cameras added operationally rather than through the site YAML."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self, camera_id: str, config: dict[str, Any], *, created_by: str = ""
    ) -> CameraRecord:
        result = await self.session.execute(
            select(CameraRecord).where(CameraRecord.camera_id == camera_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = CameraRecord(camera_id=camera_id, created_by=created_by)
            self.session.add(record)
        record.config = config
        record.enabled = bool(config.get("enabled", True))
        record.updated_at = time.time()
        await self.session.flush()
        return record

    async def list_all(self) -> list[CameraRecord]:
        result = await self.session.execute(
            select(CameraRecord).order_by(CameraRecord.camera_id)
        )
        return list(result.scalars().all())

    async def remove(self, camera_id: str) -> bool:
        result = await self.session.execute(
            delete(CameraRecord).where(CameraRecord.camera_id == camera_id)
        )
        return bool(result.rowcount)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _severities_at_least(minimum: str) -> list[str]:
    """Expand "at least this severity" into the explicit set of values."""
    order = [s.value for s in (Severity.INFO, Severity.LOW, Severity.MEDIUM,
                               Severity.HIGH, Severity.CRITICAL)]
    try:
        index = order.index(minimum)
    except ValueError:
        return order
    return order[index:]


def record_to_event(record: EventRecord) -> Event:
    """Rehydrate a stored row into a domain :class:`Event`."""
    return Event(
        camera_id=record.camera_id,
        event_type=EventType(record.event_type),
        severity=Severity(record.severity),
        timestamp=record.timestamp,
        event_id=record.event_id,
        confidence=record.confidence,
        message=record.message,
        rule_id=record.rule_id,
        zone_id=record.zone_id,
        track_ids=list(record.track_ids or []),
        boxes=[BBox(*b) for b in (record.boxes or [])],
        attributes=dict(record.attributes or {}),
        snapshot_path=record.snapshot_path,
        clip_path=record.clip_path,
        evidence_hash=record.evidence_hash,
        frame_index=record.frame_index,
    )


def _json_safe(value: Any) -> Any:
    """Coerce values into JSON-serialisable form for JSON columns."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
