"""Event query, acknowledgement, evidence retrieval and the live alert feed."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse

from ibvap.api.deps import (
    RequireOperator,
    RequireViewer,
    SessionDep,
    StateDep,
    audit,
)
from ibvap.api.schemas import (
    AcknowledgeRequest,
    EventPage,
    EventResponse,
    MessageResponse,
)
from ibvap.core.errors import AuthError
from ibvap.core.logging import get_logger
from ibvap.events.payload import event_to_payload
from ibvap.storage.repository import EventRepository

log = get_logger(__name__)
router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=EventPage)
async def query_events(
    state: StateDep,
    session: SessionDep,
    principal: RequireViewer,
    camera_id: Annotated[list[str] | None, Query()] = None,
    event_type: Annotated[list[str] | None, Query()] = None,
    severity: Annotated[list[str] | None, Query()] = None,
    min_severity: Annotated[str | None, Query()] = None,
    since: Annotated[float | None, Query()] = None,
    until: Annotated[float | None, Query()] = None,
    acknowledged: Annotated[bool | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
) -> EventPage:
    """Search stored events."""
    repository = EventRepository(session, state.settings.site_id)
    records = await repository.query(
        camera_ids=camera_id, event_types=event_type, severities=severity,
        min_severity=min_severity, since=since, until=until,
        acknowledged=acknowledged, search=search,
        limit=limit, offset=offset, order=order,
    )
    return EventPage(
        events=[EventResponse.from_record(r) for r in records],
        total=await repository.count(since=since),
        limit=limit,
        offset=offset,
    )


@router.get("/statistics")
async def event_statistics(
    state: StateDep,
    session: SessionDep,
    principal: RequireViewer,
    hours: Annotated[float, Query(gt=0, le=8760)] = 24.0,
) -> dict:
    """Aggregated counts for the dashboard."""
    return await EventRepository(session, state.settings.site_id).statistics(
        since=time.time() - hours * 3600
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: str, session: SessionDep, principal: RequireViewer) -> EventResponse:
    record = await EventRepository(session).get(event_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")
    return EventResponse.from_record(record)


@router.post("/{event_id}/acknowledge", response_model=MessageResponse)
async def acknowledge_event(
    event_id: str,
    payload: AcknowledgeRequest,
    request: Request,
    session: SessionDep,
    principal: RequireOperator,
) -> MessageResponse:
    """Record that an operator has actioned this alert."""
    repository = EventRepository(session)
    if await repository.get(event_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")
    if not await repository.acknowledge(event_id, principal.username, payload.disposition):
        raise HTTPException(status.HTTP_409_CONFLICT, "event is already acknowledged")

    await audit(
        session, principal, "event.acknowledge", target=event_id,
        detail={"disposition": payload.disposition}, request=request,
    )
    return MessageResponse(message="acknowledged")


@router.get("/{event_id}/snapshot")
async def get_snapshot(event_id: str, session: SessionDep, principal: RequireViewer):
    """Fetch the evidence snapshot for an event."""
    return await _serve_artefact(session, event_id, "snapshot")


@router.get("/{event_id}/clip")
async def get_clip(event_id: str, session: SessionDep, principal: RequireViewer):
    """Fetch the evidence clip for an event."""
    return await _serve_artefact(session, event_id, "clip")


async def _serve_artefact(session: SessionDep, event_id: str, kind: str) -> FileResponse:
    record = await EventRepository(session).get(event_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")

    raw = record.snapshot_path if kind == "snapshot" else record.clip_path
    if not raw:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no {kind} stored for this event")

    path = Path(raw)
    # Off the event loop: evidence often lives on slow or network-backed
    # storage, and a stat that blocks here stalls every other request the node
    # is serving, including live alert delivery.
    if not await asyncio.to_thread(path.is_file):
        # The row outlives the file when retention has pruned the artefact.
        raise HTTPException(
            status.HTTP_410_GONE, f"{kind} was removed by the retention policy"
        )

    return FileResponse(
        path,
        media_type="image/jpeg" if kind == "snapshot" else "video/mp4",
        filename=f"{event_id}{path.suffix}",
        headers={
            # The digest lets a reviewer verify the artefact against the
            # manifest without shell access to the node.
            "X-IBVAP-SHA256": record.evidence_hash or "",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/{event_id}/verify")
async def verify_evidence(
    event_id: str, state: StateDep, session: SessionDep, principal: RequireViewer
) -> dict:
    """Verify an event's evidence against its manifest (chain of custody)."""
    record = await EventRepository(session).get(event_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")
    if not record.snapshot_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no evidence stored for this event")

    manifest = Path(record.snapshot_path).with_suffix(".json")
    if not manifest.is_file():
        raise HTTPException(status.HTTP_410_GONE, "evidence manifest is no longer present")
    return state.evidence.verify(manifest)


# --------------------------------------------------------------------------- #
# Live alert feed
# --------------------------------------------------------------------------- #


@router.websocket("/stream")
async def event_stream(websocket: WebSocket, min_severity: str = "info") -> None:
    """Push live events to a connected console.

    Authentication happens over the socket rather than through the usual header
    dependency: browser ``WebSocket`` cannot set request headers, so the token
    is passed as a query parameter or as the first message. The connection is
    accepted first and closed immediately on a bad token, which is the only way
    to return a meaningful reason to the client.
    """
    state = getattr(websocket.app.state, "ibvap", None)
    if state is None:  # pragma: no cover
        await websocket.close(code=1011, reason="node not initialised")
        return

    await websocket.accept()

    if state.settings.security.require_auth:
        token = websocket.query_params.get("token", "")
        if not token:
            try:
                first = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                token = json.loads(first).get("token", "")
            except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
                await websocket.close(code=1008, reason="authentication required")
                return
        try:
            principal = state.tokens.verify(token)
        except AuthError as exc:
            await websocket.close(code=1008, reason=str(exc))
            return
    else:
        from ibvap.api.security import Principal, Role

        principal = Principal(username="anonymous", role=Role.ADMIN, kind="disabled")

    from ibvap.integrations.base import _ORDER

    threshold = _ORDER.get(min_severity, 0)
    # A bounded queue rather than direct sends: the subscriber callback runs on
    # a camera thread and must never block on a slow console's socket.
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()

    def on_event(event) -> None:
        if _ORDER.get(event.severity.value, 0) < threshold:
            return
        payload = event_to_payload(
            event, site_id=state.settings.site_id, site_name=state.settings.site_name
        )
        try:
            loop.call_soon_threadsafe(queue.put_nowait, payload)
        except (asyncio.QueueFull, RuntimeError):
            pass  # a console that cannot keep up simply misses frames of the feed

    unsubscribe = state.subscribe(on_event)
    log.info("alert_stream_opened", user=principal.username, min_severity=min_severity)

    try:
        await websocket.send_json({
            "type": "connected",
            "site_id": state.settings.site_id,
            "min_severity": min_severity,
        })
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=25.0)
            except TimeoutError:
                # Proxies and VSAT NAT tables drop idle connections; a periodic
                # heartbeat keeps the path open and lets the console notice a
                # dead link instead of waiting silently forever.
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
                continue
            await websocket.send_json({"type": "event", "event": payload})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - transport level
        log.debug("alert_stream_error", error=str(exc))
    finally:
        unsubscribe()
        log.info("alert_stream_closed", user=principal.username)
