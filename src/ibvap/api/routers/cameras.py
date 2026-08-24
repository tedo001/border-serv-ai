"""Camera configuration, live status and video preview."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

import cv2
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from ibvap.api.deps import (
    RequireSupervisor,
    RequireViewer,
    SessionDep,
    StateDep,
    audit,
)
from ibvap.api.schemas import CameraPayload, CameraStatusResponse, MessageResponse
from ibvap.api.status_codes import HTTP_422_UNPROCESSABLE
from ibvap.core.config import CameraConfig
from ibvap.core.errors import ConfigError
from ibvap.core.logging import get_logger
from ibvap.events.annotate import annotate_frame
from ibvap.storage.repository import CameraRepository

log = get_logger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraStatusResponse])
async def list_cameras(state: StateDep, principal: RequireViewer) -> list[CameraStatusResponse]:
    """Live status of every camera on this node."""
    if state.supervisor is None:
        return []
    return [CameraStatusResponse(**s) for s in state.supervisor.status()]


@router.get("/{camera_id}", response_model=CameraStatusResponse)
async def get_camera(
    camera_id: str, state: StateDep, principal: RequireViewer
) -> CameraStatusResponse:
    worker = state.supervisor.worker(camera_id) if state.supervisor else None
    if worker is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")
    return CameraStatusResponse(**worker.status())


@router.get("/{camera_id}/config", response_model=CameraPayload)
async def get_camera_config(
    camera_id: str, state: StateDep, principal: RequireViewer
) -> CameraPayload:
    """The full configuration of one camera, for the zone editor."""
    try:
        camera = state.settings.camera(camera_id)
    except ConfigError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return CameraPayload(**camera.model_dump())


@router.post("", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    payload: CameraPayload,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> MessageResponse:
    """Add a camera and start analytics on it immediately."""
    if state.supervisor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "supervisor is not running")
    if state.supervisor.worker(payload.id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "camera already exists")

    camera = _to_config(payload)
    try:
        state.supervisor.add_camera(camera)
    except ConfigError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    # Persist so the camera survives a restart. The site YAML remains the
    # source of truth for provisioned cameras; this table holds only those
    # added operationally.
    await CameraRepository(session).upsert(
        camera.id, camera.model_dump(mode="json"), created_by=principal.username
    )
    await audit(
        session, principal, "camera.create", target=camera.id,
        detail={"url": _redact(camera.url), "zones": len(camera.zones)}, request=request,
    )
    return MessageResponse(message=f"camera {camera.id} added and started")


@router.put("/{camera_id}", response_model=MessageResponse)
async def update_camera(
    camera_id: str,
    payload: CameraPayload,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> MessageResponse:
    """Replace a camera's configuration and restart its worker."""
    if state.supervisor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "supervisor is not running")
    if payload.id != camera_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "camera id in body must match the path")
    if state.supervisor.worker(camera_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")

    camera = _to_config(payload)
    state.supervisor.update_camera(camera)
    await CameraRepository(session).upsert(
        camera.id, camera.model_dump(mode="json"), created_by=principal.username
    )
    await audit(
        session, principal, "camera.update", target=camera_id,
        detail={"zones": len(camera.zones), "rules": len(camera.rules)}, request=request,
    )
    return MessageResponse(message=f"camera {camera_id} updated and restarted")


@router.delete("/{camera_id}", response_model=MessageResponse)
async def delete_camera(
    camera_id: str,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> MessageResponse:
    if state.supervisor is None or not state.supervisor.remove_camera(camera_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")
    await CameraRepository(session).remove(camera_id)
    await audit(session, principal, "camera.delete", target=camera_id, request=request)
    return MessageResponse(message=f"camera {camera_id} removed")


@router.post("/{camera_id}/restart", response_model=MessageResponse)
async def restart_camera(
    camera_id: str,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> MessageResponse:
    if state.supervisor is None or not state.supervisor.restart_camera(camera_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")
    await audit(session, principal, "camera.restart", target=camera_id, request=request)
    return MessageResponse(message=f"camera {camera_id} restarted")


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #


@router.get("/{camera_id}/snapshot")
async def camera_snapshot(
    camera_id: str,
    state: StateDep,
    principal: RequireViewer,
    annotate: Annotated[bool, Query()] = True,
    quality: Annotated[int, Query(ge=40, le=100)] = 80,
) -> StreamingResponse:
    """A single annotated JPEG of the camera's most recent frame."""
    worker = state.supervisor.worker(camera_id) if state.supervisor else None
    if worker is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")

    latest = worker.latest()
    if latest is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no frame available yet")

    frame, tracks = latest
    image = (
        annotate_frame(
            frame.image, tracks=tracks,
            zones=list(worker.analytics.zones.values()),
            tripwires=list(worker.analytics.tripwires.values()),
            camera_name=worker.camera.name, timestamp=frame.timestamp,
        )
        if annotate else frame.image
    )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to encode frame")

    return StreamingResponse(
        iter([encoded.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{camera_id}/stream.mjpeg")
async def camera_mjpeg(
    camera_id: str,
    state: StateDep,
    principal: RequireViewer,
    fps: Annotated[float, Query(gt=0, le=15)] = 5.0,
    quality: Annotated[int, Query(ge=40, le=95)] = 70,
    annotate: Annotated[bool, Query()] = True,
) -> StreamingResponse:
    """Live MJPEG preview.

    MJPEG rather than HLS or WebRTC because it works in every browser and every
    legacy video wall with no client-side player, no transcoding pipeline and
    no negotiation. The cost is bandwidth, which is why the frame rate is
    capped, the encode happens only while a client is attached, and the number
    of concurrent subscribers is limited - on a VSAT-connected node an
    unbounded preview would consume the same link the alerts travel over.
    """
    worker = state.supervisor.worker(camera_id) if state.supervisor else None
    if worker is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such camera")

    limit = state.settings.api.max_preview_clients
    counters: dict[str, int] = getattr(state, "_preview_clients", None) or {}
    state._preview_clients = counters  # type: ignore[attr-defined]
    if limit and sum(counters.values()) >= limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"preview client limit reached ({limit}); close another preview first",
        )
    counters[camera_id] = counters.get(camera_id, 0) + 1

    interval = 1.0 / fps
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    async def generate():
        last_index = -1
        try:
            while True:
                started = time.monotonic()
                latest = worker.latest()
                if latest is not None and latest[0].index != last_index:
                    frame, tracks = latest
                    last_index = frame.index
                    image = (
                        annotate_frame(
                            frame.image, tracks=tracks,
                            zones=list(worker.analytics.zones.values()),
                            tripwires=list(worker.analytics.tripwires.values()),
                            camera_name=worker.camera.name, timestamp=frame.timestamp,
                        )
                        if annotate else frame.image
                    )
                    # Encoding is CPU work; running it in a thread keeps the
                    # event loop free to serve alerts while a preview streams.
                    ok, encoded = await asyncio.to_thread(
                        cv2.imencode, ".jpg", image, encode_params
                    )
                    if ok:
                        chunk = encoded.tobytes()
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n"
                            + chunk + b"\r\n"
                        )
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        finally:
            counters[camera_id] = max(0, counters.get(camera_id, 1) - 1)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _to_config(payload: CameraPayload) -> CameraConfig:
    """Convert an API payload into a validated camera configuration."""
    try:
        return CameraConfig(**payload.model_dump())
    except Exception as exc:
        # Surfaces cross-field problems the payload schema cannot express, such
        # as a rule referencing a zone that is not defined on this camera.
        raise HTTPException(HTTP_422_UNPROCESSABLE, str(exc)) from exc


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"
