"""Watchlist management for faces and vehicle registrations.

Every endpoint here is audited without exception. These lists decide who gets
stopped at a border, so "who added this entry, when, and on what authority" has
to be answerable months later. That is why ``reference`` (a case file or
intelligence report) is carried on every entry.
"""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status

from ibvap.api.deps import RequireOperator, RequireSupervisor, SessionDep, StateDep, audit
from ibvap.api.schemas import (
    FaceEnrolRequest,
    FaceWatchResponse,
    MessageResponse,
    PlateWatchRequest,
    PlateWatchResponse,
)
from ibvap.api.status_codes import HTTP_413_TOO_LARGE, HTTP_422_UNPROCESSABLE
from ibvap.core.logging import get_logger
from ibvap.storage.repository import WatchlistRepository
from ibvap.vision.anpr import normalise_plate

log = get_logger(__name__)
router = APIRouter(prefix="/watchlists", tags=["watchlists"])

#: Cap on a decoded enrolment image. A face crop is small; anything larger is
#: either a mistake or an attempt to exhaust memory on an edge node.
MAX_IMAGE_BYTES = 8 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Vehicle registrations
# --------------------------------------------------------------------------- #


@router.get("/plates", response_model=list[PlateWatchResponse])
async def list_plates(session: SessionDep, principal: RequireOperator) -> list[PlateWatchResponse]:
    records = await WatchlistRepository(session).list_plates(active_only=False)
    return [
        PlateWatchResponse(
            plate=r.plate, category=r.category, reason=r.reason, reference=r.reference,
            active=r.active, created_at=r.created_at, created_by=r.created_by,
            expires_at=r.expires_at,
        )
        for r in records
    ]


@router.post("/plates", response_model=PlateWatchResponse, status_code=status.HTTP_201_CREATED)
async def add_plate(
    payload: PlateWatchRequest,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireOperator,
) -> PlateWatchResponse:
    """Add a registration of interest.

    The plate is normalised through the same grammar as a live ANPR read, so a
    typo or a spaced format cannot sit in the list silently matching nothing.
    """
    reading = normalise_plate(payload.plate, 1.0)
    if not reading.text:
        raise HTTPException(HTTP_422_UNPROCESSABLE, "plate is unreadable")
    if not reading.valid:
        # Rejected rather than accepted-with-a-warning: an ungrammatical entry
        # can never match a normalised live read, so storing it would create a
        # watchlist entry that looks active but is inert.
        raise HTTPException(
            HTTP_422_UNPROCESSABLE,
            f"'{payload.plate}' is not a recognised Indian registration format "
            f"(interpreted as '{reading.text}')",
        )

    record = await WatchlistRepository(session).upsert_plate(
        reading.text,
        category=payload.category, reason=payload.reason,
        reference=payload.reference, created_by=principal.username,
        expires_at=payload.expires_at,
    )
    await audit(
        session, principal, "watchlist.plate.add", target=reading.text,
        detail={"category": payload.category, "reference": payload.reference}, request=request,
    )
    await state.reload_watchlists()
    return PlateWatchResponse(
        plate=record.plate, category=record.category, reason=record.reason,
        reference=record.reference, active=record.active, created_at=record.created_at,
        created_by=record.created_by, expires_at=record.expires_at,
    )


@router.delete("/plates/{plate}", response_model=MessageResponse)
async def remove_plate(
    plate: str,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireOperator,
) -> MessageResponse:
    normalised = normalise_plate(plate, 1.0).text or plate.upper()
    if not await WatchlistRepository(session).remove_plate(normalised):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "plate is not on the watchlist")
    await audit(session, principal, "watchlist.plate.remove", target=normalised, request=request)
    await state.reload_watchlists()
    return MessageResponse(message=f"{normalised} removed from the watchlist")


# --------------------------------------------------------------------------- #
# Face identities
# --------------------------------------------------------------------------- #


@router.get("/faces", response_model=list[FaceWatchResponse])
async def list_faces(session: SessionDep, principal: RequireOperator) -> list[FaceWatchResponse]:
    """List watchlist identities. Embeddings are never returned.

    An embedding is biometric material: it leaves the node only as a match
    decision, never as data an API consumer could accumulate into its own
    gallery.
    """
    records = await WatchlistRepository(session).list_faces(active_only=False)
    return [
        FaceWatchResponse(
            person_id=r.person_id, name=r.name, category=r.category, notes=r.notes,
            reference=r.reference, embedding_count=r.embedding_count, active=r.active,
            created_at=r.created_at, created_by=r.created_by,
        )
        for r in records
    ]


@router.post("/faces", response_model=FaceWatchResponse, status_code=status.HTTP_201_CREATED)
async def enrol_face(
    payload: FaceEnrolRequest,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> FaceWatchResponse:
    """Enrol an identity from a face image or a pre-computed embedding.

    Requires supervisor rather than operator: adding a person to a biometric
    watchlist is a materially different act from acknowledging an alert.
    """
    if state.supervisor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "node is not running")

    embedding = (
        _embedding_from_vector(payload.embedding)
        if payload.embedding
        else await _embedding_from_image(state, payload.image_base64)
    )

    repository = WatchlistRepository(session)
    existing = next(
        (r for r in await repository.list_faces(active_only=False)
         if r.person_id == payload.person_id),
        None,
    )
    # Additional views of an already-enrolled person improve recall far more
    # than threshold tuning does, so enrolment appends rather than replaces.
    if existing is not None:
        previous = WatchlistRepository.decode_embeddings(existing)
        matrix = (
            np.vstack([previous, embedding.reshape(1, -1)])
            if previous.size else embedding.reshape(1, -1)
        )
    else:
        matrix = embedding.reshape(1, -1)

    record = await repository.upsert_face(
        payload.person_id, matrix,
        name=payload.name, category=payload.category, notes=payload.notes,
        reference=payload.reference, created_by=principal.username,
    )
    await audit(
        session, principal, "watchlist.face.enrol", target=payload.person_id,
        detail={
            "category": payload.category,
            "reference": payload.reference,
            "views": int(matrix.shape[0]),
        },
        request=request,
    )
    await state.reload_watchlists()
    return FaceWatchResponse(
        person_id=record.person_id, name=record.name, category=record.category,
        notes=record.notes, reference=record.reference,
        embedding_count=record.embedding_count, active=record.active,
        created_at=record.created_at, created_by=record.created_by,
    )


@router.delete("/faces/{person_id}", response_model=MessageResponse)
async def remove_face(
    person_id: str,
    request: Request,
    state: StateDep,
    session: SessionDep,
    principal: RequireSupervisor,
) -> MessageResponse:
    if not await WatchlistRepository(session).remove_face(person_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "identity is not on the watchlist")
    await audit(session, principal, "watchlist.face.remove", target=person_id, request=request)
    await state.reload_watchlists()
    return MessageResponse(message=f"{person_id} removed from the watchlist")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _embedding_from_vector(values: list[float]) -> np.ndarray:
    from ibvap.vision.face import l2_normalize

    vector = np.asarray(values, dtype=np.float32)
    if vector.size == 0:
        raise HTTPException(HTTP_422_UNPROCESSABLE, "embedding is empty")
    return l2_normalize(vector)


async def _embedding_from_image(state: StateDep, image_base64: str | None) -> np.ndarray:
    """Detect and embed the single face in a supplied enrolment image."""
    if not image_base64:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE,
            "provide either image_base64 or embedding",
        )

    bundle = state.supervisor.bundle if state.supervisor else None
    if bundle is None or not bundle.face_available:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "face recognition models are not loaded on this node; "
            "enrol with a pre-computed embedding instead",
        )

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(HTTP_422_UNPROCESSABLE, "image is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(HTTP_413_TOO_LARGE, "enrolment image is too large")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(HTTP_422_UNPROCESSABLE, "image could not be decoded")

    faces = bundle.face_detector.detect(image)  # type: ignore[union-attr]
    if not faces:
        raise HTTPException(HTTP_422_UNPROCESSABLE, "no face found in the image")
    if len(faces) > 1:
        # Ambiguity must be resolved by the enroller, not guessed at. Picking
        # the largest face would eventually enrol the wrong person from a group
        # photograph, and nothing downstream would ever reveal the error.
        raise HTTPException(
            HTTP_422_UNPROCESSABLE,
            f"{len(faces)} faces found; supply an image containing exactly one face",
        )

    face = faces[0]
    if not face.quality_ok(
        min_height=bundle.face_detector.min_height,  # type: ignore[union-attr]
        min_focus=bundle.face_detector.min_focus,  # type: ignore[union-attr]
    ):
        raise HTTPException(
            HTTP_422_UNPROCESSABLE,
            f"face quality is insufficient for enrolment "
            f"(height {face.pixel_height:.0f}px, focus {face.focus:.0f}); "
            "supply a sharper, closer image",
        )

    from ibvap.vision.preprocess import crop

    embedding = bundle.face_embedder.embed(  # type: ignore[union-attr]
        crop(image, face.bbox, padding=0.15), face.landmarks
    )
    if embedding.size == 0:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "failed to compute a face embedding"
        )
    return embedding
