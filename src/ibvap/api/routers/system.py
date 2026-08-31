"""Health, metrics, model registry and audit endpoints."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from ibvap.api.deps import (
    RequireAdmin,
    RequireSupervisor,
    RequireViewer,
    SessionDep,
    StateDep,
    audit,
)
from ibvap.api.schemas import AuditResponse, HealthResponse, MessageResponse
from ibvap.core.logging import get_logger
from ibvap.storage.repository import AuditRepository
from ibvap.telemetry.metrics import render_metrics

log = get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(state: StateDep) -> HealthResponse:
    """Full node health.

    Deliberately unauthenticated: a monitoring system, a load balancer or a
    duty operator's dashboard must be able to see that a node is down without
    holding a credential - and a node that is down cannot authenticate anyone
    anyway. It reports posture, never content: no event text, no watchlist
    entries, no camera URLs.
    """
    return HealthResponse(**await state.health())


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    """Process liveness for an orchestrator. Cheap and always available."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(state: StateDep, response: Response) -> dict[str, object]:
    """Readiness: whether this node can currently do useful work.

    A node with every camera offline is *live* but not *ready*, and an
    orchestrator should be told the difference rather than left to infer it.
    """
    snapshot = await state.health()
    ready = snapshot.get("status") in ("healthy", "degraded", "partial")
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": ready,
        "status": snapshot.get("status"),
        "cameras_online": snapshot.get("cameras_online", 0),
        "cameras_total": snapshot.get("cameras_total", 0),
    }


@router.get("/metrics")
async def metrics(state: StateDep) -> Response:
    """Prometheus exposition endpoint."""
    if not state.settings.telemetry.prometheus_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "metrics are disabled")
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get("/system/models")
async def list_models(state: StateDep, principal: RequireViewer) -> dict:
    """The model registry as this node sees it, including degradation state."""
    bundle = state.supervisor.bundle if state.supervisor else None
    return {
        "registry_path": str(state.settings.models.registry_path),
        "registry_loaded_from": str(state.registry.loaded_from or ""),
        "providers": state.settings.models.providers,
        "entries": state.registry.describe(),
        "active": bundle.status() if bundle else {},
    }


@router.get("/system/rules")
async def list_rule_types(principal: RequireViewer) -> dict:
    """Every analytics rule type available, for the configuration UI."""
    from ibvap.analytics import available_rules

    return {
        "rules": [
            {
                "type": name,
                "event_type": cls.event_type.value,
                "requires_tracks": cls.requires_tracks,
                "description": (cls.__doc__ or "").strip().split("\n")[0],
            }
            for name, cls in sorted(available_rules().items())
        ]
    }


@router.get("/system/config")
async def get_config(state: StateDep, principal: RequireSupervisor) -> dict:
    """The effective configuration, with secrets removed.

    Secrets are stripped rather than masked-in-place so that a copy of this
    response pasted into a ticket cannot leak anything - which is exactly how
    configuration dumps travel in practice.
    """
    config = state.settings.model_dump(mode="json")
    config["security"].pop("jwt_secret", None)
    config["security"].pop("bootstrap_admin_password", None)
    for webhook in config.get("integrations", {}).get("webhooks", []):
        webhook.pop("hmac_secret", None)
    for camera in config.get("cameras", []):
        camera["url"] = _redact(camera.get("url", ""))
    return config


@router.get("/system/audit", response_model=list[AuditResponse])
async def get_audit_log(
    session: SessionDep,
    principal: RequireAdmin,
    actor: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
    hours: Annotated[float, Query(gt=0, le=8760)] = 168.0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AuditResponse]:
    """Read the security audit trail. Administrators only."""
    records = await AuditRepository(session).query(
        actor=actor, action=action, since=time.time() - hours * 3600, limit=limit
    )
    return [AuditResponse.from_record(r) for r in records]


@router.post("/system/evidence/prune", response_model=MessageResponse)
async def prune_evidence(
    request: Request, state: StateDep, session: SessionDep, principal: RequireAdmin
) -> MessageResponse:
    """Run the retention sweep immediately rather than waiting for the janitor."""
    import asyncio

    result = await asyncio.to_thread(state.evidence.prune)
    await audit(session, principal, "evidence.prune", detail=result, request=request)
    return MessageResponse(
        message=f"removed {result['files']} artefacts",
        detail={"megabytes_freed": round(result["bytes"] / 1024**2, 1)},
    )


@router.get("/system/evidence/verify/{day}")
async def verify_evidence_chain(
    day: str, state: StateDep, principal: RequireAdmin
) -> dict:
    """Verify the evidence hash chain for one day (``YYYY-MM-DD``)."""
    import re

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "day must be formatted YYYY-MM-DD")
    return state.evidence.verify_chain(day)


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}"
