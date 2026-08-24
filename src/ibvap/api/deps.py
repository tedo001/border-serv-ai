"""FastAPI dependencies: authentication, authorisation and state access."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ibvap.api.security import Principal, Role
from ibvap.api.state import AppState
from ibvap.core.errors import AuthError
from ibvap.core.logging import get_logger
from ibvap.storage.repository import ApiKeyRepository, AuditRepository, UserRepository

log = get_logger(__name__)


def get_state(request: Request) -> AppState:
    """The application state attached at start-up."""
    state: AppState | None = getattr(request.app.state, "ibvap", None)
    if state is None:  # pragma: no cover - only if wiring is broken
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "node is not initialised")
    return state


StateDep = Annotated[AppState, Depends(get_state)]


async def get_session(state: StateDep) -> AsyncIterator[AsyncSession]:
    """A database session scoped to one request."""
    async with state.database.session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def client_ip(request: Request) -> str:
    """Best-effort source address for audit and throttling.

    ``X-Forwarded-For`` is honoured because a node is normally behind a reverse
    proxy that terminates TLS. It is trusted only for *logging and throttling* -
    never for authorisation - since a client can set it freely.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


#: Endpoints whose responses are loaded by a browser element that cannot set
#: request headers - ``<img src>``, ``<video src>`` and ``WebSocket``. Only
#: these accept a token in the query string.
_QUERY_TOKEN_SUFFIXES: tuple[str, ...] = (
    "/snapshot", "/clip", "/stream.mjpeg", "/stream",
)


def _query_token_allowed(request: Request) -> bool:
    """Whether this request may authenticate with a ``?token=`` parameter.

    A token in a URL is weaker than one in a header: it lands in proxy access
    logs, browser history and ``Referer``. It is permitted only where the
    platform genuinely has no alternative - an ``<img>`` tag cannot send an
    Authorization header - and only for read-only media GETs, so a leaked URL
    can never be replayed into a state-changing call.
    """
    return request.method == "GET" and request.url.path.endswith(_QUERY_TOKEN_SUFFIXES)


async def authenticate(
    request: Request,
    state: StateDep,
    session: SessionDep,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the caller from a bearer token, an API key, or a media query token."""
    if not state.settings.security.require_auth:
        # Explicit opt-out for isolated bench work. Loud, so it cannot be
        # mistaken for normal operation.
        return Principal(username="anonymous", role=Role.ADMIN, kind="disabled")

    if not authorization and _query_token_allowed(request):
        query_token = request.query_params.get("token", "")
        if query_token:
            authorization = f"Bearer {query_token}"

    if x_api_key:
        record = await ApiKeyRepository(session).verify(x_api_key)
        if record is None:
            _unauthorised("invalid or expired API key")
        return Principal(
            username=f"key:{record.name}", role=Role.coerce(record.role), kind="api_key"
        )

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        try:
            principal = state.tokens.verify(token)
        except AuthError as exc:
            _unauthorised(str(exc))

        # A token remains cryptographically valid after an account is disabled,
        # so the account state must be re-checked on every request rather than
        # trusted from the claim.
        user = await UserRepository(session).get(principal.username)
        if user is None or not user.active:
            _unauthorised("account is disabled or no longer exists")
        return principal

    _unauthorised("authentication required")
    raise AssertionError("unreachable")  # pragma: no cover


def _unauthorised(detail: str) -> None:
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


PrincipalDep = Annotated[Principal, Depends(authenticate)]


def require_role(role: Role):
    """Build a dependency enforcing a minimum role."""

    async def _dependency(principal: PrincipalDep) -> Principal:
        if not principal.role.satisfies(role):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"requires {role.value} role; you have {principal.role.value}",
            )
        return principal

    return _dependency


RequireViewer = Annotated[Principal, Depends(require_role(Role.VIEWER))]
RequireOperator = Annotated[Principal, Depends(require_role(Role.OPERATOR))]
RequireSupervisor = Annotated[Principal, Depends(require_role(Role.SUPERVISOR))]
RequireAdmin = Annotated[Principal, Depends(require_role(Role.ADMIN))]


async def audit(
    session: AsyncSession,
    principal: Principal,
    action: str,
    *,
    target: str = "",
    detail: dict | None = None,
    request: Request | None = None,
    success: bool = True,
) -> None:
    """Record a security-relevant action in the audit trail."""
    await AuditRepository(session).record(
        principal.username,
        action,
        target=target,
        detail=detail or {},
        source_ip=client_ip(request) if request else "",
        success=success,
    )
