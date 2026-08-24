"""Authentication and account management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from ibvap.api.deps import (
    PrincipalDep,
    RequireAdmin,
    SessionDep,
    StateDep,
    audit,
    client_ip,
)
from ibvap.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyResponse,
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    RefreshRequest,
    TokenResponse,
    UserCreateRequest,
    UserResponse,
)
from ibvap.api.security import (
    Role,
    hash_password,
    password_strength_issues,
    verify_password,
)
from ibvap.api.status_codes import HTTP_422_UNPROCESSABLE
from ibvap.core.errors import AuthError
from ibvap.core.logging import get_logger
from ibvap.storage.repository import ApiKeyRepository, UserRepository

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest, request: Request, state: StateDep, session: SessionDep
) -> TokenResponse:
    """Exchange credentials for an access and refresh token pair."""
    source = client_ip(request)
    try:
        state.throttle.check(payload.username, source)
    except AuthError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc)) from exc

    users = UserRepository(session)
    user = await users.get(payload.username)

    # One indistinguishable failure for every cause. Separate messages for
    # "no such user" and "wrong password" hand an attacker a way to enumerate
    # valid operator accounts.
    if user is None or not user.active or not verify_password(payload.password, user.password_hash):
        state.throttle.record_failure(payload.username, source)
        await audit(
            session, _anonymous(payload.username), "auth.login",
            target=payload.username, request=request, success=False,
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")

    state.throttle.record_success(payload.username, source)
    role = Role.coerce(user.role)
    access, expires_at = state.tokens.issue(user.username, role)
    refresh, _ = state.tokens.issue(user.username, role, refresh=True)

    await users.touch_login(user.username)
    await audit(session, _anonymous(user.username), "auth.login",
                target=user.username, request=request)

    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_at=expires_at,
        role=role.value,
        username=user.username,
        must_change_password=user.must_change_password,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshRequest, state: StateDep, session: SessionDep
) -> TokenResponse:
    """Exchange a refresh token for a new access token."""
    try:
        principal = state.tokens.verify(payload.refresh_token, expect_refresh=True)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    user = await UserRepository(session).get(principal.username)
    if user is None or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "account is disabled")

    role = Role.coerce(user.role)
    access, expires_at = state.tokens.issue(user.username, role)
    return TokenResponse(
        access_token=access,
        refresh_token=payload.refresh_token,
        expires_at=expires_at,
        role=role.value,
        username=user.username,
        must_change_password=user.must_change_password,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request, state: StateDep, session: SessionDep, principal: PrincipalDep
) -> MessageResponse:
    """Revoke the presented access token."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        state.tokens.revoke(header[7:].strip())
    await audit(session, principal, "auth.logout", request=request)
    return MessageResponse(message="logged out")


@router.get("/me")
async def whoami(principal: PrincipalDep) -> dict:
    """Describe the calling identity."""
    return principal.as_dict()


@router.post("/password", response_model=MessageResponse)
async def change_password(
    payload: PasswordChangeRequest,
    request: Request,
    session: SessionDep,
    principal: PrincipalDep,
) -> MessageResponse:
    """Change the caller's own password."""
    if principal.is_machine:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "API keys have no password")

    users = UserRepository(session)
    user = await users.get(principal.username)
    if user is None or not verify_password(payload.current_password, user.password_hash):
        await audit(session, principal, "auth.password_change", request=request, success=False)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "current password is incorrect")

    if issues := password_strength_issues(payload.new_password):
        raise HTTPException(
            HTTP_422_UNPROCESSABLE, f"password {'; '.join(issues)}"
        )

    await users.set_password(principal.username, hash_password(payload.new_password))
    await audit(session, principal, "auth.password_change", request=request)
    return MessageResponse(message="password changed")


# --------------------------------------------------------------------------- #
# User administration
# --------------------------------------------------------------------------- #


@router.get("/users", response_model=list[UserResponse])
async def list_users(session: SessionDep, principal: RequireAdmin) -> list[UserResponse]:
    records = await UserRepository(session).list_all()
    return [
        UserResponse(
            username=r.username, role=r.role, full_name=r.full_name, active=r.active,
            created_at=r.created_at, last_login=r.last_login,
        )
        for r in records
    ]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreateRequest,
    request: Request,
    session: SessionDep,
    principal: RequireAdmin,
) -> UserResponse:
    users = UserRepository(session)
    if await users.get(payload.username) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "username already exists")
    if issues := password_strength_issues(payload.password):
        raise HTTPException(
            HTTP_422_UNPROCESSABLE, f"password {'; '.join(issues)}"
        )

    record = await users.create(
        payload.username,
        hash_password(payload.password),
        role=payload.role,
        full_name=payload.full_name,
    )
    await audit(
        session, principal, "user.create", target=payload.username,
        detail={"role": payload.role}, request=request,
    )
    return UserResponse(
        username=record.username, role=record.role, full_name=record.full_name,
        active=record.active, created_at=record.created_at, last_login=None,
    )


@router.delete("/users/{username}", response_model=MessageResponse)
async def deactivate_user(
    username: str, request: Request, session: SessionDep, principal: RequireAdmin
) -> MessageResponse:
    """Deactivate an account. Accounts are never deleted, only disabled.

    Audit records reference the actor by username; deleting the row would
    orphan every action that account ever took.
    """
    if username == principal.username:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot deactivate your own account")
    if not await UserRepository(session).set_active(username, False):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such user")
    await audit(session, principal, "user.deactivate", target=username, request=request)
    return MessageResponse(message=f"user {username} deactivated")


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(session: SessionDep, principal: RequireAdmin) -> list[ApiKeyResponse]:
    records = await ApiKeyRepository(session).list_all()
    return [
        ApiKeyResponse(
            name=r.name, role=r.role, key_prefix=r.key_prefix, active=r.active,
            created_at=r.created_at, last_used=r.last_used, expires_at=r.expires_at,
        )
        for r in records
    ]


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    request: Request,
    session: SessionDep,
    principal: RequireAdmin,
) -> ApiKeyResponse:
    """Mint an API key. The key value is returned once and never again."""
    record, key = await ApiKeyRepository(session).create(
        payload.name, role=payload.role, created_by=principal.username, ttl_days=payload.ttl_days
    )
    await audit(
        session, principal, "apikey.create", target=payload.name,
        detail={"role": payload.role, "ttl_days": payload.ttl_days}, request=request,
    )
    return ApiKeyResponse(
        name=record.name, role=record.role, key_prefix=record.key_prefix,
        active=record.active, created_at=record.created_at,
        expires_at=record.expires_at, key=key,
    )


@router.delete("/api-keys/{name}", response_model=MessageResponse)
async def revoke_api_key(
    name: str, request: Request, session: SessionDep, principal: RequireAdmin
) -> MessageResponse:
    if not await ApiKeyRepository(session).revoke(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such API key")
    await audit(session, principal, "apikey.revoke", target=name, request=request)
    return MessageResponse(message=f"API key {name} revoked")


def _anonymous(username: str):
    """A principal stand-in for auditing pre-authentication events."""
    from ibvap.api.security import Principal

    return Principal(username=username or "anonymous", role=Role.VIEWER)
