"""Authentication, authorisation and password handling.

Two credential types, deliberately separated by purpose:

* **JWT bearer tokens** for humans. Short-lived, refreshable, carrying a role.
* **API keys** for machines. Long-lived, individually revocable, scoped by
  role, and stored only as a hash - a C2 gateway integrating with this platform
  should never require an operator's personal password.

Roles are a strict hierarchy rather than a permission matrix. Border force
structure is hierarchical, and a matrix invites misconfiguration in a system
where the difference between ``viewer`` and ``operator`` is the ability to
enrol a face on a watchlist.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import bcrypt
import jwt

from ibvap.core.config import SecurityConfig
from ibvap.core.errors import AuthError
from ibvap.core.logging import get_logger

log = get_logger(__name__)

#: bcrypt work factor. Deliberately slow - the whole point is that an attacker
#: holding a stolen database cannot test candidates quickly. 12 costs roughly
#: 250 ms on the class of CPU an edge node uses, which is imperceptible on a
#: login and ruinous for an offline guessing attack.
BCRYPT_ROUNDS = 12

#: bcrypt silently truncates input beyond 72 bytes, so a longer passphrase
#: would have its tail ignored - two different passwords sharing a 72-byte
#: prefix would both authenticate. Pre-hashing with SHA-256 removes the limit
#: entirely; base64 keeps the digest free of NUL bytes, which bcrypt also
#: truncates on.
def _prepare(password: str) -> bytes:
    import base64

    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


class Role(str, Enum):
    """Access levels, ordered from least to most privileged."""

    VIEWER = "viewer"          # read events, watch live video
    OPERATOR = "operator"      # acknowledge alerts, manage watchlists
    SUPERVISOR = "supervisor"  # configure cameras, zones and rules
    ADMIN = "admin"            # users, API keys, system settings

    @property
    def level(self) -> int:
        return _ROLE_LEVEL[self]

    def satisfies(self, required: Role) -> bool:
        """Whether this role meets or exceeds ``required``."""
        return self.level >= required.level

    @classmethod
    def coerce(cls, value: str) -> Role:
        try:
            return cls(str(value).lower())
        except ValueError:
            # An unrecognised role must fail closed to the least privilege,
            # never open. A typo in configuration should reduce access, not
            # silently grant administrative rights.
            log.warning("unknown_role_defaulting_to_viewer", value=str(value))
            return cls.VIEWER


_ROLE_LEVEL: dict[Role, int] = {
    Role.VIEWER: 0,
    Role.OPERATOR: 1,
    Role.SUPERVISOR: 2,
    Role.ADMIN: 3,
}


@dataclass(slots=True)
class Principal:
    """The authenticated identity behind a request."""

    username: str
    role: Role
    #: ``password`` for an interactive login, ``api_key`` for a machine.
    kind: str = "password"
    token_id: str = ""
    expires_at: float = 0.0

    @property
    def is_machine(self) -> bool:
        return self.kind == "api_key"

    def require(self, role: Role) -> None:
        """Raise unless this principal meets ``role``."""
        if not self.role.satisfies(role):
            raise AuthError(
                f"role {self.role.value!r} is insufficient; {role.value!r} required"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "role": self.role.value,
            "kind": self.kind,
            "expires_at": self.expires_at or None,
        }


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    if not password:
        raise AuthError("password must not be empty")
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against its stored hash, never raising."""
    if not password or not password_hash:
        return False
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode())
    except (ValueError, TypeError):
        # A malformed hash in the database must read as "wrong password",
        # not as a 500 that reveals the account exists.
        return False


def password_strength_issues(password: str) -> list[str]:
    """Return reasons a password is unacceptable. Empty means acceptable."""
    issues: list[str] = []
    if len(password) < 12:
        issues.append("must be at least 12 characters")
    if not any(c.isupper() for c in password):
        issues.append("must contain an uppercase letter")
    if not any(c.islower() for c in password):
        issues.append("must contain a lowercase letter")
    if not any(c.isdigit() for c in password):
        issues.append("must contain a digit")
    if password.lower() in _COMMON_PASSWORDS:
        issues.append("is a commonly used password")
    return issues


_COMMON_PASSWORDS: frozenset[str] = frozenset(
    {"password", "password123", "admin", "admin123", "123456789012",
     "qwertyuiop12", "changeme", "changeme123", "ibvap", "ibvap12345"}
)


def generate_password(length: int = 20) -> str:
    """Generate a strong random password for bootstrap accounts."""
    return secrets.token_urlsafe(length)[:length]


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


class TokenService:
    """Issues and validates JWTs."""

    #: Minimum signing-key length. HS256 keys shorter than the hash output
    #: weaken the MAC (RFC 7518 s3.2), and a 16-character "secret" typed into
    #: a config file is exactly the kind of thing that ships to twenty sites
    #: and never gets rotated. Rejected at construction rather than warned
    #: about, so the weak configuration cannot reach production quietly.
    MIN_SECRET_BYTES = 32

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        if not config.jwt_secret:
            raise AuthError("security.jwt_secret is not configured")
        if len(config.jwt_secret.encode()) < self.MIN_SECRET_BYTES:
            raise AuthError(
                f"security.jwt_secret must be at least {self.MIN_SECRET_BYTES} bytes; "
                f"got {len(config.jwt_secret.encode())}. Generate one with "
                "`python -m ibvap.cli secret`."
            )
        self._secret = config.jwt_secret
        self._algorithm = config.jwt_algorithm
        #: Revoked token identifiers, for explicit logout.
        self._revoked: dict[str, float] = {}

    def issue(self, username: str, role: Role, *, refresh: bool = False) -> tuple[str, float]:
        """Mint a token. Returns the token and its expiry (unix seconds)."""
        now = time.time()
        ttl = (
            self.config.refresh_token_ttl_days * 86400
            if refresh
            else self.config.access_token_ttl_minutes * 60
        )
        expires_at = now + ttl
        payload = {
            "sub": username,
            "role": role.value,
            "iat": int(now),
            "exp": int(expires_at),
            "jti": secrets.token_urlsafe(12),
            "typ": "refresh" if refresh else "access",
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm), expires_at

    def verify(self, token: str, *, expect_refresh: bool = False) -> Principal:
        """Decode and validate a token, raising :class:`AuthError` if invalid."""
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("token is invalid") from exc

        token_id = str(payload.get("jti", ""))
        if token_id in self._revoked:
            raise AuthError("token has been revoked")

        expected_type = "refresh" if expect_refresh else "access"
        if payload.get("typ") != expected_type:
            # Without this check a long-lived refresh token would be accepted
            # as an access token, defeating the point of short access TTLs.
            raise AuthError(f"expected a {expected_type} token")

        return Principal(
            username=str(payload.get("sub", "")),
            role=Role.coerce(str(payload.get("role", "viewer"))),
            kind="password",
            token_id=token_id,
            expires_at=float(payload.get("exp", 0)),
        )

    def revoke(self, token: str) -> None:
        """Revoke a token by its identifier (logout)."""
        try:
            payload = jwt.decode(
                token, self._secret, algorithms=[self._algorithm],
                options={"verify_exp": False},
            )
        except jwt.InvalidTokenError:
            return
        token_id = str(payload.get("jti", ""))
        if token_id:
            self._revoked[token_id] = float(payload.get("exp", time.time()))
        self._prune_revoked()

    def _prune_revoked(self) -> None:
        """Forget revocations for tokens that have expired anyway.

        The revocation list is in-process and would otherwise grow without
        bound on a node that runs for months.
        """
        now = time.time()
        self._revoked = {k: v for k, v in self._revoked.items() if v > now}


# --------------------------------------------------------------------------- #
# Brute-force protection
# --------------------------------------------------------------------------- #


class LoginThrottle:
    """Per-identity lockout after repeated failures.

    Keyed on username *and* source address together: keying on address alone
    locks out an entire post behind one NAT when a single operator fat-fingers
    their password, and keying on username alone lets an attacker lock a real
    operator out of their own console.
    """

    def __init__(self, max_attempts: int = 5, lockout_seconds: float = 300.0) -> None:
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}

    @staticmethod
    def _key(username: str, source_ip: str) -> str:
        return f"{username.lower()}|{source_ip}"

    def check(self, username: str, source_ip: str) -> None:
        """Raise :class:`AuthError` when this identity is locked out."""
        key = self._key(username, source_ip)
        until = self._locked_until.get(key, 0.0)
        if until > time.time():
            raise AuthError(
                f"too many failed attempts; locked for {int(until - time.time())}s"
            )

    def record_failure(self, username: str, source_ip: str) -> None:
        key = self._key(username, source_ip)
        now = time.time()
        window = [t for t in self._failures.get(key, []) if now - t < self.lockout_seconds]
        window.append(now)
        self._failures[key] = window
        if len(window) >= self.max_attempts:
            self._locked_until[key] = now + self.lockout_seconds
            log.warning(
                "login_lockout", username=username, source_ip=source_ip,
                attempts=len(window), seconds=self.lockout_seconds,
            )

    def record_success(self, username: str, source_ip: str) -> None:
        key = self._key(username, source_ip)
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)


def hash_api_key(key: str) -> str:
    """Hash an API key for storage and lookup.

    Plain SHA-256 rather than bcrypt: an API key is 256 bits of random data,
    so it has no dictionary to attack and needs no key-stretching. It is also
    presented on every request, where a deliberately slow hash would be a
    self-inflicted denial of service.
    """
    return hashlib.sha256(key.encode()).hexdigest()
