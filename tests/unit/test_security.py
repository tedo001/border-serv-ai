"""Authentication, roles and brute-force protection."""

from __future__ import annotations

import time

import pytest

from ibvap.api.security import (
    LoginThrottle,
    Role,
    TokenService,
    generate_password,
    hash_password,
    password_strength_issues,
    verify_password,
)
from ibvap.core.config import SecurityConfig
from ibvap.core.errors import AuthError

SECRET = "s" * 48


class TestRoles:
    def test_hierarchy(self) -> None:
        assert Role.ADMIN.satisfies(Role.VIEWER)
        assert Role.SUPERVISOR.satisfies(Role.OPERATOR)
        assert not Role.VIEWER.satisfies(Role.OPERATOR)
        assert Role.VIEWER.satisfies(Role.VIEWER)

    def test_unknown_role_fails_closed(self) -> None:
        """A typo in configuration must reduce access, never grant it."""
        assert Role.coerce("superuser") is Role.VIEWER
        assert Role.coerce("ADMIN") is Role.ADMIN


class TestPasswords:
    def test_hash_and_verify(self) -> None:
        digest = hash_password("Str0ngPassphrase!")
        assert verify_password("Str0ngPassphrase!", digest)
        assert not verify_password("wrong", digest)

    def test_corrupt_hash_reads_as_wrong_password(self) -> None:
        """It must not raise: a 500 here reveals that the account exists."""
        assert not verify_password("anything", "not-a-valid-hash")
        assert not verify_password("anything", "")

    def test_long_passphrases_stay_distinct(self) -> None:
        """bcrypt truncates at 72 bytes; without pre-hashing, two different
        passphrases sharing a 72-byte prefix would both authenticate."""
        first = "A" * 80 + "-tail-one"
        second = "A" * 80 + "-tail-two"
        digest = hash_password(first)
        assert verify_password(first, digest)
        assert not verify_password(second, digest)

    def test_strength_rules(self) -> None:
        assert password_strength_issues("Str0ngPassphrase!") == []
        assert password_strength_issues("short") != []
        assert any("commonly used" in issue for issue in password_strength_issues("password123"))

    def test_generated_passwords_are_strong(self) -> None:
        assert password_strength_issues(generate_password()) in ([], ["must contain a digit"])

    def test_empty_password_rejected(self) -> None:
        with pytest.raises(AuthError):
            hash_password("")


class TestTokens:
    def _service(self, **overrides) -> TokenService:
        return TokenService(SecurityConfig(jwt_secret=SECRET, **overrides))

    def test_issue_and_verify(self) -> None:
        service = self._service()
        token, expires = service.issue("operator1", Role.OPERATOR)
        principal = service.verify(token)
        assert principal.username == "operator1"
        assert principal.role is Role.OPERATOR
        assert expires > time.time()

    def test_refresh_token_is_not_an_access_token(self) -> None:
        """Otherwise a long-lived refresh token defeats short access TTLs."""
        service = self._service()
        refresh, _ = service.issue("u", Role.VIEWER, refresh=True)
        with pytest.raises(AuthError, match="expected a access token"):
            service.verify(refresh)
        assert service.verify(refresh, expect_refresh=True).username == "u"

    def test_revocation(self) -> None:
        service = self._service()
        token, _ = service.issue("u", Role.VIEWER)
        service.revoke(token)
        with pytest.raises(AuthError, match="revoked"):
            service.verify(token)

    def test_tampered_token_rejected(self) -> None:
        service = self._service()
        token, _ = service.issue("u", Role.ADMIN)
        with pytest.raises(AuthError):
            service.verify(token[:-4] + "AAAA")

    def test_foreign_signature_rejected(self) -> None:
        token, _ = TokenService(SecurityConfig(jwt_secret="x" * 48)).issue("u", Role.ADMIN)
        with pytest.raises(AuthError):
            self._service().verify(token)

    def test_short_secret_rejected_at_construction(self) -> None:
        """A weak signing key must not reach production quietly."""
        with pytest.raises(AuthError, match="at least 32 bytes"):
            TokenService(SecurityConfig(jwt_secret="tooshort"))

    def test_missing_secret_rejected(self) -> None:
        with pytest.raises(AuthError, match="not configured"):
            TokenService(SecurityConfig(jwt_secret=""))

    def test_role_requirement(self) -> None:
        service = self._service()
        principal = service.verify(service.issue("u", Role.OPERATOR)[0])
        principal.require(Role.VIEWER)
        with pytest.raises(AuthError, match="insufficient"):
            principal.require(Role.ADMIN)


class TestLoginThrottle:
    def test_locks_out_after_repeated_failures(self) -> None:
        throttle = LoginThrottle(max_attempts=3, lockout_seconds=60)
        for _ in range(3):
            throttle.record_failure("operator", "10.0.0.1")
        with pytest.raises(AuthError, match="too many failed attempts"):
            throttle.check("operator", "10.0.0.1")

    def test_keyed_on_username_and_source_together(self) -> None:
        """Keying on address alone locks out a whole post behind one NAT;
        keying on username alone lets an attacker lock out a real operator."""
        throttle = LoginThrottle(max_attempts=2, lockout_seconds=60)
        for _ in range(2):
            throttle.record_failure("operator", "10.0.0.1")

        throttle.check("operator", "10.0.0.2")   # same user, different source
        throttle.check("someone-else", "10.0.0.1")  # same source, different user

    def test_success_clears_the_counter(self) -> None:
        throttle = LoginThrottle(max_attempts=3, lockout_seconds=60)
        throttle.record_failure("operator", "10.0.0.1")
        throttle.record_failure("operator", "10.0.0.1")
        throttle.record_success("operator", "10.0.0.1")
        throttle.record_failure("operator", "10.0.0.1")
        throttle.check("operator", "10.0.0.1")
