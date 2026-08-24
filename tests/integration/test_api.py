"""End-to-end API: authentication, RBAC, events, evidence and watchlists."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from ibvap.api.app import create_app
from ibvap.core.config import Settings

pytestmark = pytest.mark.integration

API = "/api/v1"
ADMIN = {"username": "admin", "password": "TestPassphrase123!"}


@pytest_asyncio.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """A running node with its API, isolated per test."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testnode") as http:
            yield http


async def auth_headers(client: httpx.AsyncClient, **credentials) -> dict[str, str]:
    response = await client.post(f"{API}/auth/login", json=credentials or ADMIN)
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


class TestAuthentication:
    async def test_unauthenticated_requests_are_rejected(self, client: httpx.AsyncClient) -> None:
        assert (await client.get(f"{API}/events")).status_code == 401
        assert (await client.get(f"{API}/cameras")).status_code == 401

    async def test_health_is_public(self, client: httpx.AsyncClient) -> None:
        """A monitoring system must see that a node is down without holding a
        credential - and a node that is down cannot authenticate anyone."""
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["site_id"] == "test-site"
        assert "jwt_secret" not in str(body)

    async def test_login_and_whoami(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        me = (await client.get(f"{API}/auth/me", headers=headers)).json()
        assert me["username"] == "admin"
        assert me["role"] == "admin"

    async def test_bad_credentials_are_indistinguishable(self, client: httpx.AsyncClient) -> None:
        """Separate messages would let an attacker enumerate operator accounts."""
        wrong_password = await client.post(
            f"{API}/auth/login", json={"username": "admin", "password": "wrong"}
        )
        no_such_user = await client.post(
            f"{API}/auth/login", json={"username": "ghost", "password": "wrong"}
        )
        assert wrong_password.status_code == no_such_user.status_code == 401
        assert wrong_password.json()["detail"] == no_such_user.json()["detail"]

    async def test_refresh_issues_a_new_access_token(self, client: httpx.AsyncClient) -> None:
        login = (await client.post(f"{API}/auth/login", json=ADMIN)).json()
        response = await client.post(
            f"{API}/auth/refresh", json={"refresh_token": login["refresh_token"]}
        )
        assert response.status_code == 200
        assert response.json()["access_token"] != login["access_token"]

    async def test_logout_revokes_the_token(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        assert (await client.post(f"{API}/auth/logout", headers=headers)).status_code == 200
        assert (await client.get(f"{API}/events", headers=headers)).status_code == 401


class TestRbac:
    async def _viewer(self, client: httpx.AsyncClient) -> dict[str, str]:
        admin = await auth_headers(client)
        await client.post(
            f"{API}/auth/users", headers=admin,
            json={"username": "viewer1", "password": "ViewerPassphrase1!", "role": "viewer"},
        )
        return await auth_headers(
            client, username="viewer1", password="ViewerPassphrase1!"
        )

    async def test_viewer_may_read(self, client: httpx.AsyncClient) -> None:
        headers = await self._viewer(client)
        assert (await client.get(f"{API}/events", headers=headers)).status_code == 200
        assert (await client.get(f"{API}/cameras", headers=headers)).status_code == 200

    async def test_viewer_may_not_write_watchlists(self, client: httpx.AsyncClient) -> None:
        headers = await self._viewer(client)
        response = await client.post(
            f"{API}/watchlists/plates", headers=headers, json={"plate": "MH12AB1234"}
        )
        assert response.status_code == 403
        assert "operator" in response.json()["detail"]

    async def test_viewer_may_not_administer_users(self, client: httpx.AsyncClient) -> None:
        headers = await self._viewer(client)
        response = await client.post(
            f"{API}/auth/users", headers=headers,
            json={"username": "x", "password": "AnotherPassphrase1!", "role": "admin"},
        )
        assert response.status_code == 403

    async def test_deactivated_account_is_refused_immediately(
        self, client: httpx.AsyncClient
    ) -> None:
        """A token stays cryptographically valid after an account is disabled,
        so account state must be re-checked on every request."""
        admin = await auth_headers(client)
        headers = await self._viewer(client)
        assert (await client.get(f"{API}/events", headers=headers)).status_code == 200

        await client.delete(f"{API}/auth/users/viewer1", headers=admin)
        assert (await client.get(f"{API}/events", headers=headers)).status_code == 401

    async def test_api_key_authentication(self, client: httpx.AsyncClient) -> None:
        admin = await auth_headers(client)
        created = await client.post(
            f"{API}/auth/api-keys", headers=admin,
            json={"name": "sector-gateway", "role": "operator"},
        )
        key = created.json()["key"]
        assert key.startswith("ibv_")

        assert (await client.get(f"{API}/events", headers={"X-API-Key": key})).status_code == 200
        assert (
            await client.get(f"{API}/events", headers={"X-API-Key": "ibv_wrong"})
        ).status_code == 401

        # A key is shown once and is never retrievable afterwards.
        listed = (await client.get(f"{API}/auth/api-keys", headers=admin)).json()
        assert listed[0]["key"] is None

    async def test_revoked_api_key_is_refused(self, client: httpx.AsyncClient) -> None:
        admin = await auth_headers(client)
        key = (await client.post(
            f"{API}/auth/api-keys", headers=admin, json={"name": "temp", "role": "viewer"}
        )).json()["key"]
        await client.delete(f"{API}/auth/api-keys/temp", headers=admin)
        assert (await client.get(f"{API}/events", headers={"X-API-Key": key})).status_code == 401


class TestWatchlists:
    async def test_plate_is_normalised_on_entry(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        response = await client.post(
            f"{API}/watchlists/plates", headers=headers,
            json={"plate": "mh 12 ab 1234", "category": "stolen", "reference": "FIR 41/2026"},
        )
        assert response.status_code == 201
        assert response.json()["plate"] == "MH12AB1234"

    async def test_malformed_plate_is_rejected(self, client: httpx.AsyncClient) -> None:
        """An ungrammatical entry can never match a normalised live read, so
        storing it would create a watchlist entry that looks active but is inert."""
        headers = await auth_headers(client)
        response = await client.post(
            f"{API}/watchlists/plates", headers=headers, json={"plate": "NOTAPLATE99"}
        )
        assert response.status_code == 422
        assert "not a recognised Indian registration format" in response.json()["detail"]

    async def test_plate_lifecycle(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        await client.post(f"{API}/watchlists/plates", headers=headers, json={"plate": "MH12AB1234"})
        assert len((await client.get(f"{API}/watchlists/plates", headers=headers)).json()) == 1

        assert (await client.delete(
            f"{API}/watchlists/plates/MH12AB1234", headers=headers
        )).status_code == 200
        assert (await client.get(f"{API}/watchlists/plates", headers=headers)).json() == []

    async def test_face_enrolment_requires_supervisor(self, client: httpx.AsyncClient) -> None:
        """Adding a person to a biometric watchlist is a materially different
        act from acknowledging an alert."""
        admin = await auth_headers(client)
        await client.post(
            f"{API}/auth/users", headers=admin,
            json={"username": "op1", "password": "OperatorPassphrase1!", "role": "operator"},
        )
        operator = await auth_headers(client, username="op1", password="OperatorPassphrase1!")
        response = await client.post(
            f"{API}/watchlists/faces", headers=operator,
            json={"person_id": "P1", "embedding": [0.1] * 128},
        )
        assert response.status_code == 403

    async def test_face_enrolment_from_an_embedding(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        response = await client.post(
            f"{API}/watchlists/faces", headers=headers,
            json={"person_id": "P1", "name": "Subject One", "category": "wanted",
                  "embedding": [0.1] * 128},
        )
        assert response.status_code == 201
        assert response.json()["embedding_count"] == 1

        listed = (await client.get(f"{API}/watchlists/faces", headers=headers)).json()
        # Embeddings are biometric material and never leave the node as data.
        assert "embedding" not in listed[0]
        assert "embeddings" not in listed[0]


class TestEventsAndEvidence:
    async def test_events_appear_and_carry_evidence(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        for _ in range(40):
            page = (await client.get(
                f"{API}/events", headers=headers, params={"event_type": "intrusion"}
            )).json()
            if page["events"]:
                break
            await asyncio.sleep(0.5)

        assert page["events"], "no intrusion events were produced"
        event = page["events"][0]
        assert event["camera_id"] == "cam-test"
        assert event["has_snapshot"]

        snapshot = await client.get(
            f"{API}/events/{event['event_id']}/snapshot", headers=headers
        )
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.headers.get("x-ibvap-sha256")

        verification = await client.get(
            f"{API}/events/{event['event_id']}/verify", headers=headers
        )
        assert verification.json()["valid"] is True

    async def test_acknowledgement_is_recorded_once(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        for _ in range(40):
            page = (await client.get(f"{API}/events", headers=headers)).json()
            if page["events"]:
                break
            await asyncio.sleep(0.5)
        event_id = page["events"][0]["event_id"]

        first = await client.post(
            f"{API}/events/{event_id}/acknowledge", headers=headers,
            json={"disposition": "patrol dispatched"},
        )
        assert first.status_code == 200
        second = await client.post(
            f"{API}/events/{event_id}/acknowledge", headers=headers, json={"disposition": "again"}
        )
        assert second.status_code == 409

        detail = (await client.get(f"{API}/events/{event_id}", headers=headers)).json()
        assert detail["acknowledged_by"] == "admin"
        assert detail["disposition"] == "patrol dispatched"

    async def test_unknown_event_is_404(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        assert (await client.get(f"{API}/events/deadbeef", headers=headers)).status_code == 404


class TestMediaTokens:
    async def test_media_endpoints_accept_a_query_token(
        self, client: httpx.AsyncClient
    ) -> None:
        """A browser <img> cannot send an Authorization header."""
        login = (await client.post(f"{API}/auth/login", json=ADMIN)).json()
        token = login["access_token"]
        await asyncio.sleep(2)
        response = await client.get(f"{API}/cameras/cam-test/snapshot?token={token}")
        assert response.status_code in (200, 503)  # 503 only if no frame yet

    async def test_non_media_endpoints_reject_a_query_token(
        self, client: httpx.AsyncClient
    ) -> None:
        """So a leaked URL cannot be replayed into a state-changing call."""
        login = (await client.post(f"{API}/auth/login", json=ADMIN)).json()
        token = login["access_token"]
        assert (await client.get(f"{API}/events?token={token}")).status_code == 401
        assert (await client.post(
            f"{API}/watchlists/plates?token={token}", json={"plate": "MH12AB1234"}
        )).status_code == 401


class TestSystem:
    async def test_config_endpoint_strips_secrets(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        config = (await client.get(f"{API}/system/config", headers=headers)).json()
        assert "jwt_secret" not in config["security"]
        assert "bootstrap_admin_password" not in config["security"]

    async def test_audit_trail_records_actions(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        await client.post(
            f"{API}/watchlists/plates", headers=headers,
            json={"plate": "MH12AB1234", "category": "stolen"},
        )
        audit = (await client.get(f"{API}/system/audit", headers=headers)).json()
        actions = {entry["action"] for entry in audit}
        assert "watchlist.plate.add" in actions
        assert "auth.login" in actions

    async def test_metrics_are_exported(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/metrics")).text
        assert "ibvap_frames_captured_total" in body
        assert "ibvap_camera_up" in body

    async def test_rule_catalogue(self, client: httpx.AsyncClient) -> None:
        headers = await auth_headers(client)
        rules = (await client.get(f"{API}/system/rules", headers=headers)).json()["rules"]
        types = {rule["type"] for rule in rules}
        assert {"intrusion", "line_crossing", "loitering", "camera_tamper"} <= types

    async def test_readiness_reflects_camera_state(self, client: httpx.AsyncClient) -> None:
        await asyncio.sleep(2)
        readiness = (await client.get("/health/ready")).json()
        assert readiness["ready"] is True
        assert readiness["cameras_total"] == 1
