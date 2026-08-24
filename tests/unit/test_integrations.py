"""C2 sinks: filtering, signing, CEF formatting and store-and-forward."""

from __future__ import annotations

import json
import time

from ibvap.core.config import SyslogConfig, WebhookConfig
from ibvap.core.types import Event, EventType, Severity
from ibvap.events.payload import PAYLOAD_SCHEMA_VERSION, event_to_payload, payload_summary
from ibvap.integrations.base import Sink
from ibvap.integrations.dispatcher import backoff_for
from ibvap.integrations.syslog import escape_cef_value, format_cef
from ibvap.integrations.webhook import (
    WebhookSink,
    verify_signature,
)


def build_event(
    severity: Severity = Severity.HIGH,
    event_type: EventType = EventType.INTRUSION,
    **attributes,
) -> Event:
    return Event(
        camera_id="cam-north", event_type=event_type, severity=severity,
        timestamp=1_700_000_000.0, message="person entered restricted zone",
        rule_id="fence-intrusion", zone_id="fence-strip", attributes=attributes,
    )


class TestPayload:
    def test_canonical_shape(self) -> None:
        payload = event_to_payload(
            build_event(), site_id="bop-1", site_name="BOP One",
            camera_name="North Tower", latitude=26.9, longitude=88.4,
        )
        assert payload["schema_version"] == PAYLOAD_SCHEMA_VERSION
        assert payload["camera"]["latitude"] == 26.9
        assert payload["site"]["id"] == "bop-1"
        assert payload["timestamp_iso"].startswith("2023-11-14")

    def test_evidence_is_referenced_by_url_not_path(self) -> None:
        """Node storage layout must never become part of the published contract,
        and artefact access has to be authorised and audited."""
        event = build_event()
        event.snapshot_path = "/var/lib/ibvap/evidence/2026-01-01/cam/abc.jpg"
        payload = event_to_payload(event)
        assert payload["evidence"]["snapshot_url"] == f"/api/v1/events/{event.event_id}/snapshot"
        assert "/var/lib" not in json.dumps(payload)

    def test_summary_line(self) -> None:
        summary = payload_summary(event_to_payload(build_event()))
        assert "HIGH" in summary and "intrusion" in summary


class TestSinkFiltering:
    class _Recorder(Sink):
        name = "recorder"

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.delivered: list[dict] = []

        async def deliver(self, payload: dict) -> None:
            self.delivered.append(payload)

    def test_minimum_severity(self) -> None:
        sink = self._Recorder(min_severity="medium")
        assert not sink.accepts(build_event(Severity.INFO))
        assert not sink.accepts(build_event(Severity.LOW))
        assert sink.accepts(build_event(Severity.MEDIUM))
        assert sink.accepts(build_event(Severity.CRITICAL))

    def test_event_type_allowlist(self) -> None:
        """One node can feed a SIEM every detection while sending a control
        room only what needs a decision."""
        sink = self._Recorder(min_severity="info", event_types=["plate_match"])
        assert not sink.accepts(build_event(event_type=EventType.INTRUSION))
        assert sink.accepts(build_event(Severity.CRITICAL, EventType.PLATE_MATCH))

    def test_disabled_sink_accepts_nothing(self) -> None:
        assert not self._Recorder(enabled=False).accepts(build_event(Severity.CRITICAL))


class TestWebhookSigning:
    SECRET = "shared-secret-value"

    def test_signature_round_trip(self) -> None:
        body = json.dumps({"event_id": "abc"}, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        import hashlib
        import hmac

        digest = hmac.new(
            self.SECRET.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
        ).hexdigest()
        assert verify_signature(body, f"sha256={digest}", timestamp, self.SECRET)

    def test_rejects_a_tampered_body(self) -> None:
        body = b'{"event_id":"abc"}'
        timestamp = str(int(time.time()))
        import hashlib
        import hmac

        digest = hmac.new(
            self.SECRET.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
        ).hexdigest()
        assert not verify_signature(b'{"event_id":"evil"}', f"sha256={digest}", timestamp, self.SECRET)

    def test_rejects_a_replayed_delivery(self) -> None:
        """The timestamp is inside the signed material precisely so a captured
        delivery cannot be replayed later."""
        body = b'{"event_id":"abc"}'
        old = str(int(time.time()) - 4000)
        import hashlib
        import hmac

        digest = hmac.new(
            self.SECRET.encode(), f"{old}.".encode() + body, hashlib.sha256
        ).hexdigest()
        assert not verify_signature(body, f"sha256={digest}", old, self.SECRET, max_age_seconds=300)

    def test_rejects_a_wrong_secret(self) -> None:
        body = b'{"event_id":"abc"}'
        timestamp = str(int(time.time()))
        import hashlib
        import hmac

        digest = hmac.new(b"other-secret", f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
        assert not verify_signature(body, f"sha256={digest}", timestamp, self.SECRET)

    def test_malformed_timestamp_is_rejected(self) -> None:
        assert not verify_signature(b"{}", "sha256=deadbeef", "not-a-number", self.SECRET)

    def test_sink_reports_its_configuration(self) -> None:
        sink = WebhookSink(WebhookConfig(
            name="sector-c2", url="https://c2.example/alerts", hmac_secret="x"
        ))
        status = sink.status()
        assert status["kind"] == "webhook"
        assert status["signed"] is True


class TestCef:
    def test_well_formed_header(self) -> None:
        message = format_cef(event_to_payload(
            build_event(Severity.CRITICAL, EventType.PLATE_MATCH, plate="MH12AB1234"),
            site_id="bop-1", site_name="BOP One", camera_name="Gate",
        ))
        assert message.startswith("CEF:0|IBVAP|Border Video Analytics|1.0|plate_match|")
        assert message.split("|")[6] == "10"  # critical maps to CEF severity 10

    def test_custom_slots_stay_within_the_standard(self) -> None:
        """Inventing keys beyond cs1..cs6 is exactly what makes a CEF feed
        need a custom parser."""
        import re

        message = format_cef(event_to_payload(build_event(
            plate="MH12AB1234", person_id="P1", name="Subject", category="stolen",
            object_class="truck",
        )))
        extension = message.split("|", 7)[7]
        slots = re.findall(r"\bcs(\d)", extension)
        assert slots and all(int(slot) <= 6 for slot in slots)

    def test_escaping(self) -> None:
        assert escape_cef_value("a=b\\c") == "a\\=b\\\\c"
        assert "\n" not in escape_cef_value("line\nbreak")

    def test_severity_mapping(self) -> None:
        for severity, expected in (
            (Severity.INFO, "2"), (Severity.MEDIUM, "5"), (Severity.CRITICAL, "10")
        ):
            message = format_cef(event_to_payload(build_event(severity)))
            assert message.split("|")[6] == expected

    def test_sink_status(self) -> None:
        sink = __import__(
            "ibvap.integrations.syslog", fromlist=["SyslogSink"]
        ).SyslogSink(SyslogConfig(host="siem.example", port=514))
        assert sink.status()["format"] == "CEF"


class TestRetryBackoff:
    def test_front_loaded_then_stretching(self) -> None:
        """A momentary blip recovers in seconds; a multi-hour outage is not
        retried every few seconds for its whole duration."""
        delays = [backoff_for(attempt) for attempt in range(8)]
        assert delays[0] < delays[1] < delays[2]
        assert delays[0] <= 5.0
        assert delays[-1] == 3600.0
        assert delays == sorted(delays)
