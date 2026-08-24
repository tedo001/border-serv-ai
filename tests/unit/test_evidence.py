"""Evidence capture, chain of custody and retention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from ibvap.core.config import EvidenceConfig, PrivacyConfig
from ibvap.core.types import BBox, Event, EventType, Frame, Severity
from ibvap.events.evidence import EvidenceStore


def build_event(index: int = 0, timestamp: float = 1_700_000_000.0) -> Event:
    return Event(
        camera_id="cam-test", event_type=EventType.INTRUSION, severity=Severity.HIGH,
        timestamp=timestamp + index, message=f"intrusion {index}",
        boxes=[BBox(100, 80, 160, 240)], track_ids=[1],
        attributes={"zone_name": "Fence Strip", "dwell_seconds": np.float32(2.5)},
    )


@pytest.fixture
def store(workspace: Path) -> EvidenceStore:
    return EvidenceStore(
        EvidenceConfig(directory=workspace / "evidence", clip=False),
        PrivacyConfig(),
        site_id="test-site",
    )


class TestCapture:
    def test_writes_snapshot_and_manifest(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame, camera_name="Test Camera")
        assert record.snapshot_path and Path(record.snapshot_path).is_file()
        assert record.manifest_path and Path(record.manifest_path).is_file()
        assert record.snapshot_sha256 and len(record.snapshot_sha256) == 64

    def test_event_is_annotated_with_artefacts(self, store: EvidenceStore, frame: Frame) -> None:
        event = build_event()
        store.capture(event, frame)
        assert event.snapshot_path is not None
        assert event.evidence_hash is not None

    def test_numpy_attributes_are_serialisable(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame)
        manifest = json.loads(Path(record.manifest_path).read_text())
        assert manifest["attributes"]["dwell_seconds"] == 2.5

    def test_partitioned_by_day_and_camera(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame)
        path = Path(record.snapshot_path)
        assert path.parent.name == "cam-test"
        assert len(path.parent.parent.name) == 10  # YYYY-MM-DD


class TestChainOfCustody:
    def test_manifests_form_a_chain(self, store: EvidenceStore, frame: Frame) -> None:
        records = [store.capture(build_event(i), frame) for i in range(3)]
        second = json.loads(Path(records[1].manifest_path).read_text())
        assert second["previous_manifest_sha256"] == records[0].chain_hash

    def test_verification_passes_when_intact(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame)
        result = store.verify(record.manifest_path)
        assert result["valid"] and result["issues"] == []

    def test_detects_an_altered_artefact(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame)
        Path(record.snapshot_path).write_bytes(b"forged image bytes")
        result = store.verify(record.manifest_path)
        assert not result["valid"]
        assert any("digest mismatch" in issue for issue in result["issues"])

    def test_detects_altered_metadata(self, store: EvidenceStore, frame: Frame) -> None:
        record = store.capture(build_event(), frame)
        path = Path(record.manifest_path)
        manifest = json.loads(path.read_text())
        manifest["message"] = "nothing happened here"
        path.write_text(json.dumps(manifest, indent=2))
        result = store.verify(path)
        assert not result["valid"]
        assert any("metadata was altered" in issue for issue in result["issues"])

    def test_detects_a_deleted_manifest(self, store: EvidenceStore, frame: Frame) -> None:
        """Deletion, not just alteration, must be detectable - otherwise the
        simplest way to hide an incident is to remove its record."""
        records = [store.capture(build_event(i), frame) for i in range(3)]
        day = Path(records[0].manifest_path).parent.parent.name
        Path(records[0].manifest_path).unlink()

        report = store.verify_chain(day)
        assert not report["valid"]
        assert any("deleted or altered" in issue for issue in report["issues"])

    def test_chain_verifies_when_intact(self, store: EvidenceStore, frame: Frame) -> None:
        records = [store.capture(build_event(i), frame) for i in range(3)]
        day = Path(records[0].manifest_path).parent.parent.name
        report = store.verify_chain(day)
        assert report["valid"] and report["manifests"] == 3


class TestRetention:
    def test_age_based_pruning(self, workspace: Path, frame: Frame) -> None:
        store = EvidenceStore(
            EvidenceConfig(directory=workspace / "evidence", clip=False, retention_days=1),
            PrivacyConfig(),
        )
        record = store.capture(build_event(), frame)
        old = time.time() - 3 * 86400
        for path in Path(record.snapshot_path).parent.iterdir():
            import os

            os.utime(path, (old, old))

        result = store.prune()
        assert result["files"] >= 1
        assert not Path(record.snapshot_path).exists()

    def test_size_capped_pruning(self, workspace: Path, frame: Frame) -> None:
        """A full disk stops the node recording anything, which is far worse
        than losing the oldest evidence."""
        store = EvidenceStore(
            EvidenceConfig(
                directory=workspace / "evidence", clip=False,
                retention_days=0, max_bytes=20_000,
            ),
            PrivacyConfig(),
        )
        for index in range(6):
            store.capture(build_event(index), frame)
        assert store.usage()["bytes"] > 20_000

        store.prune()
        assert store.usage()["bytes"] <= 20_000

    def test_usage_reporting(self, store: EvidenceStore, frame: Frame) -> None:
        store.capture(build_event(), frame)
        usage = store.usage()
        assert usage["files"] >= 2  # snapshot plus manifest
        assert usage["bytes"] > 0


class TestDisabled:
    def test_capture_is_a_no_op_when_disabled(self, workspace: Path, frame: Frame) -> None:
        store = EvidenceStore(
            EvidenceConfig(enabled=False, directory=workspace / "evidence"), PrivacyConfig()
        )
        record = store.capture(build_event(), frame)
        assert record.snapshot_path is None
