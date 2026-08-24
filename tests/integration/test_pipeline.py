"""End-to-end pipeline: camera to alert."""

from __future__ import annotations

import time

import pytest

from ibvap.core.config import (
    AnalyticsConfig,
    CameraConfig,
    PipelineConfig,
    RuleConfig,
    Settings,
    TrackerConfig,
    TripwireConfig,
    ZoneConfig,
)
from ibvap.core.types import Event, EventType
from ibvap.pipeline.supervisor import Supervisor

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def build_settings(**overrides) -> Settings:
    camera = CameraConfig(
        id="cam-north", name="North Tower",
        url="synthetic://?width=640&height=480&fps=20&seed=1",
        target_fps=10.0, process_width=640,
        zones=[ZoneConfig(id="zone", name="Fence Strip",
                          points=[(0.35, 0.3), (1.0, 0.3), (1.0, 1.0), (0.35, 1.0)])],
        tripwires=[TripwireConfig(id="wire", name="Fence Line",
                                  start=(0.5, 0.1), end=(0.5, 0.95))],
        rules=[
            RuleConfig(id="intrusion", type="intrusion", zones=["zone"],
                       params={"confirm_frames": 2}, cooldown_seconds=2),
            RuleConfig(id="crossing", type="line_crossing", tripwires=["wire"],
                       cooldown_seconds=2),
            RuleConfig(id="presence", type="presence"),
        ],
    )
    return Settings(
        site_id="test-site",
        pipeline=PipelineConfig(queue_size=4),
        tracker=TrackerConfig(min_hits=2, max_age=15),
        analytics=AnalyticsConfig(dedup_window_seconds=1.0),
        cameras=[camera],
        **overrides,
    )


class TestSupervisor:
    def test_produces_alerts_from_a_live_camera(self) -> None:
        """The full path: decode, detect, track, evaluate rules, emit."""
        events: list[Event] = []
        supervisor = Supervisor(build_settings(), event_sink=events.append)
        supervisor.start()
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if any(e.event_type is EventType.INTRUSION for e in events):
                    break
                time.sleep(0.5)
        finally:
            supervisor.stop()

        types = {e.event_type for e in events}
        assert EventType.CAMERA_ONLINE in types
        assert EventType.INTRUSION in types, f"no intrusion raised; saw {types}"

        intrusion = next(e for e in events if e.event_type is EventType.INTRUSION)
        assert intrusion.camera_id == "cam-north"
        assert intrusion.zone_id == "zone"
        assert intrusion.boxes and intrusion.track_ids

    def test_reports_health_and_degradation(self) -> None:
        supervisor = Supervisor(build_settings())
        supervisor.start()
        try:
            time.sleep(3)
            health = supervisor.health()
            assert health["cameras_total"] == 1
            assert health["cameras_online"] == 1
            # No model artefacts in CI, so the node must say it is degraded
            # rather than let a control room believe it has full accuracy.
            assert health["models"]["degraded"] is True
            assert health["status"] == "degraded"
        finally:
            supervisor.stop()

    def test_unreachable_camera_does_not_stop_the_others(self) -> None:
        """One bad camera must never take a post's other feeds offline."""
        settings = build_settings()
        settings.cameras.append(
            CameraConfig(id="cam-dead", url="rtsp://127.0.0.1:1/dead", target_fps=4.0)
        )
        supervisor = Supervisor(settings)
        supervisor.start()
        try:
            time.sleep(5)
            health = supervisor.health()
            assert health["cameras_total"] == 2
            assert health["cameras_online"] == 1
            assert health["status"] == "partial"
        finally:
            supervisor.stop()

    def test_cameras_can_be_added_and_removed_at_runtime(self) -> None:
        supervisor = Supervisor(build_settings())
        supervisor.start()
        try:
            supervisor.add_camera(
                CameraConfig(id="cam-extra", url="synthetic://?width=320&height=240", target_fps=4.0)
            )
            time.sleep(2)
            assert len(supervisor.workers) == 2
            assert supervisor.remove_camera("cam-extra")
            assert len(supervisor.workers) == 1
        finally:
            supervisor.stop()

    def test_frames_are_sampled_to_the_target_rate(self) -> None:
        """Decoding runs at source rate to keep the decoder buffer shallow;
        only the analytics rate is sampled."""
        supervisor = Supervisor(build_settings())
        supervisor.start()
        try:
            time.sleep(6)
            worker = supervisor.worker("cam-north")
            assert worker is not None
            assert worker.reader.health.source_fps == pytest.approx(20.0)
            assert worker.reader.health.fps == pytest.approx(10.0, rel=0.25)
        finally:
            supervisor.stop()
