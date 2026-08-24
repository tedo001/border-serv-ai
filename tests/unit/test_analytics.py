"""Analytics rules and the event suppression gate."""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from tests.conftest import make_track

from ibvap.analytics.engine import AnalyticsEngine, EventGate
from ibvap.core.config import (
    AnalyticsConfig,
    CameraConfig,
    RuleConfig,
    TripwireConfig,
    ZoneConfig,
)
from ibvap.core.types import Event, EventType, Frame, ObjectClass, Severity

WIDTH, HEIGHT = 1280, 720


def build_frame(index: int, monotonic: float, *, night: bool = False, timestamp: float | None = None):
    rng = np.random.default_rng(index)
    image = rng.integers(60, 190, (HEIGHT, WIDTH, 3)).astype(np.uint8)
    return Frame(
        camera_id="cam-test", index=index, image=image,
        timestamp=timestamp if timestamp is not None else 1_700_000_000.0 + monotonic,
        monotonic=monotonic, fps=8.0, is_night=night,
    )


def build_engine(rules: list[RuleConfig], **analytics: object) -> AnalyticsEngine:
    camera = CameraConfig(
        id="cam-test", url="synthetic://",
        zones=[ZoneConfig(id="zone", name="Fence Strip",
                          points=[(0.4, 0.3), (1.0, 0.3), (1.0, 1.0), (0.4, 1.0)])],
        tripwires=[TripwireConfig(id="wire", name="Fence Line",
                                  start=(0.5, 0.05), end=(0.5, 0.95))],
        rules=rules,
    )
    config = AnalyticsConfig(dedup_window_seconds=0.0, **analytics)  # type: ignore[arg-type]
    return AnalyticsEngine(camera, config)


class TestIntrusion:
    def test_fires_after_confirmation_frames(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="intrusion", zones=["zone"], classes=["person"],
                       params={"confirm_frames": 3}, cooldown_seconds=60)
        ])
        track = make_track(1, 800, 400)
        events: list[Event] = []
        for step in range(5):
            events += engine.process(build_frame(step, step * 0.125), [track])
        assert len(events) == 1
        assert events[0].event_type is EventType.INTRUSION

    def test_single_frame_jitter_does_not_fire(self) -> None:
        """Detector jitter routinely puts one box a few pixels over a boundary."""
        engine = build_engine([
            RuleConfig(id="r", type="intrusion", zones=["zone"],
                       params={"confirm_frames": 3}, cooldown_seconds=60)
        ])
        track = make_track(1, 800, 400)
        assert engine.process(build_frame(0, 0.0), [track]) == []

    def test_membership_uses_the_foot_point(self) -> None:
        """A person just outside a fence can have a centroid inside the polygon;
        on a downward-angled camera that error is worth metres on the ground."""
        engine = build_engine([
            RuleConfig(id="r", type="intrusion", zones=["zone"],
                       params={"confirm_frames": 1}, cooldown_seconds=60)
        ])
        # Centroid inside the zone (y centre 260 > 216), feet outside (y 200 < 216).
        track = make_track(1, 800, 50, height=150)
        events = []
        for step in range(3):
            events += engine.process(build_frame(step, step * 0.125), [track])
        assert events == []

    def test_ignored_class_never_alerts(self) -> None:
        """Stray cattle are the dominant false-alarm source on rural fences."""
        engine = build_engine(
            [RuleConfig(id="r", type="intrusion", zones=["zone"],
                        params={"confirm_frames": 1}, cooldown_seconds=60)],
            ignore_classes=["animal"],
        )
        track = make_track(1, 800, 400, obj_class=ObjectClass.ANIMAL)
        events = []
        for step in range(5):
            events += engine.process(build_frame(step, step * 0.125), [track])
        assert events == []

    def test_leaving_and_returning_alerts_again(self) -> None:
        """Re-entry must re-arm the rule.

        Rule state lives on the Track object the tracker mutates in place, so
        this moves one track rather than swapping in a second object with the
        same id - which is what the pipeline actually does.
        """
        from ibvap.core.types import BBox

        engine = build_engine([
            RuleConfig(id="r", type="intrusion", zones=["zone"],
                       params={"confirm_frames": 1}, cooldown_seconds=0)
        ])
        track = make_track(1, 800, 400)

        def move_to(x: float) -> None:
            track.bbox = BBox(x, 400, x + 60, 550)
            track.trail.append(track.bbox.foot)

        first = [e for s in range(2) for e in engine.process(build_frame(s, s * 0.1), [track])]

        move_to(100)  # leaves the zone
        engine.process(build_frame(2, 0.2), [track])

        move_to(800)  # returns
        second = [e for s in range(3, 5) for e in engine.process(build_frame(s, s * 0.1), [track])]
        assert len(first) == 1
        assert len(second) == 1


class TestLineCrossing:
    def test_direction_filter(self) -> None:
        camera = CameraConfig(
            id="cam-test", url="synthetic://",
            tripwires=[TripwireConfig(id="wire", name="Fence", start=(0.5, 0.05),
                                      end=(0.5, 0.95), direction="right")],
            rules=[RuleConfig(id="r", type="line_crossing", tripwires=["wire"],
                              cooldown_seconds=0)],
        )
        engine = AnalyticsEngine(camera, AnalyticsConfig(dedup_window_seconds=0.0))

        rightward = make_track(1, 560, 400, trail=[(600, 550), (700, 550)])
        leftward = make_track(2, 560, 400, trail=[(700, 550), (600, 550)])
        assert len(engine.process(build_frame(0, 0.0), [rightward])) == 1
        assert engine.process(build_frame(1, 0.1), [leftward]) == []

    def test_one_alert_per_traversal(self) -> None:
        """A track oscillating on the line must not alert every frame."""
        engine = build_engine([
            RuleConfig(id="r", type="line_crossing", tripwires=["wire"], cooldown_seconds=0)
        ])
        track = make_track(1, 560, 400, trail=[(600, 550), (700, 550)])
        events = []
        for step in range(4):
            track.trail = [(600, 550), (700, 550)]
            events += engine.process(build_frame(step, step * 0.1), [track])
        assert len(events) == 1


class TestBehaviour:
    def test_loitering_requires_dwell_and_containment(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="loitering", zones=["zone"],
                       params={"dwell_seconds": 5, "max_displacement": 0.1},
                       cooldown_seconds=60)
        ])
        track = make_track(1, 800, 400, trail=[(830, 550)])
        events = []
        for step in range(9):
            track.trail.append((830 + step % 2, 550))
            events += engine.process(build_frame(step, float(step)), [track])
        assert len(events) == 1
        assert events[0].attributes["dwell_seconds"] >= 5

    def test_traversing_is_not_loitering(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="loitering", zones=["zone"],
                       params={"dwell_seconds": 3, "max_displacement": 0.05},
                       cooldown_seconds=60)
        ])
        track = make_track(1, 700, 400)
        events = []
        for step in range(9):
            track.bbox = make_track(1, 700 + step * 40, 400).bbox
            track.trail.append(track.bbox.foot)
            events += engine.process(build_frame(step, float(step)), [track])
        assert events == []

    def test_rapid_movement_is_resolution_independent(self) -> None:
        """Thresholds are in frame-heights per second so one rule template
        transfers between a 4 MP check-post camera and a 720p tower mast."""
        engine = build_engine([
            RuleConfig(id="r", type="rapid_movement",
                       params={"speed_threshold": 0.35, "confirm_frames": 3})
        ])
        speed = 0.5 * HEIGHT / 8.0  # 0.5 frame-heights/second at 8 fps
        track = make_track(1, 100, 300, velocity=(speed, 0))
        events = []
        for step in range(4):
            events += engine.process(build_frame(step, step * 0.125), [track])
        assert len(events) == 1
        assert events[0].attributes["speed_frame_heights_per_second"] == pytest.approx(0.5, rel=0.01)

    def test_abandoned_object_requires_no_owner_nearby(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="abandoned_object", classes=["bag"],
                       params={"static_seconds": 4, "owner_radius": 0.2}, cooldown_seconds=60)
        ])
        bag = make_track(1, 300, 500, width=40, height=40, obj_class=ObjectClass.BAG)
        owner = make_track(2, 320, 470)

        with_owner = []
        for step in range(8):
            bag.trail.append(bag.bbox.foot)
            with_owner += engine.process(build_frame(step, float(step)), [bag, owner])
        assert with_owner == []

        alone = []
        for step in range(8, 16):
            bag.trail.append(bag.bbox.foot)
            alone += engine.process(build_frame(step, float(step)), [bag])
        assert len(alone) == 1

    def test_crowd_gathering(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="crowd_gathering", zones=["zone"],
                       params={"min_people": 4, "confirm_frames": 3}, cooldown_seconds=60)
        ])
        crowd = [make_track(10 + i, 700 + i * 70, 400) for i in range(5)]
        events = []
        for step in range(4):
            events += engine.process(build_frame(step, step * 0.125), crowd)
        assert len(events) == 1
        assert events[0].attributes["person_count"] == 5


class TestSchedule:
    def test_night_rule_respects_active_hours(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="night_movement",
                       params={"min_speed": 0.01, "confirm_frames": 2},
                       active_hours=["18:00-06:00"], cooldown_seconds=60)
        ])
        track = make_track(1, 200, 300, velocity=(25, 0))

        at_noon = datetime.datetime(2026, 3, 1, 12, 0, 0).timestamp()
        day_events = [
            e for s in range(4)
            for e in engine.process(
                build_frame(s, s * 0.125, night=True, timestamp=at_noon + s * 0.125), [track]
            )
        ]
        assert day_events == []

        track.attributes.clear()
        at_night = datetime.datetime(2026, 3, 1, 23, 0, 0).timestamp()
        night_events = [
            e for s in range(4, 8)
            for e in engine.process(
                build_frame(s, s * 0.125, night=True, timestamp=at_night + s * 0.125), [track]
            )
        ]
        assert len(night_events) == 1


class TestTamper:
    def test_detects_a_covered_lens(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="camera_tamper",
                       params={"confirm_frames": 3, "warmup_frames": 2}, cooldown_seconds=60)
        ])
        events = []
        for step in range(3):
            events += engine.process(build_frame(step, float(step)), [])
        for step in range(3, 10):
            dark = Frame("cam-test", step, np.full((HEIGHT, WIDTH, 3), 2, np.uint8),
                         timestamp=1_700_000_000.0 + step, monotonic=float(step), fps=8.0)
            events += engine.process(dark, [])
        assert len(events) == 1
        assert events[0].attributes["reason"] == "lens_covered"

    def test_healthy_scene_does_not_fire(self) -> None:
        engine = build_engine([
            RuleConfig(id="r", type="camera_tamper",
                       params={"confirm_frames": 3, "warmup_frames": 2}, cooldown_seconds=60)
        ])
        events = []
        for step in range(20):
            events += engine.process(build_frame(step, float(step)), [])
        assert events == []


class TestEventGate:
    def _event(self, track_id: int = 1) -> Event:
        return Event(
            camera_id="cam", event_type=EventType.INTRUSION, severity=Severity.HIGH,
            timestamp=0.0, rule_id="r", track_ids=[track_id],
        )

    def test_deduplicates_the_same_incident(self) -> None:
        gate = EventGate(dedup_window_seconds=10.0, max_events_per_minute=100)
        assert gate.admit(self._event(), 0.0, now=0.0)
        assert not gate.admit(self._event(), 0.0, now=1.0)
        assert gate.stats.deduplicated == 1

    def test_different_tracks_remain_distinct(self) -> None:
        """Two people entering the same zone are two alerts, not one."""
        gate = EventGate(dedup_window_seconds=10.0, max_events_per_minute=100)
        assert gate.admit(self._event(1), 0.0, now=0.0)
        assert gate.admit(self._event(2), 0.0, now=0.1)

    def test_cooldown(self) -> None:
        gate = EventGate(dedup_window_seconds=0.0, max_events_per_minute=100)
        assert gate.admit(self._event(), 30.0, now=0.0)
        assert not gate.admit(self._event(), 30.0, now=10.0)
        assert gate.admit(self._event(), 30.0, now=31.0)

    def test_rate_limit_protects_the_link(self) -> None:
        """A pathological scene must not flood the C2 uplink."""
        gate = EventGate(dedup_window_seconds=0.0, max_events_per_minute=5)
        admitted = sum(gate.admit(self._event(i), 0.0, now=i * 0.1) for i in range(20))
        assert admitted == 5
        assert gate.stats.rate_limited == 15

    def test_prune_bounds_memory(self) -> None:
        """Without pruning, a busy camera leaks a key per track id forever."""
        gate = EventGate(dedup_window_seconds=1.0, max_events_per_minute=10_000)
        for i in range(500):
            gate.admit(self._event(i), 0.0, now=i * 0.01)
        gate.prune(now=10_000.0, max_age=60.0)
        assert gate._last_seen == {}
