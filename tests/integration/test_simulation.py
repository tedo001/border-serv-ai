"""End-to-end analytics driven by the scenario simulator.

These run the real pipeline - source, detector, classifier, tracker, rules -
against scripted footage, and assert on the alerts a control room would see.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pytest

from ibvap.core.config import (
    AnalyticsConfig,
    CameraConfig,
    RuleConfig,
    Settings,
    TrackerConfig,
    TripwireConfig,
    ZoneConfig,
)
from ibvap.pipeline.supervisor import Supervisor
from ibvap.vision.backends import CallableBackend
from ibvap.vision.classifier import ImageNetClassifier

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FENCE_ZONE = ZoneConfig(
    id="strip", name="Fence Strip",
    points=[(0.0, 0.46), (1.0, 0.46), (1.0, 1.0), (0.0, 1.0)],
)
FENCE_WIRE = TripwireConfig(
    id="wire", name="Fence Line", start=(0.0, 0.45), end=(1.0, 0.45),
    direction="any", left_label="exfiltration", right_label="infiltration",
)
RULES = [
    RuleConfig(id="intrusion", type="intrusion", zones=["strip"],
               params={"confirm_frames": 2}, cooldown_seconds=3),
]


def ox_classifier() -> ImageNetClassifier:
    """A stand-in that returns the ImageNet 'ox' class for any crop.

    This exercises the real decode, mapping and reclassification path. It is
    deliberately NOT a claim about what a trained MobileNet would predict on
    this synthetic footage - only that when the classifier says 'ox', the
    platform suppresses correctly.
    """
    def run(_feeds):
        logits = np.full((1, 1000), -6.0, np.float32)
        logits[0, 345] = 9.0
        return logits

    return ImageNetClassifier(
        CallableBackend(run, input_shape=(1, 3, 224, 224)), min_confidence=0.3
    )


def run_scenario(
    scenario: str, *, ignore: list[str], classifier: ImageNetClassifier | None = None,
    seconds: float = 16.0,
) -> tuple[Counter, dict]:
    """Run one scenario and return its alert counts and worker stats."""
    camera = CameraConfig(
        id="cam", url=f"sim://{scenario}?width=960&height=540&fps=15&seed=3",
        target_fps=8.0, process_width=960,
        zones=[FENCE_ZONE], tripwires=[FENCE_WIRE], rules=RULES,
        classify_objects=classifier is not None,
    )
    settings = Settings(
        site_id="sim", cameras=[camera],
        tracker=TrackerConfig(min_hits=2, max_age=15),
        analytics=AnalyticsConfig(ignore_classes=ignore, dedup_window_seconds=3),
    )
    events = []
    supervisor = Supervisor(settings, event_sink=events.append)
    if classifier is not None:
        supervisor.bundle.classifier = classifier

    supervisor.start()
    try:
        time.sleep(seconds)
        stats = supervisor.worker("cam").stats.as_dict()
    finally:
        supervisor.stop()

    counts = Counter(
        event.event_type.value for event in events
        if not event.event_type.value.startswith("camera_")
    )
    return counts, stats


class TestSimulatedAnalytics:
    def test_a_person_entering_the_zone_alarms(self) -> None:
        counts, _ = run_scenario("intrusion", ignore=["animal"])
        assert counts["intrusion"] >= 1

    def test_the_pipeline_processes_frames(self) -> None:
        _, stats = run_scenario("patrol", ignore=["animal"], seconds=10.0)
        assert stats["frames_processed"] > 20
        assert stats["last_error"] == ""


class TestLivestockSuppression:
    """The false alarm that dominates this domain, and the stage that fixes it."""

    def test_fallback_alone_misclassifies_cattle(self) -> None:
        """Documents the gap rather than hiding it.

        The classical fallback classifies by bounding-box aspect ratio, so a
        wide cow silhouette reads as a vehicle - and `ignore_classes: [animal]`
        therefore never matches it. This is one of the concrete reasons to
        install model artefacts.
        """
        counts, stats = run_scenario("cattle", ignore=["animal"])
        assert stats["reclassified"] == 0
        assert counts["intrusion"] >= 1, (
            "expected the documented false alarm; if this now passes cleanly "
            "the fallback's shape heuristic has changed and the docs need "
            "updating with it"
        )

    def test_classifier_stage_suppresses_the_false_alarm(self) -> None:
        counts, stats = run_scenario(
            "cattle", ignore=["animal"], classifier=ox_classifier()
        )
        assert stats["reclassified"] > 0
        assert counts["intrusion"] == 0

    def test_suppression_does_not_silence_people(self) -> None:
        """The whole point: cattle suppressed, person still alarms."""
        counts, _ = run_scenario("intrusion", ignore=["animal"])
        assert counts["intrusion"] >= 1


class TestScenarioCoverage:
    @pytest.mark.parametrize("scenario", ["patrol", "vehicle", "loiter", "crowd"])
    def test_scenarios_run_without_error(self, scenario: str) -> None:
        _, stats = run_scenario(scenario, ignore=["animal"], seconds=8.0)
        assert stats["last_error"] == ""
        assert stats["frames_processed"] > 10
