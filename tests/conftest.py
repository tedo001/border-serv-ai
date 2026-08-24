"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ibvap.core.config import (
    AnalyticsConfig,
    CameraConfig,
    EvidenceConfig,
    RuleConfig,
    SecurityConfig,
    Settings,
    StorageConfig,
    TelemetryConfig,
    TrackerConfig,
    TripwireConfig,
    ZoneConfig,
)
from ibvap.core.types import BBox, Detection, Frame, ObjectClass, Track, TrackState


@pytest.fixture
def workspace() -> Iterator[Path]:
    """An isolated directory for data written by a test."""
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


@pytest.fixture
def scene() -> np.ndarray:
    """A deterministic, non-uniform frame.

    Structured noise rather than a flat fill: a uniform image makes several
    detectors and the tamper rule behave unrepresentatively.
    """
    rng = np.random.default_rng(42)
    image = rng.integers(60, 190, (720, 1280, 3)).astype(np.uint8)
    image[:240] = np.clip(image[:240].astype(np.int16) + 50, 0, 255).astype(np.uint8)
    return image


@pytest.fixture
def frame(scene: np.ndarray) -> Frame:
    return Frame(
        camera_id="cam-test", index=1, image=scene,
        timestamp=1_700_000_000.0, monotonic=100.0, fps=8.0,
    )


@pytest.fixture
def camera_config() -> CameraConfig:
    """A camera with a restricted zone, a tripwire and typical rules."""
    return CameraConfig(
        id="cam-test",
        name="Test Camera",
        url="synthetic://?width=640&height=480&fps=20",
        target_fps=8.0,
        zones=[
            ZoneConfig(
                id="fence-strip", name="Fence Strip",
                points=[(0.4, 0.3), (1.0, 0.3), (1.0, 1.0), (0.4, 1.0)],
            )
        ],
        tripwires=[
            TripwireConfig(
                id="fence-line", name="Fence Line",
                start=(0.5, 0.05), end=(0.5, 0.95),
                direction="any", left_label="exfiltration", right_label="infiltration",
            )
        ],
        rules=[
            RuleConfig(
                id="intrusion", type="intrusion", zones=["fence-strip"],
                classes=["person"], cooldown_seconds=5, params={"confirm_frames": 2},
            ),
            RuleConfig(
                id="crossing", type="line_crossing",
                tripwires=["fence-line"], cooldown_seconds=5,
            ),
        ],
    )


@pytest.fixture
def settings(workspace: Path, camera_config: CameraConfig) -> Settings:
    """A fully isolated node configuration."""
    return Settings(
        site_id="test-site",
        site_name="Test Site",
        storage=StorageConfig(database_url=f"sqlite+aiosqlite:///{workspace}/test.db"),
        evidence=EvidenceConfig(directory=workspace / "evidence", clip=False),
        security=SecurityConfig(
            jwt_secret="t" * 48, bootstrap_admin_password="TestPassphrase123!"
        ),
        telemetry=TelemetryConfig(log_level="ERROR"),
        tracker=TrackerConfig(min_hits=2, max_age=10),
        analytics=AnalyticsConfig(dedup_window_seconds=1.0),
        cameras=[camera_config],
    )


def make_detection(
    x: float, y: float, width: float = 60, height: float = 150,
    obj_class: ObjectClass = ObjectClass.PERSON, score: float = 0.9,
) -> Detection:
    """Build a detection at a given position."""
    return Detection(BBox(x, y, x + width, y + height), obj_class, score)


def make_track(
    track_id: int, x: float, y: float, width: float = 60, height: float = 150,
    obj_class: ObjectClass = ObjectClass.PERSON,
    velocity: tuple[float, float] = (0.0, 0.0),
    trail: list[tuple[float, float]] | None = None,
    monotonic: float = 100.0,
) -> Track:
    """Build a confirmed track, as the tracker would emit it."""
    box = BBox(x, y, x + width, y + height)
    track = Track(
        track_id=track_id, obj_class=obj_class, bbox=box, score=0.9,
        state=TrackState.CONFIRMED, age=10, hits=10, time_since_update=0,
        velocity=velocity, start_monotonic=monotonic - 5.0, last_monotonic=monotonic,
        start_timestamp=1_700_000_000.0,
    )
    track.trail = trail if trail is not None else [box.foot]
    return track
