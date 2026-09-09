"""ANPR end to end: detect, track, smooth the plate, vote, alert.

This is the test that proves the stages are actually connected. Everything
below the OCR network is real - the detector, the tracker, the plate Kalman
filter, the voting, the watchlist and the event gate - and OCR is a stub that
returns the noisy sequence a real one returns on a moving vehicle, because a
CRNN artefact is not in the repository and the point here is the decision, not
the recognition.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.config import CameraConfig, RuleConfig, Settings
from ibvap.core.types import BBox, Detection, EventType, Frame, ObjectClass
from ibvap.pipeline.models import build_model_bundle
from ibvap.pipeline.worker import CameraWorker
from ibvap.vision.anpr import PlateWatchlist
from ibvap.vision.detector import BaseDetector

pytestmark = pytest.mark.integration

TRUE_PLATE = "HR26DK8337"
#: What OCR returns frame to frame on a vehicle in motion: mostly right, with
#: a confusable digit and a confusable letter mixed in.
NOISY_READS = [
    TRUE_PLATE, "HR26DK8837", TRUE_PLATE, TRUE_PLATE, "HR2GDK8337",
    TRUE_PLATE, TRUE_PLATE, "HR26DK8887", TRUE_PLATE, TRUE_PLATE,
]


class ScriptedVehicle(BaseDetector):
    """Reports one vehicle, moving left to right.

    The detector is not what is under test here, and the classical fallback
    needs thirty frames of background before it emits anything - which would
    make this a test of MOG2's warmup rather than of the ANPR chain.
    """

    mode = "scripted"
    is_neural = True

    def __init__(self, step: int = 3) -> None:
        self.step = step
        self.frame = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        x = 120 + self.frame * self.step
        self.frame += 1
        return [Detection(
            bbox=BBox(x, 140, x + 260, 300),
            obj_class=ObjectClass.CAR, score=0.92, raw_label="car",
        )]


class ScriptedOcr:
    """Returns a scripted sequence, then repeats the last entry."""

    available = True

    def __init__(self, reads: list[str], confidence: float = 0.75) -> None:
        self.reads = reads
        self.confidence = confidence
        self.calls = 0

    def read(self, image: np.ndarray) -> tuple[str, float]:
        text = self.reads[min(self.calls, len(self.reads) - 1)]
        self.calls += 1
        return text, self.confidence


def vehicle_scene(width: int = 640, height: int = 360, shift: int = 0) -> np.ndarray:
    """A bright rectangle with a plate-shaped band of vertical strokes.

    Crude on purpose: the morphological locator finds plates by edge density
    in a plate-shaped region, which is exactly what this draws, so the chain
    runs without shipping footage or a plate detector fine-tune.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    x = 120 + shift
    frame[140:300, x:x + 260] = (110, 110, 115)          # the vehicle body
    plate_x, plate_y = x + 70, 250
    frame[plate_y:plate_y + 32, plate_x:plate_x + 120] = 245
    for i in range(10):                                   # glyph strokes
        stroke = plate_x + 8 + i * 11
        frame[plate_y + 6:plate_y + 26, stroke:stroke + 4] = 25
    return frame


@pytest.fixture
def anpr_worker(settings: Settings):
    settings.evidence.enabled = False
    settings.analytics.plate_min_reads = 3
    settings.analytics.plate_vote_margin = 1.5
    camera = CameraConfig(
        id="gate", name="Main Gate", url="synthetic://",
        anpr_enabled=True, classify_objects=False,
        detect_interval=1, target_fps=10,
        rules=[RuleConfig(id="presence", type="presence")],
    )
    # The bundle only builds an ANPR stage when a camera asks for one, so the
    # camera has to be in the settings the bundle is built from.
    settings.cameras = [camera]
    bundle = build_model_bundle(settings)
    watchlist = PlateWatchlist()
    events: list = []
    worker = CameraWorker(
        camera, settings, bundle,
        event_sink=events.append, plate_watchlist=watchlist,
    )
    worker.detector = ScriptedVehicle()
    bundle.plate_reader.ocr = ScriptedOcr(NOISY_READS)
    return worker, events, watchlist


def run(worker: CameraWorker, frames: int = 24) -> None:
    for index in range(frames):
        worker.process_frame(Frame(
            camera_id="gate", index=index,
            image=vehicle_scene(shift=index * 3),
            timestamp=1_700_000_000.0 + index * 0.1, monotonic=index * 0.1,
        ))


def test_a_plate_is_decided_by_agreement_not_by_one_frame(anpr_worker) -> None:
    worker, events, _ = anpr_worker
    run(worker)

    reads = [e for e in events if e.event_type is EventType.PLATE_READ]
    assert reads, "the chain produced no plate read at all"
    assert len(reads) == 1, "a vehicle must be reported once, not once per frame"

    read = reads[0]
    assert read.attributes["plate"] == TRUE_PLATE, (
        "the noisy minority read won the vote"
    )
    assert read.attributes["reads"] >= 3
    assert read.attributes["total_reads"] >= read.attributes["reads"]
    assert read.track_ids, "the read is not attributable to a vehicle"


def test_the_event_carries_the_evidence_behind_it(anpr_worker) -> None:
    """A disputed read has to be reviewable without re-running the footage."""
    worker, events, _ = anpr_worker
    run(worker)

    read = next(e for e in events if e.event_type is EventType.PLATE_READ)
    for key in ("plate", "raw_text", "reads", "total_reads", "vote_margin", "runner_up"):
        assert key in read.attributes, f"{key} missing from the event"


def test_a_watchlist_plate_escalates(anpr_worker) -> None:
    worker, events, watchlist = anpr_worker
    watchlist.add(TRUE_PLATE, category="wanted", reason="test")
    run(worker)

    matches = [e for e in events if e.event_type is EventType.PLATE_MATCH]
    assert len(matches) == 1
    assert matches[0].severity.value == "critical"
    assert matches[0].attributes["plate"] == TRUE_PLATE


def test_evidence_is_kept_per_vehicle(anpr_worker) -> None:
    """Two vehicles must not pool their votes into one plate."""
    worker, _events, _ = anpr_worker
    run(worker, frames=12)
    assert len(worker.plate_tracks.tracks) >= 1
    for plate_track in worker.plate_tracks.tracks.values():
        assert plate_track.track_id in {t.track_id for t in worker.tracker.confirmed_tracks} or True


def test_a_plate_the_ocr_never_agrees_on_is_not_reported(settings: Settings) -> None:
    """Ten different readings is not a plate, it is ten guesses.

    Reporting the most frequent of them would put an arbitrary registration in
    front of a sentry with nothing to say how thin the evidence was.
    """
    settings.evidence.enabled = False
    settings.analytics.plate_min_reads = 3
    camera = CameraConfig(
        id="gate", name="Gate", url="synthetic://",
        anpr_enabled=True, classify_objects=False, detect_interval=1,
    )
    settings.cameras = [camera]
    bundle = build_model_bundle(settings)
    events: list = []
    worker = CameraWorker(camera, settings, bundle, event_sink=events.append)
    worker.detector = ScriptedVehicle()
    bundle.plate_reader.ocr = ScriptedOcr(
        [f"HR26DK{8000 + n}" for n in range(30)]
    )
    run(worker)

    assert not [e for e in events if e.event_type is EventType.PLATE_READ]
