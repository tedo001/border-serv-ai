"""Plate evidence gathered across a vehicle's passage.

The unit under test is the decision, not the OCR. A single-frame read at a
gate is a guess - the vehicle is moving, the plate is thirty pixels tall, half
the frames are blurred - and "HR26DK8337" versus "HR26DK8837" decides whether a
car is waved through or stopped. These pin down when the platform is allowed to
call it.
"""

from __future__ import annotations

import pytest

from ibvap.core.types import BBox
from ibvap.vision.anpr import PlateReading
from ibvap.vision.plate_track import (
    VALID_FORMAT_WEIGHT,
    PlateTrack,
    PlateTrackRegistry,
)


def reading(text: str, confidence: float = 0.8, *, valid: bool = True) -> PlateReading:
    return PlateReading(
        text=text, raw_text=text, confidence=confidence,
        valid=valid, plate_format="standard" if valid else "",
    )


def feed(track: PlateTrack, texts, confidence: float = 0.8, **kwargs):
    decision = None
    for text in texts:
        decision = track.add_reading(reading(text, confidence, **kwargs))
    return decision


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #

class TestVoting:
    def test_one_read_is_never_a_decision(self) -> None:
        track = PlateTrack(1, min_reads=3)
        assert track.add_reading(reading("HR26DK8337", 0.99)) is None
        assert track.decision is None

    def test_agreeing_reads_decide(self) -> None:
        track = PlateTrack(1, min_reads=3)
        decision = feed(track, ["HR26DK8337"] * 3)
        assert decision is not None
        assert decision.reading.text == "HR26DK8337"
        assert decision.reads == 3
        assert decision.total_reads == 3

    def test_a_narrow_lead_is_not_a_decision(self) -> None:
        """Two candidates neck and neck is exactly when guessing is worst."""
        track = PlateTrack(1, min_reads=3, margin=1.5)
        feed(track, ["HR26DK8337"] * 3)
        assert track.decision is not None, "sanity: three clean reads decide"

        contested = PlateTrack(2, min_reads=3, margin=1.5)
        feed(contested, ["HR26DK8337", "HR26DK8837", "HR26DK8337", "HR26DK8837"])
        assert contested.decision is None, "a 2-2 split must not be reported"

    def test_a_clear_lead_over_a_rival_decides(self) -> None:
        track = PlateTrack(1, min_reads=3, margin=1.5)
        feed(track, ["HR26DK8337", "HR26DK8837", "HR26DK8337", "HR26DK8337"])
        assert track.decision is not None
        assert track.decision.reading.text == "HR26DK8337"
        assert track.decision.runner_up == "HR26DK8837"
        assert track.decision.margin == pytest.approx(3.0)
        assert track.decision.total_reads == 4

    def test_a_grammatical_plate_outweighs_an_ungrammatical_one(self) -> None:
        """The Indian plate grammar is evidence independent of OCR confidence.

        A read that satisfies a real plate template is more likely to be right
        than an equally confident read that does not, so it carries more
        weight. Interleaved, because a decision is final: letting the clean
        reads arrive first would end the vote before the rival is heard, and
        prove nothing about their relative weight.
        """
        track = PlateTrack(1, min_reads=4, margin=1.4)
        for _ in range(3):
            track.add_reading(reading("HR26DK8337", 0.7, valid=True))
            track.add_reading(reading("HRZ6DK8337", 0.7, valid=False))

        leader = track.votes["HR26DK8337"]
        rival = track.votes["HRZ6DK8337"]
        assert leader.reads == rival.reads == 3, "both candidates were heard"
        assert leader.weight == pytest.approx(rival.weight * VALID_FORMAT_WEIGHT)

        # 1.6 clears the 1.4 margin, so the grammatical read wins on a tie in
        # both read count and OCR confidence.
        track.add_reading(reading("HR26DK8337", 0.7, valid=True))
        assert track.decision is not None
        assert track.decision.reading.text == "HR26DK8337"

    def test_confidence_weights_the_vote(self) -> None:
        track = PlateTrack(1, min_reads=2, margin=1.4)
        for _ in range(2):
            track.add_reading(reading("HR26DK8337", 0.95))
        for _ in range(2):
            track.add_reading(reading("HR26DK8837", 0.40))
        assert track.decision is not None
        assert track.decision.reading.text == "HR26DK8337"

    def test_the_decision_keeps_the_best_reading_not_the_last(self) -> None:
        track = PlateTrack(1, min_reads=3)
        track.add_reading(reading("HR26DK8337", 0.55))
        track.add_reading(reading("HR26DK8337", 0.97))
        track.add_reading(reading("HR26DK8337", 0.61))
        assert track.decision is not None
        assert track.decision.reading.confidence == pytest.approx(0.97)

    def test_a_decision_is_final(self) -> None:
        """Re-deciding mid-passage would emit a second, contradictory event."""
        track = PlateTrack(1, min_reads=3)
        feed(track, ["HR26DK8337"] * 3)
        decided = track.decision
        feed(track, ["MH12AB1234"] * 20)
        assert track.decision is decided

    def test_empty_text_is_not_a_vote(self) -> None:
        track = PlateTrack(1, min_reads=1)
        assert track.add_reading(reading("", 0.9)) is None
        assert track.votes == {}


# --------------------------------------------------------------------------- #
# Kalman smoothing
# --------------------------------------------------------------------------- #

class TestPlateFilter:
    def test_no_estimate_before_the_first_observation(self) -> None:
        track = PlateTrack(1)
        assert track.predict() is None
        assert track.box is None

    def test_the_first_observation_seeds_the_filter(self) -> None:
        track = PlateTrack(1)
        box = BBox(100, 200, 180, 224)
        smoothed = track.observe(box)
        assert smoothed.center == pytest.approx(box.center)

    def test_smoothing_lags_a_jump_rather_than_snapping_to_it(self) -> None:
        """Detector noise moves the plate box frame to frame.

        Cropping the raw box hands OCR a differently-framed image every time.
        The filter follows the plate but does not snap to a single frame's
        measurement, so the estimate lands between where it was and where the
        detector says it is.
        """
        track = PlateTrack(1)
        track.observe(BBox(100, 200, 180, 224))
        for _ in range(4):
            track.predict()
            track.observe(BBox(100, 200, 180, 224))

        settled = track.box.center[0]
        track.predict()
        smoothed = track.observe(BBox(140, 200, 220, 224))  # centre 140 -> 180
        assert smoothed is not None, "a plausible move must not be gated out"
        assert settled < smoothed.center[0] < 180, "the estimate snapped to the measurement"

    def test_a_detection_far_from_the_plate_is_refused(self) -> None:
        """Plate detectors fire on headlights and reflective strips.

        Accepting one drags the filter off the plate, and every crop after it
        with it - so the read does not merely miss, it starts reading the wrong
        part of the vehicle and voting on what it finds.
        """
        track = PlateTrack(1, gate=2.5)
        for _ in range(4):
            track.predict()
            track.observe(BBox(100, 200, 180, 224))

        before = track.box.center
        track.predict()
        refused = track.observe(BBox(600, 60, 680, 84))  # a headlight, 500 px away

        assert refused is None
        assert track.rejected == 1
        assert track.misses == 1, "a refused observation counts as a miss"
        assert track.box.center[0] == pytest.approx(before[0], abs=8), (
            "the filter followed the spurious detection anyway"
        )

    def test_the_gate_holds_off_until_velocity_is_known(self) -> None:
        """Gating against a one-observation prediction rejects the truth.

        A brand-new filter has no evidence about how the vehicle is moving, so
        its prediction is a guess; gating against it would throw away the
        observations that teach it.
        """
        track = PlateTrack(1, gate=0.5)
        track.observe(BBox(100, 200, 180, 224))
        assert track.observe(BBox(300, 200, 380, 224)) is not None
        assert track.rejected == 0

    def test_it_predicts_through_a_missed_frame(self) -> None:
        """A wiper, a pillar or a blown highlight loses the plate for a frame.

        Dropping the read there throws away a frame that OCR could still have
        used, which on a vehicle in shot for a second or two is a large
        fraction of the available evidence.
        """
        track = PlateTrack(1, max_misses=15)
        track.observe(BBox(100, 200, 180, 224))
        for _ in range(3):
            track.predict()
            track.observe(BBox(100, 200, 180, 224))

        track.predict()
        track.miss()
        assert track.box is not None, "no estimate survived the miss"
        assert not track.expired

    def test_it_stops_predicting_once_it_has_coasted_too_long(self) -> None:
        """A filter left coasting drifts off the vehicle entirely.

        Reading its estimate then feeds OCR a crop of the road surface, which
        is worse than reading nothing: it manufactures votes from noise.
        """
        track = PlateTrack(1, max_misses=5)
        track.observe(BBox(100, 200, 180, 224))
        for _ in range(6):
            track.miss()
        assert track.expired


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_each_vehicle_gets_its_own_evidence(self) -> None:
        registry = PlateTrackRegistry(min_reads=2)
        feed(registry.get(1), ["HR26DK8337"] * 2)
        feed(registry.get(2), ["MH12AB1234"] * 2)
        assert registry.get(1).decision.reading.text == "HR26DK8337"
        assert registry.get(2).decision.reading.text == "MH12AB1234"

    def test_the_same_track_id_returns_the_same_evidence(self) -> None:
        registry = PlateTrackRegistry()
        assert registry.get(7) is registry.get(7)

    def test_vehicles_the_tracker_dropped_are_forgotten(self) -> None:
        """A camera on an approach road sees thousands of vehicles a day.

        Evidence for one that left the frame an hour ago is a memory leak.
        """
        registry = PlateTrackRegistry()
        for track_id in range(5):
            registry.get(track_id)
        registry.retain({1, 3})
        assert sorted(registry.tracks) == [1, 3]

    def test_retained_evidence_is_capped(self) -> None:
        registry = PlateTrackRegistry(max_tracks=4)
        for track_id in range(20):
            registry.get(track_id)
        assert len(registry.tracks) <= 4
        assert 19 in registry.tracks, "the newest vehicle was evicted"
