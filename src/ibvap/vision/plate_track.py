"""Plate evidence accumulated across a vehicle's whole passage.

A single-frame plate read is a coin toss. At a border gate the vehicle is
moving, the plate is 30 px tall, half the frames are motion-blurred and one in
three has a headlight flaring across the glyphs. OCR on any one of those frames
produces something plausible and often wrong, and "HR26DK8337" versus
"HR26DK8837" is the difference between waving a car through and stopping it.

So a read is not a decision. This module accumulates evidence over every frame
a vehicle is tracked and decides once, on two independent grounds:

**Kalman smoothing of the plate box.** The plate is a small box inside a moving
vehicle box, and it jitters frame to frame - the detector's own noise plus the
vehicle's motion. Cropping the raw box feeds OCR a differently-framed image
every time. A constant-velocity filter over the plate box gives a stable crop,
and predicts through the frames where the plate is missed entirely (a wiper, a
pillar, a blown-out highlight) instead of dropping the read.

**Weighted voting over decoded strings.** Each frame contributes its normalised
text weighted by OCR confidence, with grammatical plates weighted above
ungrammatical ones. A decision requires both a minimum number of reads and a
margin over the runner-up - the same shape as face matching elsewhere in the
platform, and for the same reason: a leader that only just leads is not a
match, it is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ibvap.core.logging import get_logger
from ibvap.core.types import BBox
from ibvap.vision.anpr import PlateReading
from ibvap.vision.tracker import KalmanBoxTracker

log = get_logger(__name__)

#: Weight multiplier for a reading that satisfies an Indian plate template.
#: A grammatical read is worth more than an ungrammatical one at equal OCR
#: confidence, because the grammar is independent evidence.
VALID_FORMAT_WEIGHT = 1.6


@dataclass(slots=True)
class PlateVote:
    """Accumulated evidence for one candidate string."""

    text: str
    weight: float = 0.0
    reads: int = 0
    #: The single highest-confidence reading seen for this text, kept so the
    #: emitted event carries a real reading rather than a synthesised one.
    best: PlateReading | None = None

    def add(self, reading: PlateReading) -> None:
        weight = max(0.0, reading.confidence)
        if reading.valid:
            weight *= VALID_FORMAT_WEIGHT
        self.weight += weight
        self.reads += 1
        if self.best is None or reading.confidence > self.best.confidence:
            self.best = reading


@dataclass(slots=True)
class PlateDecision:
    """The outcome of a vote, once one is reached."""

    reading: PlateReading
    reads: int
    #: Total reads across every candidate, so an operator can see how much
    #: evidence the decision rests on rather than just the winner's share.
    total_reads: int
    margin: float
    runner_up: str = ""


class PlateTrack:
    """Plate evidence for one vehicle track."""

    def __init__(
        self,
        track_id: int,
        *,
        min_reads: int = 3,
        margin: float = 1.5,
        max_misses: int = 15,
        gate: float = 2.5,
    ) -> None:
        self.track_id = track_id
        self.min_reads = min_reads
        self.margin = margin
        self.max_misses = max_misses
        #: How far an observation may fall from the prediction, as a multiple
        #: of the plate's own longest side, before it is refused.
        self.gate = gate
        self.rejected = 0

        self.votes: dict[str, PlateVote] = {}
        self.decision: PlateDecision | None = None
        self.misses = 0
        self.observations = 0
        self._filter: KalmanBoxTracker | None = None
        self._predicted: BBox | None = None

    # -- geometry ---------------------------------------------------------- #

    def predict(self) -> BBox | None:
        """Advance the plate filter one frame.

        Returns the predicted plate box, or ``None`` before the first
        observation. The prediction is what makes a missed frame survivable:
        the crop still lands on the plate, so OCR gets a chance on a frame the
        detector gave up on.
        """
        if self._filter is None:
            return None
        self._predicted = self._filter.predict()
        return self._predicted

    def observe(self, box: BBox) -> BBox | None:
        """Fold a detected plate box into the filter and return the smoothed one.

        Returns ``None`` when the observation is refused. A plate detector
        occasionally fires on a headlight, a bumper sticker or a reflective
        strip elsewhere on the vehicle; accepting that box drags the filter
        off the plate and every subsequent crop with it, so an observation
        implausibly far from the prediction is treated as a miss instead.

        The gate only applies once the filter has seen enough frames to have
        evidence about velocity. Before that its prediction is a guess, and
        gating against a guess would reject the very observations that teach
        it how the vehicle is moving.
        """
        if self._filter is None:
            self._filter = KalmanBoxTracker(box)
            self._predicted = box
            self.misses = 0
            self.observations += 1
            return self._predicted

        if self.observations >= 3 and not self._within_gate(box):
            self.rejected += 1
            self.miss()
            log.debug(
                "plate_observation_rejected",
                track=self.track_id, observed=box.as_int_tuple(),
                predicted=self._predicted.as_int_tuple() if self._predicted else None,
            )
            return None

        self._filter.update(box)
        self._predicted = self._filter.bbox
        self.misses = 0
        self.observations += 1
        return self._predicted

    def _within_gate(self, box: BBox) -> bool:
        predicted = self._predicted
        if predicted is None:
            return True
        px, py = predicted.center
        ox, oy = box.center
        distance = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5
        scale = max(predicted.width, predicted.height, 1.0)
        return distance <= self.gate * scale

    def miss(self) -> None:
        self.misses += 1

    @property
    def box(self) -> BBox | None:
        """The current best estimate of where the plate is."""
        return self._predicted

    @property
    def expired(self) -> bool:
        """Whether the plate has been unseen long enough to stop predicting.

        A filter coasting indefinitely drifts away from the vehicle and starts
        handing OCR a crop of the road surface, which is worse than no crop.
        """
        return self.misses > self.max_misses

    # -- voting ------------------------------------------------------------ #

    def add_reading(self, reading: PlateReading) -> PlateDecision | None:
        """Record one frame's reading and decide if the evidence now allows it."""
        if self.decision is not None or not reading.text:
            return self.decision

        vote = self.votes.get(reading.text)
        if vote is None:
            vote = self.votes[reading.text] = PlateVote(text=reading.text)
        vote.add(reading)

        self.decision = self._decide()
        if self.decision is not None:
            log.info(
                "plate_decided",
                track=self.track_id, plate=self.decision.reading.text,
                reads=self.decision.reads, total_reads=self.decision.total_reads,
                margin=round(self.decision.margin, 2),
                runner_up=self.decision.runner_up,
            )
        return self.decision

    def _decide(self) -> PlateDecision | None:
        ranked = sorted(self.votes.values(), key=lambda v: v.weight, reverse=True)
        leader = ranked[0]
        if leader.reads < self.min_reads or leader.best is None:
            return None

        runner_up = ranked[1] if len(ranked) > 1 else None
        # No runner-up means the margin is unbounded; representing that as a
        # finite number would make the threshold comparison depend on an
        # arbitrary constant.
        ratio = (
            leader.weight / runner_up.weight
            if runner_up is not None and runner_up.weight > 0
            else float("inf")
        )
        if ratio < self.margin:
            return None

        return PlateDecision(
            reading=leader.best,
            reads=leader.reads,
            total_reads=sum(v.reads for v in self.votes.values()),
            margin=ratio,
            runner_up=runner_up.text if runner_up is not None else "",
        )

    @property
    def leader(self) -> PlateVote | None:
        """The current front-runner, decided or not - for live display."""
        if not self.votes:
            return None
        return max(self.votes.values(), key=lambda v: v.weight)


@dataclass
class PlateTrackRegistry:
    """Plate evidence for every vehicle a camera is currently tracking."""

    min_reads: int = 3
    margin: float = 1.5
    max_misses: int = 15
    #: Hard cap on retained tracks. A camera on a busy approach road can see
    #: thousands of vehicles a day, and evidence for a vehicle that left the
    #: frame an hour ago is only a memory leak.
    max_tracks: int = 256
    tracks: dict[int, PlateTrack] = field(default_factory=dict)

    def get(self, track_id: int) -> PlateTrack:
        track = self.tracks.get(track_id)
        if track is None:
            track = self.tracks[track_id] = PlateTrack(
                track_id,
                min_reads=self.min_reads,
                margin=self.margin,
                max_misses=self.max_misses,
            )
            self._evict()
        return track

    def drop(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)

    def retain(self, live_track_ids: set[int]) -> None:
        """Forget vehicles the tracker no longer reports."""
        for track_id in [t for t in self.tracks if t not in live_track_ids]:
            del self.tracks[track_id]

    def _evict(self) -> None:
        # Oldest first: dict preserves insertion order, and the oldest track is
        # the one least likely to still be in frame.
        while len(self.tracks) > self.max_tracks:
            del self.tracks[next(iter(self.tracks))]
