"""Multi-object tracking.

Implements a ByteTrack-style two-stage associator over Kalman-filtered boxes.

Why ByteTrack rather than plain SORT: its second association pass re-uses
*low-confidence* detections to continue existing tracks. On border cameras that
is not an academic nicety - a person crossing a fence at 03:00 under IR
illumination flickers between 0.6 and 0.15 confidence frame to frame. A
single-threshold tracker fragments that into six short tracks, each too brief
to trip a dwell-time rule, and the intruder is effectively invisible to
analytics. Recovering those weak frames keeps one continuous identity.

Appearance embeddings (ReID) are deliberately not used: they roughly triple the
per-frame cost, and at a BOP node with 16 cameras on a CPU that is the entire
compute budget. Motion-only association is sufficient for the fixed, mostly
sparse views this platform targets.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ibvap.core.types import BBox, Detection, ObjectClass, Track, TrackState
from ibvap.vision.nms import box_iou_matrix

#: Sentinel cost meaning "these two may never be matched".
_NO_MATCH = 1e6


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over ``[cx, cy, w, h]``.

    Noise is scaled by box height, following the ByteTrack/DeepSORT
    convention: a distant 20-pixel-tall figure and a nearby 400-pixel one
    should not share an absolute uncertainty in pixels.
    """

    #: Uncertainty in position, as a fraction of box height.
    STD_POSITION = 1.0 / 20.0
    #: Uncertainty in velocity, as a fraction of box height.
    STD_VELOCITY = 1.0 / 160.0

    def __init__(self, bbox: BBox) -> None:
        cx, cy, w, h = bbox.to_xywh()
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = [cx, cy, w, h]

        # State transition: position += velocity each frame (dt = 1 frame).
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4, dtype=np.float64)
        # Measurement matrix: we observe position and size, never velocity.
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)

        self.P = np.diag(np.square(self._initial_std(h)))

    def _initial_std(self, height: float) -> np.ndarray:
        """Covariance for a brand-new track.

        Deliberately much looser than the per-step noise: 2x on position and
        **10x on velocity**. A newly created track has exactly one observation
        and therefore no evidence at all about how fast the object is moving,
        so a tight zero-velocity prior is simply wrong. Seeding it tight makes
        the filter cling to "stationary" for several frames; the predicted box
        then lags the real object, IoU falls under the association gate, and
        the tracker forks a new identity every few frames - fragmenting exactly
        the fast movement (a runner at a fence line) that matters most.
        """
        h = max(1.0, float(height))
        p, v = self.STD_POSITION * h, self.STD_VELOCITY * h
        return np.array([2 * p, 2 * p, 2 * p, 2 * p, 10 * v, 10 * v, 10 * v, 10 * v])

    def _std(self, height: float) -> np.ndarray:
        """Per-step process noise standard deviations."""
        h = max(1.0, float(height))
        p, v = self.STD_POSITION * h, self.STD_VELOCITY * h
        return np.array([p, p, p, p, v, v, v, v], dtype=np.float64)

    def predict(self) -> BBox:
        """Advance the state one frame and return the predicted box."""
        std = self._std(self.x[3])
        Q = np.diag(np.square(std))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        # Degenerate width/height break the IoU association that follows.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.bbox

    def update(self, bbox: BBox) -> None:
        """Correct the state with an observed box."""
        z = np.array(bbox.to_xywh(), dtype=np.float64)
        std = self._std(z[3])[:4]
        R = np.diag(np.square(std))

        S = self.H @ self.P @ self.H.T + R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically rare
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        y = z - self.H @ self.x
        self.x = self.x + K @ y
        identity = np.eye(8, dtype=np.float64)
        # Joseph-free simplified form is adequate here; re-symmetrise to keep
        # P positive-definite over long-lived tracks (hours, on a fixed camera).
        self.P = (identity - K @ self.H) @ self.P
        self.P = (self.P + self.P.T) / 2.0

    @property
    def bbox(self) -> BBox:
        return BBox.from_xywh(self.x[0], self.x[1], max(1.0, self.x[2]), max(1.0, self.x[3]))

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])


class _TrackInternal:
    """A :class:`Track` plus the filter state the tracker needs privately."""

    __slots__ = ("track", "kf", "class_votes")

    def __init__(self, track: Track, kf: KalmanBoxTracker) -> None:
        self.track = track
        self.kf = kf
        # Class can flicker frame to frame (a person behind a bike reads as
        # 'bicycle' intermittently). Majority voting over the track's life is
        # far more stable than trusting the latest frame's label.
        self.class_votes: dict[ObjectClass, int] = {track.obj_class: 1}

    def vote_class(self, obj_class: ObjectClass) -> ObjectClass:
        self.class_votes[obj_class] = self.class_votes.get(obj_class, 0) + 1
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]


class ByteTracker:
    """Two-stage IoU tracker producing stable :class:`Track` identities."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou: float = 0.2,
        min_hits: int = 3,
        max_age: int = 30,
        trail_length: int = 60,
        max_distance_ratio: float = 1.2,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.max_age = max_age
        self.trail_length = trail_length
        self.max_distance_ratio = max_distance_ratio

        self._tracks: list[_TrackInternal] = []
        self._next_id = 1
        self._frame_count = 0

    # -- public API ------------------------------------------------------- #

    @property
    def tracks(self) -> list[Track]:
        """All non-removed tracks, including tentative ones."""
        return [t.track for t in self._tracks if t.track.state is not TrackState.REMOVED]

    @property
    def confirmed_tracks(self) -> list[Track]:
        """Tracks trustworthy enough to drive alerts."""
        return [t.track for t in self._tracks if t.track.state is TrackState.CONFIRMED]

    def update(
        self,
        detections: list[Detection],
        *,
        timestamp: float = 0.0,
        monotonic: float = 0.0,
    ) -> list[Track]:
        """Associate ``detections`` with existing tracks and return live tracks."""
        self._frame_count += 1

        high = [d for d in detections if d.score >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.score < self.high_threshold]

        # 1. Propagate every track forward before matching.
        for entry in self._tracks:
            predicted = entry.kf.predict()
            entry.track.bbox = predicted
            entry.track.velocity = entry.kf.velocity
            entry.track.age += 1

        active = [t for t in self._tracks if t.track.state is not TrackState.REMOVED]

        # 2. First pass: high-confidence detections against all live tracks.
        matches, unmatched_tracks, unmatched_high = self._associate(active, high, self.match_iou)
        for track_idx, det_idx in matches:
            self._apply_update(active[track_idx], high[det_idx], timestamp, monotonic)

        # 3. Second pass: low-confidence detections rescue the tracks that the
        #    first pass left unmatched. Only tracks that were healthy last
        #    frame qualify - letting long-lost tracks grab weak boxes is how a
        #    tracker starts hallucinating identities onto noise.
        recoverable = [i for i in unmatched_tracks if active[i].track.time_since_update <= 1]
        if low and recoverable:
            subset = [active[i] for i in recoverable]
            # A looser gate is appropriate here: these boxes are weak precisely
            # because the object is occluded or blurred, so their geometry is
            # noisier too.
            second, _, _ = self._associate(subset, low, self.match_iou * 0.5)
            matched_global = {recoverable[i] for i, _ in second}
            for local_idx, det_idx in second:
                self._apply_update(subset[local_idx], low[det_idx], timestamp, monotonic)
            unmatched_tracks = [i for i in unmatched_tracks if i not in matched_global]

        # 4. Third pass: proximity rescue for fast movers.
        #    An object travelling close to its own width per frame produces
        #    almost no IoU between consecutive frames - a motorcycle on a
        #    border road at 8 fps analytics is the canonical case. Before the
        #    Kalman filter has learned its velocity the IoU passes cannot hold
        #    such a track, so the tracker forks a fresh identity every frame
        #    and no dwell- or crossing-based rule ever fires on it.
        #    Matching on centre distance instead - gated tightly by object size
        #    and class - recovers those tracks without inventing associations.
        if unmatched_tracks and unmatched_high:
            fast_candidates = [i for i in unmatched_tracks if active[i].track.time_since_update <= 2]
            if fast_candidates:
                subset = [active[i] for i in fast_candidates]
                dets = [high[i] for i in unmatched_high]
                third = self._associate_by_distance(subset, dets, self.max_distance_ratio)
                matched_global = {fast_candidates[i] for i, _ in third}
                matched_dets = {unmatched_high[j] for _, j in third}
                for local_idx, det_local in third:
                    self._apply_update(subset[local_idx], dets[det_local], timestamp, monotonic)
                unmatched_tracks = [i for i in unmatched_tracks if i not in matched_global]
                unmatched_high = [i for i in unmatched_high if i not in matched_dets]

        # 5. Age out tracks that found nothing this frame.
        for idx in unmatched_tracks:
            entry = active[idx]
            entry.track.time_since_update += 1
            if entry.track.state is TrackState.TENTATIVE:
                # A tentative track that misses even one frame was probably a
                # detector false positive; drop it rather than carrying noise.
                entry.track.state = TrackState.REMOVED
            elif entry.track.time_since_update > self.max_age:
                entry.track.state = TrackState.REMOVED
            else:
                entry.track.state = TrackState.LOST

        # 6. Spawn new tracks from leftover high-confidence detections.
        for det_idx in unmatched_high:
            self._spawn(high[det_idx], timestamp, monotonic)

        self._tracks = [t for t in self._tracks if t.track.state is not TrackState.REMOVED]
        return self.tracks

    def reset(self) -> None:
        """Forget all tracks, e.g. after a stream reconnect discontinuity."""
        self._tracks.clear()
        self._frame_count = 0

    # -- internals -------------------------------------------------------- #

    def _associate(
        self,
        tracks: list[_TrackInternal],
        detections: list[Detection],
        iou_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Optimally match tracks to detections by IoU under a minimum gate."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.track.bbox.as_tuple() for t in tracks], dtype=np.float64)
        det_boxes = np.array([d.bbox.as_tuple() for d in detections], dtype=np.float64)
        iou = box_iou_matrix(track_boxes, det_boxes)

        # Forbid cross-class association outright. Without this a car passing
        # behind a stationary sentry can steal the sentry's track id, and every
        # downstream dwell timer resets.
        for ti, entry in enumerate(tracks):
            for di, det in enumerate(detections):
                if (
                    entry.track.obj_class is not ObjectClass.UNKNOWN
                    and det.obj_class is not ObjectClass.UNKNOWN
                    and entry.track.category is not det.category
                ):
                    iou[ti, di] = 0.0

        # Hungarian assignment on cost = -IoU gives the globally optimal
        # pairing; greedy nearest-match is order-dependent and demonstrably
        # worse when two people cross paths.
        track_idx, det_idx = linear_sum_assignment(-iou)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for ti, di in zip(track_idx, det_idx, strict=True):
            if iou[ti, di] >= iou_threshold:
                matches.append((int(ti), int(di)))
                matched_tracks.add(int(ti))
                matched_dets.add(int(di))

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _associate_by_distance(
        self,
        tracks: list[_TrackInternal],
        detections: list[Detection],
        max_ratio: float,
    ) -> list[tuple[int, int]]:
        """Match on size-normalised centre distance, for fast-moving objects.

        Distance is divided by the mean box diagonal so the gate means the same
        thing for a distant figure and a nearby lorry. Cross-category pairs are
        rejected outright, as in the IoU passes.
        """
        if not tracks or not detections:
            return []

        cost = np.full((len(tracks), len(detections)), _NO_MATCH, dtype=np.float64)
        for ti, entry in enumerate(tracks):
            tb = entry.track.bbox
            t_diag = float(np.hypot(tb.width, tb.height))
            for di, det in enumerate(detections):
                if (
                    entry.track.obj_class is not ObjectClass.UNKNOWN
                    and det.obj_class is not ObjectClass.UNKNOWN
                    and entry.track.category is not det.category
                ):
                    continue
                db = det.bbox
                d_diag = float(np.hypot(db.width, db.height))
                scale = max(1.0, (t_diag + d_diag) / 2.0)
                ratio = tb.distance_to(db) / scale
                if ratio <= max_ratio:
                    cost[ti, di] = ratio

        track_idx, det_idx = linear_sum_assignment(cost)
        return [
            (int(ti), int(di))
            for ti, di in zip(track_idx, det_idx, strict=True)
            if cost[ti, di] <= max_ratio
        ]

    def _apply_update(
        self,
        entry: _TrackInternal,
        detection: Detection,
        timestamp: float,
        monotonic: float,
    ) -> None:
        entry.kf.update(detection.bbox)
        track = entry.track
        track.bbox = entry.kf.bbox
        track.velocity = entry.kf.velocity
        track.score = detection.score
        track.hits += 1
        track.time_since_update = 0
        track.obj_class = entry.vote_class(detection.obj_class)
        track.last_monotonic = monotonic or track.last_monotonic

        promoted_from_tentative = (
            track.state is TrackState.TENTATIVE and track.hits >= self.min_hits
        )
        if promoted_from_tentative or track.state is TrackState.LOST:
            track.state = TrackState.CONFIRMED

        track.trail.append(track.bbox.foot)
        if len(track.trail) > self.trail_length:
            del track.trail[: len(track.trail) - self.trail_length]

    def _spawn(self, detection: Detection, timestamp: float, monotonic: float) -> None:
        track = Track(
            track_id=self._next_id,
            obj_class=detection.obj_class,
            bbox=detection.bbox,
            score=detection.score,
            state=TrackState.TENTATIVE,
            hits=1,
            start_monotonic=monotonic,
            last_monotonic=monotonic,
            start_timestamp=timestamp,
            trail=[detection.bbox.foot],
        )
        # min_hits == 1 means "trust the detector immediately"; honour it here
        # rather than forcing every track through a tentative frame.
        if self.min_hits <= 1:
            track.state = TrackState.CONFIRMED
        self._next_id += 1
        self._tracks.append(_TrackInternal(track, KalmanBoxTracker(detection.bbox)))
