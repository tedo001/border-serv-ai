"""Multi-object tracking: identity stability, occlusion and association."""

from __future__ import annotations

import pytest

from ibvap.core.types import ObjectClass, TrackState
from ibvap.vision.tracker import ByteTracker, KalmanBoxTracker
from tests.conftest import make_detection


class TestKalman:
    def test_learns_velocity(self) -> None:
        """A newly created track has no evidence about speed.

        Seeding velocity covariance too tightly makes the filter cling to
        'stationary'; the predicted box then lags, IoU falls under the gate,
        and the tracker forks a new identity every few frames.
        """
        from ibvap.core.types import BBox

        kalman = KalmanBoxTracker(BBox(100, 300, 160, 450))
        for step in range(1, 12):
            kalman.predict()
            kalman.update(BBox(100 + step * 20, 300, 160 + step * 20, 450))
        assert kalman.velocity[0] == pytest.approx(20, rel=0.25)


class TestIdentityStability:
    @pytest.mark.parametrize("speed", [5, 10, 20, 35, 50])
    def test_identity_survives_at_any_speed(self, speed: int) -> None:
        tracker = ByteTracker(min_hits=3, max_age=10)
        for step in range(15):
            tracker.update([make_detection(100 + step * speed, 300)], monotonic=step * 0.1)

        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 1, f"track fragmented at {speed} px/frame"
        assert confirmed[0].track_id == 1
        assert confirmed[0].hits == 15

    def test_recovers_after_occlusion(self) -> None:
        tracker = ByteTracker(min_hits=2, max_age=10)
        for step in range(6):
            tracker.update([make_detection(100 + step * 15, 300)], monotonic=step * 0.1)
        track_id = tracker.confirmed_tracks[0].track_id

        for step in range(5):  # occluded, within the age budget
            tracker.update([], monotonic=0.6 + step * 0.1)
        assert tracker.tracks[0].state is TrackState.LOST

        result = tracker.update([make_detection(100 + 11 * 15, 300)], monotonic=1.1)
        assert [t.track_id for t in result] == [track_id]

    def test_track_expires_beyond_max_age(self) -> None:
        tracker = ByteTracker(min_hits=2, max_age=3)
        for step in range(4):
            tracker.update([make_detection(100, 300)], monotonic=step * 0.1)
        for step in range(6):
            tracker.update([], monotonic=0.5 + step * 0.1)
        assert tracker.tracks == []

    def test_weak_detections_continue_a_track(self) -> None:
        """The ByteTrack advantage: a person under IR flickers between 0.6 and
        0.15 confidence, and a single-threshold tracker fragments them into
        stretches too short for any dwell rule to fire on."""
        tracker = ByteTracker(min_hits=2, max_age=10, high_threshold=0.5, low_threshold=0.1)
        for step in range(4):
            tracker.update([make_detection(100 + step * 15, 300, score=0.9)], monotonic=step * 0.1)
        track_id = tracker.confirmed_tracks[0].track_id

        for step in range(4, 10):
            tracker.update([make_detection(100 + step * 15, 300, score=0.22)], monotonic=step * 0.1)

        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 1
        assert confirmed[0].track_id == track_id
        assert confirmed[0].hits == 10


class TestAssociation:
    def test_no_cross_class_identity_theft(self) -> None:
        """A vehicle passing behind a stationary sentry must not steal their id,
        which would reset every dwell timer attached to it."""
        tracker = ByteTracker(min_hits=1, max_age=5)
        tracker.update([make_detection(100, 300, obj_class=ObjectClass.PERSON)], monotonic=0.0)
        result = tracker.update(
            [make_detection(102, 302, obj_class=ObjectClass.TRUCK)], monotonic=0.1
        )
        assert len({t.track_id for t in result}) == 2

    def test_distant_detections_do_not_merge(self) -> None:
        tracker = ByteTracker(min_hits=1, max_age=5)
        tracker.update([make_detection(0, 0)], monotonic=0.0)
        result = tracker.update([make_detection(900, 600)], monotonic=0.1)
        assert len({t.track_id for t in result}) == 2

    def test_crossing_paths_keep_their_identities(self) -> None:
        """Hungarian assignment, not greedy nearest-match, is what makes this work."""
        tracker = ByteTracker(min_hits=2, max_age=10)
        for step in range(12):
            tracker.update(
                [make_detection(100 + step * 20, 300), make_detection(400 - step * 20, 300)],
                monotonic=step * 0.1,
            )
        assert len(tracker.confirmed_tracks) == 2

    def test_tentative_noise_is_dropped(self) -> None:
        tracker = ByteTracker(min_hits=3)
        tracker.update([make_detection(10, 10)], monotonic=0.0)
        tracker.update([], monotonic=0.1)
        assert tracker.tracks == []

    def test_class_is_majority_voted(self) -> None:
        """Class flickers frame to frame; the latest label is not the truth."""
        tracker = ByteTracker(min_hits=1, max_age=10)
        for step in range(8):
            label = ObjectClass.BICYCLE if step == 4 else ObjectClass.MOTORCYCLE
            tracker.update([make_detection(100 + step * 5, 300, obj_class=label)], monotonic=step * 0.1)
        assert tracker.confirmed_tracks[0].obj_class is ObjectClass.MOTORCYCLE


class TestTrackGeometry:
    def test_trail_is_bounded(self) -> None:
        tracker = ByteTracker(min_hits=1, max_age=50, trail_length=10)
        for step in range(40):
            tracker.update([make_detection(100 + step * 5, 300)], monotonic=step * 0.1)
        assert len(tracker.confirmed_tracks[0].trail) == 10

    def test_displacement_and_duration(self) -> None:
        tracker = ByteTracker(min_hits=1, max_age=20, trail_length=100)
        for step in range(10):
            tracker.update([make_detection(100 + step * 20, 300)], monotonic=step * 0.5)
        track = tracker.confirmed_tracks[0]
        assert track.displacement() == pytest.approx(180, rel=0.15)
        assert track.duration == pytest.approx(4.5, rel=0.05)

    def test_reset_clears_state(self) -> None:
        tracker = ByteTracker(min_hits=1)
        tracker.update([make_detection(100, 300)], monotonic=0.0)
        tracker.reset()
        assert tracker.tracks == []
