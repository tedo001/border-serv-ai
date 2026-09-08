"""The scenario simulator and its RT-DETR-adjacent decode partner."""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.errors import StreamClosedError, StreamError
from ibvap.core.types import ObjectClass
from ibvap.ingest.simulator import (
    SCENARIOS,
    Actor,
    SimulatedSource,
    render_terrain,
)
from ibvap.ingest.source import build_source
from ibvap.vision.backends import CallableBackend
from ibvap.vision.detector import SUPPORTED_LAYOUTS, ObjectDetector
from ibvap.vision.preprocess import mean_luma


class TestScenarios:
    def test_every_scenario_builds_and_renders(self) -> None:
        for name in SCENARIOS:
            source = SimulatedSource(name, width=320, height=180, fps=10)
            source.open()
            try:
                for _ in range(30):
                    frame = source.read()
                assert frame.shape == (180, 320, 3)
            finally:
                source.close()

    def test_unknown_scenario_is_rejected(self) -> None:
        with pytest.raises(StreamError, match="unknown scenario"):
            SimulatedSource("does-not-exist")

    def test_deterministic_by_seed(self) -> None:
        """A test asserting 'this scenario raises an intrusion alert' must give
        the same answer on every machine."""
        def frames(seed: int) -> np.ndarray:
            source = SimulatedSource("intrusion", width=240, height=135, seed=seed, noise=0.0)
            source.open()
            try:
                for _ in range(20):
                    frame = source.read()
                return frame
            finally:
                source.close()

        assert np.array_equal(frames(7), frames(7))
        assert not np.array_equal(frames(7), frames(8))

    def test_cattle_scenario_contains_livestock_and_a_person(self) -> None:
        """The contrast is the point: livestock must be suppressible while a
        person in the same zone is not."""
        source = SimulatedSource("cattle", width=320, height=180, fps=10)
        source.open()
        try:
            kinds: set[str] = set()
            for _ in range(200):
                source.read()
                kinds |= {actor["kind"] for actor in source.ground_truth()}
        finally:
            source.close()
        assert ObjectClass.ANIMAL.value in kinds
        assert ObjectClass.PERSON.value in kinds

    def test_night_scenario_is_dark(self) -> None:
        day = SimulatedSource("intrusion", width=320, height=180, noise=0.0)
        night = SimulatedSource("night", width=320, height=180, noise=0.0)
        day.open()
        night.open()
        try:
            for _ in range(10):
                day_frame, night_frame = day.read(), night.read()
        finally:
            day.close()
            night.close()
        assert mean_luma(night_frame) < mean_luma(day_frame)
        assert mean_luma(night_frame) < 60  # classified as night downstream


class TestActors:
    def test_spawn_delay(self) -> None:
        actor = Actor(ObjectClass.PERSON, x=0.5, y=0.5, spawn_at=2.0)
        actor.update(0.1, elapsed=1.0)
        assert not actor.visible
        actor.update(0.1, elapsed=2.5)
        assert actor.visible

    def test_lifetime_expiry(self) -> None:
        actor = Actor(ObjectClass.BAG, x=0.5, y=0.5, lifetime=1.0)
        for _ in range(5):
            actor.update(0.1, elapsed=1.0)
        assert actor.visible
        for _ in range(10):
            actor.update(0.1, elapsed=2.0)
        assert not actor.visible

    def test_freeze_stops_movement(self) -> None:
        """Loitering and abandonment both depend on an actor stopping."""
        actor = Actor(ObjectClass.PERSON, x=0.1, y=0.5, vx=0.5, freeze_after=1.0)
        for _ in range(20):
            actor.update(0.1, elapsed=1.0)
        frozen_x = actor.x
        for _ in range(20):
            actor.update(0.1, elapsed=1.0)
        assert actor.x == pytest.approx(frozen_x)

    def test_depth_scaling(self) -> None:
        """Objects higher in frame are further away and must render smaller,
        or size-based thresholds behave nothing like they would on a camera."""
        near = Actor(ObjectClass.PERSON, x=0.5, y=0.95)
        far = Actor(ObjectClass.PERSON, x=0.5, y=0.35)
        assert near.depth_scale() > far.depth_scale()

    def test_offscreen_actors_are_not_visible(self) -> None:
        actor = Actor(ObjectClass.PERSON, x=1.9, y=0.5)
        actor.update(0.1, elapsed=1.0)
        assert not actor.visible


class TestTerrain:
    def test_has_sky_and_ground(self) -> None:
        frame = render_terrain(320, 180, night=False, seed=1)
        assert mean_luma(frame[:40]) > mean_luma(frame[120:])  # sky brighter

    def test_night_terrain_is_darker(self) -> None:
        day = render_terrain(320, 180, night=False, seed=1)
        night = render_terrain(320, 180, night=True, seed=1)
        assert mean_luma(night) < mean_luma(day)


class TestSourceFactory:
    def test_sim_url_builds_a_simulator(self) -> None:
        source = build_source("sim://cattle?width=320&height=180&fps=10&seed=2")
        assert isinstance(source, SimulatedSource)
        assert source.scenario == "cattle"
        assert source.info.width == 320

    def test_night_flag(self) -> None:
        source = build_source("sim://intrusion?night=1")
        assert isinstance(source, SimulatedSource)
        assert source.night

    def test_non_looping_source_ends(self) -> None:
        source = build_source("sim://patrol?fps=10&duration=1&loop=0&width=160&height=90")
        source.open()
        try:
            with pytest.raises(StreamClosedError):
                for _ in range(200):
                    source.read()
        finally:
            source.close()

    def test_synthetic_url_still_works(self) -> None:
        """The original synthetic source must keep working - existing
        configurations reference it."""
        source = build_source("synthetic://?width=160&height=90&frames=3")
        source.open()
        try:
            assert source.read().shape == (90, 160, 3)
        finally:
            source.close()


class TestRtdetrLayout:
    """RT-DETR decoding, which differs from YOLO in ways that fail silently."""

    @staticmethod
    def _backend(fn):
        return CallableBackend(fn, input_shape=(1, 3, 640, 640))

    def test_registered(self) -> None:
        assert "rtdetr" in SUPPORTED_LAYOUTS

    def test_normalised_boxes_are_scaled_to_pixels(self) -> None:
        """Decoded as YOLO pixels these cluster in the top-left corner -
        plausible in a list, nonsense on screen."""
        def head(_feeds):
            out = np.zeros((1, 300, 84), np.float32)
            out[0, 0, :4] = [0.5, 0.5, 0.2, 0.4]   # centre of frame
            out[0, 0, 4] = 0.95
            return out

        detector = ObjectDetector(self._backend(head), layout="rtdetr")
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))
        assert len(result) == 1
        centre_x, centre_y = result[0].bbox.center
        assert centre_x == pytest.approx(320, abs=8)
        assert centre_y == pytest.approx(320, abs=8)

    def test_logit_head_gets_a_sigmoid(self) -> None:
        def head(_feeds):
            out = np.full((1, 300, 84), -8.0, np.float32)
            out[0, 0, :4] = [0.5, 0.5, 0.2, 0.4]
            out[0, 0, 4] = 4.0  # sigmoid(4) ~ 0.98
            return out

        detector = ObjectDetector(self._backend(head), layout="rtdetr")
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))
        assert result and result[0].score == pytest.approx(0.982, abs=0.01)

    def test_no_nms_is_applied(self) -> None:
        """RT-DETR is a set predictor; duplicate suppression is learned, so a
        second NMS pass could only delete valid boxes."""
        def head(_feeds):
            out = np.zeros((1, 300, 84), np.float32)
            out[0, 0, :4] = [0.50, 0.5, 0.20, 0.4]
            out[0, 0, 4] = 0.95
            out[0, 1, :4] = [0.53, 0.5, 0.20, 0.4]  # heavy overlap, both real
            out[0, 1, 4] = 0.92
            return out

        detector = ObjectDetector(self._backend(head), layout="rtdetr")
        assert len(detector.detect(np.zeros((640, 640, 3), np.uint8))) == 2

    def test_non_square_input_scales_each_axis(self) -> None:
        def head(_feeds):
            out = np.zeros((1, 300, 84), np.float32)
            out[0, 0, :4] = [0.5, 0.5, 0.1, 0.1]
            out[0, 0, 4] = 0.9
            return out

        detector = ObjectDetector(
            CallableBackend(head, input_shape=(1, 3, 384, 640)), layout="rtdetr"
        )
        result = detector.detect(np.zeros((384, 640, 3), np.uint8))
        assert result
        centre_x, centre_y = result[0].bbox.center
        assert centre_x == pytest.approx(320, abs=10)
        assert centre_y == pytest.approx(192, abs=10)


class TestFileReplay:
    """A file has an end; a camera does not.

    Reaching the end of a clip used to raise ``StreamClosedError``, which the
    reader treats as the stream dying: it marked the camera offline,
    reconnected, reopened the file and marked it online again. Replaying a
    30-second clip therefore produced a camera_offline/camera_online alert pair
    every 30 seconds, and buried the analytics events between them.
    """

    def _clip(self, path, frames: int = 12):
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
        )
        for index in range(frames):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, index % 3] = 255
            writer.write(frame)
        writer.release()
        return path

    def test_a_clip_rewinds_instead_of_reporting_the_stream_dead(
        self, workspace
    ) -> None:
        from ibvap.ingest.source import build_source

        source = build_source(str(self._clip(workspace / "clip.mp4")))
        source.open()
        try:
            for _ in range(30):  # more than twice the clip's length
                assert source.read() is not None
            assert source.loops >= 2, "the clip never rewound"
        finally:
            source.close()

    def test_a_network_stream_never_rewinds(self) -> None:
        """Rewinding a camera would hide a real outage behind a replay."""
        from ibvap.ingest.source import OpenCVSource

        source = OpenCVSource("rtsp://10.0.0.5:554/stream")
        assert source._can_rewind is False

    def test_replay_can_be_turned_off(self, workspace) -> None:
        from ibvap.core.errors import StreamClosedError
        from ibvap.ingest.source import build_source

        source = build_source(
            str(self._clip(workspace / "once.mp4")), loop_files=False
        )
        source.open()
        try:
            with pytest.raises(StreamClosedError):
                for _ in range(30):
                    source.read()
        finally:
            source.close()
