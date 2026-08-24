"""Detector decoding, NMS, preprocessing and face matching."""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.errors import ModelError
from ibvap.core.types import BBox, ObjectClass
from ibvap.vision.backends import CallableBackend
from ibvap.vision.detector import SUPPORTED_LAYOUTS, MotionDetector, ObjectDetector
from ibvap.vision.face import FaceGallery, l2_normalize
from ibvap.vision.nms import batched_nms, box_iou_matrix, nms
from ibvap.vision.preprocess import (
    enhance_low_light,
    is_infrared,
    is_night_frame,
    letterbox,
    mean_luma,
)


class TestNms:
    def test_suppresses_overlaps(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
        assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45).tolist() == [0, 2]

    def test_class_aware_keeps_overlapping_classes(self) -> None:
        """A motorcyclist's person box overlaps their motorcycle almost exactly;
        class-agnostic NMS would delete one of them."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], float)
        keep = batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 1]), 0.45)
        assert len(keep) == 2

    def test_empty_input(self) -> None:
        assert nms(np.empty((0, 4)), np.empty(0)).shape == (0,)

    def test_iou_matrix(self) -> None:
        boxes = np.array([[0, 0, 10, 10]], float)
        assert box_iou_matrix(boxes, boxes)[0, 0] == pytest.approx(1.0)


class TestLetterbox:
    def test_preserves_aspect_ratio(self) -> None:
        image = np.zeros((1080, 1920, 3), np.uint8)
        padded, transform = letterbox(image, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert transform.scale == pytest.approx(640 / 1920)

    def test_inverse_is_exact(self) -> None:
        """A few pixels of error here is the difference between inside and
        outside a fence zone."""
        image = np.zeros((1080, 1920, 3), np.uint8)
        _, transform = letterbox(image, (640, 640))
        source = BBox(100, 200, 300, 500)
        model_space = np.array([[
            source.x1 * transform.scale + transform.pad_x,
            source.y1 * transform.scale + transform.pad_y,
            source.x2 * transform.scale + transform.pad_x,
            source.y2 * transform.scale + transform.pad_y,
        ]])
        recovered = transform.to_source_array(model_space)[0]
        assert recovered == pytest.approx(np.array(source.as_tuple()), abs=1e-6)

    def test_inverse_clamps_to_frame(self) -> None:
        """Boxes predicted inside the letterbox padding must clamp, not go
        negative - a fancy-indexed clip silently discards the result."""
        image = np.zeros((720, 1280, 3), np.uint8)
        _, transform = letterbox(image, (640, 640))
        boxes = transform.to_source_array(np.array([[-50.0, -50.0, 10_000.0, 10_000.0]]))
        assert boxes[0][0] >= 0 and boxes[0][1] >= 0
        assert boxes[0][2] <= 1280 and boxes[0][3] <= 720


class TestDetectorLayouts:
    @staticmethod
    def _backend(fn):
        return CallableBackend(fn, input_shape=(1, 3, 640, 640))

    def test_yolov8_layout(self) -> None:
        def output(_feeds):
            tensor = np.zeros((1, 84, 100), np.float32)
            tensor[0, :4, 0] = [320, 320, 100, 200]
            tensor[0, 4, 0] = 0.92
            return tensor

        detector = ObjectDetector(self._backend(output), layout="yolov8")
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))
        assert len(result) == 1
        assert result[0].obj_class is ObjectClass.PERSON

    def test_yolov5_layout(self) -> None:
        def output(_feeds):
            tensor = np.zeros((1, 200, 85), np.float32)
            tensor[0, 0, :4] = [320, 320, 100, 200]
            tensor[0, 0, 4] = 0.95
            tensor[0, 0, 5] = 0.9
            return tensor

        detector = ObjectDetector(self._backend(output), layout="yolov5")
        assert len(detector.detect(np.zeros((640, 640, 3), np.uint8))) == 1

    def test_yolo26_skips_nms(self) -> None:
        """An end-to-end head has already suppressed duplicates; running NMS
        again can only delete legitimate boxes - two people shoulder to
        shoulder, for instance."""
        def output(_feeds):
            return np.array([[
                [100, 100, 200, 400, 0.91, 0],
                [180, 100, 280, 400, 0.88, 0],   # heavy overlap, both real
            ]], np.float32)

        detector = ObjectDetector(self._backend(output), layout="yolo26")
        assert len(detector.detect(np.zeros((640, 640, 3), np.uint8))) == 2

    def test_unknown_layout_is_rejected(self) -> None:
        with pytest.raises(ModelError, match="unknown detector layout"):
            ObjectDetector(self._backend(lambda _f: np.zeros((1, 84, 10), np.float32)),
                           layout="yolo99")

    def test_class_allowlist(self) -> None:
        def output(_feeds):
            tensor = np.zeros((1, 84, 100), np.float32)
            tensor[0, :4, 0] = [320, 320, 100, 200]
            tensor[0, 4, 0] = 0.9        # person
            tensor[0, :4, 1] = [200, 300, 80, 60]
            tensor[0, 4 + 2, 1] = 0.85   # car
            return tensor

        detector = ObjectDetector(self._backend(output), allowed_classes=["person"])
        result = detector.detect(np.zeros((640, 640, 3), np.uint8))
        assert [d.obj_class for d in result] == [ObjectClass.PERSON]

    def test_all_layouts_are_named(self) -> None:
        assert {"auto", "yolov5", "yolov8", "yolo26", "nms_xyxy"} == SUPPORTED_LAYOUTS


class TestMotionFallback:
    def test_warmup_suppresses_output(self) -> None:
        """While the background model converges everything looks like
        foreground; emitting then would flood the control room on reconnect."""
        detector = MotionDetector(warmup_frames=10)
        rng = np.random.default_rng(0)
        background = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            assert detector.detect(background.copy()) == []

    def test_detects_and_shape_classifies(self) -> None:
        detector = MotionDetector(warmup_frames=5)
        rng = np.random.default_rng(0)
        background = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
        for _ in range(12):
            detector.detect(background.copy())

        moving = background.copy()
        moving[150:350, 300:360] = 255  # tall and narrow
        result = detector.detect(moving)
        assert result
        assert result[0].obj_class is ObjectClass.PERSON

    def test_confidence_is_capped(self) -> None:
        """The fallback is far weaker than a CNN, so downstream rules must stay
        appropriately sceptical of it."""
        detector = MotionDetector(warmup_frames=2)
        rng = np.random.default_rng(1)
        background = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
        for _ in range(6):
            detector.detect(background.copy())
        moving = background.copy()
        moving[100:400, 200:280] = 255
        for detection in detector.detect(moving):
            assert detection.score <= MotionDetector.MAX_SCORE

    def test_reports_degraded_mode(self) -> None:
        assert MotionDetector().is_neural is False
        assert MotionDetector().mode == "motion_fallback"


class TestNightHandling:
    def test_night_classification(self) -> None:
        rng = np.random.default_rng(0)
        bright = rng.integers(120, 200, (480, 640, 3), dtype=np.uint8)
        dark = (bright * 0.12).astype(np.uint8)
        assert not is_night_frame(bright)
        assert is_night_frame(dark)

    def test_enhancement_raises_luminance(self) -> None:
        rng = np.random.default_rng(0)
        dark = (rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) * 0.12).astype(np.uint8)
        assert mean_luma(enhance_low_light(dark)) > mean_luma(dark)

    def test_infrared_detection(self) -> None:
        rng = np.random.default_rng(0)
        grey = np.repeat(rng.integers(0, 255, (480, 640, 1), dtype=np.uint8), 3, axis=2)
        colour = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert is_infrared(grey)
        assert not is_infrared(colour)


class TestFaceGallery:
    @staticmethod
    def _embedding(seed: int, noise: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=256).astype(np.float32)
        if noise:
            vector = vector + noise * np.random.default_rng(seed + 999).normal(size=256).astype(np.float32)
        return l2_normalize(vector)

    def test_matches_the_same_identity(self) -> None:
        gallery = FaceGallery(match_threshold=0.42, min_margin=0.05)
        gallery.enroll("P1", self._embedding(1), name="Subject One")
        gallery.enroll("P2", self._embedding(2), name="Subject Two")
        match = gallery.match(self._embedding(1, 0.2))
        assert match is not None and match.matched and match.person_id == "P1"

    def test_rejects_a_stranger(self) -> None:
        gallery = FaceGallery(match_threshold=0.42)
        gallery.enroll("P1", self._embedding(1))
        match = gallery.match(self._embedding(500))
        assert match is not None and not match.matched

    def test_margin_rejects_an_ambiguous_probe(self) -> None:
        """A probe equidistant from two identities is not a match at 0.6; it is
        an ambiguous face, and naming it would put the wrong person in front of
        an armed response."""
        first, second = self._embedding(10), self._embedding(11)
        gallery = FaceGallery(match_threshold=0.3, min_margin=0.10)
        gallery.enroll("X", first)
        gallery.enroll("Y", second)
        match = gallery.match(l2_normalize(first + second))
        assert match is not None and not match.matched

    def test_enrolment_preserves_metadata(self) -> None:
        gallery = FaceGallery()
        gallery.enroll("P1", self._embedding(1), name="Subject", category="wanted")
        gallery.enroll("P1", self._embedding(5))  # a second view, no metadata
        entry = gallery.get("P1")
        assert entry is not None
        assert entry.category == "wanted"
        assert entry.name == "Subject"
        assert len(entry.embeddings) == 2

    def test_dimension_mismatch_is_handled(self) -> None:
        gallery = FaceGallery()
        gallery.enroll("P1", self._embedding(1))
        assert gallery.match(np.ones(128, np.float32)) is None

    def test_persistence_round_trip(self, workspace) -> None:
        gallery = FaceGallery()
        gallery.enroll("P1", self._embedding(1), name="Subject", category="wanted")
        gallery.save(workspace / "gallery.npz")

        restored = FaceGallery()
        restored.load(workspace / "gallery.npz")
        assert restored.size == 1
        entry = restored.get("P1")
        assert entry is not None and entry.category == "wanted"
        match = restored.match(self._embedding(1, 0.15))
        assert match is not None and match.matched
