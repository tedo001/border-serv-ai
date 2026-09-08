"""RT-DETR output decoding, in both of the forms exporters emit.

The decoder was written against the reference head - ``(queries, 4 + nc)`` of
per-class scores - and was never run against a real export until the platform
grew a way to produce one. Ultralytics emits ``(300, 6)`` instead, already
argmaxed, and reading that as per-class scores put a class *index* of up to 79
through a sigmoid: 300 phantom detections came back at score 0.51, every one
labelled class 0. These pin both readings down.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.vision.backends import CallableBackend
from ibvap.vision.detector import LAYOUT_RTDETR, ObjectDetector, _is_decoded_rtdetr


def build(output: np.ndarray, classes: list[str], **kwargs) -> ObjectDetector:
    backend = CallableBackend(lambda _feeds: [output], input_shape=(1, 3, 640, 640))
    return ObjectDetector(
        backend,
        class_names=classes,
        layout=LAYOUT_RTDETR,
        input_size=(640, 640),
        **kwargs,
    )


COCO_LIKE = ["person", "bicycle", "car", "motorcycle", "airplane", "bus"]


# --------------------------------------------------------------------------- #
# The decoded export: [cx, cy, w, h, confidence, class_id]
# --------------------------------------------------------------------------- #

def test_decoded_export_reads_confidence_and_class_from_their_columns() -> None:
    # Two real objects and one below threshold, in the layout Ultralytics
    # writes: normalised centre-form boxes, then confidence, then a class index.
    output = np.array([[[
        0.50, 0.50, 0.20, 0.40, 0.93, 0.0,     # person, centre of frame
    ], [
        0.25, 0.60, 0.30, 0.25, 0.88, 5.0,     # bus, left
    ], [
        0.80, 0.20, 0.05, 0.05, 0.05, 2.0,     # car, well under threshold
    ]]], dtype=np.float32)

    detections = build(output, COCO_LIKE, score_threshold=0.35).detect(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    assert len(detections) == 2, "the low-confidence query must be dropped"
    labels = {d.raw_label for d in detections}
    assert labels == {"person", "bus"}, f"class ids were misread: {labels}"

    person = next(d for d in detections if d.raw_label == "person")
    assert person.score == pytest.approx(0.93, abs=1e-3)
    # Normalised 0.5,0.5,0.2,0.4 over a 1280x720 frame, letterboxed and back.
    cx, cy = person.bbox.center
    assert cx == pytest.approx(640, abs=8)
    assert cy == pytest.approx(360, abs=8)
    assert person.bbox.width == pytest.approx(256, abs=12)


def test_a_class_index_is_never_pushed_through_a_sigmoid() -> None:
    """The exact failure this decoder had, as an assertion.

    Every query carries a class index in column 5. Read as a logit, sigmoid
    collapses them all onto ~0.5 and argmax picks column 0, so a frame with one
    object yields hundreds of same-class detections just above threshold.
    """
    rng = np.random.default_rng(0)
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, :, :4] = rng.uniform(0.1, 0.9, size=(300, 4))
    output[0, :, 4] = 0.02                     # nothing is confident
    output[0, :, 5] = rng.integers(0, 6, size=300)
    output[0, 7, 4] = 0.91                     # except one real object
    output[0, 7, 5] = 2.0                      # a car

    detections = build(output, COCO_LIKE, score_threshold=0.35).detect(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    assert len(detections) == 1, f"{len(detections)} phantom detections survived"
    assert detections[0].raw_label == "car"
    assert detections[0].score == pytest.approx(0.91, abs=1e-3)


# --------------------------------------------------------------------------- #
# The reference head: [cx, cy, w, h, *per-class scores]
# --------------------------------------------------------------------------- #

def test_per_class_head_is_argmaxed() -> None:
    output = np.zeros((1, 3, 4 + len(COCO_LIKE)), dtype=np.float32)
    output[0, 0, :4] = (0.5, 0.5, 0.2, 0.4)
    output[0, 0, 4 + 5] = 0.87                 # bus
    output[0, 1, :4] = (0.2, 0.3, 0.1, 0.2)
    output[0, 1, 4 + 0] = 0.62                 # person
    output[0, 2, :4] = (0.9, 0.9, 0.1, 0.1)
    output[0, 2, 4 + 2] = 0.10                 # car, under threshold

    detections = build(output, COCO_LIKE, score_threshold=0.35).detect(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    assert sorted(d.raw_label for d in detections) == ["bus", "person"]


def test_logit_head_gets_a_sigmoid() -> None:
    """A head exported without its sigmoid emits values outside [0, 1]."""
    output = np.full((1, 2, 4 + len(COCO_LIKE)), -6.0, dtype=np.float32)
    output[0, 0, :4] = (0.5, 0.5, 0.2, 0.4)
    output[0, 0, 4 + 0] = 3.0                  # sigmoid(3) ~ 0.95
    output[0, 1, :4] = (0.1, 0.1, 0.1, 0.1)
    output[0, 1, 4 + 1] = -2.0                 # sigmoid(-2) ~ 0.12

    detections = build(output, COCO_LIKE, score_threshold=0.35).detect(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )

    assert len(detections) == 1
    assert detections[0].raw_label == "person"
    assert detections[0].score == pytest.approx(0.9526, abs=1e-3)


# --------------------------------------------------------------------------- #
# Telling the two apart
# --------------------------------------------------------------------------- #

def test_six_columns_of_probabilities_is_a_two_class_head_not_an_index() -> None:
    """Six columns is ambiguous, and guessing wrong ruins both readings.

    A two-class model emits ``4 + 2`` columns of probabilities. Its second
    probability never exceeds 1.0, which is what separates it from a column of
    class indices.
    """
    two_class = np.array([[0.5, 0.5, 0.2, 0.4, 0.30, 0.80]], dtype=np.float32)
    assert not _is_decoded_rtdetr(two_class)

    decoded = np.array([
        [0.5, 0.5, 0.2, 0.4, 0.93, 0.0],
        [0.2, 0.3, 0.1, 0.2, 0.71, 17.0],
    ], dtype=np.float32)
    assert _is_decoded_rtdetr(decoded)


def test_a_fractional_class_column_is_not_an_index() -> None:
    fractional = np.array([[0.5, 0.5, 0.2, 0.4, 0.30, 1.6]], dtype=np.float32)
    assert not _is_decoded_rtdetr(fractional)


def test_end_to_end_layouts_never_run_a_second_nms_pass() -> None:
    """RT-DETR is a set predictor; suppressing again discards true positives.

    Two people shoulder to shoulder overlap heavily. NMS would drop one; the
    decoder must keep both, because the head already decided they are distinct.
    """
    output = np.array([[
        [0.50, 0.50, 0.12, 0.45, 0.92, 0.0],
        [0.55, 0.50, 0.12, 0.45, 0.90, 0.0],   # IoU well above any threshold
    ]], dtype=np.float32)

    detections = build(output, COCO_LIKE, score_threshold=0.35).detect(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    )
    assert len(detections) == 2
