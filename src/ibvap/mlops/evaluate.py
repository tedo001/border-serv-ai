"""Accuracy evaluation.

Two evaluators, because the platform makes two very different kinds of claim:

* :func:`evaluate_detection` - mAP for the object detector.
* :func:`evaluate_anpr` and :func:`evaluate_face_matching` - end-to-end accuracy
  for the recognition paths, where the operationally meaningful number is not
  a detection score at all.

For ANPR, plate-level exact-match accuracy is the figure that matters: a plate
read with one wrong character is not "90% correct", it is a different vehicle.
For face matching, false accepts are reported separately from false rejects,
because in this application they carry entirely different costs - a missed
match is a lost opportunity, a false match puts the wrong person in front of an
armed response.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Detection
from ibvap.vision.nms import box_iou_matrix

log = get_logger(__name__)


@dataclass
class GroundTruthBox:
    """One labelled object in one frame."""

    bbox: BBox
    label: str
    #: Objects marked difficult are excluded from both TP and FP accounting,
    #: the standard convention that stops heavily occluded or tiny instances
    #: from dominating the score.
    difficult: bool = False


@dataclass
class DetectionMetrics:
    """Per-class and aggregate detection accuracy."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    map50: float = 0.0
    map50_95: float = 0.0
    total_ground_truth: int = 0
    total_predictions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mAP50": round(self.map50, 4),
            "mAP50_95": round(self.map50_95, 4),
            "ground_truth": self.total_ground_truth,
            "predictions": self.total_predictions,
            "per_class": {
                name: {k: round(v, 4) for k, v in values.items()}
                for name, values in self.per_class.items()
            },
        }


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve (all-point interpolation).

    All-point rather than the legacy 11-point interpolation: 11-point is
    coarse enough to hide real differences between two candidate models, which
    is exactly the comparison this is used for.
    """
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing, right to left.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))


def evaluate_detection(
    predictions: Sequence[Sequence[Detection]],
    ground_truth: Sequence[Sequence[GroundTruthBox]],
    *,
    iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> DetectionMetrics:
    """Compute mAP over a set of frames.

    ``predictions[i]`` and ``ground_truth[i]`` describe the same frame.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must describe the same frames")

    classes = {gt.label for frame in ground_truth for gt in frame}
    classes |= {d.obj_class.value for frame in predictions for d in frame}
    metrics = DetectionMetrics()
    metrics.total_ground_truth = sum(
        1 for frame in ground_truth for gt in frame if not gt.difficult
    )
    metrics.total_predictions = sum(len(frame) for frame in predictions)

    ap_by_threshold: dict[float, list[float]] = defaultdict(list)

    for label in sorted(classes):
        ap_per_threshold: dict[float, float] = {}
        for threshold in iou_thresholds:
            ap, precision, recall = _average_precision_for_class(
                predictions, ground_truth, label, float(threshold)
            )
            ap_per_threshold[float(threshold)] = ap
            ap_by_threshold[float(threshold)].append(ap)
            if abs(threshold - 0.5) < 1e-9:
                metrics.per_class[label] = {
                    "AP50": ap, "precision": precision, "recall": recall,
                }

        if label in metrics.per_class:
            metrics.per_class[label]["AP50_95"] = float(np.mean(list(ap_per_threshold.values())))

    if ap_by_threshold:
        metrics.map50 = float(np.mean(ap_by_threshold.get(0.5, [0.0])))
        metrics.map50_95 = float(np.mean([np.mean(v) for v in ap_by_threshold.values()]))
    return metrics


def _average_precision_for_class(
    predictions: Sequence[Sequence[Detection]],
    ground_truth: Sequence[Sequence[GroundTruthBox]],
    label: str,
    iou_threshold: float,
) -> tuple[float, float, float]:
    """AP, precision and recall for one class at one IoU threshold."""
    scored: list[tuple[float, int, int]] = []  # (score, frame index, prediction index)
    for frame_index, frame in enumerate(predictions):
        for prediction_index, detection in enumerate(frame):
            if detection.obj_class.value == label:
                scored.append((detection.score, frame_index, prediction_index))

    positives = sum(
        1 for frame in ground_truth for gt in frame if gt.label == label and not gt.difficult
    )
    if positives == 0:
        return 0.0, 0.0, 0.0
    if not scored:
        return 0.0, 0.0, 0.0

    # Rank all predictions by confidence; this ordering defines the PR curve.
    scored.sort(key=lambda item: item[0], reverse=True)
    matched: dict[int, set[int]] = defaultdict(set)
    true_positives = np.zeros(len(scored))
    false_positives = np.zeros(len(scored))

    for rank, (_score, frame_index, prediction_index) in enumerate(scored):
        candidates = [
            (gt_index, gt)
            for gt_index, gt in enumerate(ground_truth[frame_index])
            if gt.label == label
        ]
        if not candidates:
            false_positives[rank] = 1
            continue

        prediction = predictions[frame_index][prediction_index]
        ious = box_iou_matrix(
            np.array([prediction.bbox.as_tuple()]),
            np.array([gt.bbox.as_tuple() for _, gt in candidates]),
        )[0]
        best = int(np.argmax(ious))
        gt_index, gt = candidates[best]

        if ious[best] < iou_threshold:
            false_positives[rank] = 1
        elif gt.difficult:
            pass  # difficult instances count as neither TP nor FP
        elif gt_index in matched[frame_index]:
            # A second prediction on the same object is a duplicate, not a hit.
            false_positives[rank] = 1
        else:
            true_positives[rank] = 1
            matched[frame_index].add(gt_index)

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    return (
        average_precision(recall, precision),
        float(precision[-1]) if len(precision) else 0.0,
        float(recall[-1]) if len(recall) else 0.0,
    )


# --------------------------------------------------------------------------- #
# Recognition accuracy
# --------------------------------------------------------------------------- #


@dataclass
class AnprMetrics:
    """End-to-end ANPR accuracy."""

    total: int = 0
    exact_matches: int = 0
    character_errors: int = 0
    total_characters: int = 0
    no_read: int = 0
    format_rejected: int = 0
    #: Reads the grammar stage repaired into the correct plate.
    corrected_to_truth: int = 0
    #: Reads the grammar stage "repaired" into the *wrong* plate - the most
    #: dangerous failure mode, since the output looks clean and confident.
    corrupted_by_correction: int = 0

    @property
    def plate_accuracy(self) -> float:
        return self.exact_matches / self.total if self.total else 0.0

    @property
    def character_accuracy(self) -> float:
        if not self.total_characters:
            return 0.0
        return 1.0 - (self.character_errors / self.total_characters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plates": self.total,
            "plate_accuracy": round(self.plate_accuracy, 4),
            "character_accuracy": round(self.character_accuracy, 4),
            "no_read": self.no_read,
            "format_rejected": self.format_rejected,
            "corrected_to_truth": self.corrected_to_truth,
            "corrupted_by_correction": self.corrupted_by_correction,
        }


def evaluate_anpr(samples: Iterable[tuple[str, str]]) -> AnprMetrics:
    """Score raw OCR output against ground truth, through the grammar stage.

    ``samples`` yields ``(raw_ocr_text, true_plate)``. Measuring *through* the
    normaliser is the point: it is the only way to see whether grammar
    correction is a net gain, and to catch the case where it confidently
    rewrites a read into the wrong registration.
    """
    from ibvap.vision.anpr import _levenshtein, normalise_plate

    metrics = AnprMetrics()
    for raw, truth in samples:
        truth = truth.upper().replace(" ", "")
        metrics.total += 1
        metrics.total_characters += len(truth)

        if not raw:
            metrics.no_read += 1
            metrics.character_errors += len(truth)
            continue

        reading = normalise_plate(raw, 1.0)
        raw_clean = raw.upper().replace(" ", "")

        if not reading.valid:
            metrics.format_rejected += 1
        if reading.text == truth:
            metrics.exact_matches += 1
            if raw_clean != truth:
                metrics.corrected_to_truth += 1
        else:
            metrics.character_errors += _levenshtein(reading.text, truth)
            if raw_clean == truth:
                # The raw read was right and normalisation broke it.
                metrics.corrupted_by_correction += 1

    log.info("anpr_evaluated", **metrics.as_dict())
    return metrics


@dataclass
class FaceMetrics:
    """Face-matching accuracy, with accepts and rejects reported separately."""

    genuine_attempts: int = 0
    impostor_attempts: int = 0
    true_accepts: int = 0
    false_rejects: int = 0
    false_accepts: int = 0
    true_rejects: int = 0
    #: Genuine matches returning the wrong identity - counted separately from a
    #: false accept against a stranger, because the operational consequence is
    #: different: the system names a specific, innocent person.
    identity_confusions: int = 0

    @property
    def true_accept_rate(self) -> float:
        return self.true_accepts / self.genuine_attempts if self.genuine_attempts else 0.0

    @property
    def false_accept_rate(self) -> float:
        return self.false_accepts / self.impostor_attempts if self.impostor_attempts else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "genuine_attempts": self.genuine_attempts,
            "impostor_attempts": self.impostor_attempts,
            "true_accept_rate": round(self.true_accept_rate, 4),
            "false_accept_rate": round(self.false_accept_rate, 4),
            "false_rejects": self.false_rejects,
            "identity_confusions": self.identity_confusions,
        }


def evaluate_face_matching(
    gallery: Any, probes: Iterable[tuple[np.ndarray, str | None]]
) -> FaceMetrics:
    """Score a face gallery against labelled probes.

    ``probes`` yields ``(embedding, true_person_id)``; a ``None`` identity
    marks an impostor who should not match anyone.
    """
    metrics = FaceMetrics()
    for embedding, truth in probes:
        match = gallery.match(embedding)
        matched = bool(match and match.matched)

        if truth is None:
            metrics.impostor_attempts += 1
            if matched:
                metrics.false_accepts += 1
            else:
                metrics.true_rejects += 1
            continue

        metrics.genuine_attempts += 1
        if not matched:
            metrics.false_rejects += 1
        elif match.person_id == truth:
            metrics.true_accepts += 1
        else:
            metrics.identity_confusions += 1
            metrics.false_accepts += 1

    log.info("face_matching_evaluated", **metrics.as_dict())
    return metrics
