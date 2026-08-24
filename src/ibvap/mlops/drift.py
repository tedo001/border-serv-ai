"""Runtime drift monitoring.

A model that was accurate at commissioning does not stay accurate. Cameras are
re-aimed, foliage grows, a new road opens, monsoon haze arrives, an IR
illuminator fails. None of these produce an error - the pipeline keeps running
and keeps emitting confident detections, just worse ones.

There is no ground truth in the field, so accuracy cannot be measured directly.
What *can* be measured is whether the model's behaviour has changed from the
period when it was known to be working. This module records a reference profile
during commissioning and compares live behaviour against it, flagging a
divergence for a human to investigate.

A drift alert is a prompt to look, never an automatic action. The platform will
not retune thresholds or disable a camera on its own: at a border post, an
analytics change has to be a decision someone made.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ibvap.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BehaviourProfile:
    """A statistical summary of one camera's analytics behaviour."""

    camera_id: str
    #: Detections per processed frame.
    detection_rate: float = 0.0
    #: Mean and standard deviation of detector confidence.
    mean_score: float = 0.0
    score_std: float = 0.0
    #: Normalised histogram of detector scores, 10 bins over [0, 1].
    score_histogram: list[float] = field(default_factory=lambda: [0.0] * 10)
    #: Share of detections by class.
    class_distribution: dict[str, float] = field(default_factory=dict)
    #: Mean object height as a fraction of frame height.
    mean_object_height: float = 0.0
    #: Mean frame luminance - catches a failed illuminator or a re-aimed camera.
    mean_luma: float = 0.0
    #: Confirmed tracks per frame.
    track_rate: float = 0.0
    #: Frames the profile was built from.
    sample_frames: int = 0
    captured_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """The comparison of live behaviour against a reference profile."""

    camera_id: str
    #: Overall divergence in [0, 1]; higher means more changed.
    score: float = 0.0
    #: Per-signal divergences, for diagnosis.
    signals: dict[str, float] = field(default_factory=dict)
    #: Human-readable findings, most significant first.
    findings: list[str] = field(default_factory=list)
    drifted: bool = False
    reference_age_days: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "score": round(self.score, 4),
            "drifted": self.drifted,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "findings": self.findings,
            "reference_age_days": round(self.reference_age_days, 1),
        }


class DriftMonitor:
    """Accumulates live statistics for one camera and compares to a reference."""

    #: Overall divergence above which a camera is reported as drifted.
    DRIFT_THRESHOLD = 0.30

    def __init__(self, camera_id: str, window_frames: int = 2000) -> None:
        self.camera_id = camera_id
        self.window_frames = window_frames
        self.reference: BehaviourProfile | None = None

        self._scores: deque[float] = deque(maxlen=window_frames * 4)
        self._heights: deque[float] = deque(maxlen=window_frames * 4)
        self._lumas: deque[float] = deque(maxlen=window_frames)
        self._classes: dict[str, int] = {}
        self._frames = 0
        self._detections = 0
        self._tracks = 0

    # -- accumulation ------------------------------------------------------ #

    def observe(
        self,
        detections: list[Any],
        tracks: list[Any],
        *,
        frame_height: int,
        luma: float,
    ) -> None:
        """Record one processed frame."""
        self._frames += 1
        self._detections += len(detections)
        self._tracks += len(tracks)
        self._lumas.append(luma)

        for detection in detections:
            self._scores.append(float(detection.score))
            if frame_height > 0:
                self._heights.append(detection.bbox.height / frame_height)
            label = detection.obj_class.value
            self._classes[label] = self._classes.get(label, 0) + 1

        # Bound the class counter's growth over a long deployment: rescale
        # rather than reset, so the distribution is preserved.
        if self._frames % (self.window_frames * 10) == 0:
            self._classes = {k: max(1, v // 2) for k, v in self._classes.items()}

    def profile(self) -> BehaviourProfile:
        """Summarise what has been observed so far."""
        scores = np.asarray(self._scores, dtype=np.float64)
        heights = np.asarray(self._heights, dtype=np.float64)
        total_classes = sum(self._classes.values()) or 1

        histogram = (
            (np.histogram(scores, bins=10, range=(0.0, 1.0))[0] / max(1, len(scores))).tolist()
            if len(scores) else [0.0] * 10
        )

        return BehaviourProfile(
            camera_id=self.camera_id,
            detection_rate=self._detections / max(1, self._frames),
            mean_score=float(scores.mean()) if len(scores) else 0.0,
            score_std=float(scores.std()) if len(scores) else 0.0,
            score_histogram=histogram,
            class_distribution={k: v / total_classes for k, v in self._classes.items()},
            mean_object_height=float(heights.mean()) if len(heights) else 0.0,
            mean_luma=float(np.mean(self._lumas)) if self._lumas else 0.0,
            track_rate=self._tracks / max(1, self._frames),
            sample_frames=self._frames,
        )

    def set_reference(self, profile: BehaviourProfile | None = None) -> BehaviourProfile:
        """Adopt the current behaviour (or a supplied profile) as the baseline.

        Called at commissioning, once an operator has confirmed the camera is
        performing acceptably. Baselining automatically on first start would
        enshrine whatever the camera was doing at that moment - including a
        misaimed view or a night with no illumination.
        """
        self.reference = profile or self.profile()
        log.info(
            "drift_reference_set",
            camera=self.camera_id, frames=self.reference.sample_frames,
        )
        return self.reference

    def reset_window(self) -> None:
        self._scores.clear()
        self._heights.clear()
        self._lumas.clear()
        self._classes.clear()
        self._frames = self._detections = self._tracks = 0

    # -- comparison -------------------------------------------------------- #

    def compare(self) -> DriftReport:
        """Compare current behaviour against the reference profile."""
        report = DriftReport(camera_id=self.camera_id)
        if self.reference is None:
            report.findings.append("no reference profile; run commissioning to establish one")
            return report
        if self._frames < 100:
            report.findings.append(
                f"only {self._frames} frames observed; too few to judge drift"
            )
            return report

        current = self.profile()
        reference = self.reference
        report.reference_age_days = (time.time() - reference.captured_at) / 86400.0

        signals: dict[str, float] = {
            "detection_rate": _relative_change(reference.detection_rate, current.detection_rate),
            "mean_score": _relative_change(reference.mean_score, current.mean_score),
            "object_height": _relative_change(
                reference.mean_object_height, current.mean_object_height
            ),
            "luma": _relative_change(reference.mean_luma, current.mean_luma),
            "track_rate": _relative_change(reference.track_rate, current.track_rate),
            "score_distribution": _jensen_shannon(
                np.asarray(reference.score_histogram), np.asarray(current.score_histogram)
            ),
            "class_distribution": _distribution_distance(
                reference.class_distribution, current.class_distribution
            ),
        }
        report.signals = signals

        # Weighted, not a plain mean: a collapse in detection rate or a shifted
        # score distribution is far stronger evidence of a real problem than a
        # luminance change, which may simply be the season.
        weights = {
            "detection_rate": 0.25, "score_distribution": 0.25, "mean_score": 0.15,
            "class_distribution": 0.15, "object_height": 0.10, "track_rate": 0.05,
            "luma": 0.05,
        }
        report.score = sum(min(1.0, signals[k]) * w for k, w in weights.items())
        report.drifted = report.score >= self.DRIFT_THRESHOLD

        findings: list[tuple[float, str]] = []
        if signals["detection_rate"] > 0.4:
            direction = "fewer" if current.detection_rate < reference.detection_rate else "more"
            findings.append((
                signals["detection_rate"],
                f"detection rate changed markedly ({direction}): "
                f"{reference.detection_rate:.2f} -> {current.detection_rate:.2f} per frame. "
                "Check the camera's aim, focus and scene for obstruction.",
            ))
        if signals["score_distribution"] > 0.3:
            findings.append((
                signals["score_distribution"],
                f"detector confidence distribution has shifted "
                f"(mean {reference.mean_score:.2f} -> {current.mean_score:.2f}). "
                "Often the first sign of degraded image quality or a changed view.",
            ))
        if signals["luma"] > 0.5:
            findings.append((
                signals["luma"],
                f"scene brightness changed substantially "
                f"({reference.mean_luma:.0f} -> {current.mean_luma:.0f}). "
                "Check for a failed IR illuminator or a re-aimed camera.",
            ))
        if signals["object_height"] > 0.4:
            findings.append((
                signals["object_height"],
                "typical object size has changed; the camera may have been "
                "moved, zoomed or re-aimed, which invalidates its zones.",
            ))
        if signals["class_distribution"] > 0.4:
            findings.append((
                signals["class_distribution"],
                "the mix of detected classes has changed; the scene's traffic "
                "pattern or the model's behaviour has shifted.",
            ))

        findings.sort(reverse=True)
        report.findings = [text for _, text in findings]
        if report.drifted and not report.findings:
            report.findings.append(
                "aggregate behaviour has drifted from the reference without a single "
                "dominant cause; review the camera against its commissioning record."
            )
        return report


def _relative_change(reference: float, current: float) -> float:
    """Symmetric relative change in ``[0, 1]``.

    Symmetric so that a halving and a doubling register as equally significant;
    a plain ratio would rate them very differently.
    """
    denominator = abs(reference) + abs(current)
    if denominator < 1e-9:
        return 0.0
    return min(1.0, abs(current - reference) / (denominator / 2.0) / 2.0)


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions, in ``[0, 1]``.

    Preferred over KL divergence because it is symmetric and finite even when
    one distribution has zero mass where the other does not - which happens
    routinely with score histograms.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p, q = p / p_sum, q / q_sum
    m = (p + q) / 2.0

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return float(np.clip((kl(p, m) + kl(q, m)) / 2.0, 0.0, 1.0))


def _distribution_distance(
    reference: dict[str, float], current: dict[str, float]
) -> float:
    """Total variation distance between two class distributions."""
    labels = set(reference) | set(current)
    if not labels:
        return 0.0
    return min(
        1.0,
        sum(abs(reference.get(label, 0.0) - current.get(label, 0.0)) for label in labels) / 2.0,
    )


def save_profiles(path: str | Path, profiles: dict[str, BehaviourProfile]) -> None:
    """Persist reference profiles so they survive a restart."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({k: v.as_dict() for k, v in profiles.items()}, indent=2), encoding="utf-8"
    )
    log.info("drift_profiles_saved", path=str(target), cameras=len(profiles))


def load_profiles(path: str | Path) -> dict[str, BehaviourProfile]:
    """Load reference profiles written by :func:`save_profiles`."""
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("drift_profiles_unreadable", path=str(source), error=str(exc))
        return {}
    return {key: BehaviourProfile(**value) for key, value in raw.items()}
