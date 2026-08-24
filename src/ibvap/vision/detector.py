"""Object detection.

Supported model families (selected per model in the registry via ``layout``):

===============  =====================  ==========================================
``layout``       Output tensor          Notes
===============  =====================  ==========================================
``yolov5``       ``(anchors, 5+nc)``    v5/v7 heads: objectness x class score
``yolov8``       ``(4+nc, anchors)``    v8/v9/v10/v11: class scores, no objectness
``yolo26``       ``(N, 6)`` xyxy        End-to-end, NMS-free head (v10-style e2e)
``nms_xyxy``     ``(N, 6)`` xyxy        Any export with NMS folded into the graph
``auto``         inferred from shape    Convenience only - prefer an explicit value
===============  =====================  ==========================================

**YOLO26** is the newest family and the recommended default for new
deployments. It is end-to-end: the NMS step is folded into the model, so the
graph emits final boxes directly. Two things follow that matter operationally.
First, the CPU-side NMS pass disappears, which is a real saving on an edge node
running sixteen cameras on four cores. Second, because the model has already
suppressed duplicates, running class-aware NMS again over its output can only
delete legitimate boxes - so this decoder skips NMS entirely for end-to-end
layouts. Confidence filtering still applies, since the threshold is a
deployment decision rather than a property of the graph.

If a YOLO26 checkpoint is exported *without* end-to-end mode, its head matches
the ``yolov8`` layout and should be configured as such; the registry entry, not
the file name, is what the loader believes.

Two detector implementations share one interface:

* :class:`ObjectDetector` - a neural detector (YOLO-family ONNX artefact).
* :class:`MotionDetector` - a classical background-subtraction detector.

The second exists because "works without model weights" is a real operational
requirement, not a test convenience. A BOP node whose model artefacts failed to
sync must still raise intrusion alerts on the fence line tonight; classical
motion detection with coarse shape classification is materially worse than a
CNN but vastly better than a blind camera. The platform reports which mode it
is in through ``/health`` and the ``ibvap_model_info`` metric so a control room
is never misled about the quality of what it is watching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import cv2
import numpy as np

from ibvap.core.errors import ModelError
from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.vision.backends import InferenceBackend
from ibvap.vision.nms import batched_nms, xywh_to_xyxy
from ibvap.vision.preprocess import LetterboxTransform, letterbox, to_nchw

log = get_logger(__name__)

#: Every output layout the decoder understands.
LAYOUT_YOLOV5 = "yolov5"
LAYOUT_YOLOV8 = "yolov8"
LAYOUT_YOLO26 = "yolo26"
LAYOUT_NMS_XYXY = "nms_xyxy"
LAYOUT_AUTO = "auto"

SUPPORTED_LAYOUTS: frozenset[str] = frozenset(
    {LAYOUT_YOLOV5, LAYOUT_YOLOV8, LAYOUT_YOLO26, LAYOUT_NMS_XYXY, LAYOUT_AUTO}
)

#: Layouts whose graph already performed suppression. Re-running NMS over
#: these can only remove valid detections, never improve them.
END_TO_END_LAYOUTS: frozenset[str] = frozenset({LAYOUT_YOLO26, LAYOUT_NMS_XYXY})

#: Standard COCO-80 ordering, the class list of every public YOLO checkpoint.
COCO80: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class BaseDetector(ABC):
    """Interface implemented by every detector the pipeline can use."""

    #: Reported through health endpoints so operators know what is running.
    mode: str = "unknown"
    #: False when the detector is a degraded fallback rather than a model.
    is_neural: bool = False

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections for one BGR frame, in source pixel coordinates."""

    def close(self) -> None:
        """Release resources held by the detector."""


class ObjectDetector(BaseDetector):
    """YOLO-family ONNX detector with automatic output-layout handling."""

    mode = "neural"
    is_neural = True

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        class_names: Sequence[str] = COCO80,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] | None = None,
        layout: str = "auto",
        allowed_classes: Sequence[str] | None = None,
        min_height_fraction: float = 0.0,
        max_detections: int = 300,
    ) -> None:
        self.backend = backend
        self.class_names = tuple(class_names)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        if layout not in SUPPORTED_LAYOUTS:
            raise ModelError(
                f"unknown detector layout {layout!r}; expected one of {sorted(SUPPORTED_LAYOUTS)}"
            )
        self.layout = layout
        self.max_detections = max_detections
        self.min_height_fraction = min_height_fraction

        # A static input size declared by the graph always wins; the config
        # value is a fallback for models exported with dynamic axes.
        self.input_size = backend.input_size() or input_size or (640, 640)

        # Resolve the allowlist once into taxonomy members, so the hot path does
        # a set lookup rather than string normalisation per detection.
        self.allowed: set[ObjectClass] | None = (
            {ObjectClass.coerce(c) for c in allowed_classes} if allowed_classes else None
        )
        self._input_name = backend.input_names[0] if backend.input_names else "images"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []

        padded, transform = letterbox(image, self.input_size)
        tensor = to_nchw(padded)
        outputs = self.backend.run({self._input_name: tensor})
        if not outputs:
            return []
        return self._decode(outputs[0], transform)

    # -- output decoding -------------------------------------------------- #

    def _decode(self, raw: np.ndarray, transform: LetterboxTransform) -> list[Detection]:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[0]  # drop the batch axis; IBVAP runs batch size 1
        if arr.ndim != 2 or arr.size == 0:
            return []

        layout = self.layout if self.layout != "auto" else self._infer_layout(arr)

        if layout in END_TO_END_LAYOUTS:
            boxes, scores, class_ids = self._decode_end_to_end(arr)
        elif layout == LAYOUT_YOLOV8:
            boxes, scores, class_ids = self._decode_v8(arr)
        else:
            boxes, scores, class_ids = self._decode_v5(arr)

        if len(boxes) == 0:
            return []

        if layout in END_TO_END_LAYOUTS:
            # The graph already suppressed duplicates. Sort by score and take
            # the top-k; a second NMS pass here would discard true positives
            # that the end-to-end head deliberately kept (two people standing
            # shoulder to shoulder, for instance).
            keep = np.argsort(scores)[::-1][: self.max_detections]
        else:
            keep = batched_nms(boxes, scores, class_ids, self.nms_threshold)
            if len(keep) > self.max_detections:
                keep = keep[: self.max_detections]

        boxes = transform.to_source_array(boxes[keep])
        scores, class_ids = scores[keep], class_ids[keep]

        min_height = self.min_height_fraction * transform.src_height
        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids, strict=True):
            raw_label = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            obj_class = ObjectClass.coerce(raw_label)
            if self.allowed is not None and obj_class not in self.allowed:
                continue
            if min_height and (y2 - y1) < min_height:
                continue
            detections.append(
                Detection(
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    obj_class=obj_class,
                    score=float(score),
                    raw_label=raw_label,
                )
            )
        return detections

    def _infer_layout(self, arr: np.ndarray) -> str:
        """Guess the output layout from tensor shape.

        Auto-detection is a convenience for ad-hoc checkpoints. Production
        artefacts should declare ``layout`` in the model registry: a guess that
        is wrong produces silently garbled boxes rather than a loud failure.
        """
        rows, cols = arr.shape
        if cols == 6 and rows <= 1000:
            # (N, 6) = x1,y1,x2,y2,score,class - an end-to-end head such as
            # YOLO26, or any export with NMS folded into the graph. The two are
            # decoded identically, so the distinction is cosmetic here.
            return LAYOUT_NMS_XYXY
        if rows < cols:
            # (4+nc, anchors) - YOLOv8/v9/v10/v11 transposed layout.
            return LAYOUT_YOLOV8
        return LAYOUT_YOLOV5

    def _decode_v8(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode ``(4 + nc, anchors)``: no objectness, class scores direct."""
        pred = arr.T  # -> (anchors, 4 + nc)
        cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(pred)), class_ids]

        mask = scores >= self.score_threshold
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
        return (
            xywh_to_xyxy(pred[mask, :4]),
            scores[mask].astype(np.float64),
            class_ids[mask].astype(np.int64),
        )

    def _decode_v5(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode ``(anchors, 5 + nc)``: objectness times class confidence."""
        if arr.shape[1] < 6:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
        objectness = arr[:, 4]
        # Cheap pre-filter before the argmax over classes: objectness alone can
        # only lower the final score, so anything below threshold here is dead.
        rough = objectness >= self.score_threshold * 0.5
        if not rough.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)

        pred = arr[rough]
        cls_scores = pred[:, 5:]
        class_ids = cls_scores.argmax(axis=1)
        scores = pred[:, 4] * cls_scores[np.arange(len(pred)), class_ids]

        mask = scores >= self.score_threshold
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64)
        return (
            xywh_to_xyxy(pred[mask, :4]),
            scores[mask].astype(np.float64),
            class_ids[mask].astype(np.int64),
        )

    def _decode_end_to_end(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode ``(N, 6)`` xyxy output from an end-to-end / pre-NMS graph.

        Boxes arrive already suppressed and in corner form, so the only work is
        thresholding on confidence.
        """
        mask = arr[:, 4] >= self.score_threshold
        pred = arr[mask]
        return (
            pred[:, :4].astype(np.float64),
            pred[:, 4].astype(np.float64),
            pred[:, 5].astype(np.int64),
        )

    def close(self) -> None:
        self.backend.close()


class MotionDetector(BaseDetector):
    """Classical fallback: MOG2 background subtraction with shape heuristics.

    **Stateful and per-camera.** The background model is learned from the frame
    history of one specific view; sharing an instance across cameras would
    corrupt it. The pipeline constructs one per camera worker.

    Accuracy is far below a CNN - it cannot distinguish a person from a swaying
    bush with confidence, and it classifies by aspect ratio alone - so every
    detection it emits is capped at a modest score to keep downstream rules
    appropriately sceptical.
    """

    mode = "motion_fallback"
    is_neural = False

    #: Ceiling on emitted confidence, signalling reduced trust downstream.
    MAX_SCORE = 0.55

    def __init__(
        self,
        *,
        history: int = 300,
        var_threshold: float = 24.0,
        detect_shadows: bool = True,
        min_area_fraction: float = 0.0008,
        max_area_fraction: float = 0.5,
        min_height_fraction: float = 0.02,
        warmup_frames: int = 30,
    ) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.min_height_fraction = min_height_fraction
        self.warmup_frames = warmup_frames
        self._frames_seen = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        self._frames_seen += 1

        mask = self._subtractor.apply(image)
        # While the background model is still converging, everything looks like
        # foreground. Emitting during that window would flood the control room
        # with alerts each time a camera reconnects.
        if self._frames_seen <= self.warmup_frames:
            return []

        # MOG2 marks shadows as 127; keep only hard foreground at 255.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(h * w)
        min_area = self.min_area_fraction * frame_area
        max_area = self.max_area_fraction * frame_area

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bh < self.min_height_fraction * h:
                continue

            obj_class = self._classify(bw, bh)
            # Fill ratio (contour area vs box area) is a crude solidity measure:
            # a compact blob is more likely a real object than scattered noise.
            fill = area / max(1.0, bw * bh)
            score = min(self.MAX_SCORE, 0.25 + 0.3 * float(fill))

            detections.append(
                Detection(
                    bbox=BBox(float(x), float(y), float(x + bw), float(y + bh)),
                    obj_class=obj_class,
                    score=score,
                    raw_label="motion",
                    attributes={"detector": "motion", "fill_ratio": round(float(fill), 3)},
                )
            )
        return detections

    @staticmethod
    def _classify(width: int, height: int) -> ObjectClass:
        """Coarse shape classification from the bounding-box aspect ratio.

        An upright human silhouette is reliably taller than wide; a vehicle is
        reliably wider than tall. Anything between is left ``UNKNOWN`` rather
        than guessed, so rules can choose whether to act on it.
        """
        aspect = width / max(1.0, float(height))
        if aspect < 0.75:
            return ObjectClass.PERSON
        if aspect > 1.5:
            return ObjectClass.CAR
        return ObjectClass.UNKNOWN

    def reset(self) -> None:
        """Discard the learned background, e.g. after a camera reconnects."""
        self._subtractor.clear()
        self._frames_seen = 0
