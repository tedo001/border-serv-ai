"""Supervision as the annotation and interchange layer.

`supervision <https://supervision.roboflow.com/>`_ is used here for the two
things it is unambiguously the best tool for: a canonical detection container
that every model ecosystem already speaks, and a set of annotators that render
a frame the way a modern CV demo looks rather than the way ``cv2.rectangle``
looks.

What it is deliberately *not* used for is tracking. Supervision's ``ByteTrack``
has been deprecated since 0.28 and is scheduled for removal, and the tracker in
:mod:`ibvap.vision.tracker` carries a third, distance-gated association pass
added specifically because figures moving faster than about 35 px/frame - which
is what a vehicle on a border road looks like at 8 fps - were fragmenting into
new identities every few frames. Swapping that out for a deprecated
implementation would be a step backwards, so identities stay ours and
supervision is handed the result.

The module imports cleanly without supervision installed; every entry point
degrades to the platform's own renderer, which is what a node without the extra
uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ibvap.core.geometry import Tripwire, Zone
from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Detection, ObjectCategory, Track

log = get_logger(__name__)

try:  # pragma: no cover - the import itself is the feature test
    import supervision as sv

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on a node without the extra
    sv = None  # type: ignore[assignment]
    AVAILABLE = False


#: RGB, matching the consoles so the wall, the analyst view and stored evidence
#: all colour an object the same way. Supervision works in RGB; OpenCV in BGR.
CATEGORY_RGB: dict[ObjectCategory, tuple[int, int, int]] = {
    ObjectCategory.HUMAN: (63, 185, 80),
    ObjectCategory.VEHICLE: (224, 138, 30),
    ObjectCategory.ANIMAL: (139, 148, 158),
    ObjectCategory.OBJECT: (188, 120, 240),
    ObjectCategory.OTHER: (139, 148, 158),
}

#: Stable ordering, so a class always maps to the same palette slot.
_CATEGORY_ORDER = list(CATEGORY_RGB)


def category_index(category: ObjectCategory) -> int:
    return _CATEGORY_ORDER.index(category) if category in CATEGORY_RGB else len(_CATEGORY_ORDER) - 1


# --------------------------------------------------------------------------- #
# Interchange
# --------------------------------------------------------------------------- #

def detections_to_sv(detections: list[Detection]) -> Any:
    """Convert platform detections into a ``sv.Detections``.

    Useful on its own: anything in the supervision ecosystem - dataset export,
    metrics, filters, slicers - accepts this type, so a model evaluated here can
    be scored with tooling this repository does not have to own.
    """
    _require()
    if not detections:
        return sv.Detections.empty()
    xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
                    dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([d.score for d in detections], dtype=np.float32),
        class_id=np.array([category_index(d.category) for d in detections], dtype=int),
        data={"label": np.array([d.obj_class.value for d in detections])},
    )


def tracks_to_sv(tracks: list[Track]) -> Any:
    """Convert live tracks into a ``sv.Detections`` carrying tracker ids.

    Tracker ids are what let supervision colour and trace by identity rather
    than by class, which is the difference between "three green boxes" and
    "person 4, person 7, person 9".
    """
    _require()
    if not tracks:
        return sv.Detections.empty()
    xyxy = np.array([[t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2] for t in tracks],
                    dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([t.score for t in tracks], dtype=np.float32),
        class_id=np.array([category_index(t.category) for t in tracks], dtype=int),
        tracker_id=np.array([t.track_id for t in tracks], dtype=int),
        data={"label": np.array([t.obj_class.value for t in tracks])},
    )


def _require() -> None:
    if not AVAILABLE:  # pragma: no cover - exercised on a node without the extra
        raise RuntimeError(
            "supervision is not installed; install the 'sv' extra: pip install -e '.[sv]'"
        )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

class SupervisionRenderer:
    """Draws tracks, zones and tripwires onto a frame.

    Holds annotator instances rather than rebuilding them per frame: the trace
    annotator in particular keeps its own history keyed by tracker id, and
    recreating it every frame would silently disable the trails it exists to
    draw.
    """

    def __init__(
        self,
        *,
        trace_length: int = 40,
        thickness: int = 2,
        text_scale: float = 0.5,
        show_traces: bool = True,
    ) -> None:
        _require()
        palette = sv.ColorPalette([sv.Color(*rgb) for rgb in CATEGORY_RGB.values()])
        lookup = sv.ColorLookup.CLASS

        self.box = sv.BoxCornerAnnotator(
            color=palette, thickness=thickness + 2, corner_length=18, color_lookup=lookup
        )
        self.label = sv.LabelAnnotator(
            color=palette, color_lookup=lookup, text_scale=text_scale,
            text_thickness=1, text_padding=6, border_radius=3, smart_position=True,
        )
        self.trace = sv.TraceAnnotator(
            color=palette, color_lookup=lookup, trace_length=trace_length,
            thickness=thickness, position=sv.Position.BOTTOM_CENTER,
        ) if show_traces else None

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        *,
        zones: list[Zone] | None = None,
        tripwires: list[Tripwire] | None = None,
        plates: dict[int, tuple[BBox, str]] | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``frame`` (BGR in, BGR out).

        ``plates`` maps a track id to its current plate box and text, so a
        vehicle carries its read on screen while the vote is still being
        gathered rather than only once the event fires.
        """
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        for zone in zones or []:
            if zone.enabled:
                _draw_polygon(canvas, zone.to_pixels(width, height))
        for wire in tripwires or []:
            if wire.enabled:
                _draw_line(canvas, wire, width, height)
        for box, text in (plates or {}).values():
            _draw_plate(canvas, box, text)

        if not tracks:
            return canvas

        detections = tracks_to_sv(tracks)
        labels = []
        for track in tracks:
            label = f"#{track.track_id} {track.obj_class.value} {track.score:.2f}"
            plate = track.attributes.get("plate") or track.attributes.get(
                "plate_candidate"
            )
            if plate:
                label = f"{label}  [{plate}]"
            labels.append(label)
        if self.trace is not None:
            canvas = self.trace.annotate(canvas, detections)
        canvas = self.box.annotate(canvas, detections)
        return self.label.annotate(canvas, detections, labels=labels)


def _draw_polygon(canvas: np.ndarray, points: list[tuple[float, float]]) -> None:
    import cv2

    if len(points) < 3:
        return
    array = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
    # Tinted fill under a solid edge: the zone has to read as a region without
    # hiding what is moving inside it.
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [array], (255, 190, 90))
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
    cv2.polylines(canvas, [array], True, (255, 190, 90), 2, cv2.LINE_AA)


def _draw_plate(canvas: np.ndarray, box: BBox, text: str) -> None:
    """Draw the plate box and its current read.

    Deliberately a different colour and weight from the object boxes: the plate
    is a region inside a vehicle box, and drawing them alike makes a crowded
    gate unreadable.
    """
    import cv2

    x1, y1, x2, y2 = box.clip(canvas.shape[1], canvas.shape[0]).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 255), 2, cv2.LINE_AA)
    if not text:
        return
    scale, thickness = 0.55, 2
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    # Flip the caption below the box when it would be cropped off the top.
    top = y1 - th - baseline - 4
    origin_y = top if top >= 0 else y2 + 2
    cv2.rectangle(
        canvas, (x1, origin_y), (x1 + tw + 8, origin_y + th + baseline + 4),
        (60, 220, 255), -1,
    )
    cv2.putText(
        canvas, text, (x1 + 4, origin_y + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA,
    )


def _draw_line(canvas: np.ndarray, wire: Tripwire, width: int, height: int) -> None:
    import cv2

    x1, y1 = int(wire.start[0] * width), int(wire.start[1] * height)
    x2, y2 = int(wire.end[0] * width), int(wire.end[1] * height)
    cv2.line(canvas, (x1, y1), (x2, y2), (60, 60, 240), 3, cv2.LINE_AA)
    cv2.circle(canvas, (x1, y1), 5, (60, 60, 240), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x2, y2), 5, (60, 60, 240), -1, cv2.LINE_AA)
    label = wire.name or wire.wire_id
    if label:
        cv2.putText(canvas, label, (x1 + 8, max(y1, 18) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 240), 1, cv2.LINE_AA)
