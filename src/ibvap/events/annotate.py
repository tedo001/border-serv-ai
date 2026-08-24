"""Frame annotation for evidence snapshots and live preview.

The same renderer serves both, so what an operator sees on the wall is exactly
what the stored evidence shows. Divergence between the two is a credibility
problem the first time a snapshot is questioned.
"""

from __future__ import annotations

import cv2
import numpy as np

from ibvap.core.geometry import Tripwire, Zone
from ibvap.core.types import BBox, Event, ObjectCategory, Severity, Track
from ibvap.core.timeutils import to_iso

#: BGR palette. Severity drives colour so a wall of thumbnails is triageable at
#: a glance without reading any text.
SEVERITY_COLOURS: dict[Severity, tuple[int, int, int]] = {
    Severity.INFO: (180, 180, 180),
    Severity.LOW: (80, 200, 80),
    Severity.MEDIUM: (40, 200, 240),
    Severity.HIGH: (40, 120, 255),
    Severity.CRITICAL: (60, 60, 255),
}

CATEGORY_COLOURS: dict[ObjectCategory, tuple[int, int, int]] = {
    ObjectCategory.HUMAN: (80, 220, 80),
    ObjectCategory.VEHICLE: (240, 180, 60),
    ObjectCategory.ANIMAL: (160, 160, 160),
    ObjectCategory.OBJECT: (200, 120, 240),
    ObjectCategory.OTHER: (180, 180, 180),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(
    image: np.ndarray,
    box: BBox,
    label: str = "",
    colour: tuple[int, int, int] = (80, 220, 80),
    *,
    thickness: int = 2,
) -> None:
    """Draw a labelled bounding box in place."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.clip(w, h).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
    if not label:
        return

    scale, text_thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(label, _FONT, scale, text_thickness)
    # Flip the label inside the frame when the box is against the top edge,
    # otherwise the most important text is the part that gets cropped away.
    top = y1 - th - baseline - 2
    if top < 0:
        top = min(h - th - baseline - 2, y2 + 2)
    cv2.rectangle(image, (x1, top), (x1 + tw + 6, top + th + baseline + 2), colour, -1)
    cv2.putText(
        image, label, (x1 + 3, top + th + 1), _FONT, scale, (20, 20, 20), text_thickness,
        cv2.LINE_AA,
    )


def draw_zone(
    image: np.ndarray,
    zone: Zone,
    *,
    colour: tuple[int, int, int] = (60, 60, 220),
    alpha: float = 0.18,
    label: bool = True,
) -> None:
    """Draw a translucent zone polygon with its outline."""
    h, w = image.shape[:2]
    points = zone.to_pixels(w, h).astype(np.int32)
    if len(points) < 3:
        return

    # Fill on an overlay then blend: filling directly would hide the very scene
    # the operator needs to judge the alert.
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], colour)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.polylines(image, [points], True, colour, 2, cv2.LINE_AA)

    if label and zone.name:
        cx, cy = points.mean(axis=0).astype(int)
        cv2.putText(image, zone.name, (cx - 40, cy), _FONT, 0.55, colour, 2, cv2.LINE_AA)


def draw_tripwire(
    image: np.ndarray,
    wire: Tripwire,
    *,
    colour: tuple[int, int, int] = (40, 200, 240),
) -> None:
    """Draw a tripwire with an arrow showing its alerting direction."""
    h, w = image.shape[:2]
    (ax, ay), (bx, by) = wire.to_pixels(w, h)
    a, b = (int(ax), int(ay)), (int(bx), int(by))
    cv2.line(image, a, b, colour, 3, cv2.LINE_AA)
    for point in (a, b):
        cv2.circle(image, point, 5, colour, -1)

    # An arrow along the wire normal makes the configured direction legible;
    # a bare line leaves the operator guessing which way triggers an alert.
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max(1.0, float(np.hypot(dx, dy)))
    nx, ny = -dy / length, dx / length  # left-hand normal
    arrow = 28
    if wire.direction.value == "left":
        tip = (int(mid[0] + nx * arrow), int(mid[1] + ny * arrow))
        cv2.arrowedLine(image, mid, tip, colour, 2, cv2.LINE_AA, tipLength=0.4)
    elif wire.direction.value == "right":
        tip = (int(mid[0] - nx * arrow), int(mid[1] - ny * arrow))
        cv2.arrowedLine(image, mid, tip, colour, 2, cv2.LINE_AA, tipLength=0.4)
    else:
        cv2.arrowedLine(image, mid, (int(mid[0] + nx * arrow), int(mid[1] + ny * arrow)),
                        colour, 2, cv2.LINE_AA, tipLength=0.4)
        cv2.arrowedLine(image, mid, (int(mid[0] - nx * arrow), int(mid[1] - ny * arrow)),
                        colour, 2, cv2.LINE_AA, tipLength=0.4)

    if wire.name:
        cv2.putText(image, wire.name, (a[0] + 6, a[1] + 18), _FONT, 0.5, colour, 1, cv2.LINE_AA)


def draw_track(image: np.ndarray, track: Track, *, show_trail: bool = True) -> None:
    """Draw one track with its class, id and motion trail."""
    colour = CATEGORY_COLOURS.get(track.category, (180, 180, 180))
    label = f"#{track.track_id} {track.obj_class.value} {track.score:.2f}"
    if plate := track.attributes.get("plate"):
        label += f" [{plate}]"
    if match := track.attributes.get("face_match"):
        label += f" <{match}>"
    draw_box(image, track.bbox, label, colour)

    if show_trail and len(track.trail) > 1:
        points = np.asarray(track.trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, colour, 2, cv2.LINE_AA)


def annotate_frame(
    image: np.ndarray,
    *,
    tracks: list[Track] | None = None,
    zones: list[Zone] | None = None,
    tripwires: list[Tripwire] | None = None,
    event: Event | None = None,
    camera_name: str = "",
    timestamp: float | None = None,
    show_trail: bool = True,
) -> np.ndarray:
    """Render a fully annotated copy of ``image``."""
    canvas = image.copy()

    for zone in zones or []:
        draw_zone(canvas, zone)
    for wire in tripwires or []:
        draw_tripwire(canvas, wire)
    for track in tracks or []:
        draw_track(canvas, track, show_trail=show_trail)

    if event is not None:
        _draw_event_banner(canvas, event)
        colour = SEVERITY_COLOURS.get(event.severity, (60, 60, 255))
        for box in event.boxes:
            draw_box(canvas, box, "", colour, thickness=3)

    _draw_status_bar(canvas, camera_name, timestamp)
    return canvas


def _draw_event_banner(image: np.ndarray, event: Event) -> None:
    """Draw a severity-coloured banner describing the event."""
    h, w = image.shape[:2]
    colour = SEVERITY_COLOURS.get(event.severity, (60, 60, 255))
    cv2.rectangle(image, (0, 0), (w, 34), colour, -1)
    text = f"{event.severity.value.upper()}  {event.event_type.value}  -  {event.message}"
    cv2.putText(image, text[:110], (10, 23), _FONT, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
    # A coloured border makes severity readable in a grid of thumbnails where
    # the banner text is far too small to resolve.
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), colour, 4)


def _draw_status_bar(image: np.ndarray, camera_name: str, timestamp: float | None) -> None:
    """Draw the camera name and capture time along the bottom edge."""
    if not camera_name and timestamp is None:
        return
    h, w = image.shape[:2]
    strip = image[h - 26 : h, 0:w]
    # Darken rather than fill, so detail behind the strip stays visible.
    image[h - 26 : h, 0:w] = (strip * 0.35).astype(image.dtype)

    if camera_name:
        cv2.putText(image, camera_name, (8, h - 8), _FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    if timestamp is not None:
        stamp = to_iso(timestamp)
        (tw, _), _ = cv2.getTextSize(stamp, _FONT, 0.5, 1)
        cv2.putText(image, stamp, (w - tw - 8, h - 8), _FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
