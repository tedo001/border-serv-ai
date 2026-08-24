"""Geometry primitives for virtual fences, zones and tripwires.

Design note - **normalised coordinates**. Every operator-drawn region is stored
in normalised ``[0, 1]`` space rather than pixels. Border deployments routinely
re-profile a camera (1080p main stream by day, 720p sub-stream when the VSAT
link degrades at night); had we stored pixels, every zone on that camera would
silently shift. Normalised regions survive resolution changes untouched and are
projected to pixels only at evaluation time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Point = tuple[float, float]

# --------------------------------------------------------------------------- #
# Polygon primitives
# --------------------------------------------------------------------------- #


def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Return True if ``(x, y)`` lies inside ``polygon`` (ray casting).

    ``polygon`` is an ``(N, 2)`` array of vertices in order. Points exactly on
    an edge may fall either way; that ambiguity is immaterial at pixel scale
    and is resolved consistently by the half-open comparison below.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Half-open test on y avoids double-counting shared vertices.
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised containment test for many points at once.

    ``points`` is ``(M, 2)``; returns a boolean array of length ``M``. Used on
    the hot path where a frame may hold dozens of tracks and a camera several
    zones - the Python-loop version costs milliseconds we cannot spare.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)

    x, y = points[:, 0], points[:, 1]
    xi, yi = polygon[:, 0], polygon[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)

    # Broadcast every point against every edge: (M, N).
    straddles = (yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])
    dy = yj - yi
    dy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)  # guard horizontal edges
    x_cross = (xj - xi)[None, :] * (y[:, None] - yi[None, :]) / dy[None, :] + xi[None, :]
    crossings = straddles & (x[:, None] < x_cross)
    return (crossings.sum(axis=1) % 2).astype(bool)


def polygon_area(polygon: np.ndarray) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(polygon: np.ndarray) -> Point:
    """Area-weighted centroid, falling back to the vertex mean for degenerates."""
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return (float(polygon[:, 0].mean()), float(polygon[:, 1].mean())) if len(polygon) else (0.0, 0.0)
    x, y = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = float(((x + xn) * cross).sum() / (6.0 * area))
    cy = float(((y + yn) * cross).sum() / (6.0 * area))
    return cx, cy


# --------------------------------------------------------------------------- #
# Line primitives
# --------------------------------------------------------------------------- #


class CrossingDirection(str, Enum):
    """Which way an object traversed a tripwire.

    Directions are expressed relative to the wire's own orientation ``A -> B``:
    ``LEFT`` means the object moved from the right-hand side of ``A -> B`` to
    the left-hand side. Deployments give these neutral names domain meaning in
    rule configuration (e.g. left = ``infiltration``, right = ``exfiltration``).
    """

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    ANY = "any"


def side_of_line(a: Point, b: Point, p: Point) -> float:
    """Signed cross product of ``AB`` and ``AP``.

    Positive when ``p`` lies to the left of the directed line ``A -> B``,
    negative to the right, zero when collinear.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True when segment ``p1p2`` properly intersects segment ``q1q2``.

    This bounded test is what makes a tripwire a *segment* rather than an
    infinite line. Relying on a side-flip alone would fire whenever an object
    crossed the wire's extension - e.g. a vehicle on a road 200 m past the end
    of a fence-line wire - which is the classic false-alarm bug in naive
    line-crossing implementations.
    """
    d1 = side_of_line(q1, q2, p1)
    d2 = side_of_line(q1, q2, p2)
    d3 = side_of_line(p1, p2, q1)
    d4 = side_of_line(p1, p2, q2)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True

    # Collinear touching cases: treat an endpoint landing on the other segment
    # as an intersection so a track that stops exactly on the wire still fires.
    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            abs(side_of_line(a, b, c)) < 1e-9
            and min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    return (
        on_segment(q1, q2, p1)
        or on_segment(q1, q2, p2)
        or on_segment(p1, p2, q1)
        or on_segment(p1, p2, q2)
    )


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point ``p`` to segment ``ab``."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# --------------------------------------------------------------------------- #
# Region objects
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Zone:
    """A named polygonal region of interest in normalised coordinates."""

    zone_id: str
    name: str
    #: Vertices as ``[(x, y), ...]`` with each component in ``[0, 1]``.
    points: list[Point]
    #: Free-form kind used by rules: ``restricted``, ``buffer``, ``mask``, ...
    kind: str = "restricted"
    enabled: bool = True

    _norm: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        if len(arr) < 3:
            raise ValueError(f"zone {self.zone_id!r} needs at least 3 points, got {len(arr)}")
        self._norm = arr

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        """Project the normalised polygon onto a frame of the given size."""
        return self._norm * np.array([width, height], dtype=np.float64)

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        """Test a *pixel* point against this zone."""
        return point_in_polygon(x, y, self.to_pixels(width, height))

    def contains_many(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        return points_in_polygon(points, self.to_pixels(width, height))

    def area_fraction(self) -> float:
        """Fraction of the frame covered by this zone, in ``[0, 1]``."""
        return polygon_area(self._norm)

    def centroid_pixels(self, width: int, height: int) -> Point:
        return polygon_centroid(self.to_pixels(width, height))


@dataclass(slots=True)
class Tripwire:
    """A directed virtual fence line in normalised coordinates."""

    wire_id: str
    name: str
    #: Start point ``A`` in normalised coordinates.
    start: Point
    #: End point ``B`` in normalised coordinates.
    end: Point
    #: Which traversal direction raises an alert.
    direction: CrossingDirection = CrossingDirection.ANY
    #: Operator-facing labels for the two traversal senses.
    left_label: str = "left"
    right_label: str = "right"
    enabled: bool = True

    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        a = (self.start[0] * width, self.start[1] * height)
        b = (self.end[0] * width, self.end[1] * height)
        return a, b

    def crossing(
        self, prev_point: Point, curr_point: Point, width: int, height: int
    ) -> CrossingDirection:
        """Classify the movement ``prev -> curr`` against this wire.

        Returns :attr:`CrossingDirection.NONE` unless the motion segment truly
        intersects the wire segment, so extensions of the line never fire.
        """
        a, b = self.to_pixels(width, height)
        if not segments_intersect(prev_point, curr_point, a, b):
            return CrossingDirection.NONE

        before = side_of_line(a, b, prev_point)
        after = side_of_line(a, b, curr_point)
        if before < 0 <= after:
            return CrossingDirection.LEFT
        if before > 0 >= after:
            return CrossingDirection.RIGHT
        return CrossingDirection.NONE

    def matches_direction(self, crossed: CrossingDirection) -> bool:
        """Whether an observed crossing satisfies this wire's alert direction."""
        if crossed is CrossingDirection.NONE:
            return False
        if self.direction in (CrossingDirection.ANY, CrossingDirection.NONE):
            return True
        return self.direction is crossed

    def label_for(self, crossed: CrossingDirection) -> str:
        if crossed is CrossingDirection.LEFT:
            return self.left_label
        if crossed is CrossingDirection.RIGHT:
            return self.right_label
        return "none"

    def length_pixels(self, width: int, height: int) -> float:
        a, b = self.to_pixels(width, height)
        return math.hypot(b[0] - a[0], b[1] - a[1])
