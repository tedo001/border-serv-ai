"""Hot-path domain types.

These objects are allocated tens of thousands of times per second per camera,
so they are ``slots``-enabled dataclasses rather than Pydantic models. Pydantic
is reserved for the API boundary (:mod:`ibvap.core.schemas`) where validation
matters more than allocation cost.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #


class ObjectClass(str, Enum):
    """Canonical object taxonomy used across the whole platform.

    Deliberately narrower than COCO: border surveillance only cares about a
    handful of classes, and collapsing the long tail keeps rule configuration
    comprehensible to an operator. ``ANIMAL`` is a first-class citizen because
    stray cattle are the single largest source of false intrusion alarms on
    rural border fences and must be suppressible by name.
    """

    PERSON = "person"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    BOAT = "boat"
    ANIMAL = "animal"
    BAG = "bag"
    UNKNOWN = "unknown"

    @property
    def category(self) -> "ObjectCategory":
        return _CLASS_TO_CATEGORY.get(self, ObjectCategory.OTHER)

    @classmethod
    def coerce(cls, value: str | "ObjectClass") -> "ObjectClass":
        """Map an arbitrary label onto the taxonomy, never raising.

        Unknown labels degrade to :attr:`UNKNOWN` so that swapping in a model
        with a different class list cannot crash a running pipeline.
        """
        if isinstance(value, cls):
            return value
        label = str(value).strip().lower()
        # Canonical taxonomy values must resolve to themselves. The alias table
        # holds only *foreign* labels, so without this a round-trip through a
        # config file ("animal" -> ObjectClass) silently produced UNKNOWN and
        # every class-based filter in the platform quietly stopped matching.
        try:
            return cls(label)
        except ValueError:
            return _LABEL_ALIASES.get(label, cls.UNKNOWN)


class ObjectCategory(str, Enum):
    """Coarse grouping used by rules that do not care about exact class."""

    HUMAN = "human"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    OBJECT = "object"
    OTHER = "other"


_CLASS_TO_CATEGORY: dict[ObjectClass, ObjectCategory] = {
    ObjectClass.PERSON: ObjectCategory.HUMAN,
    ObjectClass.BICYCLE: ObjectCategory.VEHICLE,
    ObjectClass.MOTORCYCLE: ObjectCategory.VEHICLE,
    ObjectClass.CAR: ObjectCategory.VEHICLE,
    ObjectClass.BUS: ObjectCategory.VEHICLE,
    ObjectClass.TRUCK: ObjectCategory.VEHICLE,
    ObjectClass.BOAT: ObjectCategory.VEHICLE,
    ObjectClass.ANIMAL: ObjectCategory.ANIMAL,
    ObjectClass.BAG: ObjectCategory.OBJECT,
}

# COCO-80 label aliases plus common synonyms from other public checkpoints.
_LABEL_ALIASES: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "people": ObjectClass.PERSON,
    "pedestrian": ObjectClass.PERSON,
    "human": ObjectClass.PERSON,
    "bicycle": ObjectClass.BICYCLE,
    "bike": ObjectClass.BICYCLE,
    "cycle": ObjectClass.BICYCLE,
    "motorcycle": ObjectClass.MOTORCYCLE,
    "motorbike": ObjectClass.MOTORCYCLE,
    "scooter": ObjectClass.MOTORCYCLE,
    "car": ObjectClass.CAR,
    "auto": ObjectClass.CAR,
    "van": ObjectClass.CAR,
    "jeep": ObjectClass.CAR,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
    "lorry": ObjectClass.TRUCK,
    "tractor": ObjectClass.TRUCK,
    "boat": ObjectClass.BOAT,
    "ship": ObjectClass.BOAT,
    "cow": ObjectClass.ANIMAL,
    "cattle": ObjectClass.ANIMAL,
    "buffalo": ObjectClass.ANIMAL,
    "dog": ObjectClass.ANIMAL,
    "horse": ObjectClass.ANIMAL,
    "sheep": ObjectClass.ANIMAL,
    "goat": ObjectClass.ANIMAL,
    "elephant": ObjectClass.ANIMAL,
    "bear": ObjectClass.ANIMAL,
    "bird": ObjectClass.ANIMAL,
    "cat": ObjectClass.ANIMAL,
    "backpack": ObjectClass.BAG,
    "handbag": ObjectClass.BAG,
    "suitcase": ObjectClass.BAG,
    "bag": ObjectClass.BAG,
}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates (x1,y1,x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Normalise inverted boxes rather than rejecting them: some exported
        # models emit unordered corners and a crash mid-stream is far worse
        # than silently repairing the geometry.
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre point - the ground contact point of an upright object.

        Zone membership is evaluated at the foot rather than the centroid: a
        person standing just outside a fence line has a centroid that may fall
        inside the polygon while their feet clearly do not.
        """
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 1e-6 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to centre-x, centre-y, width, height (Kalman state form)."""
        cx, cy = self.center
        return cx, cy, self.width, self.height

    @classmethod
    def from_xywh(cls, cx: float, cy: float, w: float, h: float) -> "BBox":
        hw, hh = w / 2.0, h / 2.0
        return cls(cx - hw, cy - hh, cx + hw, cy + hh)

    @classmethod
    def from_tlwh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    def iou(self, other: "BBox") -> float:
        """Intersection-over-union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw <= 0.0 or ih <= 0.0:
            return 0.0
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 1e-9 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clip(self, width: float, height: float) -> "BBox":
        """Clamp the box to a frame of the given size."""
        return BBox(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def scale(self, sx: float, sy: float | None = None) -> "BBox":
        sy = sx if sy is None else sy
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def expand(self, ratio: float) -> "BBox":
        """Grow the box outward by ``ratio`` of its size on every side."""
        dw, dh = self.width * ratio, self.height * ratio
        return BBox(self.x1 - dw, self.y1 - dh, self.x2 + dw, self.y2 + dh)

    def distance_to(self, other: "BBox") -> float:
        """Euclidean distance between box centres, in pixels."""
        (ax, ay), (bx, by) = self.center, other.center
        return math.hypot(ax - bx, ay - by)


# --------------------------------------------------------------------------- #
# Frames, detections, tracks
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Frame:
    """A decoded video frame plus the provenance needed to act on it."""

    camera_id: str
    index: int
    image: np.ndarray  # HxWx3, BGR uint8 (OpenCV convention)
    #: Wall-clock capture time (unix seconds). Used for event timestamps.
    timestamp: float = field(default_factory=time.time)
    #: Monotonic capture time. Used for all duration arithmetic, because wall
    #: clock can step backwards when NTP disciplines a drifting edge node.
    monotonic: float = field(default_factory=time.monotonic)
    #: Source frame-rate estimate at capture time, frames/second.
    fps: float = 0.0
    #: True when the frame was captured under low-light / IR conditions.
    is_night: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass(slots=True)
class Detection:
    """A single-frame detection emitted by a detector."""

    bbox: BBox
    obj_class: ObjectClass
    score: float
    #: Raw label from the underlying model, retained for debugging/evaluation.
    raw_label: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> ObjectCategory:
        return self.obj_class.category


class TrackState(str, Enum):
    """Lifecycle of a track within the multi-object tracker."""

    #: Seen once; not yet reported downstream (suppresses one-frame noise).
    TENTATIVE = "tentative"
    #: Matched on enough consecutive frames to be trusted.
    CONFIRMED = "confirmed"
    #: Recently unmatched; the Kalman filter is coasting through an occlusion.
    LOST = "lost"
    #: Unmatched beyond the retention budget; scheduled for deletion.
    REMOVED = "removed"


@dataclass(slots=True)
class Track:
    """A temporally-consistent object identity across frames."""

    track_id: int
    obj_class: ObjectClass
    bbox: BBox
    score: float
    state: TrackState = TrackState.TENTATIVE
    #: Frames since the track was created.
    age: int = 0
    #: Total number of successful detection associations.
    hits: int = 0
    #: Consecutive frames without an association.
    time_since_update: int = 0
    #: Velocity in pixels/frame of the box centre, from the Kalman filter.
    velocity: tuple[float, float] = (0.0, 0.0)
    #: Monotonic timestamp when the track was first observed.
    start_monotonic: float = 0.0
    #: Monotonic timestamp of the most recent observation.
    last_monotonic: float = 0.0
    #: Wall-clock timestamp when the track was first observed.
    start_timestamp: float = 0.0
    #: Recent foot-point trajectory, oldest first. Bounded by the tracker.
    trail: list[tuple[float, float]] = field(default_factory=list)
    #: Per-track scratch space for analytics rules (dwell timers, zone state).
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def category(self) -> ObjectCategory:
        return self.obj_class.category

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    @property
    def duration(self) -> float:
        """Seconds between first and most recent observation."""
        return max(0.0, self.last_monotonic - self.start_monotonic)

    @property
    def speed(self) -> float:
        """Instantaneous speed magnitude in pixels/frame."""
        return math.hypot(*self.velocity)

    def displacement(self) -> float:
        """Straight-line distance in pixels between trail endpoints."""
        if len(self.trail) < 2:
            return 0.0
        (x0, y0), (x1, y1) = self.trail[0], self.trail[-1]
        return math.hypot(x1 - x0, y1 - y0)

    def path_length(self) -> float:
        """Total distance travelled along the trail, in pixels."""
        if len(self.trail) < 2:
            return 0.0
        pts = np.asarray(self.trail, dtype=np.float64)
        return float(np.hypot(*(pts[1:] - pts[:-1]).T).sum())


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


class EventType(str, Enum):
    """Every alertable condition the platform can raise."""

    INTRUSION = "intrusion"                 # entered a restricted zone
    LINE_CROSSING = "line_crossing"         # crossed a virtual fence
    LOITERING = "loitering"                 # dwelled beyond a threshold
    ABANDONED_OBJECT = "abandoned_object"   # unattended bag/package
    CROWD_GATHERING = "crowd_gathering"     # density above threshold
    RAPID_MOVEMENT = "rapid_movement"       # running / erratic motion
    WRONG_DIRECTION = "wrong_direction"     # movement against permitted flow
    NIGHT_MOVEMENT = "night_movement"       # motion during curfew hours
    VEHICLE_DETECTED = "vehicle_detected"
    PERSON_DETECTED = "person_detected"
    FACE_MATCH = "face_match"               # watchlist face hit
    PLATE_READ = "plate_read"               # ANPR read
    PLATE_MATCH = "plate_match"             # watchlist plate hit
    CAMERA_TAMPER = "camera_tamper"         # lens blocked/defocused/moved
    CAMERA_OFFLINE = "camera_offline"
    CAMERA_ONLINE = "camera_online"

    @property
    def default_severity(self) -> "Severity":
        return _DEFAULT_SEVERITY.get(self, Severity.LOW)


class Severity(str, Enum):
    """Operator-facing priority. Ordered; use :meth:`rank` to compare."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

_DEFAULT_SEVERITY: dict[EventType, Severity] = {
    EventType.INTRUSION: Severity.HIGH,
    EventType.LINE_CROSSING: Severity.HIGH,
    EventType.LOITERING: Severity.MEDIUM,
    EventType.ABANDONED_OBJECT: Severity.HIGH,
    EventType.CROWD_GATHERING: Severity.MEDIUM,
    EventType.RAPID_MOVEMENT: Severity.MEDIUM,
    EventType.WRONG_DIRECTION: Severity.MEDIUM,
    EventType.NIGHT_MOVEMENT: Severity.HIGH,
    EventType.VEHICLE_DETECTED: Severity.INFO,
    EventType.PERSON_DETECTED: Severity.INFO,
    EventType.FACE_MATCH: Severity.CRITICAL,
    EventType.PLATE_READ: Severity.INFO,
    EventType.PLATE_MATCH: Severity.CRITICAL,
    EventType.CAMERA_TAMPER: Severity.HIGH,
    EventType.CAMERA_OFFLINE: Severity.MEDIUM,
    EventType.CAMERA_ONLINE: Severity.INFO,
}


@dataclass(slots=True)
class Event:
    """An actionable incident produced by the analytics layer."""

    camera_id: str
    event_type: EventType
    severity: Severity
    #: Wall-clock time of the triggering frame.
    timestamp: float
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Analytics confidence in [0, 1] - distinct from detector score.
    confidence: float = 1.0
    #: Human-readable one-line summary for the operator console.
    message: str = ""
    #: Rule instance that fired, for traceability back to configuration.
    rule_id: str = ""
    #: Zone or tripwire involved, when applicable.
    zone_id: str | None = None
    #: Track identities involved in the incident.
    track_ids: list[int] = field(default_factory=list)
    #: Bounding boxes to render on the evidence snapshot.
    boxes: list[BBox] = field(default_factory=list)
    #: Structured payload (plate text, match score, dwell seconds, ...).
    attributes: dict[str, Any] = field(default_factory=dict)
    #: Evidence artefact references populated by the evidence store.
    snapshot_path: str | None = None
    clip_path: str | None = None
    #: SHA-256 of the snapshot, for chain-of-custody verification.
    evidence_hash: str | None = None
    #: Frame index that triggered the event, for seeking in recordings.
    frame_index: int = 0

    def dedup_key(self) -> str:
        """Identity used to collapse repeats of the same ongoing incident.

        Two events collapse when the same rule fires for the same tracks on the
        same camera. Different tracks entering the same zone therefore remain
        distinct alerts, which is what an operator expects.
        """
        tracks = ",".join(str(t) for t in sorted(self.track_ids))
        return f"{self.camera_id}|{self.event_type.value}|{self.rule_id}|{self.zone_id or '-'}|{tracks}"
