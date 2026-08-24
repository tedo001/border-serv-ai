"""Rule abstractions and the rule registry.

A *rule* converts tracked objects into :class:`~ibvap.core.types.Event` objects.
Rules are stateless with respect to each other and keep any per-track memory in
``Track.attributes`` under a key namespaced by rule id, so two instances of the
same rule type on one camera (say, two different restricted zones) never
interfere.

Units are normalised on purpose. Thresholds are expressed in *fractions of
frame height per second* rather than pixels per frame, so one rule template can
be pushed to a 4 MP camera at a check post and a 720p camera on a tower mast
without retuning. A raw pixel threshold silently means something different on
every camera in the deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable

from ibvap.core.config import RuleConfig
from ibvap.core.errors import AnalyticsError
from ibvap.core.geometry import Tripwire, Zone
from ibvap.core.logging import get_logger
from ibvap.core.timeutils import TimeWindow, in_any_window, parse_windows
from ibvap.core.types import Event, EventType, Frame, ObjectClass, Severity, Track

log = get_logger(__name__)


@dataclass(slots=True)
class RuleContext:
    """Everything a rule may look at for one frame."""

    camera_id: str
    frame: Frame
    #: Confirmed tracks only - tentative ones are too unreliable to alert on.
    tracks: list[Track]
    zones: dict[str, Zone] = field(default_factory=dict)
    tripwires: dict[str, Tripwire] = field(default_factory=dict)
    #: Effective analytics frame rate, used to convert per-frame to per-second.
    fps: float = 8.0
    #: Free-form per-camera scratch space that persists across frames.
    camera_state: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.frame.width

    @property
    def height(self) -> int:
        return self.frame.height

    @property
    def now(self) -> float:
        """Monotonic time of this frame - safe for duration arithmetic."""
        return self.frame.monotonic

    @property
    def timestamp(self) -> float:
        """Wall-clock time of this frame - used for event timestamps."""
        return self.frame.timestamp

    def speed_fraction_per_second(self, track: Track) -> float:
        """Track speed as a fraction of frame height per second.

        Resolution- and frame-rate-independent, so a threshold of ``0.35``
        means the same physical briskness on every camera in the deployment.
        """
        if self.height <= 0:
            return 0.0
        return (track.speed * max(1.0, self.fps)) / float(self.height)


class Rule(ABC):
    """Base class for every analytics rule."""

    #: Registry key used in configuration (``type:`` field).
    type_name: ClassVar[str] = ""
    #: Event type this rule raises.
    event_type: ClassVar[EventType] = EventType.INTRUSION
    #: Whether the rule needs tracks at all (frame-level rules do not).
    requires_tracks: ClassVar[bool] = True

    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self.rule_id = config.id
        self.enabled = config.enabled
        self.cooldown_seconds = config.cooldown_seconds
        self._windows: list[TimeWindow] = []
        if config.active_hours:
            try:
                self._windows = parse_windows(config.active_hours)
            except ValueError as exc:
                raise AnalyticsError(f"rule {config.id!r}: {exc}") from exc

        # Resolve the class filter once into taxonomy members.
        self._classes: set[ObjectClass] | None = (
            {ObjectClass.coerce(c) for c in config.classes} if config.classes else None
        )
        self._severity: Severity | None = (
            Severity(config.severity) if config.severity else None
        )

    # -- configuration helpers -------------------------------------------- #

    def param(self, key: str, default: Any = None) -> Any:
        """Read a rule-specific tuning parameter."""
        return self.config.params.get(key, default)

    def float_param(self, key: str, default: float) -> float:
        try:
            return float(self.param(key, default))
        except (TypeError, ValueError) as exc:
            raise AnalyticsError(f"rule {self.rule_id!r}: {key!r} must be a number") from exc

    def int_param(self, key: str, default: int) -> int:
        try:
            return int(self.param(key, default))
        except (TypeError, ValueError) as exc:
            raise AnalyticsError(f"rule {self.rule_id!r}: {key!r} must be an integer") from exc

    @property
    def severity(self) -> Severity:
        return self._severity or self.event_type.default_severity

    def is_active(self, ctx: RuleContext) -> bool:
        """Whether the rule should run for this frame (schedule check)."""
        if not self.enabled:
            return False
        from datetime import datetime

        return in_any_window(self._windows, datetime.fromtimestamp(ctx.timestamp))

    def applies_to(self, track: Track) -> bool:
        """Whether this rule cares about the given track's class."""
        if self._classes is None:
            return True
        return track.obj_class in self._classes

    def selected_zones(self, ctx: RuleContext) -> list[Zone]:
        """Zones explicitly configured on this rule.

        An empty result means **no spatial restriction** - the rule applies to
        the whole frame. This is the opposite of falling back to "every zone on
        the camera", and the distinction matters: a ``rapid_movement`` rule with
        no zone configured is meant to watch the entire view, and quietly
        confining it to whichever polygons happen to exist on that camera would
        blind it everywhere else, with nothing in the configuration to explain
        why. Rules that are meaningless without a zone (intrusion) opt into the
        fallback explicitly via :meth:`all_zones`.
        """
        return [z for zid in self.config.zones if (z := ctx.zones.get(zid)) and z.enabled]

    def all_zones(self, ctx: RuleContext) -> list[Zone]:
        """Every enabled zone on the camera."""
        return [z for z in ctx.zones.values() if z.enabled]

    def zones_for_containment(self, ctx: RuleContext) -> list[Zone]:
        """Zones for a rule that requires one: configured, else all on camera."""
        return self.selected_zones(ctx) or self.all_zones(ctx)

    def within_zones(self, ctx: RuleContext, track: Track, zones: list[Zone]) -> bool:
        """Whether ``track`` satisfies the spatial filter given by ``zones``.

        An empty ``zones`` list passes everything, per the semantics above.
        """
        if not zones:
            return True
        foot = track.bbox.foot
        return any(z.contains(*foot, ctx.width, ctx.height) for z in zones)

    def selected_tripwires(self, ctx: RuleContext) -> list[Tripwire]:
        """Tripwires configured on this rule, else every enabled one."""
        if self.config.tripwires:
            return [w for wid in self.config.tripwires if (w := ctx.tripwires.get(wid)) and w.enabled]
        return [w for w in ctx.tripwires.values() if w.enabled]

    def track_state(self, track: Track) -> dict[str, Any]:
        """Per-track, per-rule scratch space that survives across frames."""
        key = f"_rule_{self.rule_id}"
        state = track.attributes.get(key)
        if state is None:
            state = {}
            track.attributes[key] = state
        return state

    # -- event construction ------------------------------------------------ #

    def make_event(
        self,
        ctx: RuleContext,
        *,
        message: str,
        tracks: Iterable[Track] = (),
        zone_id: str | None = None,
        confidence: float = 1.0,
        attributes: dict[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> Event:
        """Build an :class:`Event` with this rule's identity attached."""
        track_list = list(tracks)
        return Event(
            camera_id=ctx.camera_id,
            event_type=self.event_type,
            severity=severity or self.severity,
            timestamp=ctx.timestamp,
            confidence=round(float(confidence), 4),
            message=message,
            rule_id=self.rule_id,
            zone_id=zone_id,
            track_ids=[t.track_id for t in track_list],
            boxes=[t.bbox for t in track_list],
            attributes=attributes or {},
            frame_index=ctx.frame.index,
        )

    @abstractmethod
    def evaluate(self, ctx: RuleContext) -> list[Event]:
        """Return events raised by this rule for the current frame."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.rule_id!r} enabled={self.enabled}>"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_RULE_REGISTRY: dict[str, type[Rule]] = {}


def register_rule(cls: type[Rule]) -> type[Rule]:
    """Class decorator registering a rule under its ``type_name``."""
    if not cls.type_name:
        raise AnalyticsError(f"{cls.__name__} must define a type_name")
    if cls.type_name in _RULE_REGISTRY:
        raise AnalyticsError(f"duplicate rule type {cls.type_name!r}")
    _RULE_REGISTRY[cls.type_name] = cls
    return cls


def build_rule(config: RuleConfig) -> Rule:
    """Instantiate the rule implementation named by ``config.type``."""
    cls = _RULE_REGISTRY.get(config.type)
    if cls is None:
        raise AnalyticsError(
            f"unknown rule type {config.type!r}; available: {sorted(_RULE_REGISTRY)}"
        )
    return cls(config)


def available_rules() -> dict[str, type[Rule]]:
    """All registered rule types, for API discovery and documentation."""
    return dict(_RULE_REGISTRY)


RuleFactory = Callable[[RuleConfig], Rule]
