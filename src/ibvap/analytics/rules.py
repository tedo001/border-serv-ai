"""Concrete analytics rules.

Between them these cover the capability list in the platform brief: intrusion
and virtual fencing, loitering and abandoned objects, crowd formation, rapid
and wrong-direction movement, night-time activity, and camera tampering.

A recurring theme is **hysteresis and confirmation**. Almost every naive
implementation of these rules fires on a single frame, and a single frame is
exactly what a gust of wind through a bush, a moth crossing an IR illuminator,
or one bad detection looks like. Each rule below requires evidence to persist
before it will alert, because a control room that learns to ignore the alert
tone is worse than no analytics at all.
"""

from __future__ import annotations

import math
from typing import Any

from ibvap.analytics.base import Rule, RuleContext, register_rule
from ibvap.core.logging import get_logger
from ibvap.core.timeutils import humanise_duration
from ibvap.core.types import Event, EventType, ObjectCategory, ObjectClass, Severity, Track

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Perimeter rules
# --------------------------------------------------------------------------- #


@register_rule
class IntrusionRule(Rule):
    """Alerts when a tracked object is present inside a restricted zone.

    Membership is tested at the object's **foot point**, not its centroid: a
    person standing immediately outside a fence has a centroid that can fall
    inside the polygon while their feet plainly do not, and on a downward-angled
    border camera that error is worth metres on the ground.
    """

    type_name = "intrusion"
    event_type = EventType.INTRUSION

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        # Require the object to be inside for several consecutive frames.
        # Detector jitter routinely puts a single box a few pixels over a
        # boundary; without confirmation every zone edge becomes an alarm bell.
        confirm_frames = self.int_param("confirm_frames", 3)
        min_dwell = self.float_param("min_dwell_seconds", 0.0)

        events: list[Event] = []
        for zone in self.zones_for_containment(ctx):
            for track in ctx.tracks:
                if not self.applies_to(track):
                    continue
                state = self.track_state(track)
                key = f"zone_{zone.zone_id}"
                inside = zone.contains(*track.bbox.foot, ctx.width, ctx.height)

                entry = state.setdefault(key, {"frames": 0, "since": 0.0, "fired": False})
                if not inside:
                    # Leaving resets the confirmation counter *and* the fired
                    # latch, so the same person re-entering later alerts again.
                    entry.update(frames=0, since=0.0, fired=False)
                    continue

                if entry["frames"] == 0:
                    entry["since"] = ctx.now
                entry["frames"] += 1

                dwell = ctx.now - entry["since"]
                if entry["fired"] or entry["frames"] < confirm_frames or dwell < min_dwell:
                    continue

                entry["fired"] = True
                events.append(
                    self.make_event(
                        ctx,
                        message=(
                            f"{track.obj_class.value} entered restricted zone "
                            f"'{zone.name}'"
                        ),
                        tracks=[track],
                        zone_id=zone.zone_id,
                        confidence=track.score,
                        attributes={
                            "zone_name": zone.name,
                            "object_class": track.obj_class.value,
                            "dwell_seconds": round(dwell, 2),
                            "track_duration": round(track.duration, 2),
                        },
                    )
                )
        return events


@register_rule
class LineCrossingRule(Rule):
    """Virtual fence: alerts when a track crosses a tripwire in a watched sense.

    Crossing is detected from the track's own motion segment against the wire
    segment, so an object moving along the *extension* of the wire never fires -
    the failure mode that makes naive line-crossing unusable in the field.
    """

    type_name = "line_crossing"
    event_type = EventType.LINE_CROSSING

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        events: list[Event] = []
        for wire in self.selected_tripwires(ctx):
            for track in ctx.tracks:
                if not self.applies_to(track) or len(track.trail) < 2:
                    continue

                previous, current = track.trail[-2], track.trail[-1]
                crossed = wire.crossing(previous, current, ctx.width, ctx.height)
                if not wire.matches_direction(crossed):
                    continue

                state = self.track_state(track)
                key = f"wire_{wire.wire_id}"
                # One alert per track per wire per traversal sense. A track
                # oscillating on the line (common when someone works right at a
                # fence) must not generate an alert per frame.
                if state.get(key) == crossed.value:
                    continue
                state[key] = crossed.value

                label = wire.label_for(crossed)
                events.append(
                    self.make_event(
                        ctx,
                        message=(
                            f"{track.obj_class.value} crossed virtual fence "
                            f"'{wire.name}' ({label})"
                        ),
                        tracks=[track],
                        zone_id=wire.wire_id,
                        confidence=track.score,
                        attributes={
                            "tripwire_name": wire.name,
                            "direction": crossed.value,
                            "direction_label": label,
                            "object_class": track.obj_class.value,
                            "speed": round(ctx.speed_fraction_per_second(track), 4),
                        },
                    )
                )
        return events


@register_rule
class WrongDirectionRule(Rule):
    """Alerts on sustained movement against the permitted flow.

    Configured with ``heading_degrees`` (the permitted direction of travel, 0 =
    right, 90 = down in image space) and ``tolerance_degrees``. Used on approach
    roads and one-way corridors at check posts.
    """

    type_name = "wrong_direction"
    event_type = EventType.WRONG_DIRECTION

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        permitted = self.float_param("heading_degrees", 0.0)
        tolerance = self.float_param("tolerance_degrees", 60.0)
        min_speed = self.float_param("min_speed", 0.05)
        confirm_frames = self.int_param("confirm_frames", 5)
        zones = self.selected_zones(ctx)

        events: list[Event] = []
        for track in ctx.tracks:
            if not self.applies_to(track):
                continue
            if ctx.speed_fraction_per_second(track) < min_speed:
                continue  # too slow for a heading estimate to mean anything
            if not self.within_zones(ctx, track, zones):
                continue

            vx, vy = track.velocity
            heading = math.degrees(math.atan2(vy, vx))
            deviation = abs(_angle_difference(heading, permitted))

            state = self.track_state(track)
            if deviation <= 180.0 - tolerance:
                state["wrong_frames"] = 0
                continue

            state["wrong_frames"] = state.get("wrong_frames", 0) + 1
            if state["wrong_frames"] != confirm_frames:
                # Fire exactly once, on the frame the count is reached.
                continue

            events.append(
                self.make_event(
                    ctx,
                    message=f"{track.obj_class.value} moving against permitted flow",
                    tracks=[track],
                    confidence=track.score,
                    attributes={
                        "heading_degrees": round(heading, 1),
                        "permitted_degrees": permitted,
                        "deviation_degrees": round(deviation, 1),
                    },
                )
            )
        return events


def _angle_difference(a: float, b: float) -> float:
    """Smallest signed difference between two angles, in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Behavioural rules
# --------------------------------------------------------------------------- #


@register_rule
class LoiteringRule(Rule):
    """Alerts when an object dwells in an area beyond a time threshold.

    Dwell alone is not enough: a parked vehicle or a sentry at a post would
    trigger continuously. The rule additionally requires that the object has
    stayed within a bounded radius (``max_displacement``) - that is what
    separates *loitering* from *waiting legitimately somewhere else in frame*.
    """

    type_name = "loitering"
    event_type = EventType.LOITERING

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        dwell_threshold = self.float_param("dwell_seconds", 30.0)
        max_displacement = self.float_param("max_displacement", 0.15)
        repeat_seconds = self.float_param("repeat_seconds", 0.0)
        zones = self.selected_zones(ctx)

        events: list[Event] = []
        for track in ctx.tracks:
            if not self.applies_to(track):
                continue

            state = self.track_state(track)
            if not self.within_zones(ctx, track, zones):
                state["since"] = 0.0
                state["last_fired"] = 0.0
                continue

            if not state.get("since"):
                state["since"] = ctx.now
            dwell = ctx.now - state["since"]
            if dwell < dwell_threshold:
                continue

            # Normalised displacement keeps the threshold resolution-independent.
            displacement = track.displacement() / max(1.0, float(ctx.height))
            if displacement > max_displacement:
                # Genuinely traversing the area, not loitering in it.
                state["since"] = ctx.now
                continue

            last_fired = state.get("last_fired", 0.0)
            if last_fired and (repeat_seconds <= 0 or ctx.now - last_fired < repeat_seconds):
                continue
            state["last_fired"] = ctx.now

            zone = next(
                (z for z in zones if z.contains(*track.bbox.foot, ctx.width, ctx.height)),
                None,
            )
            events.append(
                self.make_event(
                    ctx,
                    message=(
                        f"{track.obj_class.value} loitering for "
                        f"{humanise_duration(dwell)}"
                        + (f" in '{zone.name}'" if zone else "")
                    ),
                    tracks=[track],
                    zone_id=zone.zone_id if zone else None,
                    confidence=min(1.0, dwell / (dwell_threshold * 2.0)),
                    attributes={
                        "dwell_seconds": round(dwell, 1),
                        "displacement": round(displacement, 4),
                        "object_class": track.obj_class.value,
                    },
                )
            )
        return events


@register_rule
class RapidMovementRule(Rule):
    """Alerts on running or otherwise abnormally fast movement.

    Speed is measured in frame-heights per second so the threshold transfers
    between cameras. A person walking is roughly 0.1-0.2 of frame height per
    second on a typical perimeter view; running is 0.4 and up.
    """

    type_name = "rapid_movement"
    event_type = EventType.RAPID_MOVEMENT

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        threshold = self.float_param("speed_threshold", 0.35)
        confirm_frames = self.int_param("confirm_frames", 3)
        zones = self.selected_zones(ctx)

        events: list[Event] = []
        for track in ctx.tracks:
            if not self.applies_to(track):
                continue
            if not self.within_zones(ctx, track, zones):
                continue

            speed = ctx.speed_fraction_per_second(track)
            state = self.track_state(track)
            if speed < threshold:
                state["fast_frames"] = 0
                continue

            state["fast_frames"] = state.get("fast_frames", 0) + 1
            if state["fast_frames"] != confirm_frames:
                continue

            events.append(
                self.make_event(
                    ctx,
                    message=f"{track.obj_class.value} moving rapidly ({speed:.2f} h/s)",
                    tracks=[track],
                    confidence=min(1.0, speed / max(1e-6, threshold * 2.0)),
                    attributes={
                        "speed_frame_heights_per_second": round(speed, 3),
                        "threshold": threshold,
                        "object_class": track.obj_class.value,
                    },
                )
            )
        return events


@register_rule
class AbandonedObjectRule(Rule):
    """Alerts on a static object left with no person nearby.

    Requires three conditions together, because any one alone is common and
    benign: the object must be effectively stationary, it must have been so for
    a sustained period, and no person track may be within ``owner_radius``.
    The last condition is what distinguishes an abandoned bag from a bag
    someone is standing next to.
    """

    type_name = "abandoned_object"
    event_type = EventType.ABANDONED_OBJECT

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        static_seconds = self.float_param("static_seconds", 45.0)
        max_movement = self.float_param("max_movement", 0.02)
        owner_radius = self.float_param("owner_radius", 0.2)
        zones = self.selected_zones(ctx)

        # Default to bag-like objects, but allow any class to be watched: an
        # unattended vehicle at a check post is the same pattern.
        watched = self._classes or {ObjectClass.BAG}
        radius_px = owner_radius * ctx.height
        people = [t for t in ctx.tracks if t.category is ObjectCategory.HUMAN]

        events: list[Event] = []
        for track in ctx.tracks:
            if track.obj_class not in watched:
                continue
            if not self.within_zones(ctx, track, zones):
                continue

            state = self.track_state(track)
            displacement = track.displacement() / max(1.0, float(ctx.height))
            if displacement > max_movement:
                state.update(since=0.0, fired=False)
                continue

            if not state.get("since"):
                state["since"] = ctx.now
            static_for = ctx.now - state["since"]
            if static_for < static_seconds or state.get("fired"):
                continue

            nearest = min(
                (t.bbox.distance_to(track.bbox) for t in people), default=float("inf")
            )
            if nearest <= radius_px:
                continue  # someone is with it; not abandoned

            state["fired"] = True
            events.append(
                self.make_event(
                    ctx,
                    message=(
                        f"unattended {track.obj_class.value} static for "
                        f"{humanise_duration(static_for)}"
                    ),
                    tracks=[track],
                    confidence=min(1.0, static_for / (static_seconds * 2.0)),
                    attributes={
                        "static_seconds": round(static_for, 1),
                        "nearest_person_px": None if math.isinf(nearest) else round(nearest, 1),
                        "object_class": track.obj_class.value,
                    },
                )
            )
        return events


@register_rule
class CrowdGatheringRule(Rule):
    """Alerts when people cluster beyond a threshold.

    Counts people whose foot points fall inside a watched zone (or the whole
    frame). When ``cluster_radius`` is set, the count is of the largest spatial
    cluster rather than the raw total - ten people spread across a wide view is
    a normal day at a check post; ten people within a few metres of each other
    is a gathering.
    """

    type_name = "crowd_gathering"
    event_type = EventType.CROWD_GATHERING

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        min_people = self.int_param("min_people", 5)
        confirm_frames = self.int_param("confirm_frames", 5)
        cluster_radius = self.float_param("cluster_radius", 0.0)
        zones = self.selected_zones(ctx)

        people = [t for t in ctx.tracks if t.category is ObjectCategory.HUMAN]
        events: list[Event] = []

        targets: list[tuple[str | None, str, list[Track]]] = []
        if zones:  # empty => the whole frame, per selected_zones semantics
            for zone in zones:
                inside = [
                    t for t in people
                    if zone.contains(*t.bbox.foot, ctx.width, ctx.height)
                ]
                targets.append((zone.zone_id, zone.name, inside))
        else:
            targets.append((None, "frame", people))

        for zone_id, zone_name, group in targets:
            if cluster_radius > 0:
                group = _largest_cluster(group, cluster_radius * ctx.height)

            key = f"crowd_{zone_id or 'frame'}"
            state: dict[str, Any] = ctx.camera_state.setdefault(
                f"{self.rule_id}:{key}", {"frames": 0, "fired": False}
            )

            if len(group) < min_people:
                state.update(frames=0, fired=False)
                continue

            state["frames"] += 1
            if state["frames"] < confirm_frames or state["fired"]:
                continue
            state["fired"] = True

            events.append(
                self.make_event(
                    ctx,
                    message=f"{len(group)} people gathered in '{zone_name}'",
                    tracks=group,
                    zone_id=zone_id,
                    confidence=min(1.0, len(group) / max(1, min_people * 2)),
                    attributes={
                        "person_count": len(group),
                        "threshold": min_people,
                        "zone_name": zone_name,
                        "clustered": cluster_radius > 0,
                    },
                )
            )
        return events


def _largest_cluster(tracks: list[Track], radius_px: float) -> list[Track]:
    """Return the largest group of tracks connected within ``radius_px``.

    Single-linkage clustering by union-find. At the track counts a CCTV frame
    produces this is trivially cheap, and it avoids pulling in a clustering
    dependency for what is a dozen points.
    """
    if len(tracks) < 2:
        return tracks

    parent = list(range(len(tracks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if tracks[i].bbox.distance_to(tracks[j].bbox) <= radius_px:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[Track]] = {}
    for idx, track in enumerate(tracks):
        groups.setdefault(find(idx), []).append(track)
    return max(groups.values(), key=len)


# --------------------------------------------------------------------------- #
# Schedule and scene-integrity rules
# --------------------------------------------------------------------------- #


@register_rule
class NightMovementRule(Rule):
    """Alerts on any movement during curfew hours.

    Distinct from :class:`IntrusionRule` in intent: it does not care where the
    object is, only that something is moving in a place that should be still at
    this hour. Configure ``active_hours`` on the rule (e.g. ``["22:00-05:00"]``).
    ``require_dark`` additionally gates on measured frame luminance, which
    handles the seasonal drift of actual darkness against a fixed clock window.
    """

    type_name = "night_movement"
    event_type = EventType.NIGHT_MOVEMENT

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        min_speed = self.float_param("min_speed", 0.02)
        confirm_frames = self.int_param("confirm_frames", 3)
        require_dark = bool(self.param("require_dark", True))
        zones = self.selected_zones(ctx)

        if require_dark and not ctx.frame.is_night:
            return []

        events: list[Event] = []
        for track in ctx.tracks:
            if not self.applies_to(track):
                continue
            if not self.within_zones(ctx, track, zones):
                continue
            if ctx.speed_fraction_per_second(track) < min_speed:
                continue

            state = self.track_state(track)
            state["night_frames"] = state.get("night_frames", 0) + 1
            if state["night_frames"] != confirm_frames:
                continue

            events.append(
                self.make_event(
                    ctx,
                    message=f"night-time movement: {track.obj_class.value}",
                    tracks=[track],
                    confidence=track.score,
                    attributes={
                        "object_class": track.obj_class.value,
                        "is_night_frame": ctx.frame.is_night,
                        "speed": round(ctx.speed_fraction_per_second(track), 3),
                    },
                )
            )
        return events


#: Presence maps a track's category onto the event type that names it.
_PRESENCE_EVENT: dict[ObjectCategory, EventType] = {
    ObjectCategory.HUMAN: EventType.PERSON_DETECTED,
    ObjectCategory.VEHICLE: EventType.VEHICLE_DETECTED,
}


@register_rule
class PresenceRule(Rule):
    """Informational: records that an object of interest appeared.

    Not an alarm. It exists so that a control room can build presence
    statistics (vehicle counts on a border road, footfall at a check post)
    without every appearance being escalated as an incident.
    """

    type_name = "presence"
    event_type = EventType.PERSON_DETECTED

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        zones = self.selected_zones(ctx)
        events: list[Event] = []

        for track in ctx.tracks:
            if not self.applies_to(track):
                continue
            if not self.within_zones(ctx, track, zones):
                continue

            state = self.track_state(track)
            if state.get("reported"):
                continue
            state["reported"] = True

            # Report what was actually seen. Collapsing everything that is
            # not a vehicle into `person_detected` puts cattle and unplaceable
            # objects into the one feed an operator filters on when they want
            # people, which is precisely when it matters most.
            events.append(
                Event(
                    camera_id=ctx.camera_id,
                    event_type=_PRESENCE_EVENT.get(
                        track.category, EventType.OBJECT_DETECTED
                    ),
                    severity=self._severity or Severity.INFO,
                    timestamp=ctx.timestamp,
                    confidence=track.score,
                    message=f"{track.obj_class.value} detected",
                    rule_id=self.rule_id,
                    track_ids=[track.track_id],
                    boxes=[track.bbox],
                    attributes={"object_class": track.obj_class.value},
                    frame_index=ctx.frame.index,
                )
            )
        return events


@register_rule
class TamperRule(Rule):
    """Detects lens obstruction, defocus and camera repositioning.

    Tamper detection is frame-level, not track-level: it compares the current
    frame's global statistics against a slow-moving baseline. All three failure
    modes below are ways an adversary blinds a camera *without* taking it
    offline, so a naive "is the stream up?" health check reports green while the
    camera watches a spray-painted dome.

    The baseline updates slowly and only while no tamper is asserted, so a
    genuine tamper cannot be absorbed into the baseline by simply persisting.
    """

    type_name = "camera_tamper"
    event_type = EventType.CAMERA_TAMPER
    requires_tracks = False

    #: Exponential moving average weight for the scene baseline.
    _BASELINE_ALPHA = 0.02

    def evaluate(self, ctx: RuleContext) -> list[Event]:
        from ibvap.vision.preprocess import mean_luma, sharpness

        dark_threshold = self.float_param("dark_luma", 12.0)
        blur_ratio = self.float_param("blur_ratio", 0.35)
        scene_change_ratio = self.float_param("scene_change_ratio", 0.55)
        confirm_frames = self.int_param("confirm_frames", 15)
        warmup_frames = self.int_param("warmup_frames", 30)

        state: dict[str, Any] = ctx.camera_state.setdefault(
            f"{self.rule_id}:tamper",
            {"luma": None, "focus": None, "hist": None, "frames": 0, "strikes": 0, "fired": False},
        )

        image = ctx.frame.image
        luma = mean_luma(image)
        focus = sharpness(image)
        hist = _luma_histogram(image)
        state["frames"] += 1

        if state["luma"] is None:
            state.update(luma=luma, focus=focus, hist=hist)
            return []

        reason = ""
        # 1. Blackout / covered lens.
        if luma < dark_threshold:
            reason = "lens_covered"
        # 2. Defocus - sharpness collapses relative to the learned baseline.
        elif state["focus"] > 1.0 and focus < state["focus"] * blur_ratio:
            reason = "defocused"
        # 3. Repositioned - the luma histogram decorrelates from the baseline.
        elif _histogram_correlation(hist, state["hist"]) < scene_change_ratio:
            reason = "view_changed"

        if not reason:
            state["strikes"] = max(0, state["strikes"] - 1)
            if state["strikes"] == 0:
                state["fired"] = False
            # Only drift the baseline while the scene looks healthy.
            alpha = self._BASELINE_ALPHA
            state["luma"] = (1 - alpha) * state["luma"] + alpha * luma
            state["focus"] = (1 - alpha) * state["focus"] + alpha * focus
            state["hist"] = (1 - alpha) * state["hist"] + alpha * hist
            return []

        state["strikes"] += 1
        # Ignore the settling period after a stream (re)starts, and require the
        # condition to persist: headlights, lightning and a passing lorry
        # filling the frame all look briefly like tampering.
        if (
            state["frames"] < warmup_frames
            or state["strikes"] < confirm_frames
            or state["fired"]
        ):
            return []

        state["fired"] = True
        return [
            self.make_event(
                ctx,
                message=f"camera tamper suspected: {reason.replace('_', ' ')}",
                confidence=min(1.0, state["strikes"] / (confirm_frames * 2.0)),
                severity=Severity.HIGH,
                attributes={
                    "reason": reason,
                    "luma": round(luma, 1),
                    "baseline_luma": round(float(state["luma"]), 1),
                    "focus": round(focus, 1),
                    "baseline_focus": round(float(state["focus"]), 1),
                },
            )
        ]


def _luma_histogram(image: Any, bins: int = 32) -> Any:
    """Normalised luminance histogram, sub-sampled for speed."""
    import cv2
    import numpy as np

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hist = cv2.calcHist([grey[::4, ::4]], [0], None, [bins], [0, 256]).flatten()
    total = hist.sum()
    return (hist / total) if total > 0 else np.zeros(bins, dtype=np.float32)


def _histogram_correlation(a: Any, b: Any) -> float:
    """Pearson correlation between two histograms, in ``[-1, 1]``."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_dev, b_dev = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a_dev**2).sum() * (b_dev**2).sum()))
    if denominator < 1e-12:
        return 1.0  # two flat histograms are not evidence of a changed view
    return float((a_dev * b_dev).sum() / denominator)
