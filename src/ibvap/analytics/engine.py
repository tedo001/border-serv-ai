"""Per-camera analytics engine.

Sits between the tracker and the event bus. Its job is not only to run rules
but to decide which of their outputs a human should actually see.

That second job is the one that determines whether the platform is used or
switched off. An unfiltered rule set on 32 cameras produces thousands of events
an hour, and a control room that receives thousands of alerts an hour stops
reading them within a shift. Three independent mechanisms narrow the flow:

* **Deduplication** collapses repeats of the same ongoing incident.
* **Cooldown** stops one rule re-firing on the same subject too soon.
* **Rate limiting** is a hard ceiling per camera, so a pathological scene (a
  swarm of insects on an IR illuminator) cannot flood the C2 link no matter
  what the rules decide.

Every suppressed event is counted in metrics rather than discarded silently, so
over-aggressive tuning is visible on a dashboard instead of being mistaken for
a quiet night.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from ibvap.analytics.base import Rule, RuleContext, build_rule
from ibvap.core.config import AnalyticsConfig, CameraConfig
from ibvap.core.geometry import CrossingDirection, Tripwire, Zone
from ibvap.core.logging import get_logger
from ibvap.core.types import Event, Frame, ObjectClass, Track
from ibvap.telemetry.metrics import Metrics

log = get_logger(__name__)


@dataclass
class SuppressionStats:
    """Counters explaining why events did not reach the operator."""

    deduplicated: int = 0
    cooldown: int = 0
    rate_limited: int = 0
    passed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "deduplicated": self.deduplicated,
            "cooldown": self.cooldown,
            "rate_limited": self.rate_limited,
            "passed": self.passed,
        }


class EventGate:
    """Applies deduplication, per-rule cooldown and a per-camera rate limit."""

    def __init__(
        self,
        *,
        dedup_window_seconds: float = 20.0,
        max_events_per_minute: int = 60,
    ) -> None:
        self.dedup_window = dedup_window_seconds
        self.max_events_per_minute = max_events_per_minute
        self._last_seen: dict[str, float] = {}
        self._last_fired: dict[str, float] = {}
        self._recent: deque[float] = deque()
        self.stats = SuppressionStats()

    def admit(self, event: Event, cooldown_seconds: float, now: float | None = None) -> bool:
        """Decide whether ``event`` should be delivered.

        ``now`` is monotonic. Callers pass it explicitly so that replaying a
        recorded file runs the same suppression logic on file time rather than
        wall-clock time.
        """
        moment = time.monotonic() if now is None else now
        key = event.dedup_key()

        # 1. Deduplication - the identical incident, still ongoing.
        last_seen = self._last_seen.get(key)
        if last_seen is not None and (moment - last_seen) < self.dedup_window:
            self._last_seen[key] = moment
            self.stats.deduplicated += 1
            return False

        # 2. Cooldown - the same rule re-firing on the same subject.
        last_fired = self._last_fired.get(key)
        if last_fired is not None and (moment - last_fired) < cooldown_seconds:
            self.stats.cooldown += 1
            return False

        # 3. Rate limit - a hard ceiling regardless of rule opinion.
        cutoff = moment - 60.0
        while self._recent and self._recent[0] < cutoff:
            self._recent.popleft()
        if len(self._recent) >= self.max_events_per_minute:
            self.stats.rate_limited += 1
            return False

        self._last_seen[key] = moment
        self._last_fired[key] = moment
        self._recent.append(moment)
        self.stats.passed += 1
        return True

    def prune(self, now: float | None = None, max_age: float = 600.0) -> None:
        """Drop suppression keys older than ``max_age``.

        Without this, a busy camera accumulates a dictionary entry per track id
        forever - a slow leak that only shows up after a node has been running
        for weeks, which is exactly when nobody is watching.
        """
        moment = time.monotonic() if now is None else now
        cutoff = moment - max_age
        self._last_seen = {k: v for k, v in self._last_seen.items() if v >= cutoff}
        self._last_fired = {k: v for k, v in self._last_fired.items() if v >= cutoff}


class AnalyticsEngine:
    """Runs the configured rule set for one camera."""

    def __init__(
        self,
        camera: CameraConfig,
        config: AnalyticsConfig | None = None,
        *,
        metrics: Metrics | None = None,
    ) -> None:
        self.camera = camera
        self.config = config or AnalyticsConfig()
        self.metrics = metrics

        self.zones: dict[str, Zone] = {
            z.id: Zone(
                zone_id=z.id, name=z.name, points=[tuple(p) for p in z.points],
                kind=z.kind, enabled=z.enabled,
            )
            for z in camera.zones
        }
        self.tripwires: dict[str, Tripwire] = {
            t.id: Tripwire(
                wire_id=t.id, name=t.name, start=tuple(t.start), end=tuple(t.end),
                direction=CrossingDirection(t.direction),
                left_label=t.left_label, right_label=t.right_label, enabled=t.enabled,
            )
            for t in camera.tripwires
        }

        self.rules: list[Rule] = []
        for rule_config in camera.rules:
            try:
                self.rules.append(build_rule(rule_config))
            except Exception as exc:
                # One misconfigured rule must not take the camera offline; the
                # rest of its analytics remain valuable.
                log.error(
                    "rule_build_failed",
                    camera=camera.id, rule=rule_config.id, type=rule_config.type,
                    error=str(exc),
                )

        self.gate = EventGate(
            dedup_window_seconds=self.config.dedup_window_seconds,
            max_events_per_minute=self.config.max_events_per_camera_per_minute,
        )
        self.camera_state: dict[str, Any] = {}
        self._ignored: set[ObjectClass] = {
            ObjectClass.coerce(c) for c in self.config.ignore_classes
        }
        self._frames_processed = 0
        self._last_prune = time.monotonic()

    # -- main entry point -------------------------------------------------- #

    def process(self, frame: Frame, tracks: list[Track]) -> list[Event]:
        """Evaluate all rules for one frame and return admitted events."""
        self._frames_processed += 1
        candidates = self._filter_tracks(tracks, frame)

        ctx = RuleContext(
            camera_id=self.camera.id,
            frame=frame,
            tracks=candidates,
            zones=self.zones,
            tripwires=self.tripwires,
            fps=frame.fps or self.camera.target_fps,
            camera_state=self.camera_state,
        )

        admitted: list[Event] = []
        for rule in self.rules:
            if not rule.is_active(ctx):
                continue
            # Frame-level rules (tamper) still run when there are no tracks.
            if rule.requires_tracks and not candidates:
                continue
            try:
                produced = rule.evaluate(ctx)
            except Exception as exc:
                log.error(
                    "rule_evaluation_failed",
                    camera=self.camera.id, rule=rule.rule_id, error=str(exc), exc_info=True,
                )
                continue

            for event in produced:
                if self.gate.admit(event, rule.cooldown_seconds, now=frame.monotonic):
                    admitted.append(event)
                elif self.metrics:
                    self.metrics.event_suppressed(self.camera.id, "gated")

        if self.metrics:
            for event in admitted:
                self.metrics.event(
                    self.camera.id, event.event_type.value, event.severity.value
                )

        # Periodic housekeeping; cheap and bounded.
        if frame.monotonic - self._last_prune > 300.0:
            self.gate.prune(now=frame.monotonic)
            self._last_prune = frame.monotonic

        return admitted

    def _filter_tracks(self, tracks: list[Track], frame: Frame) -> list[Track]:
        """Drop tracks that no rule should ever act on.

        Applied centrally rather than per rule so a class exclusion cannot be
        forgotten in one rule's configuration and silently re-admit the very
        false alarms it was meant to remove.
        """
        min_height = self.config.min_object_height_fraction * frame.height
        out: list[Track] = []
        for track in tracks:
            if track.obj_class in self._ignored:
                continue
            if min_height and track.bbox.height < min_height:
                continue
            out.append(track)
        return out

    # -- introspection ----------------------------------------------------- #

    @property
    def rule_ids(self) -> list[str]:
        return [r.rule_id for r in self.rules]

    def status(self) -> dict[str, Any]:
        """Snapshot for the health endpoint and operator console."""
        return {
            "camera_id": self.camera.id,
            "rules": [
                {"id": r.rule_id, "type": r.type_name, "enabled": r.enabled}
                for r in self.rules
            ],
            "zones": len(self.zones),
            "tripwires": len(self.tripwires),
            "frames_processed": self._frames_processed,
            "suppression": self.gate.stats.as_dict(),
            "ignored_classes": sorted(c.value for c in self._ignored),
        }

    def reload(self, camera: CameraConfig) -> None:
        """Rebuild rules and regions after a configuration change.

        Per-track rule state is discarded because track ids do not survive a
        reconfiguration meaningfully; the camera-level state is kept so tamper
        baselines are not thrown away by an unrelated zone edit.
        """
        preserved = {k: v for k, v in self.camera_state.items() if "tamper" in k}
        self.__init__(camera, self.config, metrics=self.metrics)  # type: ignore[misc]
        self.camera_state.update(preserved)
