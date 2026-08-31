"""Scenario-driven video simulation.

Renders synthetic border-surveillance footage: a terrain scene with a fence
line and an approach road, populated by actors that walk, drive and graze
according to a named scenario.

It exists because the alternative is worse. Validating analytics needs footage
of people crossing fences at night, cattle wandering into restricted zones and
vehicles approaching a gate - and that footage is operationally sensitive,
rarely shareable, and never available when you need a specific case. A
simulator gives every developer, reviewer and CI run the same scenarios on
demand, deterministically.

What it is good for:

* Exercising every analytics rule end to end - intrusion, line crossing,
  loitering, crowd, abandoned object, night movement.
* Demonstrating the false-alarm case that matters most: cattle walking through
  a restricted zone, which must be suppressed while a person in the same zone
  must not.
* Reproducing a scenario exactly, by seed, on any machine.

What it is **not**: a substitute for real footage when judging detection
accuracy. These are geometric figures on synthetic terrain, and a detector's
score on them says nothing about its score on a real IR frame at 40 m. Model
accuracy must be measured on real labelled data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from ibvap.core.errors import StreamClosedError, StreamError
from ibvap.core.logging import get_logger
from ibvap.core.types import ObjectClass
from ibvap.ingest.source import SourceInfo, VideoSource

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Actors
# --------------------------------------------------------------------------- #


@dataclass
class Actor:
    """One moving thing in the scene.

    Positions are normalised to ``[0, 1]``; ``y`` doubles as a depth cue, so an
    actor near the top of the frame renders smaller. Without that, everything
    is the same size regardless of distance and size-based analytics
    thresholds behave nothing like they would on a real camera.
    """

    kind: ObjectClass
    x: float
    y: float
    #: Velocity in frame-widths / frame-heights per second.
    vx: float = 0.0
    vy: float = 0.0
    #: Base size as a fraction of frame height, at the bottom of the frame.
    scale: float = 0.18
    #: Seconds after scenario start before this actor appears.
    spawn_at: float = 0.0
    #: Seconds after spawn before it disappears. 0 means it never leaves.
    lifetime: float = 0.0
    #: Stops moving after this long, for loitering and abandonment scenarios.
    freeze_after: float = 0.0
    #: Sideways wander amplitude, so paths are not unnaturally straight.
    wander: float = 0.0
    label: str = ""

    _age: float = field(default=0.0, init=False)
    _active: bool = field(default=False, init=False)

    def update(self, dt: float, elapsed: float) -> None:
        """Advance one time step."""
        if elapsed < self.spawn_at:
            self._active = False
            return
        self._active = True
        self._age += dt

        if self.freeze_after and self._age > self.freeze_after:
            return

        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.wander:
            # A slow sinusoid, not noise: real walking paths drift smoothly
            # rather than jittering frame to frame, and jitter would make the
            # tracker's job artificially easy or artificially hard.
            self.y += math.sin(self._age * 1.7) * self.wander * dt

    @property
    def visible(self) -> bool:
        if not self._active:
            return False
        if self.lifetime and self._age > self.lifetime:
            return False
        return -0.2 <= self.x <= 1.2 and -0.2 <= self.y <= 1.2

    def depth_scale(self) -> float:
        """Size multiplier from depth. Higher in frame means further away."""
        # 0.45 at the horizon to 1.0 in the foreground: a roughly linear
        # ground-plane approximation, which is what a fixed mast camera sees.
        return 0.45 + 0.55 * max(0.0, min(1.0, self.y))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

#: Daylight palette, BGR.
_DAY = {
    "sky": (188, 172, 150),
    "ground": (96, 118, 132),
    "road": (78, 80, 84),
    "fence": (120, 130, 140),
    "person": (52, 48, 46),
    "vehicle": (120, 96, 70),
    "animal": (78, 92, 110),
    "bag": (60, 70, 130),
}

#: Night / IR palette. Warm bodies read bright against cold ground, which is
#: what a thermal or IR-illuminated camera actually shows.
_NIGHT = {
    "sky": (18, 18, 20),
    "ground": (30, 32, 34),
    "road": (24, 25, 27),
    "fence": (58, 60, 64),
    "person": (210, 214, 218),
    "vehicle": (150, 155, 160),
    "animal": (170, 175, 180),
    "bag": (95, 100, 105),
}


def _palette(night: bool) -> dict[str, tuple[int, int, int]]:
    return _NIGHT if night else _DAY


def render_terrain(
    width: int, height: int, *, night: bool, seed: int, road: bool = True,
    fence: bool = True,
) -> np.ndarray:
    """Draw the static scene: sky, ground, an approach road and a fence line."""
    rng = np.random.default_rng(seed)
    colours = _palette(night)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    horizon = int(height * 0.32)
    frame[:horizon] = colours["sky"]
    frame[horizon:] = colours["ground"]

    # Ground texture, coarser in the foreground so the surface reads as
    # receding rather than as flat noise.
    for band in range(horizon, height, 8):
        depth = (band - horizon) / max(1, height - horizon)
        amount = int(6 + 18 * depth)
        noise = rng.integers(-amount, amount, (min(8, height - band), width, 3))
        frame[band:band + 8] = np.clip(
            frame[band:band + 8].astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

    if road:
        # A road narrowing towards the horizon.
        top_y, bottom_y = horizon + int(height * 0.06), height
        polygon = np.array([
            [int(width * 0.42), top_y], [int(width * 0.58), top_y],
            [int(width * 0.95), bottom_y], [int(width * 0.05), bottom_y],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [polygon], colours["road"])

    if fence:
        # Posts with a wire line, drawn with perspective so the fence recedes.
        fence_y = int(height * 0.44)
        cv2.line(frame, (0, fence_y), (width, fence_y), colours["fence"], 2)
        for post in range(0, width, max(24, width // 18)):
            cv2.line(frame, (post, fence_y - 18), (post, fence_y + 10),
                     colours["fence"], 2)

    return cv2.GaussianBlur(frame, (3, 3), 0)


def draw_actor(frame: np.ndarray, actor: Actor, *, night: bool) -> None:
    """Draw one actor with a silhouette appropriate to its class."""
    height, width = frame.shape[:2]
    colours = _palette(night)

    size = actor.scale * actor.depth_scale() * height
    cx, cy = int(actor.x * width), int(actor.y * height)

    if actor.kind is ObjectClass.PERSON:
        body_h, body_w = int(size), max(3, int(size * 0.30))
        head_r = max(2, int(size * 0.13))
        cv2.rectangle(frame, (cx - body_w // 2, cy - body_h),
                      (cx + body_w // 2, cy), colours["person"], -1)
        cv2.circle(frame, (cx, cy - body_h - head_r), head_r, colours["person"], -1)

    elif actor.kind in (ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS):
        length = int(size * (2.6 if actor.kind is ObjectClass.CAR else 3.4))
        body_h = int(size * (0.8 if actor.kind is ObjectClass.CAR else 1.1))
        cv2.rectangle(frame, (cx - length // 2, cy - body_h),
                      (cx + length // 2, cy), colours["vehicle"], -1)
        # Cabin, so the silhouette is not a bare rectangle.
        cv2.rectangle(frame, (cx - length // 6, cy - int(body_h * 1.45)),
                      (cx + length // 3, cy - body_h), colours["vehicle"], -1)
        for wheel in (-length // 3, length // 3):
            cv2.circle(frame, (cx + wheel, cy), max(2, int(size * 0.16)),
                       (20, 20, 22), -1)
        if night:
            # Headlights: the dominant feature of a vehicle on a night camera,
            # and a genuine stressor for tamper and exposure logic.
            for lamp in (-length // 2, -length // 2 + int(size * 0.5)):
                cv2.circle(frame, (cx + lamp, cy - body_h // 2),
                           max(2, int(size * 0.12)), (245, 245, 235), -1)

    elif actor.kind is ObjectClass.ANIMAL:
        body_l, body_h = int(size * 1.5), int(size * 0.62)
        cv2.ellipse(frame, (cx, cy - body_h), (body_l // 2, body_h // 2),
                    0, 0, 360, colours["animal"], -1)
        cv2.circle(frame, (cx + body_l // 2, cy - int(body_h * 1.3)),
                   max(2, int(size * 0.18)), colours["animal"], -1)
        for leg in (-body_l // 3, body_l // 3):
            cv2.line(frame, (cx + leg, cy - body_h // 2), (cx + leg, cy),
                     colours["animal"], max(1, int(size * 0.08)))

    elif actor.kind is ObjectClass.BAG:
        side = max(3, int(size * 0.42))
        cv2.rectangle(frame, (cx - side // 2, cy - side), (cx + side // 2, cy),
                      colours["bag"], -1)

    else:
        side = max(3, int(size * 0.5))
        cv2.rectangle(frame, (cx - side // 2, cy - side), (cx + side // 2, cy),
                      (110, 110, 110), -1)


def apply_conditions(
    frame: np.ndarray, *, night: bool, haze: float, noise: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply atmospheric and sensor effects."""
    out = frame
    if haze > 0:
        # Distance haze washes out contrast towards the horizon - the main
        # reason daytime long-range detection degrades.
        veil = np.full_like(out, 200 if not night else 40)
        gradient = np.linspace(haze, 0.0, out.shape[0], dtype=np.float32)[:, None, None]
        out = (out * (1 - gradient) + veil * gradient).astype(np.uint8)
    if noise > 0:
        # High-gain sensor noise, much stronger at night.
        amount = noise * (3.0 if night else 1.0)
        grain = rng.normal(0, amount * 255 * 0.06, out.shape)
        out = np.clip(out.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

ScenarioBuilder = Callable[[], list[Actor]]


def _scenario_patrol() -> list[Actor]:
    """Routine daytime activity: nothing that should alarm anyone."""
    return [
        Actor(ObjectClass.PERSON, x=-0.05, y=0.72, vx=0.06, scale=0.20,
              wander=0.01, label="patrol"),
        Actor(ObjectClass.CAR, x=1.10, y=0.86, vx=-0.10, scale=0.13,
              spawn_at=6.0, label="patrol vehicle"),
    ]


def _scenario_intrusion() -> list[Actor]:
    """A person crosses the fence line into the restricted strip."""
    return [
        Actor(ObjectClass.PERSON, x=0.12, y=0.38, vx=0.045, vy=0.055,
              scale=0.16, wander=0.008, label="intruder"),
    ]


def _scenario_cattle() -> list[Actor]:
    """The false alarm that matters: livestock in the restricted zone.

    Must be suppressed by class while a person in the same zone must still
    alarm - which is why this scenario also includes a person.
    """
    return [
        Actor(ObjectClass.ANIMAL, x=-0.05, y=0.62, vx=0.05, scale=0.17,
              wander=0.006, label="cattle"),
        Actor(ObjectClass.ANIMAL, x=-0.18, y=0.70, vx=0.05, scale=0.19,
              spawn_at=2.0, wander=0.006, label="cattle"),
        Actor(ObjectClass.PERSON, x=1.08, y=0.66, vx=-0.05, scale=0.18,
              spawn_at=10.0, label="herder"),
    ]


def _scenario_vehicle_approach() -> list[Actor]:
    """A vehicle drives up the approach road towards the gate."""
    return [
        Actor(ObjectClass.TRUCK, x=0.50, y=0.40, vy=0.045, vx=0.02,
              scale=0.10, label="approaching truck"),
    ]


def _scenario_loiter() -> list[Actor]:
    """Someone enters the zone and stops, triggering dwell-time logic."""
    return [
        Actor(ObjectClass.PERSON, x=0.30, y=0.55, vx=0.05, vy=0.02, scale=0.18,
              freeze_after=5.0, label="loiterer"),
    ]


def _scenario_abandoned() -> list[Actor]:
    """A person carries a bag in, drops it, and leaves without it."""
    return [
        Actor(ObjectClass.PERSON, x=0.10, y=0.70, vx=0.07, scale=0.18,
              lifetime=14.0, label="carrier"),
        Actor(ObjectClass.BAG, x=0.55, y=0.72, vx=0.0, scale=0.10,
              spawn_at=7.0, label="abandoned bag"),
    ]


def _scenario_crowd() -> list[Actor]:
    """A group gathers, for crowd-density logic."""
    return [
        Actor(ObjectClass.PERSON, x=0.45 + i * 0.045, y=0.66 + (i % 3) * 0.035,
              vx=0.012, scale=0.16, spawn_at=i * 0.7, wander=0.004,
              label=f"crowd-{i}")
        for i in range(7)
    ]


def _scenario_night_infiltration() -> list[Actor]:
    """Two people crossing the fence line after dark, moving quickly."""
    return [
        Actor(ObjectClass.PERSON, x=0.20, y=0.40, vx=0.05, vy=0.07, scale=0.15,
              label="infiltrator-1"),
        Actor(ObjectClass.PERSON, x=0.26, y=0.38, vx=0.05, vy=0.07, scale=0.15,
              spawn_at=1.5, label="infiltrator-2"),
    ]


#: Named scenarios, selectable from a camera URL.
SCENARIOS: dict[str, ScenarioBuilder] = {
    "patrol": _scenario_patrol,
    "intrusion": _scenario_intrusion,
    "cattle": _scenario_cattle,
    "vehicle": _scenario_vehicle_approach,
    "loiter": _scenario_loiter,
    "abandoned": _scenario_abandoned,
    "crowd": _scenario_crowd,
    "night": _scenario_night_infiltration,
}


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #


class SimulatedSource(VideoSource):
    """A :class:`VideoSource` that renders a scenario instead of decoding one.

    Deterministic for a given seed, so a test asserting "this scenario raises
    an intrusion alert" gives the same answer on every machine and in CI.
    """

    def __init__(
        self,
        scenario: str = "patrol",
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 15.0,
        seed: int = 0,
        night: bool = False,
        haze: float = 0.0,
        noise: float = 0.02,
        loop: bool = True,
        duration: float = 30.0,
    ) -> None:
        if scenario not in SCENARIOS:
            raise StreamError(
                f"unknown scenario {scenario!r}; available: {sorted(SCENARIOS)}"
            )
        self.scenario = scenario
        self.width = width
        self.height = height
        self.fps = max(1.0, fps)
        self.seed = seed
        self.night = night or scenario == "night"
        self.haze = haze
        self.noise = noise
        self.loop = loop
        self.duration = duration

        self._open = False
        self._index = 0
        self._terrain: np.ndarray | None = None
        self._actors: list[Actor] = []
        self._rng = np.random.default_rng(seed)

    # -- lifecycle --------------------------------------------------------- #

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(
            width=self.width, height=self.height, fps=self.fps,
            frame_count=0 if self.loop else int(self.duration * self.fps),
            backend=f"simulator:{self.scenario}",
        )

    def open(self) -> None:
        self._terrain = render_terrain(
            self.width, self.height, night=self.night, seed=self.seed,
            road=self.scenario in ("vehicle", "patrol"),
        )
        self._reset_actors()
        self._index = 0
        self._rng = np.random.default_rng(self.seed)
        self._open = True
        log.info(
            "simulator_opened",
            scenario=self.scenario, night=self.night,
            resolution=f"{self.width}x{self.height}", fps=self.fps,
        )

    def _reset_actors(self) -> None:
        self._actors = SCENARIOS[self.scenario]()

    def close(self) -> None:
        self._open = False
        self._terrain = None
        self._actors = []

    # -- rendering --------------------------------------------------------- #

    def read(self) -> np.ndarray:
        if not self._open or self._terrain is None:
            raise StreamError("read() called on a source that is not open")

        elapsed = self._index / self.fps
        if elapsed >= self.duration:
            if not self.loop:
                raise StreamClosedError(f"scenario {self.scenario!r} finished")
            self._index = 0
            elapsed = 0.0
            self._reset_actors()

        frame = self._terrain.copy()
        dt = 1.0 / self.fps
        for actor in self._actors:
            actor.update(dt, elapsed)
            if actor.visible:
                draw_actor(frame, actor, night=self.night)

        frame = apply_conditions(
            frame, night=self.night, haze=self.haze, noise=self.noise, rng=self._rng
        )
        self._index += 1
        return frame

    # -- introspection ----------------------------------------------------- #

    def ground_truth(self) -> list[dict[str, object]]:
        """Where each visible actor currently is, in normalised coordinates.

        Not used by the pipeline - the simulator is a *source* and the platform
        must not see labels it would never have in the field. It exists so a
        test can assert that analytics found what was actually there.
        """
        return [
            {
                "kind": actor.kind.value,
                "label": actor.label,
                "x": round(actor.x, 4),
                "y": round(actor.y, 4),
            }
            for actor in self._actors if actor.visible
        ]
