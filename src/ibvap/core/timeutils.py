"""Time-window helpers.

Border analytics are overwhelmingly schedule-driven: a fence that is a
thoroughfare by day is a hard perimeter after dusk. Windows are therefore
expressed the way a duty roster writes them - ``"18:00-06:00"`` - including
windows that wrap past midnight, which is the common case and the one naive
implementations get wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timezone

_WINDOW_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A local-time window, possibly wrapping midnight."""

    start: time
    end: time

    @property
    def wraps_midnight(self) -> bool:
        return self.start > self.end

    def contains(self, moment: time) -> bool:
        """Whether ``moment`` falls within the window.

        Boundaries are half-open ``[start, end)`` so back-to-back windows such
        as ``06:00-18:00`` and ``18:00-06:00`` partition the day exactly once.
        """
        if self.start == self.end:
            return True  # a zero-width window means "always on"
        if self.wraps_midnight:
            return moment >= self.start or moment < self.end
        return self.start <= moment < self.end

    def __str__(self) -> str:
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"


def parse_window(spec: str) -> TimeWindow:
    """Parse ``"HH:MM-HH:MM"`` into a :class:`TimeWindow`."""
    match = _WINDOW_RE.match(spec)
    if not match:
        raise ValueError(f"invalid time window {spec!r}; expected 'HH:MM-HH:MM'")
    sh, sm, eh, em = (int(g) for g in match.groups())
    for hour, minute in ((sh, sm), (eh, em)):
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"invalid time in window {spec!r}")
    return TimeWindow(time(sh, sm), time(eh, em))


def parse_windows(specs: list[str]) -> list[TimeWindow]:
    return [parse_window(s) for s in specs]


def in_any_window(windows: list[TimeWindow], moment: datetime | None = None) -> bool:
    """Whether ``moment`` (default: now, local) is inside any window.

    An empty window list means "always active", which is what an operator
    expects when they leave the schedule field blank.
    """
    if not windows:
        return True
    now = moment or datetime.now()
    return any(w.contains(now.time()) for w in windows)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(timestamp: float) -> str:
    """Render a unix timestamp as an RFC 3339 UTC string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")


def humanise_duration(seconds: float) -> str:
    """Short human-readable duration for operator-facing alert messages."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
