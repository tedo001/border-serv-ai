"""Video sources.

Ingestion is where a border deployment actually hurts. Streams arrive over
microwave links, VSAT and long copper runs; cameras brown out with the site
generator and come back with a different resolution; an RTSP server drops a
session silently and keeps the TCP socket open. A reader that assumes a healthy
network will spend its life wedged in a blocking ``read()``.

The reader below therefore treats disconnection as the normal case: every open
is bounded, every read is watched by a watchdog, and failure escalates through
capped exponential backoff rather than a tight retry loop that would hammer an
already-struggling link.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from ibvap.core.errors import StreamClosedError, StreamError
from ibvap.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class SourceInfo:
    """What a source reports about itself once opened."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    #: Total frames, when the source is a finite file. 0 for live streams.
    frame_count: int = 0
    backend: str = ""

    @property
    def is_finite(self) -> bool:
        return self.frame_count > 0


class VideoSource(ABC):
    """A source of frames. Implementations must be safe to reopen."""

    @abstractmethod
    def open(self) -> None:
        """Open the source, raising :class:`StreamError` on failure."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return the next frame, raising :class:`StreamClosedError` at EOF."""

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @property
    def info(self) -> SourceInfo:
        return SourceInfo()

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OpenCVSource(VideoSource):
    """RTSP / HTTP / file source backed by OpenCV's FFmpeg capture."""

    def __init__(
        self,
        url: str,
        *,
        rtsp_transport: str = "tcp",
        open_timeout_seconds: float = 15.0,
        read_timeout_seconds: float = 15.0,
        buffer_size: int = 1,
    ) -> None:
        self.url = url
        self.rtsp_transport = rtsp_transport
        self.open_timeout = open_timeout_seconds
        self.read_timeout = read_timeout_seconds
        self.buffer_size = buffer_size
        self._capture: cv2.VideoCapture | None = None
        self._info = SourceInfo()
        self._last_read = 0.0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def info(self) -> SourceInfo:
        return self._info

    def open(self) -> None:
        self.close()

        if self._is_rtsp:
            # FFmpeg options must be set through the environment before the
            # capture is constructed. UDP is the RTSP default and it silently
            # shreds frames over a lossy microwave hop; forcing TCP trades a
            # little latency for streams that stay decodable. The timeout is in
            # microseconds and stops a dead peer wedging the open indefinitely.
            existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            options = [
                f"rtsp_transport;{self.rtsp_transport}",
                f"stimeout;{int(self.open_timeout * 1_000_000)}",
                "reorder_queue_size;0",
                "max_delay;500000",
            ]
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(options)
            try:
                capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            finally:
                if existing:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = existing
                else:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            capture = cv2.VideoCapture(self.url)

        if not capture.isOpened():
            capture.release()
            raise StreamError(f"could not open video source: {_redact(self.url)}")

        # A shallow decoder buffer is essential for live analytics: a deep one
        # hands us frames that are already seconds stale, so every alert is late
        # by however long the buffer is.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except cv2.error:  # pragma: no cover - unsupported by some backends
            pass

        self._capture = capture
        self._info = SourceInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
            frame_count=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            backend=capture.getBackendName(),
        )
        self._last_read = time.monotonic()
        log.info(
            "source_opened",
            url=_redact(self.url), width=self._info.width, height=self._info.height,
            fps=round(self._info.fps, 2), backend=self._info.backend,
        )

    @property
    def _is_rtsp(self) -> bool:
        return self.url.lower().startswith(("rtsp://", "rtsps://"))

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise StreamError("read() called on a source that is not open")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise StreamClosedError(f"stream ended or failed: {_redact(self.url)}")
        if frame.size == 0:
            raise StreamClosedError(f"decoder returned an empty frame: {_redact(self.url)}")

        self._last_read = time.monotonic()
        return frame

    @property
    def seconds_since_read(self) -> float:
        return time.monotonic() - self._last_read if self._last_read else 0.0

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover - release is best effort
                pass
            self._capture = None


class SyntheticSource(VideoSource):
    """Deterministic generated video for drills, CI and demonstrations.

    Renders a moving figure over a static scene. Deterministic by seed, so a
    test asserting "this scenario produces an intrusion alert" gives the same
    answer on every machine - which is what makes the whole pipeline testable
    without shipping video fixtures or standing up an RTSP server.
    """

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 15.0,
        seed: int = 0,
        total_frames: int = 0,
        night: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.seed = seed
        self.total_frames = total_frames
        self.night = night
        self._index = 0
        self._open = False
        self._background: np.ndarray | None = None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(
            width=self.width, height=self.height, fps=self.fps,
            frame_count=self.total_frames, backend="synthetic",
        )

    def open(self) -> None:
        rng = np.random.default_rng(self.seed)
        base = rng.integers(70, 140, (self.height, self.width, 3)).astype(np.uint8)
        # A horizon band makes the scene look like terrain rather than noise,
        # which matters when the synthetic feed is used for operator training.
        base[: self.height // 3] = np.clip(base[: self.height // 3] + 60, 0, 255)
        if self.night:
            base = (base * 0.18).astype(np.uint8)
        self._background = cv2.GaussianBlur(base, (9, 9), 0)
        self._index = 0
        self._open = True

    def read(self) -> np.ndarray:
        if not self._open or self._background is None:
            raise StreamError("read() called on a source that is not open")
        if self.total_frames and self._index >= self.total_frames:
            raise StreamClosedError("synthetic source exhausted")

        frame = self._background.copy()
        # A figure tracking left to right across the frame, looping.
        progress = (self._index % 120) / 120.0
        x = int(progress * (self.width - 80)) + 20
        y = int(self.height * 0.55)
        colour = (40, 40, 45) if not self.night else (200, 200, 200)
        cv2.rectangle(frame, (x, y), (x + 45, y + 110), colour, -1)
        cv2.circle(frame, (x + 22, y - 15), 16, colour, -1)

        self._index += 1
        return frame

    def close(self) -> None:
        self._open = False
        self._background = None


def build_source(
    url: str,
    *,
    rtsp_transport: str = "tcp",
    read_timeout_seconds: float = 15.0,
) -> VideoSource:
    """Construct the right source for a URL.

    ``synthetic://`` URLs accept query parameters, e.g.
    ``synthetic://?width=640&height=480&fps=10&night=1``.
    """
    if url.startswith(("synthetic://", "sim://")):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        def get_int(key: str, default: int) -> int:
            try:
                return int(params.get(key, [str(default)])[0])
            except (ValueError, IndexError):
                return default

        if url.startswith("sim://"):
            # sim://<scenario>?width=..&night=1 - renders a scripted border
            # scenario rather than a single moving figure.
            from ibvap.ingest.simulator import SimulatedSource

            return SimulatedSource(
                parsed.netloc or params.get("scenario", ["patrol"])[0],
                width=get_int("width", 1280),
                height=get_int("height", 720),
                fps=float(get_int("fps", 15)),
                seed=get_int("seed", 0),
                night=bool(get_int("night", 0)),
                haze=get_int("haze", 0) / 100.0,
                noise=get_int("noise", 2) / 100.0,
                loop=bool(get_int("loop", 1)),
                duration=float(get_int("duration", 30)),
            )

        return SyntheticSource(
            width=get_int("width", 1280),
            height=get_int("height", 720),
            fps=float(get_int("fps", 15)),
            seed=get_int("seed", 0),
            total_frames=get_int("frames", 0),
            night=bool(get_int("night", 0)),
        )

    return OpenCVSource(
        url, rtsp_transport=rtsp_transport, read_timeout_seconds=read_timeout_seconds
    )


def _redact(url: str) -> str:
    """Strip credentials from a URL before it reaches a log line.

    RTSP URLs almost always embed ``user:password``. Those logs get copied into
    tickets and shipped to a central collector; the credential must not travel
    with them.
    """
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if not rest:
        return url
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***:***@{host}" if scheme else f"***@{host}"
