"""Resilient stream reader.

One :class:`StreamReader` owns one camera. It runs a decode thread that keeps a
bounded queue topped up with the *freshest* frames, reconnects on failure with
capped exponential backoff, and reports health so a supervisor can raise a
camera-offline alert without polling the network itself.

Two decisions shape everything here:

**Decode every frame, enqueue only some.** Analytics rarely needs 25 fps; 6-8
is plenty for perimeter work and costs a third of the compute. But frames must
still be *pulled* from the decoder at source rate, because an unread FFmpeg
buffer grows until the frames it hands back are seconds old. So the reader
decodes at full rate and samples on the way into the queue.

**Drop frames, never latency.** When analytics falls behind, the queue is
overwritten oldest-first. A queue that grows instead would keep every frame at
the cost of alerting on an intrusion a minute after it happened, which is not
an alert - it is a historical record.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ibvap.core.config import CameraConfig, PipelineConfig
from ibvap.core.errors import StreamError
from ibvap.core.logging import get_logger
from ibvap.core.types import Frame
from ibvap.ingest.source import VideoSource, build_source
from ibvap.telemetry.metrics import Metrics

log = get_logger(__name__)


@dataclass
class StreamHealth:
    """Live health snapshot for one camera."""

    camera_id: str
    connected: bool = False
    #: Measured analytics frame rate (frames actually enqueued per second).
    fps: float = 0.0
    #: Frame rate reported by the source.
    source_fps: float = 0.0
    frames_captured: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_monotonic: float = 0.0
    last_error: str = ""
    width: int = 0
    height: int = 0

    @property
    def seconds_since_frame(self) -> float:
        if not self.last_frame_monotonic:
            return float("inf")
        return time.monotonic() - self.last_frame_monotonic

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "connected": self.connected,
            "fps": round(self.fps, 2),
            "source_fps": round(self.source_fps, 2),
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "reconnects": self.reconnects,
            "seconds_since_frame": (
                None if self.seconds_since_frame == float("inf")
                else round(self.seconds_since_frame, 2)
            ),
            "resolution": f"{self.width}x{self.height}" if self.width else "",
            "last_error": self.last_error,
        }


class FrameQueue:
    """A bounded queue with an explicit overflow policy.

    ``queue.Queue`` is not used because its only overflow options are block or
    raise; what live video needs is *overwrite*, and expressing that through
    ``get_nowait`` + retry is both racy and easy to get subtly wrong.
    """

    def __init__(self, maxsize: int = 4, policy: str = "drop_oldest") -> None:
        self.maxsize = max(1, maxsize)
        self.policy = policy
        self._items: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped = 0
        self._closed = False

    def put(self, frame: Frame) -> bool:
        """Offer a frame. Returns False when the frame was dropped."""
        with self._not_empty:
            if self._closed:
                return False
            if len(self._items) >= self.maxsize:
                if self.policy == "drop_newest":
                    self._dropped += 1
                    return False
                if self.policy == "block":
                    while len(self._items) >= self.maxsize and not self._closed:
                        self._not_empty.wait(timeout=1.0)
                    if self._closed:
                        return False
                else:  # drop_oldest - the default, and the right one for live video
                    self._items.popleft()
                    self._dropped += 1
            self._items.append(frame)
            self._not_empty.notify()
            return True

    def get(self, timeout: float = 1.0) -> Frame | None:
        """Take the oldest frame, or ``None`` if none arrives within ``timeout``."""
        with self._not_empty:
            if not self._items and not self._closed:
                self._not_empty.wait(timeout=timeout)
            if not self._items:
                return None
            frame = self._items.popleft()
            self._not_empty.notify()
            return frame

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._items.clear()
            self._not_empty.notify_all()

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def dropped(self) -> int:
        return self._dropped


class StreamReader:
    """Decode thread for one camera, with reconnection and frame sampling."""

    def __init__(
        self,
        camera: CameraConfig,
        pipeline: PipelineConfig | None = None,
        *,
        metrics: Metrics | None = None,
        on_state_change: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline or PipelineConfig()
        self.metrics = metrics
        self.on_state_change = on_state_change

        self.queue = FrameQueue(self.pipeline.queue_size, self.pipeline.overflow_policy)
        self.health = StreamHealth(camera_id=camera.id)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._source: VideoSource | None = None
        self._frame_index = 0
        self._fps_window: deque[float] = deque(maxlen=60)
        self._last_emit = 0.0

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"reader-{self.camera.id}", daemon=True
        )
        self._thread.start()
        log.info("reader_started", camera=self.camera.id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._source:
            self._source.close()
            self._source = None
        self._set_connected(False)
        log.info("reader_stopped", camera=self.camera.id)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- decode loop ------------------------------------------------------- #

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self._source = build_source(
                    self.camera.url,
                    rtsp_transport=self.camera.rtsp_transport,
                    read_timeout_seconds=self.camera.read_timeout_seconds,
                )
                self._source.open()

                info = self._source.info
                self.health.source_fps = info.fps
                self.health.width, self.health.height = info.width, info.height
                self._set_connected(True)
                attempt = 0  # a successful open resets the backoff ladder

                self._read_loop()

            except StreamError as exc:
                self.health.last_error = str(exc)
                log.warning("stream_error", camera=self.camera.id, error=str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                self.health.last_error = f"{type(exc).__name__}: {exc}"
                log.error(
                    "reader_crashed", camera=self.camera.id, error=str(exc), exc_info=True
                )
            finally:
                if self._source:
                    self._source.close()
                    self._source = None
                self._set_connected(False)

            if self._stop.is_set():
                break

            attempt += 1
            self.health.reconnects += 1
            if self.metrics:
                self.metrics.reconnect(self.camera.id)

            delay = self._backoff(attempt)
            log.info(
                "reader_reconnecting",
                camera=self.camera.id, attempt=attempt, delay_seconds=round(delay, 1),
            )
            # Wait on the stop event rather than sleeping, so shutdown is prompt
            # even when the backoff has grown to a minute.
            self._stop.wait(delay)

    def _read_loop(self) -> None:
        assert self._source is not None
        target_interval = 1.0 / max(0.1, self.camera.target_fps)

        while not self._stop.is_set():
            frame_image = self._source.read()
            now = time.monotonic()

            # Sample down to the analytics rate. The frame was decoded either
            # way - that is what keeps the decoder buffer shallow.
            if now - self._last_emit < target_interval:
                if self.metrics:
                    self.metrics.frame_dropped(self.camera.id, "sampled")
                continue
            self._last_emit = now

            frame = self._build_frame(frame_image, now)
            if not self.queue.put(frame):
                self.health.frames_dropped += 1
                if self.metrics:
                    self.metrics.frame_dropped(self.camera.id, "queue_full")

            self.health.frames_captured += 1
            self.health.last_frame_monotonic = now
            self._update_fps(now)
            if self.metrics:
                self.metrics.frame_captured(self.camera.id)
                self.metrics.set_queue_depth(self.camera.id, self.queue.depth)

    def _build_frame(self, image: np.ndarray, monotonic: float) -> Frame:
        from ibvap.vision.preprocess import is_night_frame, resize_max_side

        # Downscale once, here, so every downstream stage shares one cost and
        # one coordinate space. Zones are normalised, so this is safe.
        if self.camera.process_width and image.shape[1] > self.camera.process_width:
            image, _ = resize_max_side(image, self.camera.process_width)

        self._frame_index += 1
        return Frame(
            camera_id=self.camera.id,
            index=self._frame_index,
            image=image,
            timestamp=time.time(),
            monotonic=monotonic,
            fps=self.health.fps or self.camera.target_fps,
            is_night=is_night_frame(image),
        )

    def _update_fps(self, now: float) -> None:
        self._fps_window.append(now)
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            if span > 0:
                self.health.fps = (len(self._fps_window) - 1) / span
                if self.metrics:
                    self.metrics.set_fps(self.camera.id, self.health.fps)

    def _backoff(self, attempt: int) -> float:
        """Capped exponential backoff with jitter.

        Jitter matters at a BOP: when the site switches to generator power, all
        sixteen cameras drop together. Without it they would reconnect in
        lockstep for as long as the outage lasts, hammering the NVR each time.
        """
        base = min(
            self.pipeline.reconnect_max_seconds,
            self.pipeline.reconnect_min_seconds * (2 ** (attempt - 1)),
        )
        return base * (0.5 + random.random() * 0.5)

    def _set_connected(self, connected: bool) -> None:
        if self.health.connected == connected:
            return
        self.health.connected = connected
        if self.metrics:
            self.metrics.set_camera_up(self.camera.id, connected)
        if self.on_state_change:
            try:
                self.on_state_change(self.camera.id, connected)
            except Exception as exc:  # pragma: no cover - callback is external
                log.error("state_callback_failed", camera=self.camera.id, error=str(exc))

    # -- consumption ------------------------------------------------------- #

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Take the next frame for analytics, or ``None`` on timeout."""
        return self.queue.get(timeout=timeout)

    def frames(self, timeout: float = 1.0):
        """Iterate frames until the reader is stopped."""
        while not self._stop.is_set():
            frame = self.queue.get(timeout=timeout)
            if frame is not None:
                yield frame
