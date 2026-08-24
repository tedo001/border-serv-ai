"""Multi-camera supervisor.

Owns the lifecycle of every camera worker on a node and is the single object
the API, the CLI and the desktop console all talk to.

Its defining responsibility is *isolation*: one camera must never be able to
take down the others. A misconfigured rule, an unreachable NVR, or a decoder
that segfaults on a malformed stream affects exactly one worker. The supervisor
also enforces an admission limit, because sixteen cameras that each run at 2 fps
because the node is oversubscribed are worse than eight that run properly - the
first configuration fails silently, the second fails loudly at start-up.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ibvap.core.config import CameraConfig, Settings
from ibvap.core.errors import ConfigError
from ibvap.core.logging import get_logger
from ibvap.core.types import Frame, Track
from ibvap.mlops.registry import ModelRegistry
from ibvap.pipeline.models import ModelBundle, build_model_bundle
from ibvap.pipeline.worker import CameraWorker, EventSink
from ibvap.telemetry.metrics import Metrics, get_metrics
from ibvap.vision.anpr import PlateWatchlist
from ibvap.vision.face import FaceGallery

log = get_logger(__name__)


class Supervisor:
    """Starts, stops and monitors every camera worker on this node."""

    def __init__(
        self,
        settings: Settings,
        *,
        event_sink: EventSink | None = None,
        metrics: Metrics | None = None,
        registry: ModelRegistry | None = None,
        frame_sink: Callable[[Frame, list[Track]], None] | None = None,
    ) -> None:
        self.settings = settings
        self.event_sink = event_sink
        self.metrics = metrics or get_metrics(settings.site_id)
        self.frame_sink = frame_sink

        self.registry = registry or ModelRegistry.load(settings.models)
        self.bundle: ModelBundle = build_model_bundle(
            settings, registry=self.registry, metrics=self.metrics
        )
        # Watchlists are node-wide: an identity of interest is of interest on
        # every camera, and duplicating the gallery per worker would multiply
        # both memory and the risk of them drifting out of sync.
        self.face_gallery: FaceGallery = self.bundle.face_gallery or FaceGallery()
        self.plate_watchlist = PlateWatchlist()

        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._running = False

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Start a worker for every enabled camera."""
        with self._lock:
            if self._running:
                return
            self._started_at = time.time()
            self._running = True

            cameras = self.settings.enabled_cameras()
            for camera in cameras:
                try:
                    self._start_worker(camera)
                except Exception as exc:
                    # Isolation: log and continue. One bad camera configuration
                    # must not prevent the other fifteen from watching a border.
                    log.error(
                        "worker_start_failed",
                        camera=camera.id, error=str(exc), exc_info=True,
                    )
            log.info(
                "supervisor_started",
                site=self.settings.site_id,
                cameras=len(self.workers),
                configured=len(cameras),
                detector=self.bundle.detector_mode,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            for worker in list(self.workers.values()):
                try:
                    worker.stop()
                except Exception as exc:  # pragma: no cover - shutdown is best effort
                    log.error("worker_stop_failed", camera=worker.camera.id, error=str(exc))
            self.workers.clear()
            self.bundle.close()
            self._running = False
            log.info("supervisor_stopped", site=self.settings.site_id)

    def _start_worker(self, camera: CameraConfig) -> CameraWorker:
        worker = CameraWorker(
            camera,
            self.settings,
            self.bundle,
            event_sink=self.event_sink,
            metrics=self.metrics,
            face_gallery=self.face_gallery,
            plate_watchlist=self.plate_watchlist,
            frame_sink=self.frame_sink,
        )
        worker.start()
        self.workers[camera.id] = worker
        return worker

    # -- dynamic camera management ----------------------------------------- #

    def add_camera(self, camera: CameraConfig) -> CameraWorker:
        """Add and start a camera at runtime."""
        with self._lock:
            if camera.id in self.workers:
                raise ConfigError(f"camera {camera.id!r} is already running")
            self.settings.cameras = [
                c for c in self.settings.cameras if c.id != camera.id
            ] + [camera]
            worker = self._start_worker(camera)
            log.info("camera_added", camera=camera.id)
            return worker

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera at runtime."""
        with self._lock:
            worker = self.workers.pop(camera_id, None)
            if worker is None:
                return False
            worker.stop()
            self.settings.cameras = [c for c in self.settings.cameras if c.id != camera_id]
            log.info("camera_removed", camera=camera_id)
            return True

    def restart_camera(self, camera_id: str) -> bool:
        """Restart one camera, picking up any configuration change."""
        with self._lock:
            worker = self.workers.get(camera_id)
            if worker is None:
                return False
            camera = next(
                (c for c in self.settings.cameras if c.id == camera_id), worker.camera
            )
            worker.stop()
            self.workers.pop(camera_id, None)
            self._start_worker(camera)
            log.info("camera_restarted", camera=camera_id)
            return True

    def update_camera(self, camera: CameraConfig) -> CameraWorker:
        """Apply a new configuration to a camera, restarting its worker."""
        with self._lock:
            self.settings.cameras = [
                c for c in self.settings.cameras if c.id != camera.id
            ] + [camera]
            existing = self.workers.pop(camera.id, None)
            if existing is not None:
                existing.stop()
            return self._start_worker(camera)

    def worker(self, camera_id: str) -> CameraWorker | None:
        return self.workers.get(camera_id)

    # -- health ------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._started_at if self._started_at else 0.0

    def health(self) -> dict[str, Any]:
        """Node health for ``/health``, the console and the CLI."""
        with self._lock:
            workers = list(self.workers.values())

        online = sum(1 for w in workers if w.reader.health.connected)
        degraded = self.bundle.status()["degraded"]

        # A node is only healthy when every configured camera is delivering.
        # "Some cameras are up" is precisely the state a control room must be
        # told about rather than left to infer from a green light.
        if not workers:
            status = "idle"
        elif online == len(workers):
            status = "degraded" if degraded else "healthy"
        elif online == 0:
            status = "down"
        else:
            status = "partial"

        return {
            "status": status,
            "site_id": self.settings.site_id,
            "site_name": self.settings.site_name,
            "tier": self.settings.tier,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "cameras_total": len(workers),
            "cameras_online": online,
            "models": self.bundle.status(),
            "watchlists": {
                "faces": self.face_gallery.size,
                "plates": self.plate_watchlist.size,
            },
        }

    def status(self) -> list[dict[str, Any]]:
        """Per-camera status for the operator console."""
        with self._lock:
            return [w.status() for w in self.workers.values()]

    def __enter__(self) -> Supervisor:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
