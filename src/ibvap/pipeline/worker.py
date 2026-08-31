"""Per-camera analytics worker.

One worker owns one camera end to end: it pulls sampled frames from that
camera's :class:`~ibvap.ingest.reader.StreamReader`, runs detection, tracking,
recognition and rules, and hands finished events to a sink.

Two throughput decisions carry most of the cost/benefit:

**Detection is strided; tracking is not.** Running the detector on every frame
is the single largest compute cost, and it is largely wasted: object motion
between consecutive frames at 8 fps is small and the Kalman filter predicts it
well. So the detector runs every ``detect_interval`` frames and the tracker
coasts in between. A forced re-detection every ``max_track_only_frames`` bounds
how long the system can drift on prediction alone.

**Recognition is opportunistic and once-per-track.** ANPR and face recognition
are expensive and only need to succeed once per object. A vehicle that has
already yielded a confident plate is not read again, which turns an
every-frame cost into an every-vehicle cost.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ibvap.analytics.engine import AnalyticsEngine
from ibvap.core.config import CameraConfig, Settings
from ibvap.core.logging import bind_context, get_logger
from ibvap.core.types import (
    Detection,
    Event,
    EventType,
    Frame,
    ObjectCategory,
    Severity,
    Track,
)
from ibvap.ingest.reader import StreamReader
from ibvap.pipeline.models import ModelBundle, build_camera_detector
from ibvap.telemetry.metrics import Metrics
from ibvap.vision.anpr import PlateWatchlist
from ibvap.vision.face import FaceGallery
from ibvap.vision.preprocess import enhance_low_light
from ibvap.vision.tracker import ByteTracker

log = get_logger(__name__)

EventSink = Callable[[Event], None]


@dataclass
class WorkerStats:
    """Runtime counters for one camera worker."""

    frames_processed: int = 0
    detections: int = 0
    events: int = 0
    plate_reads: int = 0
    face_matches: int = 0
    detector_invocations: int = 0
    #: Tracks whose class the secondary classifier corrected.
    reclassified: int = 0
    #: Exponential moving average of end-to-end processing latency, seconds.
    avg_latency: float = 0.0
    last_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "events": self.events,
            "plate_reads": self.plate_reads,
            "face_matches": self.face_matches,
            "detector_invocations": self.detector_invocations,
            "reclassified": self.reclassified,
            "avg_latency_ms": round(self.avg_latency * 1000, 2),
            "last_error": self.last_error,
        }


class CameraWorker:
    """Runs the full analytics pipeline for a single camera."""

    def __init__(
        self,
        camera: CameraConfig,
        settings: Settings,
        bundle: ModelBundle,
        *,
        event_sink: EventSink | None = None,
        metrics: Metrics | None = None,
        face_gallery: FaceGallery | None = None,
        plate_watchlist: PlateWatchlist | None = None,
        frame_sink: Callable[[Frame, list[Track]], None] | None = None,
    ) -> None:
        self.camera = camera
        self.settings = settings
        self.bundle = bundle
        self.event_sink = event_sink
        self.metrics = metrics
        self.face_gallery = face_gallery or bundle.face_gallery
        self.plate_watchlist = plate_watchlist or PlateWatchlist()
        self.frame_sink = frame_sink

        self.detector = build_camera_detector(bundle, camera)
        self.tracker = ByteTracker(
            high_threshold=settings.tracker.high_threshold,
            low_threshold=settings.tracker.low_threshold,
            match_iou=settings.tracker.match_iou,
            min_hits=settings.tracker.min_hits,
            max_age=settings.tracker.max_age,
            trail_length=settings.tracker.trail_length,
        )
        self.analytics = AnalyticsEngine(camera, settings.analytics, metrics=metrics)
        self.reader = StreamReader(
            camera, settings.pipeline, metrics=metrics,
            on_state_change=self._on_stream_state,
        )

        self.stats = WorkerStats()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frames_since_detect = 0
        #: Most recent annotated state, for live preview without re-running work.
        self._latest: tuple[Frame, list[Track]] | None = None
        self._latest_lock = threading.Lock()
        self._offline_reported = False

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.reader.start()
        self._thread = threading.Thread(
            target=self._run, name=f"worker-{self.camera.id}", daemon=True
        )
        self._thread.start()
        log.info("worker_started", camera=self.camera.id, detector=self.bundle.detector_mode)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.reader.stop(timeout=timeout)
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        log.info("worker_stopped", camera=self.camera.id, **self.stats.as_dict())

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- main loop --------------------------------------------------------- #

    def _run(self) -> None:
        bind_context(camera_id=self.camera.id)
        while not self._stop.is_set():
            frame = self.reader.read(timeout=1.0)
            if frame is None:
                self._check_offline()
                continue
            try:
                self.process_frame(frame)
            except Exception as exc:  # pragma: no cover - defensive
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                log.error(
                    "frame_processing_failed",
                    camera=self.camera.id, frame=frame.index, error=str(exc), exc_info=True,
                )

    def process_frame(self, frame: Frame) -> list[Event]:
        """Run the full pipeline for one frame. Also the unit-test entry point."""
        started = time.perf_counter()

        if self.camera.night_enhancement and frame.is_night:
            # Enhance a copy for inference but keep the original for evidence:
            # an operator reviewing a snapshot should see what the camera saw,
            # not a contrast-stretched interpretation of it.
            inference_image = enhance_low_light(frame.image)
        else:
            inference_image = frame.image

        detections = self._detect(inference_image)
        # Called for its side effect on tracker state; only confirmed tracks
        # are trustworthy enough to drive alerts, so those are read back.
        self.tracker.update(detections, timestamp=frame.timestamp, monotonic=frame.monotonic)
        confirmed = self.tracker.confirmed_tracks

        events: list[Event] = []
        events.extend(self._recognise(frame, confirmed))
        events.extend(self.analytics.process(frame, confirmed))

        self.stats.frames_processed += 1
        self.stats.detections += len(detections)
        self.stats.events += len(events)

        elapsed = time.perf_counter() - started
        # EMA rather than a running mean: an operator cares about latency *now*,
        # not the average since the node booted a fortnight ago.
        self.stats.avg_latency = (
            elapsed if self.stats.avg_latency == 0.0
            else 0.9 * self.stats.avg_latency + 0.1 * elapsed
        )

        if self.metrics:
            self.metrics.frame_processed(self.camera.id)
            self.metrics.observe_pipeline_latency(self.camera.id, elapsed)
            self.metrics.set_active_tracks(self.camera.id, len(confirmed))
            for detection in detections:
                self.metrics.detection(self.camera.id, detection.obj_class.value)

        with self._latest_lock:
            self._latest = (frame, confirmed)
        if self.frame_sink:
            try:
                self.frame_sink(frame, confirmed)
            except Exception as exc:  # pragma: no cover - callback is external
                log.debug("frame_sink_failed", error=str(exc))

        for event in events:
            self._emit(event)
        return events

    def _detect(self, image: np.ndarray) -> list[Detection]:
        """Run the detector, honouring the stride."""
        if self.detector is None:
            return []

        interval = max(1, self.camera.detect_interval)
        force = self._frames_since_detect >= self.settings.pipeline.max_track_only_frames
        if self._frames_since_detect % interval != 0 and not force:
            self._frames_since_detect += 1
            return []

        self._frames_since_detect = 1 if not force else 0
        self.stats.detector_invocations += 1

        if self.metrics:
            with self.metrics.time_inference("detector"):
                return self.detector.detect(image)
        return self.detector.detect(image)

    # -- recognition ------------------------------------------------------- #

    def _recognise(self, frame: Frame, tracks: list[Track]) -> list[Event]:
        """Run ANPR and face recognition opportunistically on eligible tracks."""
        events: list[Event] = []
        # Refine coarse classes first, so ANPR, face and every analytics rule
        # downstream sees the corrected class rather than the detector's guess.
        if self.camera.classify_objects and self.bundle.classifier_available:
            self._classify(frame, tracks)
        if self.camera.anpr_enabled and self.bundle.anpr_available:
            events.extend(self._run_anpr(frame, tracks))
        if self.camera.face_enabled and self.bundle.face_available:
            events.extend(self._run_face(frame, tracks))
        return events

    def _classify(self, frame: Frame, tracks: list[Track]) -> None:
        """Refine each new track's class with the ImageNet classifier.

        Once per track, not once per frame: the verdict cannot change between
        consecutive frames of the same object.
        """
        assert self.bundle.classifier is not None
        for track in tracks:
            if self.bundle.classifier.refine_track(frame.image, track):
                self.stats.reclassified += 1

    def _run_anpr(self, frame: Frame, tracks: list[Track]) -> list[Event]:
        assert self.bundle.plate_reader is not None
        events: list[Event] = []
        min_interval = 5  # frames between retries on the same vehicle

        for track in tracks:
            if track.category is not ObjectCategory.VEHICLE:
                continue
            state = track.attributes.setdefault("_anpr", {"done": False, "last_frame": -99})
            if state["done"] or frame.index - state["last_frame"] < min_interval:
                continue
            state["last_frame"] = frame.index

            result = self.bundle.plate_reader.read_vehicle(frame.image, track.bbox)
            if result.reading is None:
                if self.metrics:
                    self.metrics.plate_read(self.camera.id, result.reason or "no_read")
                continue

            reading = result.reading
            # Stop retrying once a confident, grammatical read is in hand.
            if reading.is_actionable:
                state["done"] = True
            track.attributes["plate"] = reading.text
            track.attributes["plate_confidence"] = reading.confidence
            self.stats.plate_reads += 1
            if self.metrics:
                self.metrics.plate_read(
                    self.camera.id, "accepted" if reading.valid else "rejected_format"
                )

            events.append(Event(
                camera_id=self.camera.id,
                event_type=EventType.PLATE_READ,
                severity=Severity.INFO,
                timestamp=frame.timestamp,
                confidence=reading.confidence,
                message=f"plate read: {reading.text}",
                rule_id="anpr",
                track_ids=[track.track_id],
                boxes=[reading.bbox or track.bbox],
                frame_index=frame.index,
                attributes={
                    "plate": reading.text,
                    "raw_text": reading.raw_text,
                    "plate_format": reading.plate_format,
                    "state_code": reading.state_code,
                    "corrections": reading.corrections,
                    "valid": reading.valid,
                    "vehicle_class": track.obj_class.value,
                },
            ))

            hit = self.plate_watchlist.check(reading)
            if hit is not None:
                events.append(Event(
                    camera_id=self.camera.id,
                    event_type=EventType.PLATE_MATCH,
                    severity=Severity.CRITICAL,
                    timestamp=frame.timestamp,
                    confidence=reading.confidence,
                    message=(
                        f"WATCHLIST VEHICLE {hit.entry.plate} "
                        f"({hit.entry.category})"
                    ),
                    rule_id="anpr_watchlist",
                    track_ids=[track.track_id],
                    boxes=[reading.bbox or track.bbox],
                    frame_index=frame.index,
                    attributes={
                        "plate": hit.entry.plate,
                        "category": hit.entry.category,
                        "reason": hit.entry.reason,
                        "reference": hit.entry.reference,
                        "exact_match": hit.exact,
                        "edit_distance": hit.distance,
                    },
                ))
        return events

    def _run_face(self, frame: Frame, tracks: list[Track]) -> list[Event]:
        assert self.bundle.face_detector is not None
        assert self.bundle.face_embedder is not None
        if self.face_gallery is None or self.face_gallery.size == 0:
            return []

        events: list[Event] = []
        min_interval = 8

        for track in tracks:
            if track.category is not ObjectCategory.HUMAN:
                continue
            state = track.attributes.setdefault("_face", {"done": False, "last_frame": -99})
            if state["done"] or frame.index - state["last_frame"] < min_interval:
                continue
            state["last_frame"] = frame.index

            faces = self.bundle.face_detector.detect_in_person(frame.image, track.bbox)
            if not faces:
                continue

            # Best available face only. Recognising the same person from a worse
            # crop cannot improve on the best one and only adds false-match risk.
            face = max(faces, key=lambda f: f.pixel_height * max(f.focus, 1.0))
            if not face.quality_ok(
                min_height=self.bundle.face_detector.min_height,
                min_focus=self.bundle.face_detector.min_focus,
            ):
                continue

            from ibvap.vision.preprocess import crop

            face_crop = crop(frame.image, face.bbox, padding=0.15)
            if face_crop.size == 0:
                continue
            embedding = self.bundle.face_embedder.embed(face_crop, face.landmarks)
            if embedding.size == 0:
                continue

            match = self.face_gallery.match(embedding)
            if match is None:
                continue
            if self.metrics:
                self.metrics.face_match(
                    self.camera.id, "match" if match.matched else "no_match"
                )
            if not match.matched:
                continue

            state["done"] = True
            track.attributes["face_match"] = match.person_id
            self.stats.face_matches += 1
            events.append(Event(
                camera_id=self.camera.id,
                event_type=EventType.FACE_MATCH,
                severity=Severity.CRITICAL,
                timestamp=frame.timestamp,
                confidence=match.similarity,
                message=f"WATCHLIST FACE: {match.name} ({match.category})",
                rule_id="face_watchlist",
                track_ids=[track.track_id],
                boxes=[face.bbox],
                frame_index=frame.index,
                attributes={
                    "person_id": match.person_id,
                    "name": match.name,
                    "category": match.category,
                    "similarity": match.similarity,
                    "margin": match.margin,
                    "face_height_px": round(face.pixel_height, 1),
                },
            ))
        return events

    # -- stream state ------------------------------------------------------ #

    def _on_stream_state(self, camera_id: str, connected: bool) -> None:
        event_type = EventType.CAMERA_ONLINE if connected else EventType.CAMERA_OFFLINE
        if connected:
            self._offline_reported = False
            # A reconnect is a discontinuity: track ids and the motion
            # background model both refer to a stream that no longer exists.
            self.tracker.reset()
            if self.detector is not None and not self.detector.is_neural:
                getattr(self.detector, "reset", lambda: None)()

        self._emit(Event(
            camera_id=camera_id,
            event_type=event_type,
            severity=event_type.default_severity,
            timestamp=time.time(),
            message=f"camera {camera_id} {'online' if connected else 'offline'}",
            rule_id="stream_health",
            attributes={"reconnects": self.reader.health.reconnects},
        ))

    def _check_offline(self) -> None:
        """Raise a camera-offline event when frames stop arriving."""
        if self._offline_reported or not self.reader.health.connected:
            return
        if self.reader.health.seconds_since_frame < self.settings.pipeline.offline_after_seconds:
            return
        self._offline_reported = True
        self._emit(Event(
            camera_id=self.camera.id,
            event_type=EventType.CAMERA_OFFLINE,
            severity=Severity.MEDIUM,
            timestamp=time.time(),
            message=(
                f"camera {self.camera.id} stalled: no frame for "
                f"{self.reader.health.seconds_since_frame:.0f}s"
            ),
            rule_id="stream_health",
            attributes={"stalled": True},
        ))

    def _emit(self, event: Event) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(event)
        except Exception as exc:  # pragma: no cover - sink is external
            log.error("event_sink_failed", camera=self.camera.id, error=str(exc))

    # -- introspection ----------------------------------------------------- #

    def latest(self) -> tuple[Frame, list[Track]] | None:
        """Most recent frame and its confirmed tracks, for live preview."""
        with self._latest_lock:
            return self._latest

    def status(self) -> dict[str, Any]:
        return {
            "camera": {
                "id": self.camera.id,
                "name": self.camera.name,
                "location": self.camera.location,
                "enabled": self.camera.enabled,
            },
            "running": self.is_running,
            "stream": self.reader.health.as_dict(),
            "stats": self.stats.as_dict(),
            "analytics": self.analytics.status(),
            "detector_mode": self.detector.mode if self.detector else "unavailable",
            "anpr_enabled": self.camera.anpr_enabled and self.bundle.anpr_available,
            "face_enabled": self.camera.face_enabled and self.bundle.face_available,
        }
