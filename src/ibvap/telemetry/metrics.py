"""Prometheus instrumentation.

The metric set is chosen so that one Grafana board answers the three questions
a control room actually asks: *are my cameras up*, *is analytics keeping up
with the streams*, and *are alerts reaching the C2 system*.

All metrics carry ``site`` so a sector-level Prometheus can scrape many BOP
nodes into one view without label collisions.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

#: A dedicated registry keeps IBVAP metrics separate from library defaults and
#: lets tests build an isolated instance without global state bleeding across.
REGISTRY = CollectorRegistry(auto_describe=True)

# Latency buckets tuned for video analytics: anything past ~2 s is a failure
# to act on, so fine resolution below that and a coarse tail above.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 5.0)

# --- Ingest ---------------------------------------------------------------- #

frames_captured = Counter(
    "ibvap_frames_captured_total",
    "Frames successfully decoded from a camera stream.",
    ["site", "camera"],
    registry=REGISTRY,
)
frames_processed = Counter(
    "ibvap_frames_processed_total",
    "Frames that completed the full analytics pipeline.",
    ["site", "camera"],
    registry=REGISTRY,
)
frames_dropped = Counter(
    "ibvap_frames_dropped_total",
    "Frames discarded before analytics, by reason.",
    ["site", "camera", "reason"],
    registry=REGISTRY,
)
stream_reconnects = Counter(
    "ibvap_stream_reconnects_total",
    "Reconnection attempts against a camera stream.",
    ["site", "camera"],
    registry=REGISTRY,
)
camera_up = Gauge(
    "ibvap_camera_up",
    "1 when the camera is delivering frames, 0 when it is offline.",
    ["site", "camera"],
    registry=REGISTRY,
)
camera_fps = Gauge(
    "ibvap_camera_fps",
    "Analytics throughput for a camera, frames per second.",
    ["site", "camera"],
    registry=REGISTRY,
)
queue_depth = Gauge(
    "ibvap_queue_depth",
    "Frames waiting in a camera's bounded analytics queue.",
    ["site", "camera"],
    registry=REGISTRY,
)

# --- Inference ------------------------------------------------------------- #

inference_latency = Histogram(
    "ibvap_inference_latency_seconds",
    "Wall-clock time of one model forward pass.",
    ["site", "model"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
pipeline_latency = Histogram(
    "ibvap_pipeline_latency_seconds",
    "Capture-to-event latency for a frame through the whole pipeline.",
    ["site", "camera"],
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
inference_errors = Counter(
    "ibvap_inference_errors_total",
    "Model invocations that raised.",
    ["site", "model"],
    registry=REGISTRY,
)
detections_total = Counter(
    "ibvap_detections_total",
    "Objects detected, by class.",
    ["site", "camera", "class"],
    registry=REGISTRY,
)
active_tracks = Gauge(
    "ibvap_active_tracks",
    "Confirmed tracks currently held by a camera's tracker.",
    ["site", "camera"],
    registry=REGISTRY,
)

# --- Events and delivery --------------------------------------------------- #

events_total = Counter(
    "ibvap_events_total",
    "Analytics events raised, by type and severity.",
    ["site", "camera", "type", "severity"],
    registry=REGISTRY,
)
events_suppressed = Counter(
    "ibvap_events_suppressed_total",
    "Events withheld by deduplication, cooldown or rate limiting.",
    ["site", "camera", "reason"],
    registry=REGISTRY,
)
delivery_attempts = Counter(
    "ibvap_delivery_attempts_total",
    "Outbound C2 delivery attempts, by sink and outcome.",
    ["site", "sink", "outcome"],
    registry=REGISTRY,
)
delivery_latency = Histogram(
    "ibvap_delivery_latency_seconds",
    "Time to deliver one event to a C2 sink.",
    ["site", "sink"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)
outbox_depth = Gauge(
    "ibvap_outbox_depth",
    "Events queued in the store-and-forward outbox awaiting delivery.",
    ["site", "sink"],
    registry=REGISTRY,
)

# --- Recognition ----------------------------------------------------------- #

plate_reads = Counter(
    "ibvap_plate_reads_total",
    "ANPR reads, by outcome (accepted / rejected_format / low_confidence).",
    ["site", "camera", "outcome"],
    registry=REGISTRY,
)
face_matches = Counter(
    "ibvap_face_matches_total",
    "Face gallery comparisons, by outcome (match / no_match).",
    ["site", "camera", "outcome"],
    registry=REGISTRY,
)

# --- Build info ------------------------------------------------------------ #

build_info = Gauge(
    "ibvap_build_info",
    "Build and model metadata; value is always 1.",
    ["site", "version", "tier"],
    registry=REGISTRY,
)
model_info = Gauge(
    "ibvap_model_info",
    "Loaded model versions; value is always 1.",
    ["site", "role", "name", "version", "backend"],
    registry=REGISTRY,
)


class Metrics:
    """Thin façade binding the ``site`` label once, at construction.

    Passing the site label at every call site is noisy and easy to get wrong;
    binding it here keeps instrumentation to a single readable line.
    """

    def __init__(self, site_id: str, version: str = "") -> None:
        self.site = site_id
        if version:
            build_info.labels(site=site_id, version=version, tier="").set(1)

    # -- ingest ---------------------------------------------------------- #
    def frame_captured(self, camera: str) -> None:
        frames_captured.labels(site=self.site, camera=camera).inc()

    def frame_processed(self, camera: str) -> None:
        frames_processed.labels(site=self.site, camera=camera).inc()

    def frame_dropped(self, camera: str, reason: str) -> None:
        frames_dropped.labels(site=self.site, camera=camera, reason=reason).inc()

    def reconnect(self, camera: str) -> None:
        stream_reconnects.labels(site=self.site, camera=camera).inc()

    def set_camera_up(self, camera: str, up: bool) -> None:
        camera_up.labels(site=self.site, camera=camera).set(1 if up else 0)

    def set_fps(self, camera: str, fps: float) -> None:
        camera_fps.labels(site=self.site, camera=camera).set(fps)

    def set_queue_depth(self, camera: str, depth: int) -> None:
        queue_depth.labels(site=self.site, camera=camera).set(depth)

    # -- inference ------------------------------------------------------- #
    @contextmanager
    def time_inference(self, model: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception:
            inference_errors.labels(site=self.site, model=model).inc()
            raise
        finally:
            inference_latency.labels(site=self.site, model=model).observe(time.perf_counter() - start)

    def observe_pipeline_latency(self, camera: str, seconds: float) -> None:
        pipeline_latency.labels(site=self.site, camera=camera).observe(seconds)

    def detection(self, camera: str, obj_class: str, count: int = 1) -> None:
        detections_total.labels(site=self.site, camera=camera, **{"class": obj_class}).inc(count)

    def set_active_tracks(self, camera: str, count: int) -> None:
        active_tracks.labels(site=self.site, camera=camera).set(count)

    def set_model_info(self, role: str, name: str, version: str, backend: str) -> None:
        model_info.labels(
            site=self.site, role=role, name=name, version=version, backend=backend
        ).set(1)

    # -- events ---------------------------------------------------------- #
    def event(self, camera: str, event_type: str, severity: str) -> None:
        events_total.labels(site=self.site, camera=camera, type=event_type, severity=severity).inc()

    def event_suppressed(self, camera: str, reason: str) -> None:
        events_suppressed.labels(site=self.site, camera=camera, reason=reason).inc()

    def delivery(self, sink: str, outcome: str, seconds: float | None = None) -> None:
        delivery_attempts.labels(site=self.site, sink=sink, outcome=outcome).inc()
        if seconds is not None:
            delivery_latency.labels(site=self.site, sink=sink).observe(seconds)

    def set_outbox_depth(self, sink: str, depth: int) -> None:
        outbox_depth.labels(site=self.site, sink=sink).set(depth)

    # -- recognition ----------------------------------------------------- #
    def plate_read(self, camera: str, outcome: str) -> None:
        plate_reads.labels(site=self.site, camera=camera, outcome=outcome).inc()

    def face_match(self, camera: str, outcome: str) -> None:
        face_matches.labels(site=self.site, camera=camera, outcome=outcome).inc()


def render_metrics() -> tuple[bytes, str]:
    """Render the registry in Prometheus text exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


_METRICS: Metrics | None = None


def get_metrics(site_id: str = "unknown", version: str = "") -> Metrics:
    """Return the process-wide metrics façade, creating it on first call."""
    global _METRICS
    if _METRICS is None or (site_id != "unknown" and _METRICS.site != site_id):
        _METRICS = Metrics(site_id, version)
    return _METRICS


def _reset_for_tests() -> None:
    """Drop the cached façade so tests can rebind the site label."""
    global _METRICS
    _METRICS = None


__all__: list[str] = [
    "Metrics",
    "REGISTRY",
    "get_metrics",
    "render_metrics",
    "CONTENT_TYPE_LATEST",
    *[n for n in dir() if not n.startswith("_")],
]
