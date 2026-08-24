"""Latency and throughput benchmarking.

Answers the question that actually decides a deployment: *how many cameras can
this box carry?* A published mAP figure does not, because the binding
constraint at a BOP is almost always compute, not accuracy.

Reports percentiles rather than a mean. Video analytics is a soft-real-time
workload; a mean of 40 ms hides a p99 of 400 ms, and it is the p99 that decides
whether frames start queueing and alerts start arriving late.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ibvap.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Timing statistics for one benchmarked component."""

    name: str
    iterations: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def throughput_fps(self) -> float:
        """Sustained single-stream throughput implied by median latency."""
        return 1000.0 / self.p50_ms if self.p50_ms > 0 else 0.0

    def cameras_supported(self, target_fps: float = 8.0, headroom: float = 0.7) -> float:
        """How many cameras this component can carry at ``target_fps``.

        Sized on p95, not the median, and with headroom left over. A node
        planned to 100% of its median capacity has no margin for the frames
        that take longer than usual - and those arrive precisely when a scene
        gets busy, which is when the analytics matter most.
        """
        if self.p95_ms <= 0:
            return 0.0
        per_camera_ms = target_fps * self.p95_ms
        return round((1000.0 * headroom) / per_camera_ms, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "cameras_at_8fps": self.cameras_supported(8.0),
        }


def benchmark(
    name: str,
    operation: Callable[[], Any],
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Time ``operation`` repeatedly and return latency statistics.

    Warm-up iterations are discarded: the first calls pay for lazy kernel
    compilation and memory-arena growth, and including them would understate
    steady-state performance by a wide margin.
    """
    for _ in range(max(0, warmup)):
        operation()

    latencies: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        latencies.append((time.perf_counter() - started) * 1000.0)

    result = BenchmarkResult(name=name, iterations=iterations, latencies_ms=latencies)
    log.info("benchmark_complete", **result.as_dict())
    return result


def benchmark_detector(
    detector: Any,
    *,
    resolution: tuple[int, int] = (1920, 1080),
    iterations: int = 50,
    warmup: int = 5,
    seed: int = 0,
) -> BenchmarkResult:
    """Benchmark a detector against synthetic frames of a given resolution."""
    rng = np.random.default_rng(seed)
    width, height = resolution
    # Structured noise rather than a flat fill: a uniform image can short-
    # circuit some inference paths and would flatter the result.
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    return benchmark(
        f"detector@{width}x{height}",
        lambda: detector.detect(frame),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_pipeline_stages(
    detector: Any,
    tracker: Any,
    *,
    resolution: tuple[int, int] = (1280, 720),
    iterations: int = 50,
) -> dict[str, BenchmarkResult]:
    """Benchmark detection and tracking separately.

    Splitting the stages is what makes the result actionable: if tracking
    dominates, raising ``detect_interval`` will not help, and the answer is a
    shorter trail or a lower resolution instead.
    """
    from ibvap.core.types import BBox, Detection, ObjectClass

    rng = np.random.default_rng(1)
    width, height = resolution
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    detections = [
        Detection(BBox(x, 200, x + 60, 360), ObjectClass.PERSON, 0.9)
        for x in range(100, 100 + 60 * 12, 60)
    ]

    return {
        "detect": benchmark(
            "detect", lambda: detector.detect(frame), iterations=iterations, warmup=5
        ),
        "track": benchmark(
            "track", lambda: tracker.update(detections), iterations=iterations, warmup=5
        ),
    }


def system_profile() -> dict[str, Any]:
    """Describe the machine, so a benchmark can be compared against another."""
    profile: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import os

        profile["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import onnxruntime as ort

        profile["onnxruntime"] = ort.__version__
        profile["providers"] = ort.get_available_providers()
    except Exception:  # pragma: no cover
        pass
    return profile


def format_report(results: dict[str, BenchmarkResult], profile: dict[str, Any] | None = None) -> str:
    """Render a benchmark run as a plain-text report."""
    lines = ["IBVAP benchmark report", "=" * 78, ""]
    if profile:
        lines += [f"{key:<16}{value}" for key, value in profile.items()]
        lines.append("")

    header = f"{'stage':<24}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'fps':>8}{'cams@8fps':>11}"
    lines += [header, "-" * len(header)]
    for result in results.values():
        lines.append(
            f"{result.name:<24}{result.p50_ms:>9.2f}{result.p95_ms:>9.2f}"
            f"{result.p99_ms:>9.2f}{result.throughput_fps:>8.1f}"
            f"{result.cameras_supported(8.0):>11.2f}"
        )
    lines += [
        "",
        "Camera counts assume 8 fps analytics per camera and 30% headroom.",
        "Sized on p95: a node planned to its median capacity has no margin for",
        "the slow frames, which arrive exactly when a scene becomes busy.",
    ]
    return "\n".join(lines)
