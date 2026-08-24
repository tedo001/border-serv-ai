"""Inference backends.

IBVAP standardises on ONNX Runtime rather than a training framework. The reason
is deployment economics: a BOP node is often a fanless mini-PC or an aging
workstation, and shipping a 2 GB PyTorch/CUDA stack to a site reachable only by
road is not viable. ONNX Runtime is ~200 MB, runs the same artefact on CPU,
CUDA, TensorRT, OpenVINO or DirectML, and lets one build serve every tier of
hardware in the deployment.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ibvap.core.errors import ModelError
from ibvap.core.logging import get_logger

log = get_logger(__name__)


class InferenceBackend(ABC):
    """Minimal contract every inference implementation satisfies."""

    #: Human-readable backend identifier used in logs and metrics.
    name: str = "base"

    @abstractmethod
    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Execute one forward pass and return the raw output tensors."""

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        ...

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Declared input shape; dynamic axes are reported as ``-1``."""
        return ()

    def input_size(self) -> tuple[int, int] | None:
        """Static ``(width, height)`` of the input, or ``None`` if dynamic."""
        shape = self.input_shape
        if len(shape) != 4:
            return None
        # NCHW is the overwhelming convention for exported detectors.
        h, w = shape[2], shape[3]
        if h is None or w is None or h < 1 or w < 1:
            return None
        return int(w), int(h)

    def warmup(self, iterations: int = 2) -> None:
        """Run dummy passes so the first real frame is not the slow one.

        Lazy kernel compilation and memory-arena growth make an uninstrumented
        first inference several times slower than steady state. Paying that at
        startup avoids a latency spike exactly when a camera comes online.
        """
        size = self.input_size()
        if not size or not self.input_names:
            return
        w, h = size
        dummy = {self.input_names[0]: np.zeros((1, 3, h, w), dtype=np.float32)}
        for _ in range(max(1, iterations)):
            try:
                self.run(dummy)
            except Exception as exc:  # a failed warmup must not block startup
                log.debug("warmup_failed", backend=self.name, error=str(exc))
                return

    def close(self) -> None:
        """Release native resources. Safe to call more than once."""

    def __enter__(self) -> "InferenceBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OnnxBackend(InferenceBackend):
    """ONNX Runtime session with graceful execution-provider fallback."""

    name = "onnxruntime"

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        graph_optimization: str = "all",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise ModelError("onnxruntime is not installed") from exc

        path = Path(model_path)
        if not path.is_file():
            raise ModelError(f"model artefact not found: {path}")

        available = set(ort.get_available_providers())
        requested = providers or ["CPUExecutionProvider"]
        # Silently skipping an unavailable provider is deliberate: the same
        # site.yaml is deployed to GPU sector nodes and CPU-only BOP nodes, and
        # a CPU node must not fail to start merely for lacking CUDA.
        selected = [p for p in requested if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"]
            log.warning(
                "no_requested_provider_available",
                requested=requested,
                available=sorted(available),
                fallback="CPUExecutionProvider",
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = {
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }.get(graph_optimization, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        if inter_op_threads > 0:
            options.inter_op_num_threads = inter_op_threads
        options.log_severity_level = 3  # warnings and above only

        try:
            self._session = ort.InferenceSession(str(path), options, providers=selected)
        except Exception as exc:
            raise ModelError(f"failed to load {path.name}: {exc}") from exc

        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        raw_shape = self._session.get_inputs()[0].shape if self._input_names else []
        # Dynamic axes surface as strings ('batch', 'height'); normalise to -1.
        self._input_shape = tuple(d if isinstance(d, int) else -1 for d in raw_shape)
        self._providers = list(self._session.get_providers())
        self._path = path
        # ORT sessions are thread-safe for run(), but serialising here keeps
        # per-session memory arenas from ballooning under many camera threads.
        self._lock = threading.Lock()
        self._closed = False

        log.info(
            "onnx_model_loaded",
            model=path.name,
            providers=self._providers,
            input_shape=self._input_shape,
            outputs=len(self._output_names),
        )

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def model_path(self) -> Path:
        return self._path

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._closed:
            raise ModelError(f"backend for {self._path.name} is closed")
        with self._lock:
            try:
                return list(self._session.run(self._output_names, feeds))
            except Exception as exc:
                raise ModelError(f"inference failed for {self._path.name}: {exc}") from exc

    def close(self) -> None:
        self._closed = True
        self._session = None  # type: ignore[assignment]


class CallableBackend(InferenceBackend):
    """Wraps a plain callable as a backend.

    Used by tests and by the evaluation harness to inject recorded or synthetic
    model outputs without producing an ONNX file.
    """

    name = "callable"

    def __init__(
        self,
        fn: Any,
        *,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        name: str = "callable",
    ) -> None:
        self._fn = fn
        self._input_names = input_names or ["images"]
        self._output_names = output_names or ["output0"]
        self._input_shape = input_shape
        self.name = name

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        result = self._fn(feeds)
        if isinstance(result, np.ndarray):
            return [result]
        return list(result)


def load_onnx_backend(
    model_path: str | Path,
    *,
    providers: list[str] | None = None,
    intra_op_threads: int = 0,
    warmup: bool = True,
) -> OnnxBackend:
    """Construct an :class:`OnnxBackend` and optionally warm it up."""
    backend = OnnxBackend(
        model_path, providers=providers, intra_op_threads=intra_op_threads
    )
    if warmup:
        backend.warmup()
    return backend
