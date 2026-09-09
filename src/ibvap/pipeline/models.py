"""Model bundle construction.

Builds the concrete vision components a camera worker needs from configuration
plus the registry, and - crucially - decides what to do when an artefact is
missing. That decision is centralised here so every subsystem degrades the same
way and reports the same thing.

The degradation ladder, best to worst:

1. Neural detector from a verified registry artefact.
2. Classical motion detection (background subtraction + shape heuristics).
3. Nothing - only if fallback is explicitly disabled in configuration.

A node in state 2 is genuinely less capable, so it says so: ``mode`` is carried
into ``/health``, the operator console and the ``ibvap_model_info`` metric. A
control room must never be left believing it has face recognition when the
node has no embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ibvap.core.config import CameraConfig, ModelsConfig, Settings
from ibvap.core.logging import get_logger
from ibvap.mlops.registry import ModelRegistry
from ibvap.telemetry.metrics import Metrics
from ibvap.vision.anpr import PlateLocator, PlateOCR, PlateReader
from ibvap.vision.classifier import ImageNetClassifier
from ibvap.vision.detector import BaseDetector, MotionDetector, ObjectDetector
from ibvap.vision.face import FaceDetector, FaceEmbedder, FaceGallery

log = get_logger(__name__)


@dataclass
class ModelBundle:
    """The vision components shared by every camera worker on a node.

    Shared rather than per-camera because ONNX Runtime sessions are the single
    largest memory consumer on an edge node - one detector session serving
    sixteen cameras instead of sixteen sessions is the difference between
    fitting in 8 GB and not. The one exception is the motion detector, whose
    background model is inherently per-view.
    """

    detector: BaseDetector | None = None
    face_detector: FaceDetector | None = None
    face_embedder: FaceEmbedder | None = None
    face_gallery: FaceGallery | None = None
    plate_reader: PlateReader | None = None
    classifier: ImageNetClassifier | None = None
    #: ``role -> {name, version, layout, mode}`` for health reporting.
    loaded: dict[str, dict[str, Any]] = field(default_factory=dict)
    registry: ModelRegistry | None = None

    @property
    def detector_mode(self) -> str:
        return self.detector.mode if self.detector else "unavailable"

    @property
    def anpr_available(self) -> bool:
        return bool(self.plate_reader and self.plate_reader.available)

    @property
    def classifier_available(self) -> bool:
        return bool(self.classifier and self.classifier.available)

    @property
    def face_available(self) -> bool:
        return bool(
            self.face_detector
            and self.face_detector.mode != "unavailable"
            and self.face_embedder
            and self.face_embedder.available
        )

    def status(self) -> dict[str, Any]:
        """Capability summary for ``/health`` and the operator console."""
        return {
            "detector_mode": self.detector_mode,
            "detector_is_neural": bool(self.detector and self.detector.is_neural),
            "anpr_available": self.anpr_available,
            "classifier_available": self.classifier_available,
            "face_available": self.face_available,
            "face_gallery_size": self.face_gallery.size if self.face_gallery else 0,
            "models": self.loaded,
            "degraded": not (self.detector and self.detector.is_neural),
        }

    def close(self) -> None:
        if self.detector:
            self.detector.close()
        if self.registry:
            self.registry.close()


def build_model_bundle(
    settings: Settings,
    *,
    registry: ModelRegistry | None = None,
    metrics: Metrics | None = None,
) -> ModelBundle:
    """Construct the node's shared model bundle from configuration."""
    models: ModelsConfig = settings.models
    registry = registry or ModelRegistry.load(models)
    bundle = ModelBundle(registry=registry)

    def record(role: str, name: str, version: str, layout: str, mode: str) -> None:
        bundle.loaded[role] = {
            "name": name, "version": version, "layout": layout, "mode": mode,
        }
        if metrics:
            metrics.set_model_info(role, name, version, mode)

    # -- object detector --------------------------------------------------- #
    _build_detector(bundle, settings, registry, record)

    # -- face pipeline ----------------------------------------------------- #
    if any(cam.face_enabled for cam in settings.cameras) or not settings.cameras:
        face_loaded = registry.load_backend(models.face_detector)
        face_backend = face_loaded[0] if face_loaded else None
        bundle.face_detector = FaceDetector(
            face_backend, score_threshold=models.face_detector.score_threshold
        )
        if face_loaded:
            record("face_detector", models.face_detector.name or "",
                   face_loaded[1].version, face_loaded[1].layout, "neural")
        else:
            record("face_detector", "cascade-fallback", "classical", "n/a",
                   bundle.face_detector.mode)

        embed_loaded = registry.load_backend(models.face_embedder)
        bundle.face_embedder = FaceEmbedder(embed_loaded[0] if embed_loaded else None)
        if embed_loaded:
            record("face_embedder", models.face_embedder.name or "",
                   embed_loaded[1].version, "embedding", "neural")

        bundle.face_gallery = FaceGallery()

    # -- ANPR -------------------------------------------------------------- #
    if any(cam.anpr_enabled for cam in settings.cameras) or not settings.cameras:
        locator = _build_plate_locator(settings, registry, record)

        ocr_loaded = registry.load_backend(models.plate_ocr)
        ocr = PlateOCR(ocr_loaded[0] if ocr_loaded else None)
        if ocr_loaded:
            record("plate_ocr", models.plate_ocr.name or "", ocr_loaded[1].version, "ctc", "neural")
        else:
            log.warning(
                "anpr_unavailable",
                detail="no OCR artefact; plates can be located but not read",
            )
        bundle.plate_reader = PlateReader(locator, ocr)

    # -- secondary classifier ---------------------------------------------- #
    if any(cam.classify_objects for cam in settings.cameras) or not settings.cameras:
        loaded = registry.load_backend(models.classifier)
        if loaded is not None:
            backend, version = loaded
            bundle.classifier = ImageNetClassifier(
                backend, min_confidence=models.classifier.score_threshold
            )
            record("classifier", models.classifier.name or "", version.version,
                   version.layout, "neural")
        else:
            log.info(
                "classifier_unavailable",
                detail="no ImageNet classifier artefact; detector classes are used as-is",
            )

    log.info("model_bundle_ready", **{k: str(v) for k, v in bundle.status().items() if k != "models"})
    return bundle


def build_camera_detector(bundle: ModelBundle, camera: CameraConfig) -> BaseDetector | None:
    """Return the detector instance this camera should use.

    Neural detectors are stateless and shared. A motion detector holds a
    background model learned from one specific view, so each camera needs its
    own - sharing one would blend sixteen scenes into a background that matches
    none of them.
    """
    if bundle.detector is None:
        return None
    if bundle.detector.is_neural:
        return bundle.detector
    return MotionDetector()


def _default_classes() -> tuple[str, ...]:
    from ibvap.vision.detector import COCO80

    return COCO80


def _build_plate_locator(
    settings: Settings,
    registry: ModelRegistry,
    record: Any,
) -> PlateLocator:
    """Choose how plates are found inside a vehicle crop.

    Same ladder as the object detector, for the same reason: a declared Torch
    or TensorRT plate model first, then a verified ONNX artefact, then the
    morphological search - which finds plate-shaped regions by edge density and
    is genuinely useful, but is not a detector and must not be reported as one.
    """
    spec = settings.models.plate_detector
    declared = registry.resolve_spec(spec)

    if declared is not None and declared.enabled and declared.runtime == "ultralytics":
        try:
            from ibvap.vision.ultralytics_detector import UltralyticsDetector

            detector = UltralyticsDetector(
                declared.file or f"{spec.name}.pt",
                score_threshold=spec.score_threshold,
                nms_threshold=spec.nms_threshold,
                imgsz=(declared.input_size or (640, 640))[0],
                models_dir=settings.models.models_dir,
            )
            record(
                "plate_detector", spec.name or "", declared.version,
                declared.layout, detector.mode,
            )
            return PlateLocator(detector=detector, score_threshold=spec.score_threshold)
        except Exception as exc:
            log.warning(
                "plate_detector_unavailable", model=spec.name, error=str(exc)
            )

    loaded = registry.load_backend(spec)
    if loaded is not None:
        record(
            "plate_detector", spec.name or "", loaded[1].version, loaded[1].layout, "neural"
        )
        return PlateLocator(loaded[0], score_threshold=spec.score_threshold)

    record("plate_detector", "morphological-fallback", "classical", "n/a", "classical")
    return PlateLocator(None, score_threshold=spec.score_threshold)


def _build_detector(
    bundle: ModelBundle,
    settings: Settings,
    registry: ModelRegistry,
    record: Any,
) -> None:
    """Choose the detector runtime, most trustworthy first.

    Three runtimes, in descending order of what they promise:

    ``ultralytics``
        A real Torch checkpoint - RT-DETR or YOLO on published weights. This
        is the development and evaluation runtime, and the source the ONNX
        artefact is exported from.
    ``onnx``
        A verified, checksummed artefact. This is what a post runs.
    motion
        Classical background subtraction. Not a model: a way to keep detecting
        *something* on a node whose artefacts never arrived, while saying so.

    Falling from one to the next is always logged. A node that silently ran the
    weakest option would report healthy while missing people.
    """
    models = settings.models
    declared = registry.resolve_spec(models.detector)

    if declared is not None and declared.enabled and declared.runtime == "ultralytics":
        try:
            from ibvap.vision.ultralytics_detector import UltralyticsDetector

            bundle.detector = UltralyticsDetector(
                declared.file or f"{models.detector.name}.pt",
                score_threshold=models.detector.score_threshold,
                nms_threshold=models.detector.nms_threshold,
                imgsz=(declared.input_size or (640, 640))[0],
                models_dir=models.models_dir,
                min_height_fraction=settings.analytics.min_object_height_fraction,
            )
            record(
                "detector", models.detector.name or "", declared.version,
                declared.layout, bundle.detector.mode,
            )
            return
        except Exception as exc:
            # A missing extra or an unreachable download is a degradation, not
            # a crash: ONNX and motion detection are still below this.
            log.warning(
                "ultralytics_detector_unavailable",
                model=models.detector.name, error=str(exc),
            )

    loaded = registry.load_backend(models.detector)
    if loaded is not None:
        backend, version = loaded
        bundle.detector = ObjectDetector(
            backend,
            class_names=version.classes or _default_classes(),
            score_threshold=models.detector.score_threshold,
            nms_threshold=models.detector.nms_threshold,
            input_size=version.input_size,
            layout=version.layout,
            min_height_fraction=settings.analytics.min_object_height_fraction,
        )
        record("detector", models.detector.name or "", version.version, version.layout, "neural")
        return

    if models.allow_stub_fallback:
        # Per-camera instances are created by the worker; this one exists so a
        # single-camera embedded use (the desktop console) works out of the box.
        bundle.detector = MotionDetector(
            min_height_fraction=settings.analytics.min_object_height_fraction
        )
        record("detector", "motion-fallback", "classical", "n/a", "motion_fallback")
        log.warning(
            "detector_degraded",
            detail="no neural detector artefact; using classical motion detection",
        )
        return

    log.error("detector_unavailable", detail="no artefact and stub fallback disabled")
