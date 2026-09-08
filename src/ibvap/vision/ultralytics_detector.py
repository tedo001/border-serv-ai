"""RT-DETR and YOLO on real pretrained weights, through Ultralytics.

The rest of the vision layer runs ONNX Runtime, for the reason set out in
:mod:`ibvap.vision.backends`: a border post is a fanless mini-PC at the end of
a dirt road, and a 2 GB training stack is not something you ship there. That
argument is about *deployment*, and it has never been an argument against
having real weights during development and evaluation.

This module is the other end of that lifecycle. It runs a genuine pretrained
detector - RT-DETR or YOLO - directly from an Ultralytics checkpoint, so a
reviewer who has just cloned the repository gets real detections on the first
run instead of the motion fallback. The same checkpoint is then exported to
ONNX by ``ibvap models fetch``, which is what actually goes to the post.

So there are two runtimes and one pipeline:

    ultralytics   development, evaluation, tuning     weights: rtdetr-l.pt
    onnx          the field                           weights: rtdetr-l.onnx

Everything downstream - tracker, classifier, rules, evidence - sees the same
:class:`~ibvap.core.types.Detection` either way, so a threshold tuned on one
runtime means the same thing on the other.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ibvap.core.errors import ModelError
from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.vision.detector import BaseDetector

log = get_logger(__name__)

#: Checkpoints this detector knows how to construct, and the Ultralytics class
#: each one needs. RT-DETR and YOLO have different heads and different Python
#: classes; loading one with the other's wrapper fails in ways that are hard to
#: read, so the mapping is explicit rather than guessed from the file name.
RTDETR_PREFIXES = ("rtdetr",)

#: Sensible published checkpoints. Anything else is still accepted - this is a
#: convenience table for the console's model picker, not an allowlist.
KNOWN_CHECKPOINTS: dict[str, str] = {
    "rtdetr-l.pt": "RT-DETR large - the accuracy/latency point this platform targets",
    "rtdetr-x.pt": "RT-DETR extra-large - slower, for a node with a GPU",
    "yolo11n.pt": "YOLO11 nano - the fastest option, for a weak CPU-only post",
    "yolo11s.pt": "YOLO11 small - a reasonable CPU default",
    "yolo11m.pt": "YOLO11 medium - GPU or a strong CPU",
}


def weights_dir(models_dir: Path | str = "models") -> Path:
    """Directory checkpoints are downloaded into.

    Ultralytics otherwise writes them to its own global settings directory,
    which puts a deployment artefact somewhere nobody thinks to look and makes
    an air-gapped install impossible to prepare in advance.
    """
    return Path(models_dir) / "weights"


def is_rtdetr(checkpoint: str) -> bool:
    return Path(checkpoint).name.lower().startswith(RTDETR_PREFIXES)


class UltralyticsDetector(BaseDetector):
    """A real pretrained detector: RT-DETR or YOLO, on Torch."""

    is_neural = True

    def __init__(
        self,
        checkpoint: str = "rtdetr-l.pt",
        *,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        half: bool = False,
        models_dir: Path | str = "models",
        allowed_classes: list[str] | None = None,
        min_height_fraction: float = 0.0,
        max_detections: int = 300,
    ) -> None:
        self.checkpoint = checkpoint
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.min_height_fraction = min_height_fraction
        self.allowed: set[ObjectClass] | None = (
            {ObjectClass.coerce(c) for c in allowed_classes} if allowed_classes else None
        )

        path = self._resolve(checkpoint, Path(models_dir))
        try:
            # Imported here, not at module scope: the package is an optional
            # extra and the rest of the vision layer must import without it.
            from ultralytics import RTDETR, YOLO
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ModelError(
                "the ultralytics runtime needs the 'torch' extra: "
                "pip install -e '.[torch]'"
            ) from exc

        loader = RTDETR if is_rtdetr(checkpoint) else YOLO
        try:
            self.model = loader(str(path))
        except Exception as exc:
            raise ModelError(f"could not load {checkpoint!r}: {exc}") from exc

        self.device = self._select_device(device)
        self.half = bool(half) and self.device != "cpu"
        # Class names come from the checkpoint itself rather than a constant in
        # this repository, so a fine-tuned model with a border-specific class
        # list works without a code change.
        names = getattr(self.model, "names", None) or {}
        self.class_names: dict[int, str] = {int(k): str(v) for k, v in dict(names).items()}
        self.mode = f"ultralytics:{'rtdetr' if is_rtdetr(checkpoint) else 'yolo'}"

        log.info(
            "ultralytics_detector_ready",
            checkpoint=checkpoint, device=self.device,
            classes=len(self.class_names), imgsz=imgsz,
        )

    # -- construction helpers ---------------------------------------------- #

    @staticmethod
    def _resolve(checkpoint: str, models_dir: Path) -> Path:
        """Return a local path, downloading the checkpoint if it is absent.

        A bare name like ``rtdetr-l.pt`` is fetched into ``models/weights`` on
        first use; an explicit path is used as given and never downloaded, so a
        site build can ship its own fine-tuned checkpoint.
        """
        given = Path(checkpoint)
        if given.parent != Path():
            if not given.is_file():
                raise ModelError(f"checkpoint {checkpoint} does not exist")
            return given

        target = weights_dir(models_dir) / given.name
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads relative to the working directory, so point it
        # at the weights directory for the duration of the fetch.
        previous = Path.cwd()
        try:
            os.chdir(target.parent)
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(given.name)
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ModelError(
                "the ultralytics runtime needs the 'torch' extra: "
                "pip install -e '.[torch]'"
            ) from exc
        except Exception as exc:
            raise ModelError(
                f"could not download {given.name}: {exc}. On an air-gapped node, "
                f"copy the checkpoint into {target.parent} before starting."
            ) from exc
        finally:
            os.chdir(previous)

        if not target.is_file():
            raise ModelError(f"download of {given.name} produced no file at {target}")
        log.info("checkpoint_downloaded", checkpoint=given.name, path=str(target))
        return target

    @staticmethod
    def _select_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:  # pragma: no cover - torch absent or broken CUDA
            pass
        return "cpu"

    # -- inference ---------------------------------------------------------- #

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]

        # `half` is only passed when it is on: Ultralytics deprecated the
        # argument, and passing it as False still prints the warning once per
        # frame, which buries every other log line on a running node.
        extra = {"half": True} if self.half else {}
        try:
            results = self.model.predict(
                image,
                conf=self.score_threshold,
                iou=self.nms_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
                **extra,
            )
        except Exception as exc:
            log.error("ultralytics_inference_failed", error=str(exc))
            return []
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        min_height = self.min_height_fraction * height
        out: list[Detection] = []
        for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, class_ids, strict=True):
            if min_height and (y2 - y1) < min_height:
                continue
            label = self.class_names.get(int(class_id), str(class_id))
            obj_class = ObjectClass.coerce(label)
            if self.allowed is not None and obj_class not in self.allowed:
                continue
            out.append(Detection(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)).clip(width, height),
                obj_class=obj_class,
                score=float(score),
                raw_label=label,
            ))
        return out

    def close(self) -> None:
        self.model = None


def export_onnx(
    checkpoint: str,
    destination: Path | str,
    *,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    models_dir: Path | str = "models",
) -> Path:
    """Export a checkpoint to ONNX, which is what the field actually runs.

    This is the bridge between the two runtimes. The exported graph carries the
    same weights the Torch runtime used, so a model evaluated in the analyst
    console is the model that goes to the post - and the post never needs Torch.
    """
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ModelError(
            "exporting needs the 'torch' extra: pip install -e '.[torch]'"
        ) from exc

    path = UltralyticsDetector._resolve(checkpoint, Path(models_dir))
    loader = RTDETR if is_rtdetr(checkpoint) else YOLO
    model = loader(str(path))
    produced = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Path(produced).replace(destination)
    log.info("onnx_exported", checkpoint=checkpoint, path=str(destination))
    return destination


def checkpoint_classes(checkpoint: str, *, models_dir: Path | str = "models") -> list[str]:
    """Ordered class names of a checkpoint, for the registry entry it produces."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ModelError("needs the 'torch' extra: pip install -e '.[torch]'") from exc

    path = UltralyticsDetector._resolve(checkpoint, Path(models_dir))
    loader = RTDETR if is_rtdetr(checkpoint) else YOLO
    names: dict[Any, Any] = dict(getattr(loader(str(path)), "names", {}) or {})
    return [str(names[key]) for key in sorted(names, key=int)]
