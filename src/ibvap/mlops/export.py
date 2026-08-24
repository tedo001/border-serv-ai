"""Model export and quantisation.

Converts a training-framework checkpoint into the ONNX artefact the platform
runs, then registers it with a checksum. This is the *only* supported route
from a trained model into a deployment: a file dropped into ``models/`` without
a registry entry will not load, by design.

Requires the ``export`` extra (``pip install ibvap[export]``). Deliberately not
a runtime dependency - a BOP node runs inference and must not carry a training
stack it will never use.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ibvap.core.errors import ModelError
from ibvap.core.logging import get_logger
from ibvap.mlops.registry import sha256_file

log = get_logger(__name__)

#: Model families and the output layout each produces when exported.
FAMILY_LAYOUTS: dict[str, str] = {
    "yolo26": "yolo26",     # end-to-end, NMS-free head
    "yolo11": "yolov8",     # v8-style transposed head
    "yolov10": "yolo26",    # also end-to-end
    "yolov9": "yolov8",
    "yolov8": "yolov8",
    "yolov5": "yolov5",
    "rtdetr": "nms_xyxy",
}


def export_ultralytics(
    weights: str | Path,
    output: str | Path,
    *,
    family: str = "yolo26",
    image_size: int = 640,
    opset: int = 17,
    half: bool = False,
    simplify: bool = True,
    end_to_end: bool | None = None,
    dynamic: bool = False,
) -> dict[str, Any]:
    """Export an Ultralytics checkpoint to ONNX.

    ``end_to_end`` controls whether NMS is folded into the graph. It defaults
    to True for families that support it, because that is what removes the
    CPU-side suppression pass - a real saving on an edge node running sixteen
    cameras on four cores. Setting it False on such a family gives a v8-style
    head instead, and the registry entry must then declare ``layout: yolov8``.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ModelError(
            "ultralytics is not installed; install the export extra: pip install 'ibvap[export]'"
        ) from exc

    weights = Path(weights)
    if not weights.is_file():
        raise ModelError(f"checkpoint not found: {weights}")

    family = family.lower()
    if family not in FAMILY_LAYOUTS:
        raise ModelError(
            f"unknown model family {family!r}; expected one of {sorted(FAMILY_LAYOUTS)}"
        )

    if end_to_end is None:
        end_to_end = FAMILY_LAYOUTS[family] in ("yolo26", "nms_xyxy")

    log.info(
        "export_started",
        weights=str(weights), family=family, image_size=image_size,
        opset=opset, end_to_end=end_to_end,
    )

    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": image_size,
        "opset": opset,
        "half": half,
        "simplify": simplify,
        "dynamic": dynamic,
    }
    if end_to_end:
        # Ultralytics exposes end-to-end export as `nms=True`; older versions
        # ignore the flag rather than failing, so the resulting layout is
        # verified below rather than assumed.
        kwargs["nms"] = True

    produced = Path(model.export(**kwargs))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != output.resolve():
        shutil.move(str(produced), str(output))

    class_names = _class_names(model)
    layout = _verify_layout(output, FAMILY_LAYOUTS[family], len(class_names))

    metadata = {
        "file": str(output),
        "sha256": sha256_file(output),
        "layout": layout,
        "input_size": [image_size, image_size],
        "classes": class_names,
        "size_mb": round(output.stat().st_size / 1024**2, 2),
        "provenance": {
            "source_checkpoint": weights.name,
            "family": family,
            "opset": opset,
            "end_to_end": end_to_end,
            "half_precision": half,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "exported_by": _tool_versions(),
        },
    }
    log.info("export_complete", **{k: v for k, v in metadata.items() if k != "classes"})
    return metadata


def _class_names(model: Any) -> list[str]:
    """Recover the ordered class list from an Ultralytics model."""
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]
    return list(names)


def _verify_layout(onnx_path: Path, expected: str, class_count: int) -> str:
    """Inspect the exported graph and confirm the layout matches expectation.

    Export flags are advisory: a toolchain that does not support end-to-end
    export may silently produce a standard head. Declaring the wrong layout in
    the registry does not fail loudly at runtime - it produces garbled boxes -
    so the graph itself is the authority here, not the flag that was passed.
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        shape = session.get_outputs()[0].shape
    except Exception as exc:
        log.warning("layout_verification_skipped", error=str(exc))
        return expected

    dims = [d if isinstance(d, int) else -1 for d in shape]
    observed = expected
    if len(dims) == 3:
        _, a, b = dims
        if b == 6:
            observed = "nms_xyxy" if expected != "yolo26" else "yolo26"
        elif a in (class_count + 4, -1) and (b == -1 or b > a):
            observed = "yolov8"
        elif b == class_count + 5:
            observed = "yolov5"

    if observed != expected:
        log.warning(
            "layout_mismatch",
            expected=expected, observed=observed, output_shape=dims,
            detail="registry entry must declare the observed layout",
        )
    return observed


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module in ("torch", "ultralytics", "onnx", "onnxruntime"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # pragma: no cover - optional
            continue
    return versions


def quantise_dynamic(
    source: str | Path, output: str | Path, *, per_channel: bool = True
) -> dict[str, Any]:
    """Quantise an ONNX model to INT8 weights (dynamic quantisation).

    Dynamic rather than static quantisation: static needs a representative
    calibration set from the deployment site, which is exactly what a new BOP
    does not have on day one. Dynamic typically gives 2-4x smaller artefacts
    and a useful CPU speed-up with a modest accuracy cost.

    **The accuracy cost must be measured, not assumed.** Run
    :mod:`ibvap.mlops.evaluate` against the quantised artefact before it is
    registered as the default for any site.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ModelError("onnxruntime quantisation tools are unavailable") from exc

    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ModelError(f"model not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)

    before = source.stat().st_size
    quantize_dynamic(
        str(source), str(output),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        extra_options={"MatMulConstBOnly": True},
    )
    after = output.stat().st_size

    result = {
        "file": str(output),
        "sha256": sha256_file(output),
        "size_before_mb": round(before / 1024**2, 2),
        "size_after_mb": round(after / 1024**2, 2),
        "compression": round(before / max(1, after), 2),
    }
    log.info("quantisation_complete", **result)
    return result


def register_model(
    registry_path: str | Path,
    name: str,
    version: str,
    metadata: dict[str, Any],
    *,
    role: str = "detector",
    make_default: bool = False,
    description: str = "",
    metrics: dict[str, float] | None = None,
) -> None:
    """Add or update an entry in ``models/registry.yaml``.

    Refuses to overwrite an existing version with different content. An
    artefact that has been deployed must never change under a version another
    site is pinning; publish a new version instead.
    """
    path = Path(registry_path)
    document: dict[str, Any] = {"version": 1, "models": {}}
    if path.is_file():
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or document
    document.setdefault("models", {})

    entry = document["models"].setdefault(
        name, {"role": role, "description": description, "versions": {}}
    )
    if description:
        entry["description"] = description
    entry["role"] = role

    existing = entry["versions"].get(version)
    if existing and existing.get("sha256") != metadata.get("sha256"):
        raise ModelError(
            f"{name}:{version} is already registered with a different checksum. "
            "Publish a new version rather than mutating a deployed one."
        )

    models_dir = path.parent
    artefact = Path(metadata["file"])
    try:
        relative = artefact.resolve().relative_to(models_dir.resolve())
        file_field = str(relative)
    except ValueError:
        file_field = str(artefact)

    entry["versions"][version] = {
        "file": file_field,
        "sha256": metadata["sha256"],
        "layout": metadata.get("layout", "auto"),
        "input_size": metadata.get("input_size"),
        "classes": metadata.get("classes", []),
        "metrics": metrics or metadata.get("metrics", {}),
        "provenance": metadata.get("provenance", {}),
        "notes": metadata.get("notes", ""),
        "enabled": True,
    }
    if make_default or not entry.get("default"):
        entry["default"] = version

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, width=100), encoding="utf-8")
    log.info(
        "model_registered",
        name=name, version=version, role=role,
        layout=metadata.get("layout"), default=entry.get("default"),
    )


def write_model_card(
    output: str | Path, name: str, version: str, metadata: dict[str, Any],
    *, metrics: dict[str, float] | None = None, intended_use: str = "",
    limitations: str = "",
) -> Path:
    """Write a model card documenting provenance, performance and limits.

    A model card is not paperwork here. This platform can put a person in front
    of an armed response; whoever authorises a deployment needs to know what
    the model was trained on, how it was measured, and where it is known to
    fail - particularly at night, at range, and on classes it confuses.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = metadata.get("provenance", {})
    metrics = metrics or metadata.get("metrics", {})

    lines = [
        f"# Model card — {name}:{version}", "",
        "## Identity", "",
        f"- **Name**: `{name}`", f"- **Version**: `{version}`",
        f"- **Artefact**: `{Path(metadata.get('file', '')).name}`",
        f"- **SHA-256**: `{metadata.get('sha256', '')}`",
        f"- **Output layout**: `{metadata.get('layout', 'auto')}`",
        f"- **Input size**: {metadata.get('input_size')}",
        f"- **Size**: {metadata.get('size_mb', '?')} MB",
        f"- **Classes** ({len(metadata.get('classes', []))}): "
        f"{', '.join(metadata.get('classes', [])[:20])}"
        + (" …" if len(metadata.get("classes", [])) > 20 else ""),
        "", "## Provenance", "",
    ]
    lines += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in provenance.items()] or ["- not recorded"]

    lines += ["", "## Measured performance", ""]
    if metrics:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in metrics.items()]
    else:
        lines.append(
            "**Not yet evaluated.** Run `ibvap evaluate` against a labelled set "
            "from a representative site before deploying this version."
        )

    lines += [
        "", "## Intended use", "",
        intended_use or (
            "Detection of people and vehicles in fixed CCTV views at border out "
            "posts, check posts and border roads, as input to the IBVAP analytics "
            "rules. Not validated for any other purpose."
        ),
        "", "## Known limitations", "",
        limitations or (
            "- Accuracy degrades at long range; objects below roughly 20 px in "
            "height are unreliable.\n"
            "- Night-time and IR performance depends on illumination and is "
            "materially worse than daytime unless the model was trained with IR "
            "imagery.\n"
            "- Livestock is a common source of false person detections on rural "
            "fence lines; the `animal` class exists so it can be suppressed by "
            "rule rather than by threshold.\n"
            "- Heavy occlusion, adverse weather (fog, heavy rain) and camera "
            "tampering all reduce recall.\n"
            "- The model reflects the distribution of its training data; "
            "performance at a given site should be re-measured after deployment."
        ),
        "", "## Operational guidance", "",
        "- Verify the checksum above matches the deployed artefact before use.",
        "- Re-evaluate after any quantisation; INT8 conversion changes accuracy.",
        "- Monitor `ibvap_detections_total` and the drift report for a shift in "
        "the score distribution, which usually precedes a measurable accuracy loss.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("model_card_written", path=str(path))
    return path
