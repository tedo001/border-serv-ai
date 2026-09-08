"""The two detector runtimes, the registry that chooses between them, and
supervision interchange.

None of these need Torch or supervision installed: what is tested is the
platform's own behaviour around them - that a declaration selects the right
runtime, that a missing extra degrades instead of crashing, and that a node
without the annotators still renders.
"""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.config import ModelsConfig, ModelSpec, Settings
from ibvap.core.errors import ModelError
from ibvap.core.types import BBox, Detection, ObjectCategory, ObjectClass, Track
from ibvap.mlops.registry import ModelRegistry, register_model
from ibvap.pipeline.models import build_model_bundle
from ibvap.vision import supervision_ops
from ibvap.vision.ultralytics_detector import KNOWN_CHECKPOINTS, is_rtdetr, weights_dir

REGISTRY = """
version: 1

# A comment that must survive `ibvap models register`.
models:

  # RT-DETR, declared but not present.
  demo-torch:
    role: detector
    description: torch runtime
    default: "coco"
    versions:
      "coco":
        file: rtdetr-l.pt
        runtime: ultralytics
        layout: rtdetr
        input_size: [640, 640]
        enabled: true
"""


@pytest.fixture
def registry_path(workspace):
    path = workspace / "registry.yaml"
    path.write_text(REGISTRY, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Runtime declaration
# --------------------------------------------------------------------------- #

def test_a_version_declares_which_runtime_loads_it(registry_path) -> None:
    registry = ModelRegistry.load(
        ModelsConfig(registry_path=registry_path, models_dir=registry_path.parent)
    )
    version = registry.resolve_spec(ModelSpec(name="demo-torch"))
    assert version is not None
    assert version.runtime == "ultralytics"


def test_onnx_is_the_default_runtime(registry_path) -> None:
    """Anything that does not say otherwise is an ONNX artefact.

    The field runs ONNX; a registry written before runtimes existed must keep
    meaning exactly what it meant.
    """
    registry_path.write_text(
        "version: 1\nmodels:\n  plain:\n    role: detector\n    default: '1'\n"
        "    versions:\n      '1': { file: d.onnx, layout: yolov8 }\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.load(
        ModelsConfig(registry_path=registry_path, models_dir=registry_path.parent)
    )
    assert registry.resolve_spec(ModelSpec(name="plain")).runtime == "onnx"


def test_torch_checkpoints_live_under_weights(registry_path) -> None:
    """Downloaded checkpoints must not land among the shipped artefacts.

    `models/` is meant to be reviewable as the set of files a site build
    carries; a checkpoint fetched at runtime is not one of them.
    """
    registry = ModelRegistry.load(
        ModelsConfig(registry_path=registry_path, models_dir=registry_path.parent)
    )
    version = registry.resolve_spec(ModelSpec(name="demo-torch"))
    assert registry.artefact_path(version).parent.name == "weights"
    assert weights_dir("models").name == "weights"


def test_an_unavailable_torch_runtime_degrades_rather_than_crashing(
    registry_path, monkeypatch
) -> None:
    """The node must still start when the optional runtime cannot load.

    A missing extra, an unreachable download or a corrupt checkpoint all end
    the same way: the bundle falls through to what is available and reports
    itself degraded. The alternative - a node that refuses to start - leaves a
    border post with no analytics at all rather than weak ones.

    The failure is injected rather than waited for: the extra is installed on
    some machines, and a unit test must not depend on that, nor pull 63 MB
    over the network to find out.
    """
    import ibvap.vision.ultralytics_detector as module

    def refuse(*_args, **_kwargs):
        raise ModelError("ultralytics is not installed")

    monkeypatch.setattr(module, "UltralyticsDetector", refuse)

    settings = Settings()
    settings.models = ModelsConfig(
        registry_path=registry_path,
        models_dir=registry_path.parent,
        detector=ModelSpec(name="demo-torch"),
        allow_stub_fallback=True,
    )
    bundle = build_model_bundle(settings)

    assert bundle.detector is not None, "the node came up with no detector at all"
    assert bundle.detector_mode == "motion_fallback"
    assert bundle.status()["degraded"] is True


def test_a_declared_torch_runtime_is_chosen_over_onnx(
    registry_path, monkeypatch
) -> None:
    """The declaration decides, not the file system.

    An ``ultralytics`` version has no ONNX artefact to verify, so the runtime
    has to be read from the declaration before any file check - otherwise the
    missing .onnx looks like a missing model and the node degrades with a real
    detector sitting right there.
    """
    import ibvap.vision.ultralytics_detector as module

    seen: dict[str, object] = {}

    class Fake:
        mode = "ultralytics:rtdetr"
        is_neural = True

        def __init__(self, checkpoint, **kwargs):
            seen["checkpoint"] = checkpoint
            seen.update(kwargs)

        def detect(self, image):
            return []

        def close(self):
            pass

    monkeypatch.setattr(module, "UltralyticsDetector", Fake)

    settings = Settings()
    settings.models = ModelsConfig(
        registry_path=registry_path,
        models_dir=registry_path.parent,
        detector=ModelSpec(name="demo-torch", score_threshold=0.5),
    )
    bundle = build_model_bundle(settings)

    assert bundle.detector_mode == "ultralytics:rtdetr"
    assert bundle.status()["degraded"] is False
    assert seen["checkpoint"] == "rtdetr-l.pt", "the declared file was not used"
    assert seen["score_threshold"] == 0.5, "the config threshold was not passed through"


# --------------------------------------------------------------------------- #
# Registering must not eat the registry's documentation
# --------------------------------------------------------------------------- #

def test_registering_preserves_every_comment(registry_path) -> None:
    """Round-tripping the document through a YAML dumper discards comments.

    The registry's comments document the layouts, the runtimes and how to
    populate the file. Losing them the first time anyone registers a model
    guts the most useful part of the artefact.
    """
    before = registry_path.read_text(encoding="utf-8")
    register_model(
        registry_path, "added", "1.0.0",
        {"file": "d.onnx", "sha256": "a" * 64, "layout": "yolov8",
         "input_size": [640, 640], "classes": ["person"]},
        role="detector",
    )
    after = registry_path.read_text(encoding="utf-8")

    for line in before.splitlines():
        if line.strip().startswith("#"):
            assert line in after, f"comment lost: {line!r}"
    assert "added:" in after
    assert "demo-torch:" in after, "the untouched entry was disturbed"


def test_registering_replaces_an_entry_in_place(registry_path) -> None:
    for version in ("1.0.0", "2.0.0"):
        register_model(
            registry_path, "added", version,
            {"file": f"d-{version}.onnx", "sha256": version.replace(".", "") * 8,
             "layout": "yolov8", "input_size": [640, 640], "classes": []},
            role="detector",
        )
    registry = ModelRegistry.load(
        ModelsConfig(registry_path=registry_path, models_dir=registry_path.parent)
    )
    assert sorted(registry.get("added").versions) == ["1.0.0", "2.0.0"]
    assert registry_path.read_text(encoding="utf-8").count("added:") == 1


# --------------------------------------------------------------------------- #
# Checkpoint naming
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("rtdetr-l.pt", True),
        ("rtdetr-x.pt", True),
        ("models/weights/rtdetr-l.pt", True),
        ("yolo11s.pt", False),
        ("yolov8n.pt", False),
    ],
)
def test_rtdetr_checkpoints_are_recognised_by_name(checkpoint, expected) -> None:
    """RT-DETR and YOLO need different Ultralytics classes.

    Loading one with the other's wrapper fails deep inside the library, in a
    message that says nothing about the real cause.
    """
    assert is_rtdetr(checkpoint) is expected


def test_the_offered_checkpoints_are_all_named_consistently() -> None:
    assert all(name.endswith(".pt") for name in KNOWN_CHECKPOINTS)
    assert any(is_rtdetr(name) for name in KNOWN_CHECKPOINTS)


# --------------------------------------------------------------------------- #
# Supervision interchange
# --------------------------------------------------------------------------- #

sv_only = pytest.mark.skipif(
    not supervision_ops.AVAILABLE, reason="the 'sv' extra is not installed"
)


def test_every_category_has_a_palette_slot() -> None:
    """A category with no colour would raise inside the annotator, per frame."""
    for category in ObjectCategory:
        assert category in supervision_ops.CATEGORY_RGB
        assert 0 <= supervision_ops.category_index(category) < len(ObjectCategory)


@sv_only
def test_detections_round_trip_into_supervision() -> None:
    detections = [
        Detection(bbox=BBox(10, 20, 60, 120), obj_class=ObjectClass.PERSON, score=0.9),
        Detection(bbox=BBox(80, 30, 200, 90), obj_class=ObjectClass.CAR, score=0.7),
    ]
    sv_detections = supervision_ops.detections_to_sv(detections)
    assert len(sv_detections) == 2
    assert sv_detections.xyxy[0].tolist() == [10, 20, 60, 120]
    assert sv_detections.confidence[1] == pytest.approx(0.7)
    assert list(sv_detections.data["label"]) == ["person", "car"]


@sv_only
def test_tracks_carry_their_identity_across() -> None:
    """Without tracker ids supervision colours by class, so three people are
    three identical boxes rather than three identities."""
    tracks = [
        Track(track_id=4, obj_class=ObjectClass.PERSON, bbox=BBox(1, 1, 9, 9), score=0.8),
        Track(track_id=9, obj_class=ObjectClass.PERSON, bbox=BBox(20, 1, 29, 9), score=0.6),
    ]
    converted = supervision_ops.tracks_to_sv(tracks)
    assert converted.tracker_id.tolist() == [4, 9]


@sv_only
def test_renderer_returns_a_new_frame_of_the_same_shape() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = [
        Track(track_id=1, obj_class=ObjectClass.PERSON, bbox=BBox(40, 40, 90, 180), score=0.9)
    ]
    out = supervision_ops.SupervisionRenderer().annotate(frame, tracks)
    assert out.shape == frame.shape
    assert out is not frame, "the source frame is reused downstream and must not be drawn on"
    assert out.any(), "nothing was drawn"


@sv_only
def test_renderer_handles_an_empty_frame() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert supervision_ops.SupervisionRenderer().annotate(frame, []).shape == frame.shape
