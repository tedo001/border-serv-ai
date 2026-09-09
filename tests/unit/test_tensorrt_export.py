"""The TensorRT path, and what the platform can honestly say about it.

None of this builds an engine: a TensorRT plan is compiled by the GPU that
will run it, so a build host without one - CI, a laptop, this container -
cannot produce one and must not pretend otherwise. What is tested is
everything around the build: that the preconditions are checked before any
work starts, that the refusal explains itself, and that a plan is never
mistaken for something portable.
"""

from __future__ import annotations

import sys

import pytest

from ibvap.core.errors import ModelError
from ibvap.vision.ultralytics_detector import (
    LOADABLE_SUFFIXES,
    UltralyticsDetector,
    export_engine,
    is_engine,
    is_published,
    tensorrt_readiness,
)


class TestEngineRecognition:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("rtdetr-l.engine", True), ("yolov8-plate.engine", True),
         ("rtdetr-l.pt", False), ("rtdetr-l.onnx", False)],
    )
    def test_engines_are_recognised_by_suffix(self, name, expected) -> None:
        assert is_engine(name) is expected

    def test_every_loadable_suffix_is_a_real_weight_format(self) -> None:
        assert ".engine" in LOADABLE_SUFFIXES
        assert ".pt" in LOADABLE_SUFFIXES
        assert all(s.startswith(".") for s in LOADABLE_SUFFIXES)


class TestReadiness:
    def test_readiness_always_explains_itself(self) -> None:
        """A bare False tells an operator nothing about what to do next."""
        ready, detail = tensorrt_readiness()
        assert isinstance(ready, bool)
        assert detail, "readiness reported no reason at all"
        if not ready:
            assert len(detail) > 20, f"unhelpfully terse: {detail!r}"

    def test_a_cpu_device_is_never_ready(self) -> None:
        ready, detail = tensorrt_readiness(device="cpu")
        assert ready is False
        assert "cpu" in detail.lower()

    def test_the_pinned_device_is_answered_without_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pinned CPU device needs nothing installed to rule out.

        Checking the import first made the answer depend on whether Torch
        happened to be present: the same call returned "device is pinned to
        cpu" on a developer's machine and "PyTorch is not installed" in CI,
        which is exactly the environment-dependent reply this function exists
        to remove.
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        ready, detail = tensorrt_readiness(device="cpu")
        assert ready is False
        assert "cpu" in detail.lower()

    def test_a_missing_torch_is_reported_as_a_missing_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        ready, detail = tensorrt_readiness()
        assert ready is False
        assert "torch" in detail.lower()

    def test_building_without_a_gpu_refuses_before_doing_any_work(
        self, workspace
    ) -> None:
        ready, _ = tensorrt_readiness()
        if ready:  # pragma: no cover - only on a CUDA build host
            pytest.skip("this machine can build engines")

        with pytest.raises(ModelError, match="cannot build a TensorRT engine"):
            export_engine("rtdetr-l.pt", workspace / "out.engine")

        assert not (workspace / "out.engine").exists()

    def test_int8_without_calibration_data_is_refused(self, workspace) -> None:
        """INT8 calibrated on the wrong images loses accuracy silently.

        Ultralytics will happily calibrate against a default image set that
        looks nothing like a border camera at night, and nothing about the
        resulting engine says so.
        """
        with pytest.raises(ModelError):
            export_engine("rtdetr-l.pt", workspace / "out.engine", int8=True)


class TestCheckpointResolution:
    """A plan is built, never fetched; a fine-tune is trained, never fetched."""

    def test_an_absent_engine_is_not_downloaded(self, workspace) -> None:
        with pytest.raises(ModelError, match="only .pt checkpoints can be downloaded"):
            UltralyticsDetector._resolve("rtdetr-l.engine", workspace)

    def test_an_absent_fine_tune_says_where_to_put_it(self, workspace) -> None:
        """The message has to name the path, because nothing else will.

        A plate detector is trained per deployment. Sending its name to the
        asset CDN returns a 404 that tells an operator nothing.
        """
        with pytest.raises(ModelError) as excinfo:
            UltralyticsDetector._resolve("yolov8-plate.pt", workspace)
        message = str(excinfo.value)
        assert "yolov8-plate.pt" in message
        assert str(workspace) in message
        assert "plate" in message.lower()

    @pytest.mark.parametrize(
        ("name", "published"),
        [("yolov8n.pt", True), ("yolo11s.pt", True), ("rtdetr-l.pt", True),
         ("rtdetr-x.pt", True), ("yolov8n-seg.pt", True),
         ("yolov8-plate.pt", False), ("yolo11s-border.pt", False),
         ("plate-detector.pt", False), ("my-model.pt", False)],
    )
    def test_only_real_published_names_are_fetched(self, name, published) -> None:
        assert is_published(name) is published

    def test_an_explicit_path_is_used_as_given(self, workspace) -> None:
        """A site ships its own weights; they are not looked up by name."""
        weights = workspace / "site" / "plate.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"not a real checkpoint")
        assert UltralyticsDetector._resolve(str(weights), workspace) == weights
