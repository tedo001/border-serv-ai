"""Model registry, benchmarking, evaluation and drift detection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ibvap.core.config import ModelsConfig, ModelSpec
from ibvap.core.errors import ChecksumMismatchError, ModelError, ModelNotFoundError
from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.mlops.benchmark import BenchmarkResult, benchmark, format_report
from ibvap.mlops.drift import DriftMonitor, load_profiles, save_profiles
from ibvap.mlops.evaluate import (
    GroundTruthBox,
    average_precision,
    evaluate_anpr,
    evaluate_detection,
    evaluate_face_matching,
)
from ibvap.mlops.registry import ModelRegistry, _version_key, register_model, sha256_file, write_model_card


@pytest.fixture
def registry_root(workspace: Path) -> Path:
    (workspace / "detector").mkdir()
    artefact = workspace / "detector" / "model.onnx"
    artefact.write_bytes(b"fake onnx artefact")
    return workspace


def build_metadata(artefact: Path) -> dict:
    return {
        "file": str(artefact),
        "sha256": hashlib.sha256(artefact.read_bytes()).hexdigest(),
        "layout": "yolo26",
        "input_size": [640, 640],
        "classes": ["person", "car"],
        "size_mb": 0.01,
        "provenance": {"family": "yolo26", "end_to_end": True},
    }


class TestRegistry:
    def test_register_and_resolve(self, registry_root: Path) -> None:
        artefact = registry_root / "detector" / "model.onnx"
        registry_path = registry_root / "registry.yaml"
        register_model(
            registry_path, "det", "1.0.0", build_metadata(artefact),
            role="detector", make_default=True,
        )
        registry = ModelRegistry.load(
            ModelsConfig(registry_path=registry_path, models_dir=registry_root)
        )
        version = registry.get("det").resolve("latest")
        assert version.version == "1.0.0"
        assert version.layout == "yolo26"
        assert version.classes == ["person", "car"]

    def test_checksum_mismatch_is_fatal(self, registry_root: Path) -> None:
        """An artefact truncated by a failed sync still loads in ONNX Runtime
        and still returns tensors - they are simply wrong."""
        artefact = registry_root / "detector" / "model.onnx"
        registry_path = registry_root / "registry.yaml"
        register_model(registry_path, "det", "1.0.0", build_metadata(artefact))

        artefact.write_bytes(b"TAMPERED")
        registry = ModelRegistry.load(
            ModelsConfig(registry_path=registry_path, models_dir=registry_root)
        )
        with pytest.raises(ChecksumMismatchError):
            registry.verify("det", "1.0.0")

    def test_published_versions_are_immutable(self, registry_root: Path) -> None:
        """An artefact another site is pinning must never change underneath it."""
        artefact = registry_root / "detector" / "model.onnx"
        registry_path = registry_root / "registry.yaml"
        metadata = build_metadata(artefact)
        register_model(registry_path, "det", "1.0.0", metadata)

        with pytest.raises(ModelError, match="already registered with a different checksum"):
            register_model(registry_path, "det", "1.0.0", {**metadata, "sha256": "deadbeef"})

    def test_missing_model_is_reported(self, registry_root: Path) -> None:
        registry = ModelRegistry.load(
            ModelsConfig(registry_path=registry_root / "registry.yaml", models_dir=registry_root)
        )
        with pytest.raises(ModelNotFoundError):
            registry.get("nope")

    def test_absent_registry_yields_an_empty_one(self, workspace: Path) -> None:
        """A fresh node has no artefacts yet and must still start."""
        registry = ModelRegistry.load(ModelsConfig(registry_path=workspace / "none.yaml"))
        assert registry.entries == {}

    def test_missing_artefact_degrades_when_permitted(self, registry_root: Path) -> None:
        registry_path = registry_root / "registry.yaml"
        artefact = registry_root / "detector" / "model.onnx"
        register_model(registry_path, "det", "1.0.0", build_metadata(artefact))
        artefact.unlink()

        registry = ModelRegistry.load(ModelsConfig(
            registry_path=registry_path, models_dir=registry_root, allow_stub_fallback=True
        ))
        assert registry.load_backend(ModelSpec(name="det")) is None

    def test_missing_artefact_raises_when_fallback_disabled(self, registry_root: Path) -> None:
        registry_path = registry_root / "registry.yaml"
        artefact = registry_root / "detector" / "model.onnx"
        register_model(registry_path, "det", "1.0.0", build_metadata(artefact))
        artefact.unlink()

        registry = ModelRegistry.load(ModelsConfig(
            registry_path=registry_path, models_dir=registry_root, allow_stub_fallback=False
        ))
        with pytest.raises(ModelNotFoundError):
            registry.load_backend(ModelSpec(name="det"))

    def test_version_sorting_is_numeric(self) -> None:
        assert sorted(["1.0.0", "1.10.0", "1.2.0", "2.0.0"], key=_version_key) == [
            "1.0.0", "1.2.0", "1.10.0", "2.0.0"
        ]

    def test_describe_reports_presence(self, registry_root: Path) -> None:
        artefact = registry_root / "detector" / "model.onnx"
        registry_path = registry_root / "registry.yaml"
        register_model(registry_path, "det", "1.0.0", build_metadata(artefact))
        registry = ModelRegistry.load(
            ModelsConfig(registry_path=registry_path, models_dir=registry_root)
        )
        described = registry.describe()[0]
        assert described["present"] is True
        assert described["is_default"] is True

    def test_checksum_helper(self, registry_root: Path) -> None:
        artefact = registry_root / "detector" / "model.onnx"
        assert sha256_file(artefact) == hashlib.sha256(artefact.read_bytes()).hexdigest()


class TestModelCard:
    def test_documents_provenance_and_limits(self, registry_root: Path) -> None:
        artefact = registry_root / "detector" / "model.onnx"
        path = write_model_card(
            registry_root / "card.md", "det", "1.0.0", build_metadata(artefact),
            metrics={"mAP50": 0.74},
        )
        text = path.read_text()
        assert "det:1.0.0" in text
        assert "SHA-256" in text
        assert "Known limitations" in text
        assert "mAP50" in text

    def test_flags_an_unevaluated_model(self, registry_root: Path) -> None:
        artefact = registry_root / "detector" / "model.onnx"
        path = write_model_card(
            registry_root / "card.md", "det", "1.0.0", build_metadata(artefact)
        )
        assert "Not yet evaluated" in path.read_text()


class TestBenchmark:
    def test_reports_percentiles(self) -> None:
        result = benchmark("noop", lambda: None, iterations=30, warmup=2)
        assert result.iterations == 30
        assert result.p50_ms <= result.p95_ms <= result.p99_ms
        assert result.throughput_fps > 0

    def test_capacity_is_sized_on_p95(self) -> None:
        """A node planned to its median capacity has no margin for the slow
        frames, which arrive exactly when a scene becomes busy."""
        result = BenchmarkResult("x", 10, [10.0] * 9 + [100.0])
        assert result.cameras_supported(8.0) < 1000 / (8.0 * result.p50_ms)

    def test_report_formatting(self) -> None:
        results = {"detect": benchmark("detect", lambda: None, iterations=5, warmup=1)}
        report = format_report(results, {"platform": "test"})
        assert "detect" in report and "cams@8fps" in report


class TestDetectionEvaluation:
    def test_perfect_detector_scores_one(self) -> None:
        truth = [[GroundTruthBox(BBox(10, 10, 60, 110), "person")]]
        predictions = [[Detection(BBox(10, 10, 60, 110), ObjectClass.PERSON, 0.95)]]
        assert evaluate_detection(predictions, truth).map50 == pytest.approx(1.0)

    def test_missed_detection_scores_zero(self) -> None:
        truth = [[GroundTruthBox(BBox(10, 10, 60, 110), "person")]]
        assert evaluate_detection([[]], truth).map50 == pytest.approx(0.0)

    def test_offset_boxes_lose_strict_iou(self) -> None:
        truth = [[GroundTruthBox(BBox(10, 10, 60, 110), "person")]]
        predictions = [[Detection(BBox(18, 18, 68, 118), ObjectClass.PERSON, 0.95)]]
        metrics = evaluate_detection(predictions, truth)
        assert metrics.map50 == pytest.approx(1.0)
        assert metrics.map50_95 < 0.6

    def test_duplicate_prediction_is_a_false_positive(self) -> None:
        truth = [[GroundTruthBox(BBox(10, 10, 60, 110), "person")]]
        predictions = [[
            Detection(BBox(10, 10, 60, 110), ObjectClass.PERSON, 0.95),
            Detection(BBox(11, 11, 61, 111), ObjectClass.PERSON, 0.90),
        ]]
        assert evaluate_detection(predictions, truth).per_class["person"]["precision"] == pytest.approx(0.5)

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate_detection([[]], [[], []])

    def test_average_precision_helper(self) -> None:
        assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


class TestAnprEvaluation:
    def test_measures_through_the_grammar_stage(self) -> None:
        """The only way to see whether grammar correction is a net gain."""
        metrics = evaluate_anpr([
            ("MH12AB1234", "MH12AB1234"),   # clean
            ("MHI2A8I234", "MH12AB1234"),   # repaired to truth
            ("0L8CAF1234", "DL8CAF1234"),   # repaired to truth
            ("", "TN07CD5043"),             # no read
        ])
        assert metrics.total == 4
        assert metrics.exact_matches == 3
        assert metrics.corrected_to_truth == 2
        assert metrics.no_read == 1
        assert metrics.corrupted_by_correction == 0

    def test_detects_correction_that_corrupts_a_good_read(self) -> None:
        """The most dangerous failure mode: output that looks clean but names
        a different vehicle."""
        metrics = evaluate_anpr([("MH12AB1234", "MH12AB1234")])
        assert metrics.corrupted_by_correction == 0
        assert metrics.plate_accuracy == pytest.approx(1.0)


class TestFaceEvaluation:
    def test_separates_false_accepts_from_false_rejects(self) -> None:
        """They carry entirely different operational costs."""
        from ibvap.vision.face import FaceGallery, l2_normalize

        def embedding(seed: int, noise: float = 0.0) -> np.ndarray:
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=128).astype(np.float32)
            if noise:
                vector += noise * np.random.default_rng(seed + 7).normal(size=128).astype(np.float32)
            return l2_normalize(vector)

        gallery = FaceGallery(match_threshold=0.45, min_margin=0.05)
        for index in range(4):
            gallery.enroll(f"P{index}", embedding(index))

        probes = [(embedding(i, 0.2), f"P{i}") for i in range(4)]
        probes += [(embedding(500 + i), None) for i in range(4)]

        metrics = evaluate_face_matching(gallery, probes)
        assert metrics.genuine_attempts == 4
        assert metrics.impostor_attempts == 4
        assert metrics.false_accept_rate == pytest.approx(0.0)
        assert metrics.identity_confusions == 0


class TestDrift:
    @staticmethod
    def _feed(monitor: DriftMonitor, frames: int, per_frame: int, score: float,
              height: float, luma: float, obj_class: ObjectClass = ObjectClass.PERSON,
              seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        for _ in range(frames):
            detections = [
                Detection(BBox(0, 0, 50, height * 720), obj_class,
                          float(np.clip(rng.normal(score, 0.06), 0, 1)))
                for _ in range(per_frame)
            ]
            monitor.observe(detections, detections, frame_height=720, luma=luma)

    def test_steady_state_shows_no_drift(self) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 600, 2, 0.82, 0.25, 130)
        monitor.set_reference()
        monitor.reset_window()

        self._feed(monitor, 600, 2, 0.82, 0.25, 130, seed=1)
        report = monitor.compare()
        assert not report.drifted
        assert report.score < 0.1

    def test_detects_a_failed_illuminator(self) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 600, 2, 0.82, 0.25, 130)
        monitor.set_reference()
        monitor.reset_window()

        self._feed(monitor, 600, 0, 0.0, 0.25, 8, seed=2)
        report = monitor.compare()
        assert report.drifted
        assert any("brightness" in finding for finding in report.findings)

    def test_detects_a_re_aimed_camera(self) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 600, 2, 0.82, 0.25, 130)
        monitor.set_reference()
        monitor.reset_window()

        self._feed(monitor, 600, 5, 0.55, 0.05, 120, obj_class=ObjectClass.CAR, seed=3)
        report = monitor.compare()
        assert report.drifted
        assert report.findings

    def test_refuses_to_judge_on_too_few_frames(self) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 600, 2, 0.82, 0.25, 130)
        monitor.set_reference()
        monitor.reset_window()

        self._feed(monitor, 20, 2, 0.82, 0.25, 130)
        assert any("too few" in finding for finding in monitor.compare().findings)

    def test_requires_a_reference(self) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 200, 2, 0.8, 0.25, 130)
        assert any("no reference" in finding for finding in monitor.compare().findings)

    def test_profiles_persist(self, workspace: Path) -> None:
        monitor = DriftMonitor("cam")
        self._feed(monitor, 300, 2, 0.8, 0.25, 130)
        reference = monitor.set_reference()

        save_profiles(workspace / "profiles.json", {"cam": reference})
        restored = load_profiles(workspace / "profiles.json")
        assert restored["cam"].detection_rate == pytest.approx(reference.detection_rate)

    def test_missing_profile_file_is_empty(self, workspace: Path) -> None:
        assert load_profiles(workspace / "absent.json") == {}
