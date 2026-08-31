"""Secondary ImageNet classification and the taxonomy mapping."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import make_track

from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.vision.backends import CallableBackend
from ibvap.vision.classifier import (
    LIVESTOCK_INDICES,
    Classification,
    ImageNetClassifier,
    imagenet_to_ibvap,
    looks_like_livestock,
)


def logit_head(index: int, magnitude: float = 9.0):
    """A backend that confidently predicts one ImageNet class."""
    def run(_feeds):
        logits = np.full((1, 1000), -6.0, np.float32)
        logits[0, index] = magnitude
        return logits
    return CallableBackend(run, input_shape=(1, 3, 224, 224))


@pytest.fixture
def crop_image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (140, 70, 3)).astype(np.uint8)


class TestTaxonomyMapping:
    @pytest.mark.parametrize("index", sorted(LIVESTOCK_INDICES))
    def test_livestock_maps_to_animal(self, index: int) -> None:
        """The classes that actually walk into a border fence at night."""
        assert imagenet_to_ibvap(index) is ObjectClass.ANIMAL

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (207, ObjectClass.ANIMAL),      # golden retriever
            (285, ObjectClass.ANIMAL),      # Egyptian cat
            (717, ObjectClass.CAR),         # pickup
            (468, ObjectClass.CAR),         # cab
            (867, ObjectClass.TRUCK),       # trailer truck
            (779, ObjectClass.BUS),         # school bus
            (670, ObjectClass.MOTORCYCLE),  # motor scooter
            (671, ObjectClass.BICYCLE),     # mountain bike
            (814, ObjectClass.BOAT),        # speedboat
            (414, ObjectClass.BAG),         # backpack
        ],
    )
    def test_known_classes(self, index: int, expected: ObjectClass) -> None:
        assert imagenet_to_ibvap(index) is expected

    @pytest.mark.parametrize("index", [559, 950, 999, 700])
    def test_irrelevant_classes_are_unknown(self, index: int) -> None:
        """Furniture and food have no surveillance meaning and must not be
        forced onto the taxonomy."""
        assert imagenet_to_ibvap(index) is ObjectClass.UNKNOWN

    def test_a_large_share_of_imagenet_is_animal(self) -> None:
        """The reason this backbone is useful here at all."""
        animals = sum(
            1 for i in range(1000) if imagenet_to_ibvap(i) is ObjectClass.ANIMAL
        )
        assert 300 < animals < 500


class TestClassification:
    def test_classifies_and_maps(self, crop_image: np.ndarray) -> None:
        classifier = ImageNetClassifier(logit_head(345))  # ox
        verdict = classifier.classify(crop_image)
        assert verdict is not None
        assert verdict.index == 345
        assert verdict.obj_class is ObjectClass.ANIMAL
        assert verdict.is_animal
        assert verdict.confidence > 0.99

    def test_topk_is_returned_for_audit(self, crop_image: np.ndarray) -> None:
        verdict = ImageNetClassifier(logit_head(345)).classify(crop_image, topk=3)
        assert verdict is not None
        assert len(verdict.topk) == 3
        assert verdict.topk[0][0] == 345

    def test_tiny_crops_are_refused(self) -> None:
        """Upscaling a 12-pixel crop to 224 invents detail the classifier will
        confidently interpret; refusing is the honest answer."""
        classifier = ImageNetClassifier(logit_head(345))
        assert classifier.classify(np.zeros((10, 8, 3), np.uint8)) is None

    def test_unavailable_without_a_backend(self) -> None:
        classifier = ImageNetClassifier()
        assert not classifier.available
        assert classifier.classify(np.zeros((100, 100, 3), np.uint8)) is None

    def test_empty_input(self) -> None:
        assert ImageNetClassifier(logit_head(1)).classify(np.empty((0, 0, 3), np.uint8)) is None


class TestRefinement:
    def test_reclassifies_a_wrong_detection(self, scene: np.ndarray) -> None:
        """The headline case: the fallback detector reads a wide cow silhouette
        as a vehicle, and suppression by class therefore never matches it."""
        classifier = ImageNetClassifier(logit_head(345))  # ox
        detection = Detection(BBox(100, 100, 260, 200), ObjectClass.CAR, 0.5)

        refined = classifier.refine_detection(scene, detection)
        assert refined.obj_class is ObjectClass.ANIMAL
        assert refined.attributes["reclassified_from"] == "car"
        assert refined.attributes["imagenet_index"] == 345

    def test_leaves_unmapped_classes_alone(self, scene: np.ndarray) -> None:
        classifier = ImageNetClassifier(logit_head(559))  # folding chair
        detection = Detection(BBox(100, 100, 200, 300), ObjectClass.PERSON, 0.8)
        assert classifier.refine_detection(scene, detection).obj_class is ObjectClass.PERSON

    def test_low_confidence_is_ignored(self, scene: np.ndarray) -> None:
        # A nearly flat distribution: no class is credible.
        classifier = ImageNetClassifier(logit_head(345, magnitude=0.01), min_confidence=0.5)
        detection = Detection(BBox(100, 100, 260, 200), ObjectClass.CAR, 0.5)
        assert classifier.refine_detection(scene, detection).obj_class is ObjectClass.CAR

    def test_track_refinement_runs_once(self, scene: np.ndarray) -> None:
        """Once per track, not per frame: the verdict cannot change between
        consecutive frames of the same object."""
        calls = {"n": 0}

        def counting(_feeds):
            calls["n"] += 1
            logits = np.full((1, 1000), -6.0, np.float32)
            logits[0, 345] = 9.0
            return logits

        classifier = ImageNetClassifier(CallableBackend(counting, input_shape=(1, 3, 224, 224)))
        track = make_track(1, 100, 100, width=160, height=100, obj_class=ObjectClass.CAR)

        assert classifier.refine_track(scene, track) is True
        assert track.obj_class is ObjectClass.ANIMAL
        for _ in range(5):
            classifier.refine_track(scene, track)
        assert calls["n"] == 1

    def test_never_promotes_to_person(self, scene: np.ndarray) -> None:
        """ImageNet-1k has no person class, so a PERSON verdict could only come
        from inferring absence. No index may produce one."""
        assert all(
            imagenet_to_ibvap(index) is not ObjectClass.PERSON for index in range(1000)
        )


class TestLivestockHelper:
    def test_confident_livestock(self) -> None:
        verdict = Classification(345, 0.9, ObjectClass.ANIMAL)
        assert looks_like_livestock(verdict)

    def test_marginal_verdict_rejected(self) -> None:
        assert not looks_like_livestock(Classification(345, 0.1, ObjectClass.ANIMAL))

    def test_none_is_safe(self) -> None:
        assert not looks_like_livestock(None)
