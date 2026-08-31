"""Secondary classification with an ImageNet-pretrained backbone.

The detector says *where* something is and gives a coarse class. This stage
looks at the crop again with a lightweight image classifier (MobileNetV3 by
default) and refines that answer.

It earns its place in two specific situations:

**Rescuing the classical fallback.** When no detector artefact is present, the
platform falls back to background subtraction, which classifies purely by
bounding-box aspect ratio - tall means person, wide means vehicle. That is a
guess, not a classification. A 5 MB MobileNet over the crop turns it into a
real one at a few milliseconds per object.

**Separating livestock from people.** Stray cattle on a rural fence line are
the single largest source of false intrusion alarms in this domain, and
ImageNet is unusually strong here: roughly 400 of its 1000 classes are animals,
covering cattle, buffalo, dogs, goats and sheep in detail. Confirming "this is
an ox, not a person" is exactly what the suppression rules need.

**An honest limitation, and it is a large one.** ImageNet-1k has **no person
class**. It cannot confirm that something is a human, only that it is probably
one of the animals or objects it does know. So this stage is used to *demote*
and *reclassify* - never to promote something to PERSON. A crop it cannot place
is left with whatever the detector said.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ibvap.core.logging import get_logger
from ibvap.core.types import Detection, ObjectClass, Track
from ibvap.vision.backends import InferenceBackend
from ibvap.vision.preprocess import crop

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# ImageNet-1k -> IBVAP taxonomy
# --------------------------------------------------------------------------- #

#: Contiguous ImageNet-1k index ranges that are unambiguously animals.
#: Expressed as ranges because they genuinely are contiguous in the standard
#: class ordering - listing 400 indices individually would be unreadable and
#: no more precise.
_ANIMAL_RANGES: tuple[tuple[int, int], ...] = (
    (0, 116),     # birds, reptiles, amphibians, fish, invertebrates
    (118, 150),   # crustaceans, marine invertebrates
    (151, 275),   # dogs (151-268), then wolves, foxes and wild canids
    (276, 299),   # hyenas, cats, big cats
    (300, 319),   # insects
    (321, 397),   # butterflies, hares, rodents, ungulates, primates
    (398, 400),   # remaining mammals before the object classes begin
)

#: Livestock and large mammals worth naming, because these are the ones that
#: actually walk into a border fence line at night.
LIVESTOCK_INDICES: frozenset[int] = frozenset({
    339,  # sorrel (horse)
    345,  # ox
    346,  # water buffalo
    347,  # bison
    348,  # ram
    349,  # bighorn sheep
    350,  # ibex
    353,  # gazelle
    354,  # Arabian camel
    386,  # African elephant
})

#: Vehicle classes, grouped to the IBVAP taxonomy. ImageNet is fine-grained
#: here (it distinguishes a limousine from a cab), which is more resolution
#: than this platform needs, so several indices collapse onto one class.
_VEHICLE_INDICES: dict[ObjectClass, frozenset[int]] = {
    ObjectClass.CAR: frozenset({
        407,  # ambulance
        436,  # beach wagon
        468,  # cab
        511,  # convertible
        609,  # jeep
        627,  # limousine
        656,  # minivan
        661,  # Model T
        705,  # passenger car
        717,  # pickup
        734,  # police van
        751,  # racer
        757,  # recreational vehicle
        817,  # sports car
    }),
    ObjectClass.TRUCK: frozenset({
        555,  # fire engine
        561,  # forklift
        569,  # garbage truck
        586,  # half track
        675,  # moving van
        803,  # snowplough
        864,  # tow truck
        867,  # trailer truck
        913,  # wreck
    }),
    ObjectClass.BUS: frozenset({
        654,  # minibus
        779,  # school bus
        874,  # trolleybus
        829,  # streetcar
    }),
    ObjectClass.MOTORCYCLE: frozenset({
        665,  # moped
        670,  # motor scooter
        671,  # mountain bike is cycled below; kept out deliberately
    }) - {671},
    ObjectClass.BICYCLE: frozenset({
        444,  # bicycle-built-for-two
        671,  # mountain bike
        870,  # tricycle
    }),
    ObjectClass.BOAT: frozenset({
        472,  # canoe
        484,  # catamaran
        554,  # fireboat
        625,  # lifeboat
        628,  # liner
        724,  # pirate ship
        814,  # speedboat
        833,  # submarine
        871,  # trimaran
        914,  # yawl
    }),
    ObjectClass.BAG: frozenset({
        414,  # backpack
        636,  # mailbag
        728,  # plastic bag
        797,  # sleeping bag
        831,  # studio couch is not a bag; excluded below
        868,  # tray
    }) - {831, 868},
}


def imagenet_to_ibvap(index: int) -> ObjectClass:
    """Map an ImageNet-1k class index onto the IBVAP taxonomy.

    Returns :attr:`ObjectClass.UNKNOWN` for the many classes that have no
    surveillance meaning (furniture, food, instruments) - and, importantly,
    for anything that might be a person, since ImageNet cannot express that.
    """
    if index in LIVESTOCK_INDICES:
        return ObjectClass.ANIMAL
    for low, high in _ANIMAL_RANGES:
        if low <= index <= high:
            return ObjectClass.ANIMAL
    for obj_class, indices in _VEHICLE_INDICES.items():
        if index in indices:
            return obj_class
    return ObjectClass.UNKNOWN


@dataclass(slots=True)
class Classification:
    """One classifier verdict on one crop."""

    #: Winning ImageNet class index.
    index: int
    #: Probability of that class after softmax.
    confidence: float
    #: The IBVAP class it maps to, possibly UNKNOWN.
    obj_class: ObjectClass
    #: Top-k indices and probabilities, for audit and debugging.
    topk: list[tuple[int, float]] = field(default_factory=list)

    @property
    def is_animal(self) -> bool:
        return self.obj_class is ObjectClass.ANIMAL


class ImageNetClassifier:
    """MobileNet-style ImageNet classifier over object crops.

    Defaults match the torchvision MobileNetV3 preprocessing contract
    (224x224 RGB, ImageNet channel statistics). A model exported with
    different statistics must be given its own, or every prediction will be
    subtly wrong in a way that still looks like plausible output.
    """

    #: Standard ImageNet channel statistics.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    DEFAULT_INPUT = (224, 224)

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        *,
        input_size: tuple[int, int] | None = None,
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
        min_confidence: float = 0.35,
        min_crop_pixels: int = 24,
    ) -> None:
        self.backend = backend
        self.input_size = (backend.input_size() if backend else None) or input_size or self.DEFAULT_INPUT
        self.mean = mean
        self.std = std
        self.min_confidence = min_confidence
        self.min_crop_pixels = min_crop_pixels
        self._input_name = backend.input_names[0] if backend and backend.input_names else "input"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def classify(self, image: np.ndarray, *, topk: int = 5) -> Classification | None:
        """Classify one crop. Returns None when it cannot be classified."""
        if self.backend is None or image is None or image.size == 0:
            return None
        if min(image.shape[:2]) < self.min_crop_pixels:
            # Upscaling a 12-pixel crop to 224 invents detail the classifier
            # will confidently interpret. Refusing is the honest answer.
            return None

        width, height = self.input_size
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1] if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))[None, ...]

        outputs = self.backend.run({self._input_name: tensor})
        if not outputs:
            return None

        logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if logits.size == 0:
            return None

        # Softmax, numerically stabilised. Applied unconditionally: a graph
        # that already ends in softmax is idempotent under it to within
        # floating-point noise, so this cannot make a correct output wrong.
        shifted = logits - logits.max()
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        order = np.argsort(probabilities)[::-1][:topk]
        best = int(order[0])
        return Classification(
            index=best,
            confidence=float(probabilities[best]),
            obj_class=imagenet_to_ibvap(best),
            topk=[(int(i), float(probabilities[i])) for i in order],
        )

    # -- refinement -------------------------------------------------------- #

    def refine_detection(self, frame: np.ndarray, detection: Detection) -> Detection:
        """Return ``detection`` with its class corrected where warranted."""
        verdict = self.classify(crop(frame, detection.bbox, padding=0.08))
        if verdict is None or verdict.confidence < self.min_confidence:
            return detection
        if verdict.obj_class is ObjectClass.UNKNOWN:
            return detection

        # Never promote to PERSON - ImageNet has no such class, so a "person"
        # here could only ever be an inference from absence.
        if verdict.obj_class is ObjectClass.PERSON:  # pragma: no cover - unreachable
            return detection

        detection.attributes["imagenet_index"] = verdict.index
        detection.attributes["imagenet_confidence"] = round(verdict.confidence, 4)
        if verdict.obj_class is not detection.obj_class:
            detection.attributes["reclassified_from"] = detection.obj_class.value
            detection.obj_class = verdict.obj_class
        return detection

    def refine_track(self, frame: np.ndarray, track: Track) -> bool:
        """Refine a track's class once. Returns True when it was changed.

        Runs once per track, not once per frame: the answer will not change
        between consecutive frames of the same object, and repeating it would
        turn a per-object cost into a per-frame one.
        """
        state = track.attributes.setdefault("_classified", {"done": False})
        if state["done"]:
            return False

        verdict = self.classify(crop(frame, track.bbox, padding=0.08))
        if verdict is None:
            return False
        state["done"] = True

        if verdict.confidence < self.min_confidence or verdict.obj_class is ObjectClass.UNKNOWN:
            return False

        track.attributes["imagenet_index"] = verdict.index
        track.attributes["imagenet_confidence"] = round(verdict.confidence, 4)
        if verdict.obj_class is not track.obj_class:
            log.debug(
                "track_reclassified",
                track=track.track_id, was=track.obj_class.value,
                now=verdict.obj_class.value, confidence=round(verdict.confidence, 3),
            )
            track.attributes["reclassified_from"] = track.obj_class.value
            track.obj_class = verdict.obj_class
            return True
        return False


def looks_like_livestock(verdict: Classification | None, *, min_confidence: float = 0.3) -> bool:
    """Whether a verdict indicates livestock specifically.

    Separate from the general animal test because livestock is the case the
    suppression rules are actually written for, and a confident "ox" deserves
    more weight than a marginal "some animal".
    """
    if verdict is None or verdict.confidence < min_confidence:
        return False
    return verdict.index in LIVESTOCK_INDICES or verdict.is_animal
