"""Face detection, embedding and watchlist matching.

Biometric identification is the most consequential capability in this platform
and the one with the least tolerance for a careless false positive: a wrong hit
puts a real person in front of an armed response. Three design choices follow
from that:

* **Quality gating before matching.** A face that is too small, too blurred or
  too far off-frontal is rejected outright rather than matched at low
  confidence. Most catastrophic false matches come from comparing degraded
  probes, not from a bad threshold.
* **A margin requirement, not just a threshold.** A probe must beat the
  runner-up in the gallery by a clear margin. A probe that is 0.61 against two
  different people is not a match at 0.6; it is an ambiguous face.
* **Explicit privacy controls.** Embeddings of passers-by are never persisted
  by default; only deliberate watchlist enrolments are stored.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ibvap.core.errors import StorageError
from ibvap.core.logging import get_logger
from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.vision.backends import InferenceBackend
from ibvap.vision.preprocess import crop, sharpness

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FaceDetection:
    """A detected face with the quality signals used to gate recognition."""

    bbox: BBox
    score: float
    #: Five-point landmarks (eyes, nose, mouth corners) when the model emits them.
    landmarks: np.ndarray | None = None
    #: Variance-of-Laplacian focus measure of the face crop.
    focus: float = 0.0
    #: Face height in pixels - the dominant predictor of recognition accuracy.
    pixel_height: float = 0.0

    def quality_ok(self, *, min_height: float = 48.0, min_focus: float = 25.0) -> bool:
        """Whether this face is good enough to attempt identification."""
        return self.pixel_height >= min_height and self.focus >= min_focus


class FaceDetector:
    """Face localisation with a neural backend and a Haar-cascade fallback.

    The cascade fallback is frontal-only and markedly weaker, but it keeps face
    detection functioning on a node whose model artefacts have not synced. As
    with the motion detector, the degraded mode is reported rather than hidden.
    """

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        min_height: float = 48.0,
        min_focus: float = 25.0,
    ) -> None:
        self.backend = backend
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (backend.input_size() if backend else None) or input_size
        self.min_height = min_height
        self.min_focus = min_focus
        self._cascade: cv2.CascadeClassifier | None = None
        if backend is None:
            self._cascade = self._load_cascade()

    @property
    def is_neural(self) -> bool:
        return self.backend is not None

    @property
    def mode(self) -> str:
        if self.backend is not None:
            return "neural"
        return "haar_fallback" if self._cascade is not None else "unavailable"

    #: Cascade XMLs ship with some OpenCV builds and not others - the
    #: ``headless`` wheels from OpenCV 5 dropped them - so several well-known
    #: locations are probed before giving up.
    _CASCADE_SEARCH_PATHS: tuple[str, ...] = (
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
    )

    @classmethod
    def _load_cascade(cls) -> cv2.CascadeClassifier | None:
        filename = "haarcascade_frontalface_default.xml"
        candidates: list[Path] = []
        try:
            candidates.append(Path(cv2.data.haarcascades) / filename)
        except AttributeError:  # pragma: no cover - very old/minimal builds
            pass
        candidates.extend(Path(d) / filename for d in cls._CASCADE_SEARCH_PATHS)

        for path in candidates:
            try:
                if not path.is_file():
                    continue
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    log.info("haar_cascade_loaded", path=str(path))
                    return cascade
            except Exception as exc:  # pragma: no cover - build dependent
                log.debug("haar_cascade_load_failed", path=str(path), error=str(exc))

        # Not an error: it simply means face detection needs its model artefact.
        log.info("haar_cascade_unavailable", searched=len(candidates))
        return None

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image, in source pixel coordinates."""
        if image is None or image.size == 0:
            return []
        faces = self._detect_neural(image) if self.backend else self._detect_cascade(image)

        for face in faces:
            face.pixel_height = face.bbox.height
            face_crop = crop(image, face.bbox)
            face.focus = sharpness(face_crop) if face_crop.size else 0.0
        return faces

    def _detect_neural(self, image: np.ndarray) -> list[FaceDetection]:
        from ibvap.vision.detector import ObjectDetector

        detector = ObjectDetector(
            self.backend,  # type: ignore[arg-type]
            class_names=("face",),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            input_size=self.input_size,
        )
        return [FaceDetection(bbox=d.bbox, score=d.score) for d in detector.detect(image)]

    def _detect_cascade(self, image: np.ndarray) -> list[FaceDetection]:
        if self._cascade is None:
            return []
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        grey = cv2.equalizeHist(grey)
        rects = self._cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(int(self.min_height), int(self.min_height))
        )
        # A cascade returns no score, so a fixed conservative value is used;
        # it must never look as trustworthy as a neural detection downstream.
        return [
            FaceDetection(bbox=BBox(float(x), float(y), float(x + w), float(y + h)), score=0.5)
            for (x, y, w, h) in rects
        ]

    def detect_in_person(self, image: np.ndarray, person_box: BBox) -> list[FaceDetection]:
        """Detect faces within a person track's box.

        Restricting the search to a tracked person is much cheaper than a
        full-frame sweep at 1080p and removes the entire class of false faces
        found in foliage and brickwork.
        """
        # The head is in the top third of an upright person box.
        head_region = BBox(
            person_box.x1, person_box.y1,
            person_box.x2, person_box.y1 + person_box.height * 0.45,
        ).expand(0.1)
        region = crop(image, head_region)
        if region.size == 0:
            return []

        offset_x, offset_y = max(0.0, head_region.x1), max(0.0, head_region.y1)
        faces = self.detect(region)
        for face in faces:
            face.bbox = BBox(
                face.bbox.x1 + offset_x, face.bbox.y1 + offset_y,
                face.bbox.x2 + offset_x, face.bbox.y2 + offset_y,
            )
        return faces


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

#: ArcFace canonical 5-point template for a 112x112 aligned face.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray, landmarks: np.ndarray | None, output_size: int = 112
) -> np.ndarray:
    """Similarity-transform a face onto the canonical template.

    Alignment is worth several points of recognition accuracy: embedding models
    are trained on aligned crops, and feeding them raw boxes at a border camera's
    steep downward angle is a needless handicap. Without landmarks the crop is
    returned resized, which still works but less well.
    """
    if image is None or image.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    if landmarks is None or len(landmarks) < 5:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    src = np.asarray(landmarks[:5], dtype=np.float32).reshape(5, 2)
    dst = _ARCFACE_TEMPLATE * (output_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return cv2.warpAffine(image, matrix, (output_size, output_size), borderValue=0)


class FaceEmbedder:
    """Maps an aligned face crop to a unit-length embedding vector."""

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        *,
        input_size: int = 112,
        embedding_dim: int = 512,
    ) -> None:
        self.backend = backend
        size = (backend.input_size() if backend else None)
        self.input_size = size[0] if size else input_size
        self.embedding_dim = embedding_dim
        self._input_name = backend.input_names[0] if backend and backend.input_names else "input"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def embed(self, face_image: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        """Return a unit-norm embedding, or an empty array when unavailable."""
        if self.backend is None or face_image is None or face_image.size == 0:
            return np.empty(0, dtype=np.float32)

        aligned = align_face(face_image, landmarks, self.input_size)
        if aligned.size == 0:
            return np.empty(0, dtype=np.float32)

        rgb = aligned[:, :, ::-1] if aligned.ndim == 3 else cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
        # ArcFace convention: scale to [-1, 1].
        tensor = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]

        outputs = self.backend.run({self._input_name: np.ascontiguousarray(tensor)})
        if not outputs:
            return np.empty(0, dtype=np.float32)
        return l2_normalize(np.asarray(outputs[0], dtype=np.float32).reshape(-1))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length, leaving a zero vector untouched."""
    norm = float(np.linalg.norm(vector))
    return vector if norm < 1e-9 else (vector / norm).astype(np.float32)


# --------------------------------------------------------------------------- #
# Watchlist gallery
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FaceMatch:
    """Result of comparing a probe embedding against the gallery."""

    person_id: str
    name: str
    #: Cosine similarity in ``[-1, 1]``; for unit vectors this is the dot product.
    similarity: float
    #: Gap to the next-best identity. Small margins indicate an ambiguous face.
    margin: float
    #: Operator-facing watchlist category, e.g. ``wanted``, ``staff``.
    category: str = ""
    matched: bool = False


@dataclass(slots=True)
class GalleryEntry:
    """A watchlist identity and its enrolled embeddings."""

    person_id: str
    name: str
    category: str = "watchlist"
    notes: str = ""
    embeddings: list[np.ndarray] = field(default_factory=list)


class FaceGallery:
    """In-memory watchlist with vectorised cosine matching.

    A flat NumPy matrix is used rather than an ANN index: watchlists at a BOP
    are hundreds of identities, not millions, and an exhaustive dot product over
    a few thousand 512-d vectors costs well under a millisecond. Adding a FAISS
    dependency for that would be pure operational burden.
    """

    def __init__(
        self,
        *,
        match_threshold: float = 0.42,
        min_margin: float = 0.05,
    ) -> None:
        self.match_threshold = match_threshold
        self.min_margin = min_margin
        self._entries: dict[str, GalleryEntry] = {}
        # Flattened view rebuilt on mutation; matching is far more frequent
        # than enrolment, so the cost belongs on the write side.
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._owner_ids: list[str] = []
        self._lock = threading.RLock()

    # -- enrolment -------------------------------------------------------- #

    def enroll(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        name: str | None = None,
        category: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Add an embedding for an identity. Multiple enrolments are encouraged.

        Several views per person (frontal, three-quarter, with and without
        headwear) raise recall far more than tuning the threshold does.
        """
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if vector.size == 0:
            raise ValueError("cannot enroll an empty embedding")

        with self._lock:
            entry = self._entries.get(person_id)
            if entry is None:
                entry = GalleryEntry(
                    person_id=person_id,
                    name=name or person_id,
                    category=category or "watchlist",
                    notes=notes or "",
                )
                self._entries[person_id] = entry
            # Only overwrite metadata the caller actually supplied. Adding a
            # second view of an already-enrolled person must not silently reset
            # their watchlist category back to the default.
            if name is not None:
                entry.name = name
            if category is not None:
                entry.category = category
            if notes is not None:
                entry.notes = notes
            entry.embeddings.append(vector)
            self._rebuild()

    def remove(self, person_id: str) -> bool:
        """Remove an identity and all its embeddings."""
        with self._lock:
            if person_id not in self._entries:
                return False
            del self._entries[person_id]
            self._rebuild()
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._rebuild()

    def _rebuild(self) -> None:
        """Flatten the per-identity embeddings into one matrix."""
        vectors: list[np.ndarray] = []
        owners: list[str] = []
        for entry in self._entries.values():
            for vector in entry.embeddings:
                vectors.append(vector)
                owners.append(entry.person_id)
        self._matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), np.float32)
        self._owner_ids = owners

    # -- query ------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Number of enrolled identities."""
        return len(self._entries)

    @property
    def embedding_count(self) -> int:
        return len(self._owner_ids)

    def identities(self) -> list[GalleryEntry]:
        with self._lock:
            return list(self._entries.values())

    def get(self, person_id: str) -> GalleryEntry | None:
        return self._entries.get(person_id)

    def match(self, embedding: np.ndarray) -> FaceMatch | None:
        """Find the best gallery identity for a probe embedding.

        Returns ``None`` only when matching is impossible (empty gallery or
        malformed probe). Otherwise a :class:`FaceMatch` is always returned,
        with ``matched`` recording whether it cleared both the similarity
        threshold and the margin requirement - callers frequently want the
        near-miss for audit even when it is not an alert.
        """
        probe = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        with self._lock:
            if probe.size == 0 or self._matrix.size == 0:
                return None
            if probe.shape[0] != self._matrix.shape[1]:
                log.warning(
                    "embedding_dim_mismatch",
                    probe_dim=int(probe.shape[0]),
                    gallery_dim=int(self._matrix.shape[1]),
                )
                return None

            # Unit vectors, so a dot product is the cosine similarity.
            similarities = self._matrix @ probe

            # Score each identity by its best-matching enrolment.
            per_identity: dict[str, float] = {}
            for owner, score in zip(self._owner_ids, similarities, strict=True):
                value = float(score)
                if value > per_identity.get(owner, -2.0):
                    per_identity[owner] = value

            ranked = sorted(per_identity.items(), key=lambda kv: kv[1], reverse=True)
            best_id, best_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = best_score - runner_up

            entry = self._entries[best_id]
            matched = best_score >= self.match_threshold and (
                len(ranked) == 1 or margin >= self.min_margin
            )
            return FaceMatch(
                person_id=best_id,
                name=entry.name,
                similarity=round(best_score, 4),
                margin=round(margin, 4),
                category=entry.category,
                matched=matched,
            )

    # -- persistence ------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the gallery as an ``.npz`` plus a JSON sidecar of metadata."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            arrays: dict[str, np.ndarray] = {}
            metadata: dict[str, dict[str, str]] = {}
            for entry in self._entries.values():
                if entry.embeddings:
                    arrays[entry.person_id] = np.vstack(entry.embeddings).astype(np.float32)
                metadata[entry.person_id] = {
                    "name": entry.name, "category": entry.category, "notes": entry.notes,
                }
            try:
                np.savez_compressed(target, **arrays)
                target.with_suffix(".json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                raise StorageError(f"failed to save gallery to {target}: {exc}") from exc

    def load(self, path: str | Path) -> None:
        """Replace the gallery contents from a saved ``.npz`` + sidecar."""
        source = Path(path)
        if not source.exists() and source.with_suffix(".npz").exists():
            source = source.with_suffix(".npz")
        if not source.exists():
            raise StorageError(f"gallery file not found: {source}")

        meta_path = source.with_suffix(".json")
        metadata: dict[str, dict[str, str]] = {}
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("gallery_metadata_unreadable", path=str(meta_path), error=str(exc))

        with self._lock:
            self._entries.clear()
            with np.load(source) as data:
                for person_id in data.files:
                    info = metadata.get(person_id, {})
                    vectors = [l2_normalize(v) for v in np.atleast_2d(data[person_id])]
                    self._entries[person_id] = GalleryEntry(
                        person_id=person_id,
                        name=info.get("name", person_id),
                        category=info.get("category", "watchlist"),
                        notes=info.get("notes", ""),
                        embeddings=vectors,
                    )
            self._rebuild()
        log.info("gallery_loaded", identities=self.size, embeddings=self.embedding_count)


def blur_face(image: np.ndarray, box: BBox, *, strength: int = 25) -> np.ndarray:
    """Blur a face region in place-safe fashion, for privacy-preserving evidence.

    Used when ``privacy.blur_unmatched_faces`` is set, so an evidence snapshot
    of an intrusion does not incidentally build a biometric record of every
    uninvolved person in frame.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.clip(w, h).as_int_tuple()
    if x2 <= x1 or y2 <= y1:
        return image
    out = image.copy()
    region = out[y1:y2, x1:x2]
    # Kernel must be odd and scale with the face, or small faces stay readable.
    kernel = max(3, (min(region.shape[:2]) // strength) * 2 + 1)
    out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return out


def face_to_detection(face: FaceDetection) -> Detection:
    """Adapt a :class:`FaceDetection` to the generic detection type."""
    return Detection(
        bbox=face.bbox,
        obj_class=ObjectClass.PERSON,
        score=face.score,
        raw_label="face",
        attributes={"kind": "face", "focus": round(face.focus, 2)},
    )
