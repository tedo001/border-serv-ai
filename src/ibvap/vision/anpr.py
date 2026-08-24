"""Automatic Number Plate Recognition.

The pipeline is the conventional three stages - locate the plate, rectify it,
read the glyphs - followed by a fourth that carries most of the field value:
**grammar-constrained normalisation** against the Indian number plate format.

That fourth stage matters more than model accuracy. A raw OCR pass on a plate
captured at 40 m through haze at 03:00 routinely returns ``MHI2A8I234``. Every
one of those errors is a glyph/digit confusion at a position where the format
already tells us which class of character is legal. Reading the same string as
``MH12AB1234`` is not guesswork; it is decoding under a known grammar. Without
it, ANPR output cannot be matched against a watchlist at all, because a single
wrong glyph makes the lookup miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from ibvap.core.logging import get_logger
from ibvap.core.types import BBox
from ibvap.vision.backends import InferenceBackend
from ibvap.vision.preprocess import crop, sharpness

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Indian number plate grammar
# --------------------------------------------------------------------------- #

#: RTO state/UT codes. Union territories reorganised in 2020 (DD+DN -> DD,
#: LA created for Ladakh), so both historical and current codes are accepted -
#: vehicles carrying the older plates are still very much on the roads.
STATE_CODES: frozenset[str] = frozenset(
    """AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP
       MZ NL OD OR PB PY RJ SK TG TN TR TS UA UK UP WB""".split()
)

#: Characters an OCR model confuses with a *letter* when the position demands
#: one, and vice versa. Derived from the visual similarity of the glyphs in the
#: IND font, which is where essentially all real ANPR errors come from.
_TO_LETTER: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
    "6": "G", "7": "T", "8": "B", "9": "P",
}
_TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6",
    "T": "7", "B": "8", "P": "9",
}

#: Several digits resemble more than one letter - ``0`` reads as O, D or Q
#: depending on the plate font and how the glyph is clipped. ``_TO_LETTER``
#: holds only the single most likely target, which is right for the body of a
#: plate but wrong for the two-character state code, where an external
#: authority (the RTO code list) can tell us which alternative is correct.
#: These alternates are searched only at those two positions, so the cost stays
#: bounded at a handful of comparisons.
_LETTER_ALTERNATES: dict[str, tuple[str, ...]] = {
    "0": ("O", "D", "Q", "C"),
    "1": ("I", "L", "T"),
    "2": ("Z",),
    "4": ("A",),
    "5": ("S",),
    "6": ("G", "C"),
    "7": ("T", "Z"),
    "8": ("B", "R"),
    "9": ("P", "G"),
}


def _resolve_state_code(candidate: str, original: str) -> str | None:
    """Try to repair an unrecognised two-letter state code.

    Only the first two characters are varied, and only where the *original*
    OCR character was a digit that the confusion table says could be several
    letters. Returns the repaired plate, or ``None`` when no substitution
    yields a real RTO code - in which case the plate stays flagged invalid
    rather than being forced onto a plausible-looking state.
    """
    if len(candidate) < 2 or len(original) < 2:
        return None

    options: list[tuple[str, ...]] = []
    for pos in (0, 1):
        source = original[pos]
        if source.isdigit():
            options.append(_LETTER_ALTERNATES.get(source, (candidate[pos],)))
        else:
            options.append((candidate[pos],))

    for first in options[0]:
        for second in options[1]:
            if first + second in STATE_CODES:
                return first + second + candidate[2:]
    return None

_ALNUM_RE = re.compile(r"[^A-Z0-9]")

#: Templates are strings of ``A`` (letter) and ``N`` (digit), plus literals.
#: Generated rather than hand-listed because the standard format has two
#: variable-length segments and enumerating 8 combinations by hand invites
#: exactly the kind of omission that silently drops valid plates.
def _standard_templates() -> list[str]:
    out = []
    for district_len in (1, 2):
        for series_len in (0, 1, 2, 3):
            out.append("AA" + "N" * district_len + "A" * series_len + "NNNN")
    return out


#: ``format_name -> template``. Order matters only for tie-breaking.
PLATE_TEMPLATES: list[tuple[str, str]] = [
    *[("standard", t) for t in _standard_templates()],
    # Bharat series (2021-): YY BH NNNN XX - one national plate, no state code.
    ("bharat", "NNBHNNNNA"),
    ("bharat", "NNBHNNNNAA"),
]


@dataclass(slots=True)
class PlateReading:
    """One normalised plate read with its provenance."""

    #: Grammar-corrected plate text, e.g. ``MH12AB1234``.
    text: str
    #: Raw OCR output before normalisation, retained for audit and retraining.
    raw_text: str
    #: Combined confidence in ``[0, 1]``: OCR confidence penalised by the
    #: number of grammar corrections that had to be applied.
    confidence: float
    #: Which template matched: ``standard``, ``bharat`` or ``unknown``.
    plate_format: str = "unknown"
    #: Number of characters changed by grammar correction.
    corrections: int = 0
    #: True when the text satisfies a known format and (for ``standard``) the
    #: state code is a real RTO code.
    valid: bool = False
    #: Plate location within the source frame.
    bbox: BBox | None = None
    #: Two-letter state code when derivable - useful for cross-border profiling.
    state_code: str = ""

    @property
    def is_actionable(self) -> bool:
        """Whether this read should be matched against a watchlist.

        An invalid or very low confidence read is worse than no read: it can
        collide with a real registration and put an innocent vehicle on an
        alert list.
        """
        return self.valid and self.confidence >= 0.5


def _coerce_to_template(text: str, template: str) -> tuple[str, int]:
    """Force ``text`` onto ``template``, returning the result and edit count.

    Returns ``("", -1)`` when a character cannot legally be coerced - e.g. a
    letter with no digit lookalike sitting in a digit position.
    """
    if len(text) != len(template):
        return "", -1

    out: list[str] = []
    edits = 0
    for ch, slot in zip(text, template, strict=True):
        if slot == "A":
            if ch.isalpha():
                out.append(ch)
            elif ch in _TO_LETTER:
                out.append(_TO_LETTER[ch])
                edits += 1
            else:
                return "", -1
        elif slot == "N":
            if ch.isdigit():
                out.append(ch)
            elif ch in _TO_DIGIT:
                out.append(_TO_DIGIT[ch])
                edits += 1
            else:
                return "", -1
        else:  # literal, e.g. the 'BH' of a Bharat-series plate
            if ch == slot:
                out.append(ch)
            elif slot in _TO_LETTER.values() and _TO_LETTER.get(ch) == slot:
                out.append(slot)
                edits += 1
            else:
                return "", -1
    return "".join(out), edits


def normalise_plate(raw: str, ocr_confidence: float = 1.0) -> PlateReading:
    """Normalise raw OCR text into a grammatically valid Indian plate.

    Every candidate template of the right length is tried; the one requiring
    the fewest character corrections wins, with a valid RTO state code breaking
    ties. Confidence is reduced for each correction so that a heavily "repaired"
    read cannot masquerade as a clean one downstream.
    """
    cleaned = _ALNUM_RE.sub("", (raw or "").upper())
    if not cleaned:
        return PlateReading(text="", raw_text=raw or "", confidence=0.0)

    best: tuple[int, bool, str, str] | None = None  # (edits, state_ok, text, fmt)
    for fmt, template in PLATE_TEMPLATES:
        candidate, edits = _coerce_to_template(cleaned, template)
        if edits < 0:
            continue
        state_ok = fmt != "standard" or candidate[:2] in STATE_CODES
        if not state_ok:
            repaired = _resolve_state_code(candidate, cleaned)
            if repaired is not None:
                # Same number of edits - we are choosing a better target for a
                # correction we were already making, not making another one.
                candidate, state_ok = repaired, True
        # Rank: fewest edits first, then prefer a real state code. A candidate
        # with an unknown state code is kept only if nothing better exists, so
        # a genuinely novel RTO code still produces output rather than nothing.
        key = (edits, not state_ok, candidate, fmt)
        if best is None or key[:2] < best[:2]:
            best = key

    if best is None:
        # No template fits: return the cleaned text, explicitly marked invalid.
        return PlateReading(
            text=cleaned,
            raw_text=raw,
            confidence=ocr_confidence * 0.3,
            plate_format="unknown",
            valid=False,
        )

    edits, state_bad, text, fmt = best
    # Each correction costs 8% confidence; four or more corrections means the
    # read is mostly invention and should not reach a watchlist.
    confidence = max(0.0, ocr_confidence * (1.0 - 0.08 * edits))
    valid = not state_bad or fmt != "standard"
    state_code = text[:2] if fmt == "standard" and text[:2] in STATE_CODES else ""

    return PlateReading(
        text=text,
        raw_text=raw,
        confidence=round(confidence, 4),
        plate_format=fmt,
        corrections=edits,
        valid=valid,
        state_code=state_code,
    )


def plates_match(a: str, b: str, *, max_distance: int = 0) -> bool:
    """Compare two plate strings, optionally tolerating N character edits.

    A tolerance of 1 is often justified for watchlist matching at long range;
    anything higher produces false hits against unrelated registrations.
    """
    a_clean = _ALNUM_RE.sub("", (a or "").upper())
    b_clean = _ALNUM_RE.sub("", (b or "").upper())
    if a_clean == b_clean:
        return True
    if max_distance <= 0 or abs(len(a_clean) - len(b_clean)) > max_distance:
        return False
    return _levenshtein(a_clean, b_clean) <= max_distance


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, row-wise to keep memory at O(min(len))."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------- #
# Plate localisation
# --------------------------------------------------------------------------- #


#: Width/height ratio bounds. Indian plates come in one-row (long) and two-row
#: (squat) layouts; both must pass or every truck and two-wheeler is missed.
ONE_ROW_ASPECT = (2.0, 6.5)
TWO_ROW_ASPECT = (0.9, 2.0)


class PlateLocator:
    """Locates candidate plate regions inside a vehicle crop.

    Uses a neural detector when one is configured, and falls back to a
    classical morphological search otherwise. The classical path exploits the
    single most distinctive property of a number plate in gradient space: a
    dense horizontal band of near-vertical strokes (the glyph edges) enclosed
    in a strong rectangular boundary.
    """

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        *,
        score_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        max_candidates: int = 5,
    ) -> None:
        self.backend = backend
        self.score_threshold = score_threshold
        self.input_size = (backend.input_size() if backend else None) or input_size
        self.max_candidates = max_candidates
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))

    @property
    def is_neural(self) -> bool:
        return self.backend is not None

    def locate(self, image: np.ndarray) -> list[BBox]:
        """Return candidate plate boxes in ``image`` coordinates, best first."""
        if image is None or image.size == 0:
            return []
        if self.backend is not None:
            return self._locate_neural(image)
        return self._locate_classical(image)

    def _locate_neural(self, image: np.ndarray) -> list[BBox]:
        from ibvap.vision.detector import ObjectDetector

        # A plate detector is a single-class detector; reuse the generic
        # decoding path rather than duplicating YOLO output handling.
        detector = ObjectDetector(
            self.backend,  # type: ignore[arg-type]
            class_names=("plate",),
            score_threshold=self.score_threshold,
            input_size=self.input_size,
        )
        found = detector.detect(image)
        found.sort(key=lambda d: d.score, reverse=True)
        return [d.bbox for d in found[: self.max_candidates]]

    def _locate_classical(self, image: np.ndarray) -> list[BBox]:
        h, w = image.shape[:2]
        if h < 16 or w < 32:
            return []

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        grey = cv2.bilateralFilter(grey, 7, 40, 40)

        # Sobel in x only: plate glyphs are dominated by vertical strokes, and
        # ignoring horizontal gradient suppresses bumpers, shadows and skylines.
        grad = cv2.Sobel(grey, cv2.CV_8U, 1, 0, ksize=3)
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(h * w)

        scored: list[tuple[float, BBox]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 24 or bh < 8:
                continue
            area_fraction = (bw * bh) / frame_area
            if not (0.002 <= area_fraction <= 0.4):
                continue
            aspect = bw / float(bh)
            if not (
                ONE_ROW_ASPECT[0] <= aspect <= ONE_ROW_ASPECT[1]
                or TWO_ROW_ASPECT[0] <= aspect <= TWO_ROW_ASPECT[1]
            ):
                continue

            # Rank by edge density: a real plate is far busier than a body panel.
            region = binary[y : y + bh, x : x + bw]
            density = float(region.mean()) / 255.0
            # Plates sit low on a vehicle; bias towards the lower half of the crop.
            vertical_bias = 0.5 + 0.5 * ((y + bh / 2.0) / h)
            scored.append((density * vertical_bias, BBox(x, y, x + bw, y + bh)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [box for _, box in scored[: self.max_candidates]]


def deskew_plate(image: np.ndarray, *, max_angle: float = 25.0) -> np.ndarray:
    """Rotate a plate crop to horizontal using its dominant text orientation.

    Border cameras look down from mast height, so plates arrive rotated by
    5-20 degrees. OCR accuracy degrades sharply with skew, and this single
    correction typically recovers more reads than any OCR hyper-parameter.
    """
    if image is None or image.size == 0 or min(image.shape[:2]) < 8:
        return image

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(255 - binary)
    if coords is None or len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV reports the angle in (0, 90]; map it to the nearest horizontal.
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #

#: Blank is index 0 by CTC convention; the rest is the Indian plate alphabet.
#: 'I' and 'O' are legal in RTO series codes and so cannot be dropped, despite
#: being the worst confusion pairs - which is precisely why grammar
#: normalisation, not alphabet pruning, is the right place to resolve them.
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ctc_greedy_decode(
    logits: np.ndarray, charset: str = DEFAULT_CHARSET
) -> tuple[str, float]:
    """Greedy CTC decode of ``(T, C)`` logits into text and mean confidence.

    Collapses repeated labels and strips blanks, per the standard CTC rule.
    Confidence is the mean softmax probability of the emitted (non-blank)
    timesteps - averaging over blanks would inflate it towards 1.0 on short
    plates, since blanks dominate the sequence.
    """
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        return "", 0.0

    # Softmax over the class axis, numerically stabilised.
    shifted = arr - arr.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    indices = probs.argmax(axis=1)
    confidences = probs[np.arange(len(indices)), indices]

    chars: list[str] = []
    kept: list[float] = []
    previous = -1
    for idx, conf in zip(indices, confidences, strict=True):
        idx = int(idx)
        if idx != previous and idx != 0:
            char_pos = idx - 1  # index 0 is the CTC blank
            if 0 <= char_pos < len(charset):
                chars.append(charset[char_pos])
                kept.append(float(conf))
        previous = idx

    mean_conf = float(np.mean(kept)) if kept else 0.0
    return "".join(chars), mean_conf


class PlateOCR:
    """CRNN-style ONNX plate reader with CTC decoding."""

    #: Typical CRNN plate-recognition input geometry (width, height).
    DEFAULT_INPUT = (160, 48)

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        *,
        charset: str = DEFAULT_CHARSET,
        input_size: tuple[int, int] | None = None,
        min_sharpness: float = 12.0,
    ) -> None:
        self.backend = backend
        self.charset = charset
        self.input_size = (backend.input_size() if backend else None) or input_size or self.DEFAULT_INPUT
        self.min_sharpness = min_sharpness
        self._input_name = backend.input_names[0] if backend and backend.input_names else "input"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def read(self, plate_image: np.ndarray) -> tuple[str, float]:
        """Read a rectified plate crop, returning raw text and confidence."""
        if self.backend is None or plate_image is None or plate_image.size == 0:
            return "", 0.0

        # Reject crops too blurred to carry glyph information. Feeding these to
        # OCR does not fail loudly - it returns confident nonsense, which then
        # gets "corrected" by the grammar stage into a plausible wrong plate.
        if sharpness(plate_image) < self.min_sharpness:
            return "", 0.0

        w, h = self.input_size
        resized = cv2.resize(plate_image, (w, h), interpolation=cv2.INTER_CUBIC)
        grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        tensor = (grey.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = tensor[None, None, :, :]  # NCHW, single channel

        outputs = self.backend.run({self._input_name: tensor})
        if not outputs:
            return "", 0.0
        return ctc_greedy_decode(outputs[0], self.charset)


@dataclass(slots=True)
class AnprResult:
    """Everything ANPR learned about one vehicle in one frame."""

    reading: PlateReading | None = None
    candidates: list[PlateReading] = field(default_factory=list)
    #: Why no plate was produced, when ``reading`` is ``None``.
    reason: str = ""


class PlateReader:
    """End-to-end ANPR: locate, rectify, read and normalise."""

    def __init__(
        self,
        locator: PlateLocator,
        ocr: PlateOCR,
        *,
        min_confidence: float = 0.45,
        deskew: bool = True,
        upscale_to_height: int = 64,
    ) -> None:
        self.locator = locator
        self.ocr = ocr
        self.min_confidence = min_confidence
        self.deskew = deskew
        self.upscale_to_height = upscale_to_height

    @property
    def available(self) -> bool:
        """Whether a full read is possible. Localisation alone is not enough."""
        return self.ocr.available

    def read_vehicle(self, frame: np.ndarray, vehicle_box: BBox) -> AnprResult:
        """Run ANPR on the region of ``frame`` occupied by one vehicle."""
        # Expand slightly: detector boxes routinely clip the bumper, and the
        # plate is usually right at that edge.
        vehicle_crop = crop(frame, vehicle_box, padding=0.05)
        if vehicle_crop.size == 0:
            return AnprResult(reason="empty_crop")
        if not self.ocr.available:
            return AnprResult(reason="ocr_unavailable")

        boxes = self.locator.locate(vehicle_crop)
        if not boxes:
            return AnprResult(reason="no_plate_located")

        origin_x, origin_y = vehicle_box.expand(0.05).x1, vehicle_box.expand(0.05).y1
        candidates: list[PlateReading] = []

        for box in boxes:
            plate_crop = crop(vehicle_crop, box, padding=0.03)
            if plate_crop.size == 0:
                continue
            plate_crop = self._prepare(plate_crop)

            raw_text, ocr_conf = self.ocr.read(plate_crop)
            if not raw_text:
                continue

            reading = normalise_plate(raw_text, ocr_conf)
            # Translate the plate box from vehicle-crop space to frame space so
            # the evidence snapshot can draw it in the right place.
            reading.bbox = BBox(
                box.x1 + max(0.0, origin_x),
                box.y1 + max(0.0, origin_y),
                box.x2 + max(0.0, origin_x),
                box.y2 + max(0.0, origin_y),
            )
            candidates.append(reading)

        if not candidates:
            return AnprResult(reason="no_text_read")

        # Prefer valid plates, then confidence: a clean read of a real format
        # always beats a high-confidence read of something ungrammatical.
        candidates.sort(key=lambda r: (r.valid, r.confidence), reverse=True)
        best = candidates[0]
        if best.confidence < self.min_confidence:
            return AnprResult(candidates=candidates, reason="low_confidence")
        return AnprResult(reading=best, candidates=candidates)

    def _prepare(self, plate_crop: np.ndarray) -> np.ndarray:
        """Deskew and upscale a plate crop ahead of OCR."""
        if self.deskew:
            plate_crop = deskew_plate(plate_crop)
        h = plate_crop.shape[0]
        if 0 < h < self.upscale_to_height:
            # Small plates benefit markedly from cubic upscaling before OCR;
            # the network was trained at a fixed height and degrades on crops
            # that are heavily under-sampled relative to it.
            scale = self.upscale_to_height / h
            plate_crop = cv2.resize(
                plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        return plate_crop


# --------------------------------------------------------------------------- #
# Plate watchlist
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PlateWatchEntry:
    """A registration of interest and why it is being watched."""

    plate: str
    #: Operator-facing category: ``wanted``, ``stolen``, ``permitted``, ...
    category: str = "wanted"
    reason: str = ""
    #: Reference to the originating case file or intelligence report.
    reference: str = ""
    added_at: float = 0.0


@dataclass(slots=True)
class PlateWatchHit:
    """A watchlist match against a read plate."""

    entry: PlateWatchEntry
    reading: PlateReading
    #: Character edits between the read plate and the watchlist entry.
    distance: int = 0
    #: True when the plate matched exactly rather than fuzzily.
    exact: bool = True


class PlateWatchlist:
    """Registrations of interest, with optional fuzzy matching.

    Fuzzy matching is off by default. It is genuinely useful at long range,
    where one glyph is routinely lost - but each additional edit of tolerance
    multiplies the space of registrations that can collide, and a false hit
    here stops the wrong vehicle. One edit is the most that can be justified,
    and only when the read is otherwise clean.
    """

    def __init__(self, *, max_distance: int = 0, min_confidence: float = 0.5) -> None:
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self._entries: dict[str, PlateWatchEntry] = {}

    def add(
        self,
        plate: str,
        *,
        category: str = "wanted",
        reason: str = "",
        reference: str = "",
    ) -> PlateWatchEntry:
        """Add or replace a watchlist registration.

        The plate is normalised through the same grammar as a live read, so a
        mistyped entry cannot sit in the list never matching anything.
        """
        normalised = normalise_plate(plate, 1.0)
        key = normalised.text or _ALNUM_RE.sub("", plate.upper())
        entry = PlateWatchEntry(
            plate=key, category=category, reason=reason, reference=reference
        )
        self._entries[key] = entry
        return entry

    def remove(self, plate: str) -> bool:
        key = normalise_plate(plate, 1.0).text or _ALNUM_RE.sub("", plate.upper())
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def entries(self) -> list[PlateWatchEntry]:
        return list(self._entries.values())

    def check(self, reading: PlateReading) -> PlateWatchHit | None:
        """Test a read against the watchlist.

        Only actionable reads are tested. Matching a grammatically invalid or
        low-confidence read risks flagging a vehicle whose registration was
        never actually established.
        """
        if not reading.is_actionable or reading.confidence < self.min_confidence:
            return None

        exact = self._entries.get(reading.text)
        if exact is not None:
            return PlateWatchHit(entry=exact, reading=reading, distance=0, exact=True)

        if self.max_distance <= 0:
            return None

        best: tuple[int, PlateWatchEntry] | None = None
        for key, entry in self._entries.items():
            if abs(len(key) - len(reading.text)) > self.max_distance:
                continue
            distance = _levenshtein(key, reading.text)
            if distance <= self.max_distance and (best is None or distance < best[0]):
                best = (distance, entry)

        if best is None:
            return None
        return PlateWatchHit(entry=best[1], reading=reading, distance=best[0], exact=False)
