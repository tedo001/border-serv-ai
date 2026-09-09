"""Indian number-plate grammar, OCR decoding and the plate watchlist."""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.core.types import BBox, Detection, ObjectClass
from ibvap.vision.anpr import (
    DEFAULT_CHARSET,
    PlateLocator,
    PlateReader,
    PlateWatchlist,
    ctc_greedy_decode,
    normalise_plate,
    plates_match,
)
from ibvap.vision.backends import CallableBackend
from ibvap.vision.detector import BaseDetector


class TestPlateNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("MH12AB1234", "MH12AB1234"),      # clean standard
            ("mh12ab1234", "MH12AB1234"),      # case
            ("MH 12 AB 1234", "MH12AB1234"),   # spaced
            ("MH-12-AB-1234", "MH12AB1234"),   # hyphenated
            ("DL8CAF1234", "DL8CAF1234"),      # 1-digit district, 2-letter series
            ("KA05MG2345", "KA05MG2345"),
            ("PB65Z9876", "PB65Z9876"),        # single-letter series
            ("22BH1234AB", "22BH1234AB"),      # Bharat series
        ],
    )
    def test_valid_plates_pass_through(self, raw: str, expected: str) -> None:
        reading = normalise_plate(raw, 1.0)
        assert reading.text == expected
        assert reading.valid

    @pytest.mark.parametrize(
        ("raw", "expected", "edits"),
        [
            ("MHI2A8I234", "MH12AB1234", 3),   # I->1, 8->B, I->1
            ("TN07CD5O43", "TN07CD5043", 1),   # O in a digit position
            ("0L8CAF1234", "DL8CAF1234", 1),   # 0 -> D, resolved by state code
            ("6J01AB1234", "GJ01AB1234", 1),   # 6 -> G, resolved by state code
        ],
    )
    def test_ocr_confusions_are_repaired(self, raw: str, expected: str, edits: int) -> None:
        """Glyph/digit confusions account for nearly all real ANPR errors."""
        reading = normalise_plate(raw, 1.0)
        assert reading.text == expected
        assert reading.valid
        assert reading.corrections == edits

    def test_confidence_is_penalised_per_correction(self) -> None:
        clean = normalise_plate("MH12AB1234", 0.9)
        repaired = normalise_plate("MHI2A8I234", 0.9)
        assert repaired.confidence < clean.confidence
        assert repaired.confidence > 0.5  # three edits is still usable

    def test_unknown_state_code_is_invalid(self) -> None:
        reading = normalise_plate("XX99ZZ9999", 0.9)
        assert not reading.valid
        assert not reading.is_actionable

    def test_garbage_is_rejected(self) -> None:
        reading = normalise_plate("!!!", 0.9)
        assert reading.text == ""
        assert not reading.valid

    def test_low_confidence_is_not_actionable(self) -> None:
        """A shaky read must never reach a watchlist: it could name a real,
        uninvolved vehicle."""
        assert not normalise_plate("MH12AB1234", 0.2).is_actionable
        assert normalise_plate("MH12AB1234", 0.95).is_actionable

    def test_state_code_extracted(self) -> None:
        assert normalise_plate("MH12AB1234", 1.0).state_code == "MH"
        assert normalise_plate("22BH1234AB", 1.0).state_code == ""  # Bharat has none


class TestPlateMatching:
    def test_exact(self) -> None:
        assert plates_match("MH12AB1234", "mh 12 ab 1234")

    def test_fuzzy_within_tolerance(self) -> None:
        assert plates_match("MH12AB1234", "MH12AB1284", max_distance=1)

    def test_fuzzy_rejects_beyond_tolerance(self) -> None:
        assert not plates_match("MH12AB1234", "MH99XY9999", max_distance=1)

    def test_no_fuzzy_by_default(self) -> None:
        assert not plates_match("MH12AB1234", "MH12AB1284")


class TestCtcDecode:
    def test_collapses_repeats_and_strips_blanks(self) -> None:
        charset = DEFAULT_CHARSET
        sequence = [0, "M", "M", 0, "H", "1", 0, "2", 0, 0]
        logits = np.full((len(sequence), len(charset) + 1), -8.0)
        for step, symbol in enumerate(sequence):
            index = 0 if symbol == 0 else charset.index(symbol) + 1
            logits[step, index] = 8.0

        text, confidence = ctc_greedy_decode(logits, charset)
        assert text == "MH12"
        assert confidence > 0.99

    def test_empty_input(self) -> None:
        assert ctc_greedy_decode(np.empty((0, 0))) == ("", 0.0)

    def test_all_blank(self) -> None:
        logits = np.full((5, 37), -8.0)
        logits[:, 0] = 8.0
        assert ctc_greedy_decode(logits)[0] == ""


class TestPlateWatchlist:
    def test_entries_are_normalised_on_entry(self) -> None:
        """A mistyped entry must not sit in the list silently matching nothing."""
        watchlist = PlateWatchlist()
        watchlist.add("mh 12 ab 1234", category="stolen")
        assert watchlist.entries()[0].plate == "MH12AB1234"

    def test_matches_a_repaired_read(self) -> None:
        watchlist = PlateWatchlist()
        watchlist.add("MH12AB1234", category="stolen", reference="FIR 41/2026")
        hit = watchlist.check(normalise_plate("MHI2A8I234", 0.95))
        assert hit is not None
        assert hit.entry.category == "stolen"
        assert hit.entry.reference == "FIR 41/2026"
        assert hit.exact

    def test_ignores_unknown_and_weak_reads(self) -> None:
        watchlist = PlateWatchlist()
        watchlist.add("MH12AB1234")
        assert watchlist.check(normalise_plate("KA05MG2345", 0.9)) is None
        assert watchlist.check(normalise_plate("MH12AB1234", 0.1)) is None

    def test_fuzzy_matching_when_enabled(self) -> None:
        watchlist = PlateWatchlist(max_distance=1)
        watchlist.add("MH12AB1234")
        hit = watchlist.check(normalise_plate("MH12AB1284", 0.9))
        assert hit is not None and not hit.exact and hit.distance == 1

    def test_removal(self) -> None:
        watchlist = PlateWatchlist()
        watchlist.add("MH12AB1234")
        assert watchlist.remove("mh 12 ab 1234")
        assert watchlist.size == 0


class TestPlateLocatorRuntimes:
    """Localisation runs on whatever detector the registry declared.

    A plate detector is a single-class YOLOv8 fine-tune, and it must be able to
    arrive on any runtime the platform supports - Torch, TensorRT or ONNX -
    without the ANPR code knowing which.
    """

    class _Stub(BaseDetector):
        mode = "stub"
        is_neural = True

        def __init__(self, boxes):
            self.boxes = boxes
            self.calls = 0

        def detect(self, image):
            self.calls += 1
            return [
                Detection(bbox=box, obj_class=ObjectClass.UNKNOWN, score=score,
                          raw_label="plate")
                for box, score in self.boxes
            ]

    def test_any_detector_can_locate_plates(self) -> None:
        stub = self._Stub([(BBox(10, 40, 90, 64), 0.9)])
        locator = PlateLocator(detector=stub)
        boxes = locator.locate(np.zeros((120, 200, 3), dtype=np.uint8))

        assert locator.is_neural
        assert locator.mode == "stub"
        assert [b.as_int_tuple() for b in boxes] == [(10, 40, 90, 64)]

    def test_candidates_come_back_best_first(self) -> None:
        stub = self._Stub([
            (BBox(0, 0, 20, 10), 0.30),
            (BBox(10, 40, 90, 64), 0.95),
            (BBox(5, 5, 40, 20), 0.60),
        ])
        boxes = PlateLocator(detector=stub).locate(
            np.zeros((120, 200, 3), dtype=np.uint8)
        )
        assert boxes[0].as_int_tuple() == (10, 40, 90, 64)

    def test_the_detector_is_built_once_not_per_frame(self) -> None:
        """Localisation runs on every vehicle in every frame.

        Constructing the detector inside `locate` re-resolved the class
        allowlist and input size each time, on the hottest path ANPR has.
        """
        backend = CallableBackend(
            lambda _feeds: [np.zeros((1, 5, 6), dtype=np.float32)],
            input_shape=(1, 3, 640, 640),
        )
        locator = PlateLocator(backend)
        first = locator.detector
        locator.locate(np.zeros((120, 200, 3), dtype=np.uint8))
        locator.locate(np.zeros((120, 200, 3), dtype=np.uint8))
        assert locator.detector is first

    def test_without_a_detector_it_says_so(self) -> None:
        """The morphological search is useful, but it is not a detector.

        Reporting it as one would tell a control room its ANPR is neural when
        it is edge density and an aspect-ratio filter.
        """
        locator = PlateLocator(None)
        assert locator.is_neural is False
        assert locator.mode == "morphological"


class TestReadBox:
    def test_reading_a_known_box_skips_localisation(self) -> None:
        """This is what a Kalman-smoothed plate track buys.

        On a frame where the detector loses the plate, the filter still says
        where it is, and OCR gets its chance on a frame that would otherwise
        have contributed nothing to the vote.
        """
        class Ocr:
            available = True

            def read(self, image):
                return "MH12AB1234", 0.88

        reader = PlateReader(PlateLocator(None), Ocr())
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        reading = reader.read_box(frame, BBox(100, 150, 200, 180))

        assert reading is not None
        assert reading.text == "MH12AB1234"
        assert reading.bbox is not None
        assert reading.bbox.as_int_tuple() == (100, 150, 200, 180)

    def test_no_ocr_means_no_read(self) -> None:
        class Ocr:
            available = False

            def read(self, image):  # pragma: no cover - must never be called
                raise AssertionError("OCR ran while unavailable")

        reader = PlateReader(PlateLocator(None), Ocr())
        assert reader.read_box(np.zeros((240, 320, 3), np.uint8), BBox(1, 1, 9, 9)) is None

    def test_a_box_outside_the_frame_reads_nothing(self) -> None:
        class Ocr:
            available = True

            def read(self, image):  # pragma: no cover - must never be called
                raise AssertionError("OCR ran on an empty crop")

        reader = PlateReader(PlateLocator(None), Ocr())
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert reader.read_box(frame, BBox(500, 500, 560, 520)) is None
