"""Indian number-plate grammar, OCR decoding and the plate watchlist."""

from __future__ import annotations

import numpy as np
import pytest

from ibvap.vision.anpr import (
    DEFAULT_CHARSET,
    PlateWatchlist,
    ctc_greedy_decode,
    normalise_plate,
    plates_match,
)


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
