"""Live analysis console.

These exercise the window's logic, not its pixels: what URL each source mode
produces, which rules drawn geometry creates, how the risk threshold filters
the view without touching the export, and that the frame bridge coalesces
rather than queues. Qt runs on the offscreen platform, so they need no display.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the desktop extra is not installed")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ibvap.core.types import Event, EventType, Severity  # noqa: E402
from ibvap.desktop.analyst import (  # noqa: E402
    AnalystWindow,
    PipelineBridge,
    event_risk,
)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app):
    win = AnalystWindow()
    yield win
    win.close()


def make_event(severity: Severity, confidence: float = 1.0, **kwargs) -> Event:
    return Event(
        camera_id="analyst",
        event_type=kwargs.pop("event_type", EventType.INTRUSION),
        severity=severity,
        timestamp=1_700_000_000.0,
        confidence=confidence,
        message=kwargs.pop("message", "test event"),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Risk scoring
# --------------------------------------------------------------------------- #

def test_risk_is_ordered_by_severity() -> None:
    scores = [event_risk(make_event(s)) for s in Severity]
    assert scores == sorted(scores), "severity must dominate the risk ordering"


def test_confidence_moves_risk_within_its_band_only() -> None:
    """A low-confidence critical still outranks a certain info event.

    That ordering is the reason risk is a product of a severity band and a
    confidence term rather than either one alone: an analyst raising the
    threshold must never lose a critical to a confident nuisance.
    """
    unsure_critical = event_risk(make_event(Severity.CRITICAL, confidence=0.0))
    certain_info = event_risk(make_event(Severity.INFO, confidence=1.0))
    assert unsure_critical > certain_info

    assert event_risk(make_event(Severity.HIGH, 1.0)) > event_risk(
        make_event(Severity.HIGH, 0.2)
    )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def test_simulation_source_builds_a_sim_url(window: AnalystWindow) -> None:
    window.radio_sim.setChecked(True)
    index = [
        window.scenario_combo.itemData(i) for i in range(window.scenario_combo.count())
    ].index("cattle")
    window.scenario_combo.setCurrentIndex(index)
    window.night_check.setChecked(True)

    url = window._source_url()
    assert url is not None
    assert url.startswith("sim://cattle?")
    assert "night=1" in url


def test_cctv_source_returns_the_typed_url(window: AnalystWindow) -> None:
    window.radio_cctv.setChecked(True)
    window.rtsp_input.setText("rtsp://10.20.0.11:554/Streaming/Channels/101")
    assert window._source_url() == "rtsp://10.20.0.11:554/Streaming/Channels/101"


def test_file_source_without_a_file_refuses_rather_than_guessing(
    window: AnalystWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ibvap.desktop import analyst

    monkeypatch.setattr(analyst.QMessageBox, "warning", lambda *a, **k: None)
    window.radio_file.setChecked(True)
    window._video_path = ""
    assert window._source_url() is None


# --------------------------------------------------------------------------- #
# Geometry to rules
# --------------------------------------------------------------------------- #

def test_drawing_a_fence_line_creates_a_line_crossing_rule(
    window: AnalystWindow,
) -> None:
    window._arm("tripwire")
    window._on_point(0.5, 0.1)
    window._on_point(0.5, 0.9)

    assert len(window.tripwires) == 1
    wire = window.tripwires[0]
    assert wire.start == (0.5, 0.1)
    assert wire.end == (0.5, 0.9)

    rules = window._build_rules()
    crossing = [r for r in rules if r.type == "line_crossing"]
    assert len(crossing) == 1
    assert crossing[0].tripwires == [wire.id]
    # Drawing must disarm, or the next click starts a second line by accident.
    assert window._draw_mode == ""


def test_drawing_a_zone_creates_an_intrusion_rule(window: AnalystWindow) -> None:
    window._arm("zone")
    for point in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)):
        window._on_point(*point)
    window._finish_zone()

    assert len(window.zones) == 1
    intrusion = [r for r in window._build_rules() if r.type == "intrusion"]
    assert len(intrusion) == 1
    assert intrusion[0].zones == [window.zones[0].id]


def test_a_zone_needs_three_corners(
    window: AnalystWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ibvap.desktop import analyst

    monkeypatch.setattr(analyst.QMessageBox, "information", lambda *a, **k: None)
    window._arm("zone")
    window._on_point(0.1, 0.1)
    window._on_point(0.9, 0.9)
    window._finish_zone()
    assert window.zones == []


def test_clearing_geometry_removes_its_rules(window: AnalystWindow) -> None:
    window._arm("tripwire")
    window._on_point(0.2, 0.2)
    window._on_point(0.8, 0.8)
    assert any(r.type == "line_crossing" for r in window._build_rules())

    window._clear_geometry()
    assert window.tripwires == []
    assert not any(r.type == "line_crossing" for r in window._build_rules())


# --------------------------------------------------------------------------- #
# The risk slider filters the view, never the record
# --------------------------------------------------------------------------- #

def test_risk_threshold_hides_rows_but_keeps_every_alert(
    window: AnalystWindow,
) -> None:
    window.risk_slider.setValue(0)
    window._on_event(make_event(Severity.INFO, 1.0, message="quiet"))
    window._on_event(make_event(Severity.CRITICAL, 1.0, message="loud"))
    assert window.alert_table.rowCount() == 2

    window.risk_slider.setValue(70)
    assert window.alert_table.rowCount() == 1
    # The export is the record of what happened, so it keeps both.
    assert len(window.alerts) == 2
    assert window.suppressed == 1

    window.risk_slider.setValue(0)
    assert window.alert_table.rowCount() == 2


def test_alert_rows_are_newest_first(window: AnalystWindow) -> None:
    window.risk_slider.setValue(0)
    window._on_event(make_event(Severity.HIGH, message="first"))
    window._on_event(make_event(Severity.HIGH, message="second"))
    assert window.alert_table.item(0, 4).text() == "second"


# --------------------------------------------------------------------------- #
# Frame bridge
# --------------------------------------------------------------------------- #

def fake_frame(index: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        index=index,
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        timestamp=1_700_000_000.0 + index,
        monotonic=float(index),
    )


def test_bridge_coalesces_frames_instead_of_queueing_them(qt_app) -> None:
    """Rendering must never fall minutes behind the video.

    The worker produces frames faster than the GUI paints them. If each one
    were queued the window would drift ever further behind; instead the newest
    frame replaces the unpainted one and the older is counted as dropped.
    """
    bridge = PipelineBridge()
    emitted: list[int] = []
    bridge.frameReady.connect(lambda: emitted.append(1))

    for index in range(5):
        bridge.on_frame(fake_frame(index), [])
    qt_app.processEvents()

    assert sum(emitted) == 1, "one signal, not one per frame"
    assert bridge.dropped == 4

    latest = bridge.take()
    assert latest is not None
    _image, stats = latest
    assert stats["frame_index"] == 4, "the newest frame survives, not the oldest"
    assert bridge.take() is None, "taking twice must not replay a stale frame"


def test_bridge_re_arms_after_the_gui_catches_up(qt_app) -> None:
    bridge = PipelineBridge()
    emitted: list[int] = []
    bridge.frameReady.connect(lambda: emitted.append(1))

    bridge.on_frame(fake_frame(0), [])
    qt_app.processEvents()
    bridge.take()
    bridge.on_frame(fake_frame(1), [])
    qt_app.processEvents()

    assert sum(emitted) == 2


def test_bridge_counts_tracks_by_category(qt_app) -> None:
    from ibvap.core.types import BBox, ObjectClass, Track

    bridge = PipelineBridge()
    tracks = [
        Track(track_id=1, obj_class=ObjectClass.PERSON, bbox=BBox(1, 1, 9, 9), score=0.9),
        Track(track_id=2, obj_class=ObjectClass.CAR, bbox=BBox(1, 1, 9, 9), score=0.8),
        Track(track_id=3, obj_class=ObjectClass.ANIMAL, bbox=BBox(1, 1, 9, 9), score=0.7),
    ]
    bridge.on_frame(fake_frame(0), tracks)
    latest = bridge.take()
    assert latest is not None
    counts = latest[1]["counts"]
    assert (counts["human"], counts["vehicle"], counts["animal"]) == (1, 1, 1)


# --------------------------------------------------------------------------- #
# Degradation is announced before the run, not discovered during it
# --------------------------------------------------------------------------- #

def test_a_missing_artefact_is_announced_in_the_model_note(
    window: AnalystWindow,
) -> None:
    """The registry ships declarations; the .onnx files are not in the tree.

    An analyst who does not know they are on the motion fallback will read its
    output as the model's, so the window has to say so before Start is pressed.
    """
    present = [e for e in window.detectors if e["present"]]
    if present:
        pytest.skip("detector artefacts are installed in this checkout")
    window._describe_model()
    assert "falls back" in window.model_note.text()
