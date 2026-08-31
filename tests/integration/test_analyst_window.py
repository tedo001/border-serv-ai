"""The live analysis console, driven end to end against the simulator.

This is the one test that proves the window is wired to the real pipeline
rather than to a mock of it: it presses Start on a simulated source and waits
for annotated frames and live counters to arrive through the same callbacks
the node uses.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6", reason="the desktop extra is not installed")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ibvap.desktop.analyst import AnalystWindow  # noqa: E402

pytestmark = pytest.mark.integration

#: Model loading, stream opening and the detector's warmup all happen before
#: the first frame can be painted. Generous, because a cold CI runner is slow.
STARTUP_BUDGET = 30.0


def pump(app: QApplication, seconds: float, until=None) -> None:
    """Run the Qt event loop for up to ``seconds``, stopping early on ``until``."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if until is not None and until():
            return
        time.sleep(0.02)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_start_runs_the_real_pipeline_and_paints_frames(qt_app) -> None:
    window = AnalystWindow()
    window.show()
    try:
        window.radio_sim.setChecked(True)
        index = [
            window.scenario_combo.itemData(i)
            for i in range(window.scenario_combo.count())
        ].index("intrusion")
        window.scenario_combo.setCurrentIndex(index)

        # A fence line down the middle: the scenario's infiltrator walks
        # across it, so this is the geometry the analytics has to act on.
        window._arm("tripwire")
        window._on_point(0.5, 0.05)
        window._on_point(0.5, 0.95)

        window._start()
        pump(qt_app, STARTUP_BUDGET, until=lambda: window._frames > 20)

        assert window.worker is not None, "the worker never started"
        assert window._frames > 20, "no frames reached the view"
        assert window.figures["fps"].value.text() != "0"
        assert "mode=" in window.detector_line.text()

        # Every event carries the site and the rule that produced it, which is
        # what makes the JSON export usable as a record rather than a log dump.
        assert window.alerts, "the pipeline produced no events at all"
        first = window.alerts[0]
        assert {"event_id", "type", "severity", "risk", "rule_id"} <= set(first)

        window._stop()
        pump(qt_app, 10.0, until=lambda: window.worker is None
             and window.start_button.isEnabled())
        assert window.start_button.isEnabled(), "the window never returned to idle"
    finally:
        window.close()
