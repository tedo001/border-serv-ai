"""Live analysis console - one source, one model, one screen.

The operator console in :mod:`ibvap.desktop.main_window` watches a running
node: many cameras, stored events, watchlists, evidence. This window is the
other half of the job. An analyst sits in front of *one* video, picks a
detector, moves the thresholds and watches what the pipeline makes of it -
with no node, no database and no login.

Everything on screen comes from the production pipeline. The window builds a
:class:`~ibvap.pipeline.worker.CameraWorker` exactly as the supervisor does
and subscribes to its frame and event callbacks. Nothing here re-implements
detection, tracking or analytics, so a threshold that looks right in this
window is a threshold that behaves the same way on the node.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ibvap.core.config import (
    CameraConfig,
    ModelSpec,
    RuleConfig,
    Settings,
    TripwireConfig,
    ZoneConfig,
)
from ibvap.core.logging import configure_logging, get_logger
from ibvap.core.types import Event, Severity, Track
from ibvap.events.annotate import annotate_frame
from ibvap.ingest.simulator import SCENARIOS
from ibvap.mlops.registry import ModelRegistry
from ibvap.pipeline.models import build_model_bundle
from ibvap.vision import supervision_ops

log = get_logger(__name__)

#: Palette, shared with the operator console so the two read as one product.
INK = "#0d1117"
PANEL = "#161b22"
FIELD = "#0a0e14"
BORDER = "#2a313c"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#2f81f7"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"

#: How much each severity contributes to an event's risk score. Risk is what
#: the analyst's threshold slider acts on, so it has to be an explicit,
#: readable number rather than a hidden weighting.
SEVERITY_RISK: dict[Severity, float] = {
    Severity.INFO: 0.20,
    Severity.LOW: 0.40,
    Severity.MEDIUM: 0.60,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 1.00,
}

SEVERITY_COLOURS = {
    "info": MUTED, "low": GOOD, "medium": WARN, "high": "#f0883e", "critical": BAD,
}

#: One-line description per simulator scenario, for the source picker.
SCENARIO_HINTS = {
    "patrol": "routine patrol inside the fence",
    "intrusion": "single infiltrator crossing the fence line",
    "cattle": "livestock plus one person - the false-alarm case",
    "vehicle": "vehicle approaching the post on the border road",
    "loiter": "figure stopping and waiting near the fence",
    "abandoned": "bag left behind and its carrier walking away",
    "crowd": "group movement along the fence",
    "night": "IR night infiltration in low light",
}

ANALYST_STYLE = f"""
QMainWindow, QWidget {{ background: {INK}; color: {TEXT}; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px;
    padding: 10px 8px 8px 8px; background: {PANEL};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 5px;
    color: {ACCENT}; font-weight: 700;
}}
QPushButton {{
    background: #21262d; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 7px 12px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #56606b; background: #191e25; }}
QPushButton#start {{ background: #1a7f37; border: none; font-weight: 700; padding: 11px; }}
QPushButton#start:hover {{ background: #238636; }}
QPushButton#start:disabled {{ background: #1b2a20; color: #56606b; }}
QPushButton#stop {{ background: #b62324; border: none; font-weight: 700; padding: 11px; }}
QPushButton#stop:hover {{ background: #da3633; }}
QPushButton#stop:disabled {{ background: #2a1b1c; color: #56606b; }}
QPushButton#browse {{ background: {ACCENT}; border: none; font-weight: 700; padding: 10px; }}
QPushButton#armed {{ border-color: {ACCENT}; background: #1f2a3a; }}
QLineEdit, QComboBox, QTextEdit {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 7px; color: {TEXT};
}}
QComboBox QAbstractItemView {{ background: {FIELD}; selection-background-color: #1f6feb; }}
QSlider::groove:horizontal {{ height: 5px; background: #21262d; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 13px; margin: -5px 0; border-radius: 7px;
}}
QTableWidget {{
    background: {FIELD}; alternate-background-color: #10151c;
    gridline-color: #1c222b; border: 1px solid {BORDER};
}}
QHeaderView::section {{
    background: {PANEL}; color: {MUTED}; padding: 5px;
    border: none; border-bottom: 1px solid {BORDER};
}}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical {{ background: {INK}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; }}
QRadioButton, QCheckBox {{ padding: 3px 0; spacing: 7px; background: transparent; }}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 13px; height: 13px; border: 1px solid #4d5661; background: {FIELD};
}}
QRadioButton::indicator {{ border-radius: 11px; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator:checked {{
    border: 4px solid {ACCENT}; border-radius: 11px; background: {FIELD};
}}
QCheckBox::indicator:checked {{ border-color: {ACCENT}; background: {ACCENT}; }}
QLabel {{ background: transparent; }}
"""


def _narrow(combo: QComboBox) -> None:
    """Stop a combo box from sizing itself to its longest entry.

    Left to itself a combo demands the width of its widest item, which drags
    the whole control column past its viewport and silently clips whatever
    sits to the right of it.
    """
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(16)


def event_risk(event: Event) -> float:
    """Risk score in ``[0, 1]`` for one event.

    Severity sets the band and analytics confidence moves it within that band,
    so a low-confidence critical still outranks a confident info. The analyst's
    risk slider is a floor on this number.
    """
    return SEVERITY_RISK.get(event.severity, 0.5) * (0.5 + 0.5 * event.confidence)


# --------------------------------------------------------------------------- #
# Worker/GUI bridge
# --------------------------------------------------------------------------- #

class PipelineBridge(QObject):
    """Marshals camera-worker callbacks onto the Qt event loop.

    The worker runs on its own thread and calls back per frame. Emitting a
    signal per frame is safe, but queueing every frame is not: if rendering
    falls behind the decoder the queue grows without bound and the window ends
    up minutes behind the video. So frames are coalesced - the newest one
    replaces any frame not yet painted, and the signal is emitted only when
    the GUI is ready for another.
    """

    frameReady = pyqtSignal()
    eventRaised = pyqtSignal(object)
    noticed = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, dict[str, Any]] | None = None
        self._pending = False
        self._last_frame_at = 0.0
        self._fps = 0.0
        self.dropped = 0
        #: Overlay geometry, replaced whenever the analyst edits it.
        self._zones: tuple[Any, ...] = ()
        self._tripwires: tuple[Any, ...] = ()
        # Supervision draws corner boxes, rounded labels and identity traces.
        # Held for the life of the run rather than rebuilt per frame: the trace
        # annotator keeps its own history keyed by tracker id, so a fresh one
        # each frame would silently draw no trails at all.
        self.renderer = (
            supervision_ops.SupervisionRenderer()
            if supervision_ops.AVAILABLE else None
        )

    def on_frame(self, frame: Any, tracks: list[Track]) -> None:
        """Called on the worker thread once per processed frame."""
        now = time.perf_counter()
        if self._last_frame_at:
            interval = now - self._last_frame_at
            if interval > 0:
                instant = 1.0 / interval
                self._fps = instant if self._fps == 0.0 else 0.8 * self._fps + 0.2 * instant
        self._last_frame_at = now

        counts = {"human": 0, "vehicle": 0, "animal": 0, "object": 0, "other": 0}
        for track in tracks:
            counts[track.category.value] = counts.get(track.category.value, 0) + 1

        stats = {
            "frame_index": frame.index,
            "tracks": len(tracks),
            "fps": self._fps,
            "counts": counts,
            "resolution": f"{frame.image.shape[1]}x{frame.image.shape[0]}",
        }
        if self.renderer is not None:
            canvas = self.renderer.annotate(
                frame.image, tracks,
                zones=list(self._zones), tripwires=list(self._tripwires),
            )
        else:
            canvas = annotate_frame(
                frame.image, tracks=tracks,
                zones=list(self._zones), tripwires=list(self._tripwires),
                timestamp=frame.timestamp,
            )
        with self._lock:
            if self._latest is not None:
                self.dropped += 1
            self._latest = (canvas, stats)
            emit = not self._pending
            self._pending = True
        if emit:
            self.frameReady.emit()

    def set_overlay(self, zones: Any, tripwires: Any) -> None:
        self._zones = tuple(zones)
        self._tripwires = tuple(tripwires)

    def take(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Collect the newest frame and re-arm the signal. GUI thread only."""
        with self._lock:
            latest, self._latest, self._pending = self._latest, None, False
        return latest

    def on_event(self, event: Event) -> None:
        self.eventRaised.emit(event)

    def on_state(self, state: str, detail: str = "") -> None:
        self.noticed.emit(state, detail)


# --------------------------------------------------------------------------- #
# Video surface
# --------------------------------------------------------------------------- #

class LiveView(QLabel):
    """The annotated video surface, and the canvas the analyst draws on.

    Clicks are reported in normalised frame coordinates rather than widget
    pixels, because that is what zones and tripwires are stored in - the same
    line then means the same thing at any window size or stream resolution.
    """

    pointPicked = pyqtSignal(float, float)

    def __init__(self) -> None:
        super().__init__("no video\n\nchoose a source and press Start Analysis")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background:#000;color:{MUTED};border:1px solid {BORDER};")
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self._target = QRect()
        self._drawing = False

    def set_drawing(self, drawing: bool) -> None:
        self._drawing = drawing
        self.setCursor(
            Qt.CursorShape.CrossCursor if drawing else Qt.CursorShape.ArrowCursor
        )

    def show_frame(self, bgr: np.ndarray) -> None:
        height, width = bgr.shape[:2]
        # Qt wants RGB and a contiguous buffer; the copy also detaches the
        # image from the worker's frame, which is reused downstream.
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._render()

    def clear_frame(self, message: str) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message)

    def _render(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Remember where the frame actually landed: the label centres the
        # pixmap and letterboxes the rest, so widget pixels are not frame
        # pixels and a click has to be corrected for the offset.
        self._target = QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(), scaled.height(),
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if not self._drawing or self._pixmap is None or self._target.isEmpty():
            return
        position = event.position().toPoint()
        if not self._target.contains(position):
            return
        x = (position.x() - self._target.x()) / self._target.width()
        y = (position.y() - self._target.y()) / self._target.height()
        self.pointPicked.emit(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))


class StatFigure(QFrame):
    """One large figure in the counter strip."""

    def __init__(self, caption: str, tone: str = TEXT) -> None:
        super().__init__()
        self.setStyleSheet(f"QFrame {{ background:{PANEL}; border:1px solid {BORDER}; }}")
        self._tone = tone
        self.value = QLabel("0")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setStyleSheet(
            f"font-size:24px;font-weight:700;font-family:monospace;color:{tone};border:none;"
        )
        label = QLabel(caption.upper())
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color:{MUTED};font-size:9px;letter-spacing:1px;border:none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(2)
        layout.addWidget(self.value)
        layout.addWidget(label)

    def set_value(self, value: Any, tone: str | None = None) -> None:
        self.value.setText(str(value))
        self.value.setStyleSheet(
            "font-size:24px;font-weight:700;font-family:monospace;"
            f"color:{tone or self._tone};border:none;"
        )


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #

#: Rules the analyst can switch on without drawing anything. Each is
#: ``(key, caption, rule type, params)``; drawn zones and tripwires add their
#: own intrusion and line-crossing rules on top.
OPTIONAL_RULES: list[tuple[str, str, str, dict[str, Any]]] = [
    ("presence", "Presence (anything in view)", "presence", {"min_confidence": 0.4}),
    ("loitering", "Loitering", "loitering", {"dwell_seconds": 12.0}),
    ("night", "Night movement", "night_movement", {"require_dark": True}),
    ("abandoned", "Abandoned object", "abandoned_object", {}),
    ("tamper", "Camera tamper", "camera_tamper", {"warmup_frames": 30}),
]


class AnalystWindow(QMainWindow):
    """Single-source live analysis: source, model, thresholds, video, alerts."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IBVAP - Live Analysis")
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(ANALYST_STYLE)

        self.settings = Settings()
        self.registry = ModelRegistry.load(self.settings.models)
        self.detectors = [
            entry for entry in self.registry.describe()
            if entry["role"] == "detector" and entry["enabled"]
        ]

        self.bridge = PipelineBridge()
        self.bridge.frameReady.connect(self._on_frame)
        self.bridge.eventRaised.connect(self._on_event)
        self.bridge.noticed.connect(self._on_notice)

        self.worker: Any = None
        self.bundle: Any = None
        self.zones: list[ZoneConfig] = []
        self.tripwires: list[TripwireConfig] = []
        self.alerts: list[dict[str, Any]] = []
        self.suppressed = 0
        self._draw_mode = ""
        self._pending_points: list[tuple[float, float]] = []
        self._frames = 0
        self._video_path = ""

        self._build_ui()

        self.poll = QTimer(self)
        self.poll.timeout.connect(self._refresh_stream_state)
        self.poll.start(1000)

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_stage())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 900])

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
        title = QLabel("Border AI Dashboard")
        title.setStyleSheet(f"font-size:19px;font-weight:800;color:{TEXT};")
        edition = QLabel("IBVAP  ·  LIVE ANALYSIS")
        edition.setStyleSheet(f"color:{ACCENT};font-weight:700;letter-spacing:1px;")

        self.state_chip = QLabel("  IDLE  ")
        self.state_chip.setStyleSheet(
            f"background:#21262d;color:{MUTED};font-weight:700;padding:5px 10px;border-radius:4px;"
        )

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(edition)
        layout.addStretch(1)
        layout.addWidget(self.state_chip)
        return header

    # -- left column ------------------------------------------------------- #

    def _build_controls(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(self._group_source())
        layout.addWidget(self._group_model())
        layout.addWidget(self._group_settings())
        layout.addWidget(self._group_geometry())
        layout.addWidget(self._group_run())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(column)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(460)
        return scroll

    def _group_source(self) -> QGroupBox:
        box = QGroupBox("1.  Video source")
        layout = QVBoxLayout(box)

        self.source_group = QButtonGroup(self)
        self.radio_cctv = QRadioButton("CCTV  (RTSP / HTTP stream)")
        self.radio_file = QRadioButton("Local video file")
        self.radio_sim = QRadioButton("Simulation  (no hardware needed)")
        for index, radio in enumerate((self.radio_cctv, self.radio_file, self.radio_sim)):
            self.source_group.addButton(radio, index)
        self.radio_sim.setChecked(True)

        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://10.20.0.11:554/Streaming/Channels/101")

        self.browse_button = QPushButton("Browse local video")
        self.browse_button.setObjectName("browse")
        self.browse_button.clicked.connect(self._browse)
        self.file_label = QLabel("no file selected")
        self.file_label.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self.file_label.setWordWrap(True)

        self.scenario_combo = QComboBox()
        for name in SCENARIOS:
            self.scenario_combo.addItem(name, name)
        _narrow(self.scenario_combo)
        self.scenario_note = QLabel()
        self.scenario_note.setWordWrap(True)
        self.scenario_note.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self.scenario_combo.currentIndexChanged.connect(self._describe_scenario)
        self._describe_scenario()
        self.night_check = QCheckBox("Render as IR night footage")

        layout.addWidget(self.radio_cctv)
        layout.addWidget(self.rtsp_input)
        layout.addSpacing(4)
        layout.addWidget(self.radio_file)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.file_label)
        layout.addSpacing(4)
        layout.addWidget(self.radio_sim)
        layout.addWidget(self.scenario_combo)
        layout.addWidget(self.scenario_note)
        layout.addWidget(self.night_check)
        return box

    def _group_model(self) -> QGroupBox:
        box = QGroupBox("2.  Model")
        layout = QVBoxLayout(box)

        self.model_combo = QComboBox()
        for entry in self.detectors:
            self.model_combo.addItem(f"{entry['name']}  [{entry['layout']}]", entry["name"])
        if not self.detectors:
            self.model_combo.addItem("no detector declared", None)
        _narrow(self.model_combo)

        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet(f"color:{MUTED};font-size:11px;")
        self.model_combo.currentIndexChanged.connect(self._describe_model)

        self.classify_check = QCheckBox("Refine classes with MobileNet / ImageNet")
        self.classify_check.setChecked(True)
        self.classify_check.setToolTip(
            "Runs the secondary classifier once per track. This is what tells "
            "cattle from people, and it is the difference between suppressing "
            "livestock by class and not being able to."
        )
        self.livestock_check = QCheckBox("Suppress livestock (no animal alerts)")
        self.livestock_check.setChecked(True)

        layout.addWidget(self.model_combo)
        layout.addWidget(self.model_note)
        layout.addWidget(self.classify_check)
        layout.addWidget(self.livestock_check)
        self._describe_model()
        return box

    def _slider(
        self, layout: QGridLayout, row: int, caption: str,
        low: int, high: int, value: int, *, scale: float = 0.01, suffix: str = "",
    ) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        readout = QLabel()
        readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        readout.setMinimumWidth(56)
        readout.setStyleSheet(f"color:{WARN};font-weight:700;font-family:monospace;")

        def show(raw: int) -> None:
            readout.setText(f"{raw * scale:.2f}{suffix}" if scale != 1 else f"{raw}{suffix}")

        slider.valueChanged.connect(show)
        show(value)

        name = QLabel(caption)
        name.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(name, row, 0)
        layout.addWidget(readout, row, 1)
        layout.addWidget(slider, row + 1, 0, 1, 2)
        return slider

    def _group_settings(self) -> QGroupBox:
        box = QGroupBox("3.  Settings")
        grid = QGridLayout(box)
        grid.setColumnStretch(0, 1)

        self.conf_slider = self._slider(grid, 0, "Detection confidence", 5, 95, 35)
        self.iou_slider = self._slider(grid, 2, "NMS IoU", 10, 90, 45)
        self.risk_slider = self._slider(grid, 4, "Risk threshold", 0, 100, 30)
        self.interval_slider = self._slider(
            grid, 6, "Detect every N frames", 1, 10, 1, scale=1
        )
        self.fps_slider = self._slider(grid, 8, "Analytics FPS", 1, 30, 8, scale=1, suffix=" fps")

        # Confidence and IoU are the two an analyst moves while watching, so
        # they are applied to the running detector rather than needing a restart.
        self.conf_slider.valueChanged.connect(self._apply_live_thresholds)
        self.iou_slider.valueChanged.connect(self._apply_live_thresholds)
        self.risk_slider.valueChanged.connect(lambda _v: self._repopulate_alerts())
        return box

    def _group_geometry(self) -> QGroupBox:
        box = QGroupBox("4.  Fence line and zones")
        layout = QVBoxLayout(box)

        self.line_button = QPushButton("Draw fence line  (2 clicks)")
        self.line_button.clicked.connect(lambda: self._arm("tripwire"))
        self.zone_button = QPushButton("Draw zone  (click corners)")
        self.zone_button.clicked.connect(lambda: self._arm("zone"))
        self.finish_button = QPushButton("Finish zone")
        self.finish_button.clicked.connect(self._finish_zone)
        self.finish_button.setEnabled(False)
        self.clear_button = QPushButton("Clear all geometry")
        self.clear_button.clicked.connect(self._clear_geometry)

        self.geometry_label = QLabel("none drawn - rules watch the whole frame")
        self.geometry_label.setWordWrap(True)
        self.geometry_label.setStyleSheet(f"color:{MUTED};font-size:11px;")

        for widget in (self.line_button, self.zone_button,
                       self.finish_button, self.clear_button):
            layout.addWidget(widget)
        layout.addWidget(self.geometry_label)

        self.rule_checks: dict[str, QCheckBox] = {}
        for key, caption, _type, _params in OPTIONAL_RULES:
            check = QCheckBox(caption)
            check.setChecked(key in ("presence", "night"))
            self.rule_checks[key] = check
            layout.addWidget(check)
        return box

    def _group_run(self) -> QGroupBox:
        box = QGroupBox("5.  Run")
        layout = QVBoxLayout(box)

        self.start_button = QPushButton("Start Analysis")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)

        self.run_label = QLabel("idle")
        self.run_label.setWordWrap(True)
        self.run_label.setStyleSheet(f"color:{MUTED};font-size:11px;")

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.run_label)
        return box

    # -- right column ------------------------------------------------------ #

    def _build_stage(self) -> QWidget:
        self.view = LiveView()
        self.view.pointPicked.connect(self._on_point)

        self.figures = {
            "frames": StatFigure("frames"),
            "persons": StatFigure("persons", GOOD),
            "vehicles": StatFigure("vehicles", WARN),
            "animals": StatFigure("animals"),
            "alerts": StatFigure("alerts", BAD),
            "fps": StatFigure("fps", GOOD),
            "tracks": StatFigure("tracks", ACCENT),
        }
        strip = QHBoxLayout()
        strip.setSpacing(6)
        for figure in self.figures.values():
            strip.addWidget(figure)

        self.detector_line = QLabel("DETECTOR:  not started")
        self.detector_line.setStyleSheet(
            f"color:{MUTED};font-family:monospace;font-size:11px;padding:4px 2px;"
        )

        self.alert_table = QTableWidget(0, 5)
        self.alert_table.setHorizontalHeaderLabels(
            ["Time", "Type", "Severity", "Risk", "Detail"]
        )
        self.alert_table.setAlternatingRowColors(True)
        self.alert_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.alert_table.setColumnWidth(0, 90)
        self.alert_table.setColumnWidth(1, 140)
        self.alert_table.setColumnWidth(2, 80)
        self.alert_table.setColumnWidth(3, 60)
        self.alert_table.horizontalHeader().setStretchLastSection(True)
        self.alert_table.verticalHeader().setVisible(False)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(90)
        self.console.setStyleSheet(
            f"background:{FIELD};color:{TEXT};font-family:monospace;font-size:11px;"
        )

        self.export_button = QPushButton("Export alerts (JSON)")
        self.export_button.clicked.connect(self._export)
        self.export_button.setEnabled(False)
        self.suppressed_label = QLabel("")
        self.suppressed_label.setStyleSheet(f"color:{MUTED};font-size:11px;")

        log_box = QGroupBox("Alert log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.alert_table, 3)
        log_layout.addWidget(self.console, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.suppressed_label)
        footer.addStretch(1)
        footer.addWidget(self.export_button)
        log_layout.addLayout(footer)

        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.addLayout(strip)
        lower_layout.addWidget(self.detector_line)
        lower_layout.addWidget(log_box, 1)

        stage = QSplitter(Qt.Orientation.Vertical)
        stage.addWidget(self.view)
        stage.addWidget(lower)
        stage.setStretchFactor(0, 3)
        stage.setSizes([440, 320])

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(stage)
        return page

    # -- helpers ----------------------------------------------------------- #

    def _note(self, message: str, tone: str = MUTED) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.console.append(f'<span style="color:{tone}">[{stamp}] {message}</span>')

    def _set_chip(self, text: str, colour: str) -> None:
        self.state_chip.setText(f"  {text}  ")
        self.state_chip.setStyleSheet(
            f"background:#21262d;color:{colour};font-weight:700;"
            "padding:5px 10px;border-radius:4px;"
        )

    def _describe_model(self) -> None:
        name = self.model_combo.currentData()
        entry = next((e for e in self.detectors if e["name"] == name), None)
        if entry is None:
            self.model_note.setText(
                "No detector artefact is declared. Analysis still runs, on "
                "classical motion detection, and reports itself as degraded."
            )
            return
        if entry["present"]:
            self.model_note.setText(
                f"layout {entry['layout']} · {entry['classes']} classes · "
                f"version {entry['version']}"
            )
            self.model_note.setStyleSheet(f"color:{MUTED};font-size:11px;")
        else:
            # Saying this before the run starts is the whole point: an analyst
            # who does not know they are on the fallback will read its output
            # as the model's.
            self.model_note.setText(
                f"layout {entry['layout']} · artefact not present, so this run "
                "falls back to classical motion detection and is marked degraded."
            )
            self.model_note.setStyleSheet(f"color:{WARN};font-size:11px;")

    def _describe_scenario(self) -> None:
        name = self.scenario_combo.currentData()
        self.scenario_note.setText(SCENARIO_HINTS.get(name, ""))

    def _browse(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose a video file", "",
            "Video (*.mp4 *.avi *.mkv *.mov *.m4v *.webm);;All files (*)",
        )
        if not path:
            return
        self._video_path = path
        self.file_label.setText(Path(path).name)
        self.radio_file.setChecked(True)

    def _source_url(self) -> str | None:
        if self.radio_cctv.isChecked():
            url = self.rtsp_input.text().strip()
            if not url:
                QMessageBox.warning(self, "No stream", "Enter an RTSP or HTTP stream URL.")
                return None
            return url
        if self.radio_file.isChecked():
            if not self._video_path:
                QMessageBox.warning(self, "No file", "Choose a video file first.")
                return None
            return self._video_path
        scenario = self.scenario_combo.currentData()
        night = 1 if self.night_check.isChecked() else 0
        return f"sim://{scenario}?width=1280&height=720&fps=15&night={night}"

    def _build_rules(self) -> list[RuleConfig]:
        rules: list[RuleConfig] = []
        for zone in self.zones:
            rules.append(RuleConfig(
                id=f"{zone.id}-intrusion", type="intrusion",
                zones=[zone.id], params={"confirm_frames": 2}, cooldown_seconds=8.0,
            ))
        for wire in self.tripwires:
            rules.append(RuleConfig(
                id=f"{wire.id}-crossing", type="line_crossing",
                tripwires=[wire.id], cooldown_seconds=8.0,
            ))
        for key, _caption, rule_type, params in OPTIONAL_RULES:
            if self.rule_checks[key].isChecked():
                rules.append(RuleConfig(
                    id=f"analyst-{key}", type=rule_type,
                    params=dict(params), cooldown_seconds=10.0,
                ))
        return rules

    # -- run --------------------------------------------------------------- #

    def _start(self) -> None:
        url = self._source_url()
        if url is None:
            return

        settings = Settings()
        settings.site_id = "analyst"
        settings.site_name = "Live Analysis"
        settings.models.detector = ModelSpec(
            name=self.model_combo.currentData(),
            score_threshold=self.conf_slider.value() / 100.0,
            nms_threshold=self.iou_slider.value() / 100.0,
        )
        settings.analytics.ignore_classes = (
            ["animal"] if self.livestock_check.isChecked() else []
        )
        # Evidence writing belongs to a node with a database and a retention
        # policy. This window analyses; it does not become a second, unmanaged
        # evidence store.
        settings.evidence.enabled = False

        camera = CameraConfig(
            id="analyst", name="Live Analysis", url=url,
            target_fps=float(self.fps_slider.value()),
            detect_interval=self.interval_slider.value(),
            zones=list(self.zones), tripwires=list(self.tripwires),
            rules=self._build_rules(),
            classify_objects=self.classify_check.isChecked(),
        )
        self.settings = settings

        self._frames = 0
        self.alerts.clear()
        self.suppressed = 0
        self.alert_table.setRowCount(0)
        for figure in self.figures.values():
            figure.set_value(0)

        self.start_button.setEnabled(False)
        self.run_label.setText("loading models…")
        self._set_chip("STARTING", WARN)
        self._note(f"source {url}")

        # Model loading reads and verifies artefacts, which takes seconds on a
        # cold cache. Doing it inline would freeze the window at exactly the
        # moment the analyst is watching for feedback.
        threading.Thread(
            target=self._start_worker, args=(camera, settings), daemon=True
        ).start()

    def _start_worker(self, camera: CameraConfig, settings: Settings) -> None:
        from ibvap.pipeline.worker import CameraWorker

        try:
            bundle = build_model_bundle(settings, registry=self.registry)
            worker = CameraWorker(
                camera, settings, bundle,
                frame_sink=self.bridge.on_frame,
                event_sink=self.bridge.on_event,
            )
            self.bridge.set_overlay(
                worker.analytics.zones.values(), worker.analytics.tripwires.values()
            )
            worker.start()
        except Exception as exc:
            log.error("analyst_start_failed", error=str(exc), exc_info=True)
            self.bridge.on_state("failed", f"{type(exc).__name__}: {exc}")
            return

        self.bundle = bundle
        self.worker = worker
        status = bundle.status()
        self.bridge.on_state(
            "running",
            f"{status['detector_mode']}|{len(camera.rules)}|"
            f"{'degraded' if status['degraded'] else 'full'}",
        )

    def _stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.run_label.setText("stopping…")
        worker, bundle = self.worker, self.bundle
        self.worker = self.bundle = None

        def shutdown() -> None:
            if worker is not None:
                worker.stop()
            if bundle is not None:
                bundle.close()
            self.bridge.on_state("stopped", "")

        threading.Thread(target=shutdown, daemon=True).start()

    def _apply_live_thresholds(self) -> None:
        """Push the two tunable thresholds into the running detector."""
        worker = self.worker
        if worker is None:
            return
        detector = worker.detector
        # The motion fallback has neither threshold; moving the sliders while
        # it is running is a no-op rather than an error.
        if hasattr(detector, "score_threshold"):
            detector.score_threshold = self.conf_slider.value() / 100.0
        if hasattr(detector, "nms_threshold"):
            detector.nms_threshold = self.iou_slider.value() / 100.0

    # -- live updates ------------------------------------------------------ #

    def _on_frame(self) -> None:
        latest = self.bridge.take()
        if latest is None:
            return
        image, stats = latest
        self.view.show_frame(image)

        # The frame *index* rather than a count of paints: frames are coalesced
        # when rendering falls behind, and an analyst reading "frames" means
        # position in the video, not how many made it to the screen.
        self._frames = stats["frame_index"]
        counts = stats["counts"]
        self.figures["frames"].set_value(self._frames)
        self.figures["persons"].set_value(counts.get("human", 0))
        self.figures["vehicles"].set_value(counts.get("vehicle", 0))
        self.figures["animals"].set_value(counts.get("animal", 0))
        self.figures["tracks"].set_value(stats["tracks"])
        self.figures["fps"].set_value(f"{stats['fps']:.1f}")

    def _on_event(self, event: Event) -> None:
        risk = event_risk(event)
        record = {
            "event_id": event.event_id,
            "time": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            "timestamp": event.timestamp,
            "type": event.event_type.value,
            "severity": event.severity.value,
            "risk": round(risk, 2),
            "rule_id": event.rule_id,
            "zone_id": event.zone_id,
            "track_ids": list(event.track_ids),
            "confidence": round(event.confidence, 3),
            "message": event.message,
            "frame_index": event.frame_index,
            "attributes": dict(event.attributes),
        }
        self.alerts.append(record)
        self.export_button.setEnabled(True)
        if risk < self.risk_slider.value() / 100.0:
            self.suppressed += 1
        self._repopulate_alerts()

    def _repopulate_alerts(self) -> None:
        """Re-render the alert log for the current risk threshold.

        The threshold filters the *view*, not the pipeline: every event stays
        in the export, so raising the slider can never quietly discard evidence
        that something happened.
        """
        floor = self.risk_slider.value() / 100.0
        shown = [a for a in self.alerts if a["risk"] >= floor]
        self.suppressed = len(self.alerts) - len(shown)

        self.alert_table.setRowCount(len(shown))
        for row, alert in enumerate(reversed(shown)):
            colour = SEVERITY_COLOURS.get(alert["severity"], MUTED)
            cells = [
                alert["time"], alert["type"], alert["severity"].upper(),
                f"{alert['risk']:.2f}", alert["message"],
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setForeground(Qt.GlobalColor.white)
                    item.setToolTip(f"rule {alert['rule_id']}")
                self.alert_table.setItem(row, column, item)
            severity_item = self.alert_table.item(row, 2)
            if severity_item is not None:
                severity_item.setForeground(QColor(colour))

        self.figures["alerts"].set_value(len(shown))
        self.suppressed_label.setText(
            f"{self.suppressed} below the risk threshold (kept in the export)"
            if self.suppressed else f"{len(self.alerts)} alerts recorded"
        )

    def _on_notice(self, state: str, detail: str) -> None:
        if state == "running":
            mode, rules, degraded = detail.split("|")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.run_label.setText("processing…")
            self._set_chip("RUNNING", GOOD if degraded == "full" else WARN)
            self.detector_line.setText(
                f"DETECTOR:  {self.model_combo.currentData() or 'none'}   "
                f"mode={mode}   rules={rules}   {degraded}"
            )
            self._note(f"analysis started - detector {mode}, {rules} rules",
                       GOOD if degraded == "full" else WARN)
            if degraded != "full":
                self._note(
                    "no neural detector artefact: running classical motion "
                    "detection. Object classes are inferred from shape.", WARN
                )
        elif state == "stopped":
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.run_label.setText("idle")
            self._set_chip("STOPPED", MUTED)
            self.view.clear_frame("stopped")
            self._note("analysis stopped")
        elif state == "failed":
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.run_label.setText("failed - see the log")
            self._set_chip("FAILED", BAD)
            self._note(detail, BAD)

    def _refresh_stream_state(self) -> None:
        worker = self.worker
        if worker is None:
            return
        health = worker.reader.health.as_dict()
        if not health["connected"]:
            self._set_chip("CONNECTING", WARN)
            if health["last_error"]:
                self.run_label.setText(str(health["last_error"])[:120])
            return
        self._set_chip("RUNNING", GOOD)
        stats = worker.stats
        self.run_label.setText(
            f"{health['resolution'] or '?'} · {health['fps']:.1f} fps in · "
            f"{stats.avg_latency * 1000:.0f} ms/frame · "
            f"{stats.reclassified} reclassified · {health['frames_dropped']} dropped"
        )

    # -- geometry ---------------------------------------------------------- #

    def _arm(self, mode: str) -> None:
        self._draw_mode = "" if self._draw_mode == mode else mode
        self._pending_points.clear()
        self.view.set_drawing(bool(self._draw_mode))
        self.finish_button.setEnabled(self._draw_mode == "zone")
        for name, button in (("tripwire", self.line_button), ("zone", self.zone_button)):
            button.setObjectName("armed" if self._draw_mode == name else "")
            button.setStyleSheet(button.styleSheet())  # force a restyle
        self.setStyleSheet(ANALYST_STYLE)
        if self._draw_mode:
            self._note(f"click on the video to place the {self._draw_mode}")

    def _on_point(self, x: float, y: float) -> None:
        self._pending_points.append((x, y))
        if self._draw_mode == "tripwire" and len(self._pending_points) == 2:
            start, end = self._pending_points
            wire = TripwireConfig(
                id=f"wire-{len(self.tripwires) + 1}",
                name=f"Fence line {len(self.tripwires) + 1}",
                start=start, end=end, direction="any",
                left_label="exfiltration", right_label="infiltration",
            )
            self.tripwires.append(wire)
            self._pending_points.clear()
            self._arm("")
            self._geometry_changed(f"fence line {wire.id} placed")
        else:
            self.geometry_label.setText(
                f"{len(self._pending_points)} point(s) placed - "
                + ("click once more" if self._draw_mode == "tripwire"
                   else "press Finish zone when the outline is closed")
            )

    def _finish_zone(self) -> None:
        if len(self._pending_points) < 3:
            QMessageBox.information(self, "Zone", "A zone needs at least three corners.")
            return
        zone = ZoneConfig(
            id=f"zone-{len(self.zones) + 1}",
            name=f"Zone {len(self.zones) + 1}",
            points=list(self._pending_points),
        )
        self.zones.append(zone)
        self._pending_points.clear()
        self._arm("")
        self._geometry_changed(f"zone {zone.id} placed")

    def _clear_geometry(self) -> None:
        self.zones.clear()
        self.tripwires.clear()
        self._pending_points.clear()
        self._arm("")
        self._geometry_changed("geometry cleared")

    def _geometry_changed(self, message: str) -> None:
        self.geometry_label.setText(
            f"{len(self.zones)} zone(s), {len(self.tripwires)} fence line(s)"
            if (self.zones or self.tripwires)
            else "none drawn - rules watch the whole frame"
        )
        self._note(message)
        self._reload_analytics()

    def _reload_analytics(self) -> None:
        """Apply edited geometry to a running analysis without restarting it.

        Rebuilding the engine resets per-rule state (dwell timers, crossing
        sides), which is correct: those states were accumulated against
        geometry that no longer exists.
        """
        worker = self.worker
        if worker is None:
            return
        from ibvap.analytics.engine import AnalyticsEngine

        worker.camera.zones = list(self.zones)
        worker.camera.tripwires = list(self.tripwires)
        worker.camera.rules = self._build_rules()
        engine = AnalyticsEngine(worker.camera, self.settings.analytics)
        worker.analytics = engine
        self.bridge.set_overlay(engine.zones.values(), engine.tripwires.values())
        self._note(f"analytics reloaded with {len(worker.camera.rules)} rules")

    # -- export ------------------------------------------------------------ #

    def _export(self) -> None:
        if not self.alerts:
            return
        default = f"ibvap-alerts-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export alerts", default, "JSON (*.json)"
        )
        if not path:
            return
        payload = {
            "site_id": self.settings.site_id,
            "exported_at": time.time(),
            "source": self._source_url() or "",
            "detector": self.model_combo.currentData(),
            "thresholds": {
                "confidence": self.conf_slider.value() / 100.0,
                "nms_iou": self.iou_slider.value() / 100.0,
                "risk": self.risk_slider.value() / 100.0,
            },
            "alerts": self.alerts,
        }
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._note(f"exported {len(self.alerts)} alerts to {path}", GOOD)

    # -- lifecycle --------------------------------------------------------- #

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if self.worker is not None:
            self.worker.stop()
        if self.bundle is not None:
            self.bundle.close()
        self.registry.close()
        super().closeEvent(event)


def main() -> int:
    """Entry point for ``ibvap-analyst`` and ``ibvap analyst``."""
    configure_logging()
    app = QApplication.instance() or QApplication([])
    window = AnalystWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
