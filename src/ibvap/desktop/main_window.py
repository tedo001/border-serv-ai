"""Main window of the IBVAP desktop console."""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QAction, QImage, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ibvap.core.logging import get_logger
from ibvap.desktop.client import NodeClient
from ibvap.desktop.widgets import SeverityBadge, StatCard, VideoTile
from ibvap.desktop.zone_editor import MODE_SELECT, MODE_TRIPWIRE, MODE_ZONE, ZoneCanvas

log = get_logger(__name__)

#: Severity ranking, for filtering.
SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

DARK_STYLE = """
QMainWindow, QWidget { background: #0d1117; color: #e6edf3; }
QTabWidget::pane { border: 1px solid #2a313c; }
QTabBar::tab {
    background: #161b22; color: #8b949e; padding: 7px 16px;
    border: 1px solid #2a313c; border-bottom: none;
}
QTabBar::tab:selected { background: #1f2630; color: #e6edf3; }
QFrame { background: #161b22; border: 1px solid #2a313c; border-radius: 4px; }
QPushButton {
    background: #21262d; border: 1px solid #2a313c; border-radius: 4px;
    padding: 5px 12px; color: #e6edf3;
}
QPushButton:hover { border-color: #2f81f7; }
QPushButton:disabled { color: #56606b; }
QPushButton#primary { background: #2f81f7; border: none; font-weight: 600; }
QLineEdit, QComboBox, QTextEdit {
    background: #0a0e14; border: 1px solid #2a313c; border-radius: 4px;
    padding: 4px 7px; color: #e6edf3;
}
QTableWidget {
    background: #0d1117; alternate-background-color: #12171f;
    gridline-color: #1c222b; border: 1px solid #2a313c;
}
QHeaderView::section {
    background: #161b22; color: #8b949e; padding: 5px;
    border: none; border-bottom: 1px solid #2a313c;
}
QStatusBar { background: #161b22; color: #8b949e; }
QSplitter::handle { background: #2a313c; }
QScrollBar:vertical { background: #0d1117; width: 10px; }
QScrollBar::handle:vertical { background: #2a313c; border-radius: 5px; }
"""


class MainWindow(QMainWindow):
    """The operator's main workspace."""

    def __init__(self, client: NodeClient) -> None:
        super().__init__()
        self.client = client
        self.cameras: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.tiles: dict[str, VideoTile] = {}
        self._selected_camera: str | None = None

        self.setWindowTitle("IBVAP — Intelligent Border Video Analytics Platform")
        self.resize(1500, 940)
        self.setStyleSheet(DARK_STYLE)

        self._build_ui()
        self._connect_signals()

        # Status is polled; alerts are pushed. Polling status over a slow link
        # is cheap, whereas polling for alerts would add latency to precisely
        # the thing that has to be immediate.
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh_status)
        self.poll.start(5000)

        self.refresh_status()
        self.refresh_alerts()
        self.client.open_stream()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_wall_tab(), "Video wall")
        self.tabs.addTab(self._build_alerts_tab(), "Alerts")
        self.tabs.addTab(self._build_zones_tab(), "Zone editor")
        self.tabs.addTab(self._build_watchlist_tab(), "Watchlists")
        self.tabs.addTab(self._build_system_tab(), "System")
        self.setCentralWidget(self.tabs)

        self.link_label = QLabel("connecting")
        self.health_label = QLabel("—")
        self.user_label = QLabel(f"{self.client.username} ({self.client.role})")
        status = QStatusBar()
        status.addPermanentWidget(self.link_label)
        status.addPermanentWidget(self.health_label)
        status.addPermanentWidget(self.user_label)
        self.setStatusBar(status)
        self.statusBar().showMessage(f"Connected to {self.client.base_url}")

        refresh = QAction("Refresh", self)
        refresh.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh.triggered.connect(lambda: (self.refresh_status(), self.refresh_alerts()))
        self.addAction(refresh)

    def _build_wall_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.grid_columns = QComboBox()
        self.grid_columns.addItems(["1 column", "2 columns", "3 columns", "4 columns"])
        self.grid_columns.setCurrentIndex(1)
        self.grid_columns.currentIndexChanged.connect(self._relayout_wall)

        self.overlays_check = QCheckBox("Analytics overlays")
        self.overlays_check.setChecked(True)
        self.overlays_check.stateChanged.connect(self._restart_streams)

        self.wall_summary = QLabel("")
        self.wall_summary.setStyleSheet("color:#8b949e;")

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Layout"))
        toolbar.addWidget(self.grid_columns)
        toolbar.addWidget(self.overlays_check)
        toolbar.addStretch(1)
        toolbar.addWidget(self.wall_summary)

        self.wall_container = QWidget()
        self.wall_grid = QGridLayout(self.wall_container)
        self.wall_grid.setSpacing(8)

        layout.addLayout(toolbar)
        layout.addWidget(self.wall_container, 1)
        return page

    def _build_alerts_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.severity_filter = QComboBox()
        self.severity_filter.addItems(
            ["all", "low and above", "medium and above", "high and above", "critical only"]
        )
        self.severity_filter.setCurrentIndex(2)
        self.severity_filter.currentIndexChanged.connect(self.refresh_alerts)

        self.camera_filter = QComboBox()
        self.camera_filter.addItem("all cameras", "")
        self.camera_filter.currentIndexChanged.connect(self.refresh_alerts)

        self.unack_check = QCheckBox("Unacknowledged only")
        self.unack_check.stateChanged.connect(self.refresh_alerts)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search messages…")
        self.search_box.returnPressed.connect(self.refresh_alerts)

        toolbar = QHBoxLayout()
        for widget in (
            QLabel("Severity"), self.severity_filter, QLabel("Camera"),
            self.camera_filter, self.unack_check, self.search_box,
        ):
            toolbar.addWidget(widget)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_alerts)
        toolbar.addWidget(refresh_button)

        self.alert_table = QTableWidget(0, 5)
        self.alert_table.setHorizontalHeaderLabels(["Time", "Severity", "Type", "Camera", "Message"])
        self.alert_table.setAlternatingRowColors(True)
        self.alert_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.alert_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.alert_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.alert_table.itemSelectionChanged.connect(self._on_alert_selected)

        self.evidence_label = QLabel("Select an alert to review its evidence.")
        self.evidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.evidence_label.setMinimumHeight(260)
        self.evidence_label.setStyleSheet("background:#000;color:#8b949e;")

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(190)

        self.verify_button = QPushButton("Verify evidence integrity")
        self.verify_button.clicked.connect(self._verify_evidence)
        self.verify_button.setEnabled(False)

        self.ack_note = QLineEdit()
        self.ack_note.setPlaceholderText("Disposition — what action was taken?")
        self.ack_button = QPushButton("Acknowledge")
        self.ack_button.setObjectName("primary")
        self.ack_button.clicked.connect(self._acknowledge)
        self.ack_button.setEnabled(False)

        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.addWidget(self.evidence_label, 1)
        detail_layout.addWidget(self.detail_text)
        detail_layout.addWidget(self.verify_button)
        ack_row = QHBoxLayout()
        ack_row.addWidget(self.ack_note, 1)
        ack_row.addWidget(self.ack_button)
        detail_layout.addLayout(ack_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.alert_table)
        splitter.addWidget(detail_panel)
        splitter.setSizes([880, 560])

        layout.addLayout(toolbar)
        layout.addWidget(splitter, 1)
        return page

    def _build_zones_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.zone_camera = QComboBox()
        self.zone_camera.currentIndexChanged.connect(self._load_zone_camera)

        self.mode_select = QPushButton("Select")
        self.mode_zone = QPushButton("Draw zone")
        self.mode_wire = QPushButton("Draw tripwire")
        for button, mode in (
            (self.mode_select, MODE_SELECT), (self.mode_zone, MODE_ZONE),
            (self.mode_wire, MODE_TRIPWIRE),
        ):
            button.setCheckable(True)
            button.clicked.connect(lambda _c, m=mode: self._set_draw_mode(m))
        self.mode_select.setChecked(True)

        self.snapshot_button = QPushButton("Refresh frame")
        self.snapshot_button.clicked.connect(self._load_zone_snapshot)
        self.save_zones_button = QPushButton("Save to camera")
        self.save_zones_button.setObjectName("primary")
        self.save_zones_button.clicked.connect(self._save_zones)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Camera"))
        toolbar.addWidget(self.zone_camera)
        toolbar.addWidget(self.mode_select)
        toolbar.addWidget(self.mode_zone)
        toolbar.addWidget(self.mode_wire)
        toolbar.addWidget(self.snapshot_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.save_zones_button)

        self.canvas = ZoneCanvas()
        self.canvas.shape_completed.connect(self._on_shape_completed)
        self.canvas.cursor_moved.connect(
            lambda x, y: self.statusBar().showMessage(f"cursor  x={x:.3f}  y={y:.3f}")
        )

        self.shape_table = QTableWidget(0, 4)
        self.shape_table.setHorizontalHeaderLabels(["Kind", "ID", "Name", "Detail"])
        self.shape_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.shape_table.setMaximumHeight(170)
        self.shape_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        remove_button = QPushButton("Remove selected")
        remove_button.clicked.connect(self._remove_shape)

        hint = QLabel(
            "Left-click to place points.  Right-click closes a zone.  "
            "Esc clears the shape in progress.  Backspace removes the last point."
        )
        hint.setStyleSheet("color:#8b949e;font-size:11px;")

        layout.addLayout(toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(hint)
        layout.addWidget(self.shape_table)
        layout.addWidget(remove_button)
        return page

    def _build_watchlist_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        # -- vehicles --
        plates = QWidget()
        plate_layout = QVBoxLayout(plates)
        plate_layout.addWidget(QLabel("Vehicle registrations"))

        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("MH12AB1234")
        self.plate_category = QComboBox()
        self.plate_category.addItems(["wanted", "stolen", "suspect", "banned", "permitted"])
        self.plate_reference = QLineEdit()
        self.plate_reference.setPlaceholderText("Case reference")
        add_plate = QPushButton("Add")
        add_plate.setObjectName("primary")
        add_plate.clicked.connect(self._add_plate)

        plate_form = QHBoxLayout()
        for widget in (self.plate_input, self.plate_category, self.plate_reference, add_plate):
            plate_form.addWidget(widget)
        plate_layout.addLayout(plate_form)

        self.plate_table = QTableWidget(0, 4)
        self.plate_table.setHorizontalHeaderLabels(["Plate", "Category", "Reference", "Added by"])
        self.plate_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.plate_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        plate_layout.addWidget(self.plate_table, 1)

        remove_plate = QPushButton("Remove selected")
        remove_plate.clicked.connect(self._remove_plate)
        plate_layout.addWidget(remove_plate)

        # -- faces --
        faces = QWidget()
        face_layout = QVBoxLayout(faces)
        face_layout.addWidget(QLabel("Face identities"))
        note = QLabel(
            "Enrolment requires supervisor authority and is recorded in the audit log."
        )
        note.setStyleSheet("color:#8b949e;font-size:11px;")
        face_layout.addWidget(note)

        self.face_table = QTableWidget(0, 5)
        self.face_table.setHorizontalHeaderLabels(["ID", "Name", "Category", "Views", "Added by"])
        self.face_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.face_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        face_layout.addWidget(self.face_table, 1)

        enrol_button = QPushButton("Enrol from image file…")
        enrol_button.clicked.connect(self._enrol_face)
        face_layout.addWidget(enrol_button)
        remove_face = QPushButton("Remove selected")
        remove_face.clicked.connect(self._remove_face)
        face_layout.addWidget(remove_face)

        layout.addWidget(plates, 1)
        layout.addWidget(faces, 1)
        return page

    def _build_system_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.stat_cards = {
            key: StatCard(label)
            for key, label in (
                ("cameras", "cameras online"), ("status", "node status"),
                ("events", "events (24 h)"), ("unack", "unacknowledged"),
                ("uptime", "uptime"), ("evidence", "evidence stored"),
            )
        }
        cards = QHBoxLayout()
        for card in self.stat_cards.values():
            cards.addWidget(card)

        self.camera_table = QTableWidget(0, 6)
        self.camera_table.setHorizontalHeaderLabels(
            ["Camera", "State", "FPS", "Latency", "Detector", "Frames"]
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.system_text = QTextEdit()
        self.system_text.setReadOnly(True)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.camera_table)
        splitter.addWidget(self.system_text)
        splitter.setSizes([900, 540])

        layout.addLayout(cards)
        layout.addWidget(splitter, 1)
        return page

    def _connect_signals(self) -> None:
        self.client.event_received.connect(self.on_live_event)
        self.client.link_state_changed.connect(self.on_link_state)
        self.client.error.connect(
            lambda message: self.statusBar().showMessage(f"⚠ {message}", 8000)
        )

    # -- status refresh ---------------------------------------------------- #

    def refresh_status(self) -> None:
        self.client.get("/api/v1/cameras", self._on_cameras)
        self.client.get("/health", self._on_health)

    @pyqtSlot(object)
    def _on_cameras(self, cameras: Any) -> None:
        if not isinstance(cameras, list):
            return
        rebuild = {c["camera"]["id"] for c in cameras} != set(self.tiles)
        self.cameras = cameras

        if rebuild:
            self._rebuild_wall()
            self._populate_camera_selectors()

        online = 0
        for camera in cameras:
            identifier = camera["camera"]["id"]
            stream = camera["stream"]
            if stream["connected"]:
                online += 1
            if tile := self.tiles.get(identifier):
                tile.set_status(
                    stream["connected"], stream["fps"], camera["stats"]["avg_latency_ms"]
                )
        self.wall_summary.setText(f"{online} of {len(cameras)} cameras online")
        self._populate_camera_table()

    def _on_health(self, health: Any) -> None:
        if not isinstance(health, dict):
            return
        status = health.get("status", "unknown")
        colour = {"healthy": "#3fb950", "degraded": "#d29922", "partial": "#d29922"}.get(
            status, "#f85149"
        )
        self.health_label.setText(f"  {status.upper()}  ")
        self.health_label.setStyleSheet(f"color:{colour};font-weight:600;")

        models = health.get("models", {})
        self.stat_cards["cameras"].set_value(
            f"{health.get('cameras_online', 0)}/{health.get('cameras_total', 0)}",
            "ok" if health.get("cameras_online") == health.get("cameras_total") else "warn",
        )
        self.stat_cards["status"].set_value(status, "ok" if status == "healthy" else "warn")
        self.stat_cards["uptime"].set_value(f"{health.get('uptime_seconds', 0) / 3600:.1f}h")
        evidence = (health.get("storage") or {}).get("evidence", {})
        self.stat_cards["evidence"].set_value(f"{evidence.get('megabytes', 0)} MB")

        lines = [
            f"site            {health.get('site_name')} ({health.get('site_id')})",
            f"tier            {health.get('tier')}",
            f"detector        {models.get('detector_mode')}",
            f"neural          {'yes' if models.get('detector_is_neural') else 'no — degraded fallback'}",
            f"ANPR            {'available' if models.get('anpr_available') else 'unavailable'}",
            f"face            {'available' if models.get('face_available') else 'unavailable'}",
            f"watchlists      {health.get('watchlists')}",
        ]
        if integrations := health.get("integrations"):
            lines.append(f"C2 sinks        {[s.get('name') for s in integrations.get('sinks', [])]}")
            lines.append(f"delivered       {integrations.get('delivered')}")
        if models.get("degraded"):
            lines.append("")
            lines.append(
                "WARNING: this node is running classical fallback detection. "
                "Accuracy is materially reduced until model artefacts are installed."
            )
        self.system_text.setPlainText("\n".join(lines))
        self.client.get("/api/v1/events/statistics?hours=24", self._on_statistics)

    def _on_statistics(self, stats: Any) -> None:
        if not isinstance(stats, dict):
            return
        self.stat_cards["events"].set_value(stats.get("total", 0))
        unacknowledged = stats.get("unacknowledged", 0)
        self.stat_cards["unack"].set_value(unacknowledged, "warn" if unacknowledged else "ok")

    # -- video wall -------------------------------------------------------- #

    def _rebuild_wall(self) -> None:
        for tile in self.tiles.values():
            tile.stop()
            tile.setParent(None)
        self.tiles.clear()

        for camera in self.cameras:
            identifier = camera["camera"]["id"]
            tile = VideoTile(identifier, camera["camera"].get("name") or identifier)
            tile.clicked.connect(self._on_tile_clicked)
            self.tiles[identifier] = tile
        self._relayout_wall()
        self._restart_streams()

    def _relayout_wall(self) -> None:
        while self.wall_grid.count():
            self.wall_grid.takeAt(0)
        columns = self.grid_columns.currentIndex() + 1
        for index, tile in enumerate(self.tiles.values()):
            self.wall_grid.addWidget(tile, index // columns, index % columns)

    def _restart_streams(self) -> None:
        annotate = 1 if self.overlays_check.isChecked() else 0
        for identifier, tile in self.tiles.items():
            tile.start(
                self.client.media_url(
                    f"/api/v1/cameras/{identifier}/stream.mjpeg?fps=6&annotate={annotate}"
                )
            )

    def _on_tile_clicked(self, camera_id: str) -> None:
        self._selected_camera = camera_id
        self.statusBar().showMessage(f"selected {camera_id}", 3000)

    # -- alerts ------------------------------------------------------------ #

    def refresh_alerts(self) -> None:
        parts = ["limit=200"]
        severity_index = self.severity_filter.currentIndex()
        if severity_index > 0:
            parts.append(f"min_severity={['', 'low', 'medium', 'high', 'critical'][severity_index]}")
        if camera := self.camera_filter.currentData():
            parts.append(f"camera_id={camera}")
        if self.unack_check.isChecked():
            parts.append("acknowledged=false")
        if search := self.search_box.text().strip():
            parts.append(f"search={search}")
        self.client.get(f"/api/v1/events?{'&'.join(parts)}", self._on_alerts)

    def _on_alerts(self, page: Any) -> None:
        if not isinstance(page, dict):
            return
        self.alerts = page.get("events", [])
        self._populate_alert_table()

    def _populate_alert_table(self) -> None:
        table = self.alert_table
        table.setRowCount(len(self.alerts))
        for row, alert in enumerate(self.alerts):
            stamp = time.strftime("%H:%M:%S", time.localtime(alert["timestamp"]))
            table.setItem(row, 0, QTableWidgetItem(stamp))
            table.setCellWidget(row, 1, SeverityBadge(alert["severity"]))
            table.setItem(row, 2, QTableWidgetItem(alert["event_type"]))
            table.setItem(row, 3, QTableWidgetItem(alert["camera_id"]))
            message = alert["message"] + ("  ✓" if alert.get("acknowledged") else "")
            table.setItem(row, 4, QTableWidgetItem(message))
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

    def _selected_alert(self) -> dict[str, Any] | None:
        rows = self.alert_table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return self.alerts[index] if 0 <= index < len(self.alerts) else None

    def _on_alert_selected(self) -> None:
        alert = self._selected_alert()
        if alert is None:
            return

        self.ack_button.setEnabled(
            not alert.get("acknowledged")
            and self.client.role in ("operator", "supervisor", "admin")
        )
        self.verify_button.setEnabled(bool(alert.get("has_snapshot")))
        self.verify_button.setText("Verify evidence integrity")
        self.verify_button.setStyleSheet("")

        rows = [
            ("severity", alert["severity"]), ("type", alert["event_type"]),
            ("camera", alert["camera_id"]),
            ("time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert["timestamp"]))),
            ("confidence", f"{alert['confidence']:.2f}"), ("rule", alert.get("rule_id") or "—"),
            ("zone", alert.get("zone_id") or "—"),
            ("tracks", ", ".join(map(str, alert.get("track_ids", []))) or "—"),
        ]
        rows += [(k.replace("_", " "), str(v)) for k, v in (alert.get("attributes") or {}).items()]
        if alert.get("acknowledged"):
            rows.append(("acknowledged by", alert.get("acknowledged_by") or "—"))
            rows.append(("disposition", alert.get("disposition") or "—"))
        width = max(len(k) for k, _ in rows) + 2
        self.detail_text.setPlainText("\n".join(f"{k:<{width}}{v}" for k, v in rows))

        if alert.get("has_snapshot"):
            self._load_evidence(alert["event_id"])
        else:
            self.evidence_label.setText("no evidence stored for this alert")
            self.evidence_label.setPixmap(__import__("PyQt6.QtGui", fromlist=["QPixmap"]).QPixmap())

    def _load_evidence(self, event_id: str) -> None:
        """Fetch and display the evidence snapshot for an alert."""
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtNetwork import QNetworkRequest

        request = QNetworkRequest(
            self.client.media_url(f"/api/v1/events/{event_id}/snapshot")
        )
        reply = self.client._network.get(request)

        def done() -> None:
            data = bytes(reply.readAll().data())
            image = QImage()
            if data and image.loadFromData(data):
                self.evidence_label.setPixmap(
                    QPixmap.fromImage(image).scaled(
                        self.evidence_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                self.evidence_label.setText("evidence could not be loaded")
            reply.deleteLater()

        reply.finished.connect(done)

    def _verify_evidence(self) -> None:
        alert = self._selected_alert()
        if alert is None:
            return

        def done(result: Any) -> None:
            if not isinstance(result, dict):
                return
            if result.get("valid"):
                self.verify_button.setText("✓ Evidence verified — unaltered")
                self.verify_button.setStyleSheet("color:#3fb950;")
            else:
                self.verify_button.setText(f"✗ {'; '.join(result.get('issues', []))}")
                self.verify_button.setStyleSheet("color:#f85149;")

        self.client.get(f"/api/v1/events/{alert['event_id']}/verify", done)

    def _acknowledge(self) -> None:
        alert = self._selected_alert()
        if alert is None:
            return

        def done(_result: Any) -> None:
            self.ack_note.clear()
            self.statusBar().showMessage("alert acknowledged", 4000)
            self.refresh_alerts()

        self.client.post(
            f"/api/v1/events/{alert['event_id']}/acknowledge",
            {"disposition": self.ack_note.text()},
            done,
        )

    @pyqtSlot(dict)
    def on_live_event(self, payload: dict[str, Any]) -> None:
        """Insert a live event at the top of the alert table."""
        threshold = self.severity_filter.currentIndex()
        if threshold and SEVERITY_ORDER.get(payload.get("severity", "info"), 0) < threshold:
            return

        alert = {
            "event_id": payload.get("event_id", ""),
            "camera_id": (payload.get("camera") or {}).get("id", ""),
            "event_type": payload.get("event_type", ""),
            "severity": payload.get("severity", "info"),
            "timestamp": payload.get("timestamp", time.time()),
            "confidence": payload.get("confidence", 0.0),
            "message": payload.get("message", ""),
            "rule_id": payload.get("rule_id", ""),
            "zone_id": payload.get("zone_id"),
            "track_ids": payload.get("track_ids", []),
            "attributes": payload.get("attributes", {}),
            "acknowledged": False,
            "has_snapshot": bool((payload.get("evidence") or {}).get("snapshot_url")),
        }
        self.alerts.insert(0, alert)
        del self.alerts[200:]
        self._populate_alert_table()

        if tile := self.tiles.get(alert["camera_id"]):
            if alert["severity"] in ("high", "critical"):
                tile.flash_alert()
                QTimer.singleShot(5000, tile.clear_alert)

    def on_link_state(self, state: str) -> None:
        colour = {"live": "#3fb950", "connecting": "#d29922", "reconnecting": "#d29922"}.get(
            state, "#f85149"
        )
        self.link_label.setText(f"  {state.upper()}  ")
        self.link_label.setStyleSheet(f"color:{colour};font-weight:600;")

    # -- zone editor ------------------------------------------------------- #

    def _populate_camera_selectors(self) -> None:
        for combo, include_all in ((self.camera_filter, True), (self.zone_camera, False)):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if include_all:
                combo.addItem("all cameras", "")
            for camera in self.cameras:
                combo.addItem(camera["camera"].get("name") or camera["camera"]["id"],
                              camera["camera"]["id"])
            if current:
                index = combo.findData(current)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        if self.zone_camera.count() and not self.canvas.shapes:
            self._load_zone_camera()

    def _set_draw_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.mode_select.setChecked(mode == MODE_SELECT)
        self.mode_zone.setChecked(mode == MODE_ZONE)
        self.mode_wire.setChecked(mode == MODE_TRIPWIRE)

    def _load_zone_camera(self) -> None:
        camera_id = self.zone_camera.currentData()
        if not camera_id:
            return
        self._load_zone_snapshot()
        self.client.get(f"/api/v1/cameras/{camera_id}/config", self._on_zone_config)

    def _load_zone_snapshot(self) -> None:
        camera_id = self.zone_camera.currentData()
        if not camera_id:
            return
        from PyQt6.QtNetwork import QNetworkRequest

        request = QNetworkRequest(
            self.client.media_url(f"/api/v1/cameras/{camera_id}/snapshot?annotate=0")
        )
        reply = self.client._network.get(request)

        def done() -> None:
            data = bytes(reply.readAll().data())
            image = QImage()
            if data and image.loadFromData(data):
                self.canvas.set_background(image)
            reply.deleteLater()

        reply.finished.connect(done)

    def _on_zone_config(self, config: Any) -> None:
        if not isinstance(config, dict):
            return
        shapes: list[dict[str, Any]] = []
        for zone in config.get("zones", []):
            shapes.append({
                "kind": MODE_ZONE, "id": zone["id"], "name": zone.get("name", zone["id"]),
                "points": [tuple(p) for p in zone["points"]], "kind_detail": zone.get("kind", "restricted"),
            })
        for wire in config.get("tripwires", []):
            shapes.append({
                "kind": MODE_TRIPWIRE, "id": wire["id"], "name": wire.get("name", wire["id"]),
                "points": [tuple(wire["start"]), tuple(wire["end"])],
                "direction": wire.get("direction", "any"),
            })
        self.canvas.set_shapes(shapes)
        self._camera_config = config
        self._populate_shape_table()

    def _on_shape_completed(self, kind: str, points: list[tuple[float, float]]) -> None:
        existing = {s["id"] for s in self.canvas.shapes}
        prefix = "zone" if kind == MODE_ZONE else "wire"
        index = 1
        while f"{prefix}{index}" in existing:
            index += 1

        name, ok = QInputDialog.getText(
            self, f"Name this {'zone' if kind == MODE_ZONE else 'tripwire'}",
            "Name:", text=f"{'Zone' if kind == MODE_ZONE else 'Tripwire'} {index}",
        )
        if not ok:
            self.canvas.clear_draft()
            return

        shape: dict[str, Any] = {
            "kind": kind, "id": f"{prefix}{index}", "name": name or f"{prefix}{index}",
            "points": points,
        }
        if kind == MODE_TRIPWIRE:
            direction, ok = QInputDialog.getItem(
                self, "Alert direction",
                "Which crossing direction raises an alert?",
                ["any", "left", "right"], 0, False,
            )
            shape["direction"] = direction if ok else "any"

        self.canvas.set_shapes([*self.canvas.shapes, shape])
        self._populate_shape_table()

    def _populate_shape_table(self) -> None:
        table = self.shape_table
        table.setRowCount(len(self.canvas.shapes))
        for row, shape in enumerate(self.canvas.shapes):
            table.setItem(row, 0, QTableWidgetItem(shape["kind"]))
            table.setItem(row, 1, QTableWidgetItem(shape["id"]))
            table.setItem(row, 2, QTableWidgetItem(shape["name"]))
            detail = (
                f"direction={shape.get('direction', 'any')}"
                if shape["kind"] == MODE_TRIPWIRE
                else f"{len(shape['points'])} points"
            )
            table.setItem(row, 3, QTableWidgetItem(detail))

    def _remove_shape(self) -> None:
        rows = self.shape_table.selectionModel().selectedRows()
        if not rows:
            return
        index = rows[0].row()
        shapes = list(self.canvas.shapes)
        if 0 <= index < len(shapes):
            del shapes[index]
            self.canvas.set_shapes(shapes)
            self._populate_shape_table()

    def _save_zones(self) -> None:
        """Write the drawn geometry back to the camera's configuration."""
        camera_id = self.zone_camera.currentData()
        config = getattr(self, "_camera_config", None)
        if not camera_id or not config:
            return

        config = dict(config)
        config["zones"] = [
            {
                "id": s["id"], "name": s["name"],
                "points": [list(p) for p in s["points"]],
                "kind": s.get("kind_detail", "restricted"), "enabled": True,
            }
            for s in self.canvas.shapes if s["kind"] == MODE_ZONE
        ]
        config["tripwires"] = [
            {
                "id": s["id"], "name": s["name"],
                "start": list(s["points"][0]), "end": list(s["points"][1]),
                "direction": s.get("direction", "any"),
                "left_label": "exfiltration", "right_label": "infiltration",
                "enabled": True,
            }
            for s in self.canvas.shapes if s["kind"] == MODE_TRIPWIRE
        ]
        # Drop rules that reference geometry the operator has just deleted;
        # the node would otherwise reject the whole update as inconsistent.
        zone_ids = {z["id"] for z in config["zones"]}
        wire_ids = {w["id"] for w in config["tripwires"]}
        kept = []
        for rule in config.get("rules", []):
            rule = dict(rule)
            rule["zones"] = [z for z in rule.get("zones", []) if z in zone_ids]
            rule["tripwires"] = [w for w in rule.get("tripwires", []) if w in wire_ids]
            kept.append(rule)
        config["rules"] = kept

        def done(_result: Any) -> None:
            QMessageBox.information(
                self, "Saved",
                f"Geometry saved to {camera_id}. The camera has been restarted "
                "to apply the new configuration.",
            )
            # Re-read from the node rather than trusting the local draft: this
            # is what proves the save actually landed, and it picks up any
            # normalisation the node applied on the way in.
            self.client.get(f"/api/v1/cameras/{camera_id}/config", self._on_zone_config)
            self.refresh_status()

        # PUT, not POST: POST /cameras creates, PUT /cameras/{id} replaces.
        self.client.put(f"/api/v1/cameras/{camera_id}", config, done)

    # -- watchlists -------------------------------------------------------- #

    def refresh_watchlists(self) -> None:
        self.client.get("/api/v1/watchlists/plates", self._on_plates)
        self.client.get("/api/v1/watchlists/faces", self._on_faces)

    def _on_plates(self, plates: Any) -> None:
        if not isinstance(plates, list):
            return
        self.plate_table.setRowCount(len(plates))
        for row, plate in enumerate(plates):
            for column, key in enumerate(("plate", "category", "reference", "created_by")):
                self.plate_table.setItem(row, column, QTableWidgetItem(str(plate.get(key, ""))))

    def _on_faces(self, faces: Any) -> None:
        if not isinstance(faces, list):
            return
        self.face_table.setRowCount(len(faces))
        for row, face in enumerate(faces):
            for column, key in enumerate(
                ("person_id", "name", "category", "embedding_count", "created_by")
            ):
                self.face_table.setItem(row, column, QTableWidgetItem(str(face.get(key, ""))))

    def _add_plate(self) -> None:
        plate = self.plate_input.text().strip()
        if not plate:
            return

        def done(_result: Any) -> None:
            self.plate_input.clear()
            self.plate_reference.clear()
            self.statusBar().showMessage("registration added to the watchlist", 4000)
            self.refresh_watchlists()

        self.client.post(
            "/api/v1/watchlists/plates",
            {
                "plate": plate,
                "category": self.plate_category.currentText(),
                "reference": self.plate_reference.text().strip(),
            },
            done,
        )

    def _remove_plate(self) -> None:
        rows = self.plate_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.plate_table.item(rows[0].row(), 0)
        if item is None:
            return
        plate = item.text()
        if QMessageBox.question(
            self, "Remove registration", f"Remove {plate} from the watchlist?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.client.delete(
            f"/api/v1/watchlists/plates/{plate}", lambda _r: self.refresh_watchlists()
        )

    def _enrol_face(self) -> None:
        import base64

        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image containing exactly one face",
            "", "Images (*.jpg *.jpeg *.png)",
        )
        if not path:
            return
        person_id, ok = QInputDialog.getText(self, "Enrol identity", "Person ID:")
        if not ok or not person_id:
            return
        name, _ = QInputDialog.getText(self, "Enrol identity", "Name:")
        category, ok = QInputDialog.getItem(
            self, "Enrol identity", "Category:",
            ["wanted", "suspect", "staff", "permitted", "banned"], 0, False,
        )
        if not ok:
            return

        try:
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode()
        except OSError as exc:
            QMessageBox.warning(self, "Enrolment failed", str(exc))
            return

        def done(_result: Any) -> None:
            self.statusBar().showMessage(f"{person_id} enrolled", 5000)
            self.refresh_watchlists()

        self.client.post(
            "/api/v1/watchlists/faces",
            {
                "person_id": person_id, "name": name, "category": category,
                "image_base64": encoded,
            },
            done,
        )

    def _remove_face(self) -> None:
        rows = self.face_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.face_table.item(rows[0].row(), 0)
        if item is None:
            return
        person_id = item.text()
        if QMessageBox.question(
            self, "Remove identity", f"Remove {person_id} from the watchlist?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.client.delete(
            f"/api/v1/watchlists/faces/{person_id}", lambda _r: self.refresh_watchlists()
        )

    # -- system ------------------------------------------------------------ #

    def _populate_camera_table(self) -> None:
        table = self.camera_table
        table.setRowCount(len(self.cameras))
        for row, camera in enumerate(self.cameras):
            stream, stats = camera["stream"], camera["stats"]
            values = [
                camera["camera"].get("name") or camera["camera"]["id"],
                "online" if stream["connected"] else "offline",
                f"{stream['fps']:.1f}",
                f"{stats['avg_latency_ms']} ms",
                camera.get("detector_mode", "—"),
                str(stats["frames_processed"]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    from PyQt6.QtGui import QColor

                    item.setForeground(QColor("#3fb950" if stream["connected"] else "#f85149"))
                table.setItem(row, column, item)

    # -- lifecycle --------------------------------------------------------- #

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self.refresh_watchlists()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.poll.stop()
        for tile in self.tiles.values():
            tile.stop()
        self.client.close_stream()
        super().closeEvent(event)
