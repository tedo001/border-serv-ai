"""The control panel, as a real window.

``app.py`` at the repository root is the launcher's engine: it creates the
virtual environment, installs the project, writes a configuration, generates a
signing key and starts the node. It is deliberately standard-library only,
because it has to run on a bare checkout before anything is installed - and
that constraint is why its own window is Tk, which looks like 1998.

Once the environment exists that constraint is gone. This module is the same
control panel with the same actions, drawn in Qt and styled like the operator
and analyst consoles, so the three surfaces read as one product. It shells out
to ``app.py`` for every action rather than reimplementing any of them: one
implementation, two faces, and no way for the window and the terminal to
disagree about what "Start node" does.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ibvap.core.logging import configure_logging, get_logger

log = get_logger(__name__)

INK = "#0d1117"
PANEL = "#161b22"
FIELD = "#0a0e14"
BORDER = "#2a313c"
TEXT = "#e6edf3"
MUTED = "#8b949e"
FAINT = "#6e7681"
ACCENT = "#2f81f7"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"

HEAD = "Arial"

#: Actions, grouped the way an operator reaches for them. Each entry is
#: ``(action, label, tooltip)`` and ``action`` is passed straight to ``app.py``.
GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("RUN", [
        ("setup", "Set up environment", "Create .venv and install the project"),
        ("start", "Start node and open console", "Bring the platform up and open the browser console"),
        ("stop", "Stop node", "Shut the analytics node down"),
    ]),
    ("CONSOLES", [
        ("analyst", "Live analysis console", "One source, one detector - no node needed"),
        ("desktop", "Desktop operator console", "Video wall, alerts, zone editor"),
        ("console", "Browser console", "The same console in a web browser"),
    ]),
    ("CHECKS", [
        ("status", "Status", "What is running and how it is doing"),
        ("test", "Run tests", "The full automated suite"),
        ("validate", "Validate configuration", "Check configs/site.yaml before deploying it"),
        ("benchmark", "Benchmark this machine", "How many cameras this hardware can carry"),
    ]),
    ("MAINTENANCE", [
        ("logs", "Show node log", "Tail the node's log file"),
        ("clean", "Clean runtime data", "Remove the local database, evidence and logs"),
    ]),
]

STYLE = f"""
QMainWindow, QWidget {{ background: {INK}; color: {TEXT}; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
/* QLabel subclasses QFrame, so a card's border rule reaches the labels inside
   it and draws a box around every one. An explicit type rule takes it back. */
QLabel {{ border: none; background: transparent; }}
QPushButton {{
    background: #21262d; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 9px 12px; color: {TEXT}; text-align: left;
}}
QPushButton:hover {{ border-color: {ACCENT}; background: #263041; }}
QPushButton:disabled {{ color: #56606b; background: #191e25; border-color: #222833; }}
QPushButton#primary {{ background: {ACCENT}; border: none; font-weight: 700; }}
QPushButton#primary:hover {{ background: #4a92f7; }}
QPushButton#primary:disabled {{ background: #1b2b45; color: #56606b; }}
QPushButton#danger:hover {{ border-color: {BAD}; }}
QTextEdit {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 6px;
    color: {TEXT}; padding: 6px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: {PANEL}; color: {MUTED}; padding: 7px 16px;
    border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: #1f2630; color: {TEXT}; }}
QTableWidget {{
    background: {FIELD}; alternate-background-color: #10151c;
    gridline-color: #1c222b; border: none;
}}
QHeaderView::section {{
    background: {PANEL}; color: {MUTED}; padding: 5px;
    border: none; border-bottom: 1px solid {BORDER};
}}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical {{ background: {INK}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; }}
"""


def find_app_py() -> Path | None:
    """Locate ``app.py``, which owns every action this window offers.

    An editable install puts the package three levels below the checkout root;
    a copied tree may not, so the working directory is checked too. Returning
    ``None`` is handled by the caller rather than raising: a window that opens
    and explains the problem beats a traceback on a machine with no terminal.
    """
    candidates = [
        Path(__file__).resolve().parents[3] / "app.py",
        Path.cwd() / "app.py",
    ]
    override = os.environ.get("IBVAP_APP")
    if override:
        candidates.insert(0, Path(override))
    return next((c for c in candidates if c.is_file()), None)


class Tile(QFrame):
    """One live figure in the node strip."""

    def __init__(self, caption: str) -> None:
        super().__init__()
        self.setObjectName("card")
        self.value = QLabel("—")
        self.value.setStyleSheet(
            f"font-size:20px;font-weight:700;font-family:monospace;color:{TEXT};border:none;"
        )
        label = QLabel(caption.upper())
        label.setStyleSheet(f"color:{FAINT};font-size:9px;letter-spacing:1px;border:none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        layout.addWidget(self.value)
        layout.addWidget(label)

    def set(self, value: Any, colour: str = TEXT) -> None:
        self.value.setText(str(value))
        self.value.setStyleSheet(
            f"font-size:20px;font-weight:700;font-family:monospace;color:{colour};border:none;"
        )


class LauncherWindow(QMainWindow):
    """Control panel: every action, the node's live state, and the log."""

    appended = pyqtSignal(str)
    #: Carries a health poll back from its worker thread. It has to be a
    #: signal: `QTimer.singleShot` schedules onto the *calling* thread's event
    #: loop, and a bare worker thread has none, so the callback never ran and
    #: the panel sat on "CHECKING" forever.
    polled = pyqtSignal(object)
    #: Per-camera status, which needs a token that `/health` does not.
    cameras = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IBVAP — Control Panel")
        self.setMinimumSize(1080, 680)
        self.setStyleSheet(STYLE)

        self.app_py = find_app_py()
        self.host = "127.0.0.1"
        self.port = int(os.environ.get("IBVAP_PORT", "8080"))
        self.base_url = f"http://{self.host}:{self.port}"

        self.process: QProcess | None = None
        self.buttons: dict[str, QPushButton] = {}
        self._token: str | None = None

        self._build_ui()
        self.polled.connect(self._apply)
        self.cameras.connect(self._apply_cameras)

        if self.app_py is None:
            self._say(
                "app.py was not found. Run this window from the repository root, "
                "or set IBVAP_APP to its path.", BAD,
            )
            for button in self.buttons.values():
                button.setEnabled(False)
        else:
            self._say(f"control panel ready — {self.app_py}", MUTED)
            self._say("press 'Start node and open console' to bring the platform up.")

        self.poll = QTimer(self)
        self.poll.timeout.connect(self.refresh)
        self.poll.start(4000)
        self.refresh()

    # -- construction ------------------------------------------------------ #

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_actions())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 780])

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_footer())
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        # Scoped by object name: a bare `background/border` rule on a container
        # cascades to every child, which drew the header's bottom border
        # underneath each of its own labels.
        header.setStyleSheet(
            f"QWidget#header {{ background:{PANEL}; border-bottom:1px solid {BORDER}; }}"
        )

        mark = QLabel("IBVAP")
        mark.setFont(QFont(HEAD, 17, QFont.Weight.Bold))
        mark.setStyleSheet(f"color:{ACCENT};letter-spacing:2px;background:transparent;")
        subtitle = QLabel("Intelligent Border Video Analytics Platform")
        subtitle.setStyleSheet(f"color:{MUTED};font-size:11px;background:transparent;")

        left = QVBoxLayout()
        left.setSpacing(1)
        left.addWidget(mark)
        left.addWidget(subtitle)

        self.chip = QLabel("  CHECKING  ")
        self.chip.setStyleSheet(
            f"background:#21262d;color:{MUTED};font-weight:700;padding:6px 12px;border-radius:5px;"
        )
        self.chip_detail = QLabel("")
        self.chip_detail.setStyleSheet(
            f"color:{MUTED};font-size:11px;background:transparent;"
        )

        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.addLayout(left)
        layout.addStretch(1)
        layout.addWidget(self.chip_detail)
        layout.addSpacing(12)
        layout.addWidget(self.chip)
        return header

    def _build_actions(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(14, 14, 8, 14)
        layout.setSpacing(12)

        for caption, actions in GROUPS:
            card = QFrame()
            card.setObjectName("card")
            inner = QVBoxLayout(card)
            inner.setContentsMargins(10, 10, 10, 10)
            inner.setSpacing(6)

            title = QLabel(caption)
            title.setStyleSheet(
                f"color:{FAINT};font-size:9px;font-weight:700;letter-spacing:1.5px;border:none;"
            )
            inner.addWidget(title)

            for action, label, tip in actions:
                button = QPushButton(label)
                button.setToolTip(tip)
                if action == "start":
                    button.setObjectName("primary")
                elif action in ("stop", "clean"):
                    button.setObjectName("danger")
                button.clicked.connect(lambda _c, a=action: self.run(a))
                inner.addWidget(button)
                self.buttons[action] = button

            layout.addWidget(card)
        layout.addStretch(1)
        return column

    def _build_right(self) -> QWidget:
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background:{FIELD};border:1px solid {BORDER};border-radius:6px;"
            f"color:{TEXT};font-family:monospace;font-size:12px;padding:8px;"
        )
        self.appended.connect(self._append)

        clear = QPushButton("Clear")
        clear.setMaximumWidth(90)
        clear.clicked.connect(self.log.clear)
        open_console = QPushButton("Open console in browser")
        open_console.setMaximumWidth(220)
        open_console.clicked.connect(
            lambda: QDesktopServices.openUrl(_url(self.base_url))
        )
        bar = QHBoxLayout()
        bar.addWidget(clear)
        bar.addWidget(open_console)
        bar.addStretch(1)

        activity = QWidget()
        activity_layout = QVBoxLayout(activity)
        activity_layout.setContentsMargins(8, 8, 8, 8)
        activity_layout.addWidget(self.log, 1)
        activity_layout.addLayout(bar)

        self.tiles = {
            key: Tile(caption) for key, caption in (
                ("status", "node status"), ("cameras", "cameras online"),
                ("detector", "detector"), ("uptime", "uptime"),
                ("faces", "watchlist faces"), ("plates", "watchlist plates"),
            )
        }
        grid = QGridLayout()
        grid.setSpacing(8)
        for index, tile in enumerate(self.tiles.values()):
            grid.addWidget(tile, index // 3, index % 3)

        self.camera_table = QTableWidget(0, 5)
        self.camera_table.setHorizontalHeaderLabels(
            ["Camera", "State", "FPS", "Latency", "Detector"]
        )
        self.camera_table.setAlternatingRowColors(True)
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.camera_table.horizontalHeader().setStretchLastSection(True)

        node = QWidget()
        node_layout = QVBoxLayout(node)
        node_layout.setContentsMargins(8, 8, 8, 8)
        node_layout.addLayout(grid)
        node_layout.addWidget(self.camera_table, 1)

        tabs = QTabWidget()
        tabs.addTab(activity, "Activity")
        tabs.addTab(node, "Node")
        self.tabs = tabs
        return tabs

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footer")
        footer.setStyleSheet(
            f"QWidget#footer {{ background:{PANEL}; border-top:1px solid {BORDER}; }}"
        )
        self.footer_label = QLabel("")
        self.footer_label.setStyleSheet(
            f"color:{MUTED};font-size:11px;font-family:monospace;background:transparent;"
        )
        root = QLabel(str(self.app_py.parent) if self.app_py else "")
        root.setStyleSheet(
            f"color:{FAINT};font-size:11px;font-family:monospace;background:transparent;"
        )
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 6, 18, 6)
        layout.addWidget(self.footer_label)
        layout.addStretch(1)
        layout.addWidget(root)
        self._refresh_footer()
        return footer

    # -- running actions ---------------------------------------------------- #

    def run(self, action: str) -> None:
        """Run one ``app.py`` action, streaming its output into the log."""
        if self.app_py is None or self.process is not None:
            return
        self.tabs.setCurrentIndex(0)
        self._say(f"\n$ python app.py {action}", ACCENT)
        for button in self.buttons.values():
            button.setEnabled(False)

        process = QProcess(self)
        process.setWorkingDirectory(str(self.app_py.parent))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._drain)
        process.finished.connect(lambda code, _s: self._finished(action, code))
        # Unbuffered, or a long action's output arrives only when it exits and
        # the window looks frozen for the minutes an install takes.
        environment = process.processEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        self.process = process
        process.start(sys.executable, [str(self.app_py), action])

    def _drain(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in chunk.splitlines():
            self.appended.emit(line)

    def _finished(self, action: str, code: int) -> None:
        self.process = None
        for button in self.buttons.values():
            button.setEnabled(True)
        self._say(
            f"[{action}] finished, exit code {code}", GOOD if code == 0 else BAD
        )
        self._refresh_footer()
        self.refresh()

    def _say(self, message: str, colour: str = TEXT) -> None:
        self.appended.emit(f'<span style="color:{colour}">{message}</span>')

    def _append(self, line: str) -> None:
        self.log.append(line if line.startswith("<span") else _escape(line))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    # -- live state --------------------------------------------------------- #

    def refresh(self) -> None:
        """Poll the node, off the GUI thread.

        ``urlopen`` blocks for its whole timeout when nothing is listening,
        which is exactly the state this poll exists to detect; doing it inline
        would freeze the window for seconds at a time during setup.
        """
        import threading

        def probe() -> None:
            health = _get_json(f"{self.base_url}/health")
            self.polled.emit(health)
            self.cameras.emit(self._fetch_cameras() if health else None)

        threading.Thread(target=probe, daemon=True).start()

    def _apply(self, health: dict[str, Any] | None) -> None:
        if health is None:
            self.chip.setText("  NODE STOPPED  ")
            self.chip.setStyleSheet(
                f"background:#21262d;color:{MUTED};font-weight:700;"
                "padding:6px 12px;border-radius:5px;"
            )
            self.chip_detail.setText("")
            for tile in self.tiles.values():
                tile.set("—", MUTED)
            self.camera_table.setRowCount(0)
            return

        status = str(health.get("status", "?")).upper()
        colour = {"HEALTHY": GOOD, "DEGRADED": WARN, "PARTIAL": WARN,
                  "IDLE": MUTED}.get(status, BAD)
        self.chip.setText(f"  {status}  ")
        self.chip.setStyleSheet(
            f"background:#21262d;color:{colour};font-weight:700;"
            "padding:6px 12px;border-radius:5px;"
        )
        self.chip_detail.setText(
            f"{health.get('site_name', '')}  ·  {self.base_url}"
        )

        models = health.get("models") or {}
        watchlists = health.get("watchlists") or {}
        online = health.get("cameras_online", 0)
        total = health.get("cameras_total", 0)
        self.tiles["status"].set(status.title(), colour)
        self.tiles["cameras"].set(
            f"{online}/{total}", GOOD if online == total and total else WARN
        )
        self.tiles["detector"].set(
            models.get("detector_mode", "—"),
            WARN if models.get("degraded") else GOOD,
        )
        self.tiles["uptime"].set(_duration(health.get("uptime_seconds", 0)))
        self.tiles["faces"].set(watchlists.get("faces", 0))
        self.tiles["plates"].set(watchlists.get("plates", 0))

    def _fetch_cameras(self) -> list[dict[str, Any]] | None:
        """Per-camera status, signing in with the credentials app.py generated.

        The launcher wrote the bootstrap password into .env itself, so it can
        read the node the same way an operator does rather than asking for it a
        second time. Worker thread only - it blocks.
        """
        for attempt in range(2):
            if self._token is None:
                self._token = _login(self.base_url, self._credentials())
                if self._token is None:
                    return None
            rows = _get_json(
                f"{self.base_url}/api/v1/cameras", token=self._token
            )
            if rows is not None:
                return rows if isinstance(rows, list) else []
            self._token = None  # expired or revoked; sign in once more
            if attempt:
                return None
        return None

    def _credentials(self) -> tuple[str, str]:
        env = (self.app_py.parent / ".env") if self.app_py else None
        password = "ChangeThisPassphrase1!"
        user = "admin"
        if env and env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                if key.strip() == "IBVAP_SECURITY__BOOTSTRAP_ADMIN_PASSWORD":
                    password = value.strip()
                elif key.strip() == "IBVAP_SECURITY__BOOTSTRAP_ADMIN_USER":
                    user = value.strip()
        return user, password

    def _apply_cameras(self, rows: list[dict[str, Any]] | None) -> None:
        rows = rows or []
        self.camera_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            camera = row.get("camera", {})
            stream = row.get("stream", {})
            stats = row.get("stats", {})
            connected = bool(stream.get("connected"))
            cells = [
                str(camera.get("name") or camera.get("id", "")),
                "ONLINE" if connected else "OFFLINE",
                f"{float(stream.get('fps') or 0):.1f}",
                f"{float(stats.get('avg_latency_ms') or 0):.0f} ms",
                str(row.get("detector_mode", "")),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 1:
                    item.setForeground(QColor(GOOD if connected else BAD))
                self.camera_table.setItem(index, column, item)

    def _refresh_footer(self) -> None:
        if self.app_py is None:
            return
        root = self.app_py.parent
        venv = root / ".venv"
        config = root / "configs" / "site.yaml"
        self.footer_label.setText(
            f"Python {sys.version_info.major}.{sys.version_info.minor}"
            f"   ·   .venv {'ready' if venv.is_dir() else 'missing'}"
            f"   ·   site.yaml {'present' if config.is_file() else 'not written'}"
            f"   ·   port {self.port}"
        )

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if self.process is not None:
            self.process.kill()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        or "&nbsp;"
    )


def _duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _get_json(url: str, timeout: float = 3.0, token: str | None = None) -> Any:
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url)
    if token:
        request.add_header("authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _login(base_url: str, credentials: tuple[str, str], timeout: float = 4.0) -> str | None:
    import json
    import urllib.error
    import urllib.request

    user, password = credentials
    body = json.dumps({"username": user, "password": password}).encode()
    request = urllib.request.Request(
        f"{base_url}/api/v1/auth/login", data=body, method="POST"
    )
    request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode()).get("access_token")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _url(text: str) -> Any:
    from PyQt6.QtCore import QUrl

    return QUrl(text)


def main() -> int:
    """Entry point for ``ibvap-panel`` and ``python -m ibvap.desktop.launcher``."""
    configure_logging()
    app = QApplication.instance() or QApplication([])
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
