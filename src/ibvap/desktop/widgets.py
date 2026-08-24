"""Reusable widgets for the desktop console."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ibvap.core.logging import get_logger

log = get_logger(__name__)

#: Severity to colour, matching the browser console so the two agree.
SEVERITY_COLOURS: dict[str, str] = {
    "info": "#8b949e",
    "low": "#3fb950",
    "medium": "#d29922",
    "high": "#f0883e",
    "critical": "#f85149",
}

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


class MjpegStream(QObject):
    """Decodes a ``multipart/x-mixed-replace`` MJPEG stream into frames.

    Rather than parsing multipart boundaries, this scans for JPEG start/end
    markers directly. Boundary handling varies between servers and proxies -
    some omit the trailing CRLF, some re-wrap the parts - whereas SOI/EOI are
    part of the JPEG format itself and are the same everywhere.
    """

    frame_received = pyqtSignal(QImage)
    state_changed = pyqtSignal(str)

    #: Guard against a corrupt stream growing the buffer without bound: if no
    #: complete JPEG appears within this many bytes, the buffer is reset.
    MAX_BUFFER = 8 * 1024 * 1024

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._network = QNetworkAccessManager(self)
        self._reply: QNetworkReply | None = None
        self._buffer = bytearray()

    def start(self, url: QUrl) -> None:
        self.stop()
        request = QNetworkRequest(url)
        request.setAttribute(
            QNetworkRequest.Attribute.CacheLoadControlAttribute,
            QNetworkRequest.CacheLoadControl.AlwaysNetwork,
        )
        reply = self._network.get(request)
        reply.readyRead.connect(self._on_data)
        reply.errorOccurred.connect(lambda _e: self.state_changed.emit("error"))
        reply.finished.connect(lambda: self.state_changed.emit("closed"))
        self._reply = reply
        self._buffer.clear()
        self.state_changed.emit("connecting")

    def stop(self) -> None:
        if self._reply is not None:
            self._reply.readyRead.disconnect()
            self._reply.abort()
            self._reply.deleteLater()
            self._reply = None
        self._buffer.clear()

    def _on_data(self) -> None:
        if self._reply is None:
            return
        self._buffer.extend(self._reply.readAll().data())

        while True:
            start = self._buffer.find(JPEG_START)
            if start < 0:
                break
            end = self._buffer.find(JPEG_END, start + 2)
            if end < 0:
                break

            jpeg = bytes(self._buffer[start : end + 2])
            del self._buffer[: end + 2]

            image = QImage()
            if image.loadFromData(jpeg, "JPG"):
                self.frame_received.emit(image)
                self.state_changed.emit("live")

        if len(self._buffer) > self.MAX_BUFFER:
            log.warning("mjpeg_buffer_overflow", bytes=len(self._buffer))
            self._buffer.clear()


class VideoTile(QFrame):
    """One camera tile: live video, name, state and throughput."""

    clicked = pyqtSignal(str)

    def __init__(self, camera_id: str, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.setObjectName("videoTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._alerting = False
        self._stream = MjpegStream(self)
        self._stream.frame_received.connect(self._on_frame)
        self._stream.state_changed.connect(self._on_state)

        self.video = QLabel("connecting…")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background:#000;color:#8b949e;")
        self.video.setMinimumHeight(180)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight:600;")
        self.state_label = QLabel("—")
        self.meta_label = QLabel("")
        self.meta_label.setStyleSheet("color:#8b949e;font-family:monospace;font-size:11px;")

        caption = QHBoxLayout()
        caption.setContentsMargins(6, 3, 6, 3)
        caption.addWidget(self.name_label)
        caption.addWidget(self.state_label)
        caption.addStretch(1)
        caption.addWidget(self.meta_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video, 1)
        layout.addLayout(caption)

        self._pixmap: QPixmap | None = None

    def start(self, url: QUrl) -> None:
        self._stream.start(url)

    def stop(self) -> None:
        self._stream.stop()

    def _on_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self.video.setPixmap(
            self._pixmap.scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._render()

    def _on_state(self, state: str) -> None:
        if state in ("error", "closed"):
            self.video.setText("camera offline")
            self.video.setPixmap(QPixmap())
            self._pixmap = None

    def set_status(self, connected: bool, fps: float, latency_ms: float) -> None:
        self.state_label.setText("LIVE" if connected else "OFFLINE")
        self.state_label.setStyleSheet(
            f"color:{'#3fb950' if connected else '#f85149'};font-weight:600;font-size:11px;"
        )
        self.meta_label.setText(f"{fps:.1f} fps · {latency_ms:.1f} ms")
        if not connected:
            self.video.setText("camera offline")

    def flash_alert(self) -> None:
        """Highlight this tile when its camera raises a serious alert."""
        self._alerting = True
        self.setStyleSheet("#videoTile { border: 2px solid #f85149; }")
        self.update()

    def clear_alert(self) -> None:
        self._alerting = False
        self.setStyleSheet("")

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)


class SeverityBadge(QLabel):
    """A coloured severity chip."""

    def __init__(self, severity: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.set_severity(severity)

    def set_severity(self, severity: str) -> None:
        colour = SEVERITY_COLOURS.get(severity, "#8b949e")
        self.setText(severity.upper())
        self.setStyleSheet(
            f"color:{colour};font-weight:700;font-size:10px;"
            f"border:1px solid {colour};border-radius:3px;padding:1px 5px;"
        )


class StatCard(QFrame):
    """A large single-figure card for the dashboard."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.value_label = QLabel("—")
        self.value_label.setStyleSheet("font-size:26px;font-weight:600;font-family:monospace;")
        self.caption = QLabel(label.upper())
        self.caption.setStyleSheet("color:#8b949e;font-size:10px;letter-spacing:0.5px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption)

    def set_value(self, value: Any, tone: str = "") -> None:
        self.value_label.setText(str(value))
        colour = {"ok": "#3fb950", "warn": "#d29922", "bad": "#f85149"}.get(tone, "#e6edf3")
        self.value_label.setStyleSheet(
            f"font-size:26px;font-weight:600;font-family:monospace;color:{colour};"
        )
