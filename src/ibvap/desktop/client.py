"""Client for a remote IBVAP node, used by the desktop console.

Uses Qt's own networking rather than ``requests``/``httpx``: everything stays
on the Qt event loop, so no request can block the UI thread, and the same
signal/slot mechanics carry both HTTP replies and WebSocket frames.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWebSockets import QWebSocket

from ibvap.core.logging import get_logger

log = get_logger(__name__)


class NodeClient(QObject):
    """Talks to one IBVAP node's REST API and live alert stream."""

    #: Emitted with the decoded payload of every live event.
    event_received = pyqtSignal(dict)
    #: Emitted with a human-readable connection state.
    link_state_changed = pyqtSignal(str)
    #: Emitted when a request fails, with a message for the status bar.
    error = pyqtSignal(str)

    def __init__(self, base_url: str = "http://127.0.0.1:8080", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.token: str = ""
        self.refresh_token: str = ""
        self.username: str = ""
        self.role: str = ""

        self._network = QNetworkAccessManager(self)
        self._socket: QWebSocket | None = None
        self._reconnect = QTimer(self)
        self._reconnect.setSingleShot(True)
        self._reconnect.timeout.connect(self.open_stream)
        self._reconnect_delay = 1000

    # -- REST -------------------------------------------------------------- #

    def _request(self, path: str, *, authenticated: bool = True) -> QNetworkRequest:
        request = QNetworkRequest(QUrl(f"{self.base_url}{path}"))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        if authenticated and self.token:
            request.setRawHeader(b"Authorization", f"Bearer {self.token}".encode())
        return request

    def get(self, path: str, callback: Callable[[Any], None]) -> None:
        """GET ``path`` and hand the decoded JSON to ``callback``."""
        reply = self._network.get(self._request(path))
        reply.finished.connect(lambda: self._complete(reply, callback))

    def post(
        self, path: str, body: dict[str, Any], callback: Callable[[Any], None],
        *, authenticated: bool = True,
    ) -> None:
        reply = self._network.post(
            self._request(path, authenticated=authenticated),
            json.dumps(body).encode(),
        )
        reply.finished.connect(lambda: self._complete(reply, callback))

    def put(self, path: str, body: dict[str, Any], callback: Callable[[Any], None]) -> None:
        """PUT ``path`` - used to replace an existing resource such as a camera."""
        reply = self._network.put(self._request(path), json.dumps(body).encode())
        reply.finished.connect(lambda: self._complete(reply, callback))

    def delete(self, path: str, callback: Callable[[Any], None]) -> None:
        reply = self._network.deleteResource(self._request(path))
        reply.finished.connect(lambda: self._complete(reply, callback))

    def _complete(self, reply: QNetworkReply, callback: Callable[[Any], None]) -> None:
        try:
            payload = bytes(reply.readAll().data())
            status = reply.attribute(
                QNetworkRequest.Attribute.HttpStatusCodeAttribute
            )
            if reply.error() != QNetworkReply.NetworkError.NoError:
                detail = reply.errorString()
                try:
                    detail = json.loads(payload).get("detail", detail)
                except (json.JSONDecodeError, ValueError):
                    pass
                self.error.emit(f"{status or ''} {detail}".strip())
                return
            callback(json.loads(payload) if payload else None)
        except (json.JSONDecodeError, ValueError) as exc:
            self.error.emit(f"malformed response from node: {exc}")
        finally:
            # Qt does not free replies automatically; without this a console
            # left open for a shift leaks one object per poll.
            reply.deleteLater()

    def media_url(self, path: str) -> QUrl:
        """URL for a media endpoint, carrying the token as a query parameter."""
        separator = "&" if "?" in path else "?"
        return QUrl(f"{self.base_url}{path}{separator}token={self.token}")

    # -- authentication ---------------------------------------------------- #

    def login(self, username: str, password: str, callback: Callable[[bool, str], None]) -> None:
        def handle(data: Any) -> None:
            if not data:
                callback(False, "no response from node")
                return
            self.token = data.get("access_token", "")
            self.refresh_token = data.get("refresh_token", "")
            self.username = data.get("username", username)
            self.role = data.get("role", "viewer")
            callback(True, self.role)

        failed = [False]

        def on_error(message: str) -> None:
            if not failed[0]:
                failed[0] = True
                callback(False, message)

        self.error.connect(on_error)
        self.post(
            "/api/v1/auth/login",
            {"username": username, "password": password},
            handle,
            authenticated=False,
        )
        QTimer.singleShot(6000, lambda: self.error.disconnect(on_error) if not failed[0] else None)

    # -- live alert stream -------------------------------------------------- #

    def open_stream(self) -> None:
        """Open (or reopen) the WebSocket alert feed."""
        if not self.token:
            return
        self.close_stream()

        socket = QWebSocket()
        socket.connected.connect(self._on_stream_open)
        socket.disconnected.connect(self._on_stream_close)
        socket.textMessageReceived.connect(self._on_stream_message)
        socket.errorOccurred.connect(
            lambda _err: self.link_state_changed.emit("link error")
        )

        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        socket.open(QUrl(f"{scheme}://{host}/api/v1/events/stream?token={self.token}"))
        self._socket = socket
        self.link_state_changed.emit("connecting")

    def close_stream(self) -> None:
        if self._socket is not None:
            self._socket.disconnected.disconnect()
            self._socket.close()
            self._socket.deleteLater()
            self._socket = None

    def _on_stream_open(self) -> None:
        self._reconnect_delay = 1000
        self.link_state_changed.emit("live")

    def _on_stream_close(self) -> None:
        self.link_state_changed.emit("reconnecting")
        # Capped backoff: a console left running overnight must not hammer a
        # node that is down for maintenance.
        self._reconnect.start(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 30_000)

    def _on_stream_message(self, message: str) -> None:
        try:
            frame = json.loads(message)
        except json.JSONDecodeError:
            return
        if frame.get("type") == "event":
            self.event_received.emit(frame.get("event", {}))
