"""Desktop console entry point."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ibvap.core.logging import configure_logging, get_logger
from ibvap.desktop.client import NodeClient
from ibvap.desktop.main_window import DARK_STYLE, MainWindow

log = get_logger(__name__)


class LoginDialog(QDialog):
    """Collects the node address and operator credentials."""

    def __init__(self, default_url: str = "http://127.0.0.1:8080") -> None:
        super().__init__()
        self.setWindowTitle("IBVAP — Sign in")
        self.setMinimumWidth(380)
        self.setStyleSheet(DARK_STYLE)
        self.client: NodeClient | None = None

        title = QLabel("IBVAP")
        title.setStyleSheet("font-size:22px;font-weight:700;color:#2f81f7;letter-spacing:2px;")
        subtitle = QLabel("Intelligent Border Video Analytics Platform")
        subtitle.setStyleSheet("color:#8b949e;font-size:11px;")

        self.url_input = QLineEdit(default_url)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("operator")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)

        form = QFormLayout()
        form.addRow("Node", self.url_input)
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color:#f85149;font-size:11px;")
        self.error_label.setWordWrap(True)

        self.sign_in = QPushButton("Sign in")
        self.sign_in.setObjectName("primary")
        self.sign_in.setDefault(True)
        self.sign_in.clicked.connect(self._attempt)
        self.password_input.returnPressed.connect(self._attempt)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(self.sign_in)

    def _attempt(self) -> None:
        url = self.url_input.text().strip()
        if not url:
            self.error_label.setText("a node address is required")
            return

        self.sign_in.setEnabled(False)
        self.error_label.setText("signing in…")
        client = NodeClient(url)

        def done(ok: bool, detail: str) -> None:
            self.sign_in.setEnabled(True)
            if ok:
                self.client = client
                self.accept()
            else:
                self.error_label.setText(detail or "sign-in failed")

        client.login(self.username_input.text(), self.password_input.text(), done)


def main(argv: list[str] | None = None) -> int:
    """Run the desktop console."""
    argv = list(argv if argv is not None else sys.argv)
    configure_logging("INFO", "text")

    application = QApplication(argv)
    application.setApplicationName("IBVAP Console")
    application.setOrganizationName("IBVAP")
    application.setStyle("Fusion")

    default_url = "http://127.0.0.1:8080"
    for index, argument in enumerate(argv):
        if argument in ("--node", "-n") and index + 1 < len(argv):
            default_url = argv[index + 1]

    dialog = LoginDialog(default_url)
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.client is None:
        return 0

    window = MainWindow(dialog.client)
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
