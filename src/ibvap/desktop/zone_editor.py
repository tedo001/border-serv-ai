"""Interactive zone and tripwire editor.

An operator draws a virtual fence by clicking on a still frame from the camera
it belongs to. Two properties make this trustworthy:

* **Draw on what the camera actually sees.** The background is a live snapshot,
  not a map or a blank canvas, so the fence line is placed against the real
  scene - the actual fence, the actual track, the actual dead ground.
* **Store normalised coordinates.** Clicks are converted from widget pixels to
  the ``[0, 1]`` space the platform uses everywhere, via the *displayed image*
  rectangle rather than the widget rectangle. Those differ whenever the image
  is letterboxed inside the widget, and conflating them puts every zone the
  operator draws slightly off - consistently, and invisibly.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QWidget

#: Editing modes.
MODE_SELECT = "select"
MODE_ZONE = "zone"
MODE_TRIPWIRE = "tripwire"


class ZoneCanvas(QWidget):
    """A canvas for drawing polygons and lines over a camera still."""

    #: Emitted when a shape is completed: (kind, points in normalised space).
    shape_completed = pyqtSignal(str, list)
    #: Emitted when the operator's pointer moves, with normalised coordinates.
    cursor_moved = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._background: QPixmap | None = None
        self.mode = MODE_SELECT
        #: Vertices of the shape currently being drawn, in normalised space.
        self._draft: list[tuple[float, float]] = []
        #: Existing shapes: ``{"kind", "id", "name", "points", "direction"}``.
        self.shapes: list[dict[str, Any]] = []
        self._hover: tuple[float, float] | None = None

    # -- content ----------------------------------------------------------- #

    def set_background(self, image: QImage) -> None:
        self._background = QPixmap.fromImage(image)
        self.update()

    def set_shapes(self, shapes: list[dict[str, Any]]) -> None:
        self.shapes = list(shapes)
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._draft.clear()
        self.update()

    def clear_draft(self) -> None:
        self._draft.clear()
        self.update()

    # -- coordinate mapping ------------------------------------------------ #

    def _image_rect(self) -> QRect:
        """Rectangle the background occupies inside this widget.

        When no background has loaded, the whole widget is used so the editor
        still works against a blank canvas.
        """
        if self._background is None or self._background.isNull():
            return self.rect()

        widget = self.rect()
        scaled = self._background.size().scaled(
            widget.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        x = widget.x() + (widget.width() - scaled.width()) // 2
        y = widget.y() + (widget.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def to_normalised(self, point: QPoint) -> tuple[float, float]:
        """Widget pixel -> normalised ``[0, 1]`` image coordinate."""
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return 0.0, 0.0
        x = (point.x() - rect.x()) / rect.width()
        y = (point.y() - rect.y()) / rect.height()
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def to_widget(self, point: tuple[float, float]) -> QPointF:
        """Normalised image coordinate -> widget pixel."""
        rect = self._image_rect()
        return QPointF(
            rect.x() + point[0] * rect.width(),
            rect.y() + point[1] * rect.height(),
        )

    # -- interaction ------------------------------------------------------- #

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self.mode == MODE_SELECT:
            return
        position = self.to_normalised(event.position().toPoint())

        if event.button() == Qt.MouseButton.LeftButton:
            self._draft.append(position)
            # A tripwire is exactly two points, so it completes itself.
            if self.mode == MODE_TRIPWIRE and len(self._draft) == 2:
                self.shape_completed.emit(MODE_TRIPWIRE, list(self._draft))
                self._draft.clear()
        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click closes a polygon - the standard idiom, and it means
            # the operator never has to hit a small "finish" target precisely.
            if self.mode == MODE_ZONE and len(self._draft) >= 3:
                self.shape_completed.emit(MODE_ZONE, list(self._draft))
            self._draft.clear()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._hover = self.to_normalised(event.position().toPoint())
        self.cursor_moved.emit(*self._hover)
        if self._draft:
            self.update()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._draft.clear()
            self.update()
        elif event.key() == Qt.Key.Key_Backspace and self._draft:
            self._draft.pop()
            self.update()
        else:
            super().keyPressEvent(event)

    # -- painting ---------------------------------------------------------- #

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._image_rect()

        painter.fillRect(self.rect(), QColor("#0a0e14"))
        if self._background is not None and not self._background.isNull():
            painter.drawPixmap(rect, self._background)
        else:
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "waiting for a frame from this camera…",
            )

        for shape in self.shapes:
            self._draw_shape(painter, shape)
        self._draw_draft(painter)
        painter.end()

    def _draw_shape(self, painter: QPainter, shape: dict[str, Any]) -> None:
        points = [self.to_widget(p) for p in shape.get("points", [])]
        if len(points) < 2:
            return

        if shape.get("kind") == MODE_ZONE:
            colour = QColor("#f85149")
            polygon = QPolygonF(points)
            fill = QColor(colour)
            fill.setAlpha(48)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(colour, 2))
            painter.drawPolygon(polygon)
        else:
            colour = QColor("#d29922")
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(colour, 3))
            painter.drawLine(points[0], points[1])
            self._draw_direction_arrow(painter, points[0], points[1], shape.get("direction", "any"), colour)

        painter.setPen(QPen(colour, 1))
        painter.setBrush(QBrush(colour))
        for point in points:
            painter.drawEllipse(point, 4, 4)

        if name := shape.get("name"):
            painter.setPen(QPen(colour))
            painter.drawText(points[0] + QPointF(6, -6), name)

    def _draw_direction_arrow(
        self, painter: QPainter, a: QPointF, b: QPointF, direction: str, colour: QColor
    ) -> None:
        """Show which way across the wire raises an alert."""
        import math

        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / length, dx / length  # left-hand normal
        size = 22.0

        painter.setPen(QPen(colour, 2))
        for sign in ((1,) if direction == "left" else (-1,) if direction == "right" else (1, -1)):
            tip = QPointF(mid.x() + nx * size * sign, mid.y() + ny * size * sign)
            painter.drawLine(mid, tip)
            # Arrow head, drawn from the tip back along the shaft.
            angle = math.atan2(tip.y() - mid.y(), tip.x() - mid.x())
            for offset in (2.6, -2.6):
                painter.drawLine(
                    tip,
                    QPointF(
                        tip.x() + 8 * math.cos(angle + offset),
                        tip.y() + 8 * math.sin(angle + offset),
                    ),
                )

    def _draw_draft(self, painter: QPainter) -> None:
        if not self._draft:
            return
        colour = QColor("#2f81f7")
        points = [self.to_widget(p) for p in self._draft]

        painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        # Rubber-band segment from the last placed vertex to the cursor.
        if self._hover is not None:
            painter.drawLine(points[-1], self.to_widget(self._hover))
            if self.mode == MODE_ZONE and len(points) >= 2:
                # Preview the closing edge so the operator can see the shape
                # they will get before committing to it.
                painter.setPen(QPen(colour, 1, Qt.PenStyle.DotLine))
                painter.drawLine(self.to_widget(self._hover), points[0])

        painter.setPen(QPen(colour, 1))
        painter.setBrush(QBrush(colour))
        for point in points:
            painter.drawEllipse(point, 4, 4)
