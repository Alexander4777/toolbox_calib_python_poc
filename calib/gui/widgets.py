"""The image viewer with the interactive corner picking.

The widget shows an image and lets the user click the corners of the
calibration pattern.  It supports zooming and panning, and displays the
picked corners with their index.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .picking import CornerPicker

__all__ = ["ImageView"]


def _to_qimage(image: np.ndarray) -> QImage:
    """Convert a NumPy image to a ``QImage``.

    Parameters
    ----------
    image : numpy.ndarray
        The image.  Grayscale ``(H, W)`` and colour ``(H, W, 3)`` arrays
        are supported, as well as ``uint8`` and ``float`` arrays.

    Returns
    -------
    PySide6.QtGui.QImage
        The converted image.
    """
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim == 3:
        if image.shape[2] != 3:
            image = image[:, :, 0]
        else:
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return QImage(
                np.ascontiguousarray(image),
                image.shape[1],
                image.shape[0],
                image.shape[1] * 3,
                QImage.Format_RGB888,
            )

    # Grayscale.
    gray = image
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    return QImage(
        gray.data,
        gray.shape[1],
        gray.shape[0],
        gray.shape[1],
        QImage.Format_Grayscale8,
    )


class ImageView(QWidget):
    """A widget that shows an image and collects the pattern corners.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    """

    #: Emitted when a corner is picked.
    cornerPicked = Signal(float, float)
    #: Emitted when the last corner is removed.
    cornerRemoved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._pixmap: QPixmap | None = None
        self._image_size = (0, 0)

        self._picker: CornerPicker | None = None
        self._points: list[tuple[float, float]] = []

        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)

        self._panning = False
        self._pan_start = QPointF()

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------
    def set_image(self, image: np.ndarray | None) -> None:
        """Show an image.

        Parameters
        ----------
        image : numpy.ndarray or None
            The image to show, or ``None`` to clear the view.
        """
        if image is None:
            self._pixmap = None
            self._image_size = (0, 0)
        else:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            qimage = _to_qimage(array)
            self._pixmap = QPixmap.fromImage(qimage)
            self._image_size = (array.shape[1], array.shape[0])
            self.fit_to_window()

        self.update()

    def set_picker(self, picker: CornerPicker | None) -> None:
        """Set the corner picker that defines the expected pattern.

        Parameters
        ----------
        picker : CornerPicker or None
            The picker.  When it is ``None`` the corners are not drawn.
        """
        self._picker = picker
        self._points = []
        self.update()

    def set_points(self, points) -> None:
        """Set the corners to draw.

        Parameters
        ----------
        points : array_like, shape (2, N)
            The corners, in image coordinates.
        """
        array = np.asarray(points, dtype=float)
        if array.size == 0:
            self._points = []
        else:
            self._points = [(float(x), float(y)) for x, y in array.T]
        self.update()

    def clear_points(self) -> None:
        """Remove every drawn corner."""
        self._points = []
        self.update()

    # ------------------------------------------------------------------
    # Zoom and pan
    # ------------------------------------------------------------------
    def fit_to_window(self) -> None:
        """Zoom so that the whole image is visible."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        w, h = self._image_size
        if w <= 0 or h <= 0:
            return

        sx = self.width() / w
        sy = self.height() / h
        self._scale = max(min(sx, sy), 1e-6)
        self._offset = QPointF(0.0, 0.0)
        self.update()

    def zoom(self, factor: float, center: QPointF | None = None) -> None:
        """Zoom in or out.

        Parameters
        ----------
        factor : float
            The zoom factor, greater than one to zoom in.
        center : QPointF, optional
            The point that stays fixed.  The centre of the widget by
            default.
        """
        if center is None:
            center = QPointF(self.width() / 2.0, self.height() / 2.0)

        new_scale = self._scale * factor
        new_scale = max(min(new_scale, 40.0), 1e-4)

        # Keep the center fixed.
        image_center = (center - self._offset) / self._scale
        self._scale = new_scale
        self._offset = center - image_center * self._scale

        self.update()

    def wheelEvent(self, event) -> None:
        """Handle the mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self.zoom(factor, self._map_from_global(event.position()))
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Pick a corner, or start panning."""
        if event.button() == Qt.LeftButton and self._picker is not None:
            if self._picker.is_complete:
                return
            point = self._map_from_global(event.position())
            self._points.append((point.x(), point.y()))
            self.cornerPicked.emit(point.x(), point.y())
            self.update()
            event.accept()
            return

        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Pan the image."""
        if self._panning:
            delta = event.position() - self._pan_start
            self._offset += delta
            self._pan_start = event.position()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Stop panning."""
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self.unsetCursor()
            event.accept()

    def keyPressEvent(self, event) -> None:
        """Handle the keyboard shortcuts."""
        if event.key() == Qt.Key_Backspace and self._points:
            self._points.pop()
            self.cornerRemoved.emit()
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key_F:
            self.fit_to_window()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def _map_from_global(self, position: QPointF) -> QPointF:
        """Convert a widget position into image coordinates."""
        return (position - self._offset) / self._scale

    def _map_to_widget(self, x: float, y: float) -> QPointF:
        """Convert image coordinates into a widget position."""
        return QPointF(x * self._scale + self._offset.x(),
                       y * self._scale + self._offset.y())

    def paintEvent(self, event) -> None:
        """Draw the image and the corners."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(45, 45, 48))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(160, 160, 165))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No image loaded\n\nUse File ▸ Add images to load them",
            )
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        target = QRectF(
            self._offset.x(),
            self._offset.y(),
            self._image_size[0] * self._scale,
            self._image_size[1] * self._scale,
        )
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        self._draw_corners(painter)

        painter.end()

    def _draw_corners(self, painter: QPainter) -> None:
        """Draw the picked corners and the expected pattern."""
        if self._picker is None:
            return

        # The pattern outline, once every corner has been picked.
        if len(self._points) == self._picker.n_expected and self._picker.nx > 1:
            polygon = QPolygonF(
                [self._map_to_widget(x, y) for x, y in self._points]
            )
            painter.setPen(QPen(QColor(80, 200, 120), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolygon(polygon)

        # The individual corners.
        radius = max(3.0, 5.0 / max(self._scale, 1e-3))
        for i, (x, y) in enumerate(self._points):
            centre = self._map_to_widget(x, y)

            painter.setPen(QPen(QColor(20, 20, 20), 1.0))
            painter.setBrush(QBrush(QColor(255, 80, 80)))
            painter.drawEllipse(centre, radius, radius)

            # The index of the corner.
            if self._scale > 0.25:
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    QPointF(centre.x() + radius + 2, centre.y() - radius - 2),
                    str(i + 1),
                )

        # The points that are still expected are drawn in grey.
        if not self._picker.is_complete and self._pixmap is not None:
            remaining = self._picker.n_expected - len(self._points)
            if remaining > 0:
                # The counter is drawn in a small panel in the corner so
                # that it does not overlap the pattern.
                text = f"{len(self._points)} / {self._picker.n_expected} corners picked"
                box = QRectF(8, 8, 210, 24)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 160))
                painter.drawRoundedRect(box, 4, 4)
                painter.setPen(QColor(235, 235, 235))
                painter.drawText(box, Qt.AlignCenter, text)
