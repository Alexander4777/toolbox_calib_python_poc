"""Tests for the graphical interface.

The tests run with the ``offscreen`` Qt platform so that they need no
display.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from calib.gui.picking import (  # noqa: E402
    CornerPicker,
    detect_grid_corners,
    order_corners,
)
from calib.gui.session import CalibrationSession  # noqa: E402


@pytest.fixture(scope="module")
def app():
    """The Qt application, shared by the tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def make_grid_image(nx: int = 13, ny: int = 9, width: int = 640, height: int = 480) -> np.ndarray:
    """Return an image with a regular grid of bright squares on black."""
    image = np.zeros((height, width), dtype=np.uint8)

    step_x = (width - 80) / max(nx - 1, 1)
    step_y = (height - 80) / max(ny - 1, 1)

    xs = 40 + np.arange(nx) * step_x
    ys = 40 + np.arange(ny) * step_y

    for x in xs:
        for y in ys:
            x0, y0 = int(round(x)) - 4, int(round(y)) - 4
            image[y0 : y0 + 9, x0 : x0 + 9] = 255

    return image


# ======================================================================
# The picking helpers
# ======================================================================
def test_order_corners_sorts_a_grid():
    """A shuffled set of grid points is put back in order.

    The order is the one of the pattern: the top row of the image first,
    and the points of a row from the left to the right.
    """
    nx, ny = 4, 3
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    grid = np.vstack(
        [(xs.ravel() * 10.0), (ys.ravel() * 10.0)]
    ).T

    rng = np.random.default_rng(0)
    shuffled = grid[rng.permutation(grid.shape[0])]

    ordered = order_corners(shuffled, nx, ny)

    assert ordered.shape == grid.shape
    # The ordered points must match the original grid, up to the flip of
    # the vertical axis (the first row may be the bottom one).
    direct = np.allclose(ordered, grid)
    flipped = np.allclose(ordered, grid[::-1])
    assert direct or flipped


def test_order_corners_of_a_tilted_grid():
    """The order of a tilted grid is still the order of the pattern."""
    nx, ny = 5, 4
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    grid = np.vstack(
        [(xs.ravel() * 10.0), (ys.ravel() * 10.0)]
    ).T

    # A rotation of the grid, as seen by a tilted camera.
    angle = np.deg2rad(20.0)
    rot = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    tilted = grid @ rot.T + np.array([50.0, 30.0])

    rng = np.random.default_rng(1)
    shuffled = tilted[rng.permutation(tilted.shape[0])]

    ordered = order_corners(shuffled, nx, ny)

    # The order must match the tilted grid, up to the flip of the rows.
    direct = np.allclose(ordered, tilted, atol=1e-6)
    flipped = np.allclose(ordered, tilted[::-1], atol=1e-6)
    assert direct or flipped


def test_corner_picker_state():
    """The picker tracks the picked corners."""
    picker = CornerPicker(3, 2)

    assert picker.n_expected == 6
    assert not picker.is_complete

    for i in range(6):
        picker.add(i, i)

    assert picker.is_complete
    assert picker.as_array().shape == (2, 6)

    # A seventh corner is not accepted.
    picker.add(9, 9)
    assert len(picker.points) == 6

    picker.undo()
    assert not picker.is_complete
    assert len(picker.points) == 5

    picker.clear()
    assert picker.as_array().shape == (2, 0)


def test_detect_grid_corners_on_a_synthetic_pattern():
    """The automatic detector finds the corners of a grid of squares."""
    image = make_grid_image(nx=5, ny=4)

    corners = detect_grid_corners(image, 5, 4)

    assert corners is not None
    # The corners are returned as a (2, N) array, like everywhere else in
    # the toolbox.
    assert corners.shape == (2, 20)

    # The detected corners must be close to the true ones.
    xs = 40 + np.arange(5) * (640 - 80) / 4
    ys = 40 + np.arange(4) * (480 - 80) / 3
    ideal = np.vstack([np.repeat(xs, 4), np.tile(ys, 5)])

    dist = np.linalg.norm(
        ideal[:, :, None] - corners[:, None, :], axis=0
    )
    assert dist.min(axis=0).max() < 4.0


def test_detect_grid_corners_returns_none_when_absent():
    """A blank image has no corners."""
    image = np.zeros((120, 160), dtype=np.uint8)
    assert detect_grid_corners(image, 3, 3) is None


# ======================================================================
# The session
# ======================================================================
def test_session_grid_points():
    """The pattern coordinates follow the geometry of the grid."""
    session = CalibrationSession()
    session.grid_nx = 4
    session.grid_ny = 3
    session.grid_spacing = 25.0
    session.grid_z = 5.0

    grid = session.grid_points()

    assert grid.shape == (3, 12)
    assert grid[2].min() == 5.0
    assert set(np.unique(grid[0])) == {0.0, 25.0, 50.0, 75.0}
    assert set(np.unique(grid[1])) == {0.0, 25.0, 50.0}


def test_session_build_data_requires_corners():
    """Building the data without any corner is an error."""
    session = CalibrationSession()
    with pytest.raises(ValueError, match="no corners"):
        session.build_data()


def test_session_build_data(tmp_path):
    """The data holds the views that have corners."""
    import cv2

    session = CalibrationSession()
    session.grid_nx = 3
    session.grid_ny = 2

    for i in range(3):
        path = tmp_path / f"view{i}.png"
        image = np.zeros((100, 120), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        session.add_images([path])

    assert len(session.views) == 3

    # Give the corners of the first two views.
    session.views[0].corners = np.zeros((2, 6))
    session.views[1].corners = np.ones((2, 6))

    data = session.build_data()

    assert data.n_ima == 2
    assert data.image_size == (120, 100)
    assert data.points_3d[0].shape == (3, 6)
    assert data.active_images.tolist() == [True, True]


def test_session_build_data_checks_the_sizes(tmp_path):
    """Images of different sizes are rejected."""
    import cv2

    session = CalibrationSession()

    for i, (w, h) in enumerate([(120, 100), (140, 100)]):
        path = tmp_path / f"view{i}.png"
        cv2.imwrite(str(path), np.zeros((h, w), dtype=np.uint8))
        session.add_images([path])

    session.views[0].corners = np.zeros((2, 6))
    session.views[1].corners = np.zeros((2, 6))

    with pytest.raises(ValueError, match="same size"):
        session.build_data()


def test_session_reprojection(tmp_path):
    """The session reprojects the corners with the calibration result."""
    import cv2

    from calib import CalibrationOptions, calibrate

    session = CalibrationSession()
    session.grid_nx = 4
    session.grid_ny = 3

    rng = np.random.default_rng(0)
    grid = session.grid_points()

    for i in range(4):
        path = tmp_path / f"view{i}.png"
        cv2.imwrite(str(path), np.zeros((480, 640), dtype=np.uint8))

        om = rng.normal(scale=0.2, size=3)
        T = np.array([rng.uniform(-40, 40), rng.uniform(-40, 40), 500.0])

        from calib import project_points

        proj = project_points(grid, om, T, [600.0, 600.0], [320.0, 240.0], np.zeros(5))
        session.add_images([path])[0].corners = proj

    data = session.build_data()
    result = calibrate(data, CalibrationOptions(verbose=False))
    session.set_result(result, data)

    reproj = session.reproject(0)
    assert reproj is not None
    assert reproj.shape == (2, 12)

    errors = session.error_per_view()
    assert len(errors) == 4
    assert all(rms < 1e-3 for _, rms in errors)


# ======================================================================
# The widgets
# ======================================================================
def test_image_view_picks_corners(app):
    """Clicking on the image reports the corner in image coordinates."""
    from calib.gui.widgets import ImageView

    image = np.zeros((100, 200), dtype=np.uint8)
    image[10:90, 10:190] = 255

    view = ImageView()
    view.resize(400, 300)
    view.set_image(image)
    view.set_picker(CornerPicker(3, 3))

    picked = []
    view.cornerPicked.connect(lambda x, y: picked.append((x, y)))

    # The widget shows the image scaled; the click position must be
    # converted back to the image coordinates.
    view.zoom(1.0)
    centre = view._map_to_widget(50.0, 50.0)

    from PySide6.QtCore import QEvent, QPointF as QPointF2
    from PySide6.QtGui import QMouseEvent

    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF2(centre),
        QPointF2(centre),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    view.mousePressEvent(event)

    assert len(picked) == 1
    assert abs(picked[0][0] - 50.0) < 1.0
    assert abs(picked[0][1] - 50.0) < 1.0


def test_image_view_set_points(app):
    """The corners can be set and cleared."""
    from calib.gui.widgets import ImageView

    view = ImageView()
    view.set_image(np.zeros((50, 50), dtype=np.uint8))
    view.set_points(np.array([[1.0, 2.0], [3.0, 4.0]]))

    assert len(view._points) == 2

    view.clear_points()
    assert view._points == []


def test_main_window_construction(app):
    """The main window builds with all its tabs."""
    from calib.gui import CalibrationWindow

    window = CalibrationWindow()
    window.show()

    assert window.tabs.count() == 4
    assert window.tabs.tabText(0) == "&Images"

    window.close()


def test_main_window_add_images_and_pick(app, tmp_path):
    """Images can be added and corners picked through the window."""
    import cv2

    from calib.gui import CalibrationWindow

    window = CalibrationWindow()
    window.show()

    path = tmp_path / "view.png"
    cv2.imwrite(str(path), np.zeros((480, 640), dtype=np.uint8))

    window.session.add_images([path])
    window._refresh_table()
    window.image_table.setCurrentCell(0, 0)

    assert window.image_table.rowCount() == 1

    window.on_corner_picked(10.0, 20.0)
    assert window.session.views[0].corners.shape == (2, 1)

    window.on_corner_removed()
    assert window.session.views[0].corners.shape == (2, 0)

    window.close()
