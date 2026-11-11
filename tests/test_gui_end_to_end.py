"""End-to-end test of the graphical interface.

The test renders realistic views of a calibration pattern, loads them in
the interface, detects the corners automatically, runs the calibration and
checks that the results match the ground truth.

The pattern is a grid of squares drawn at the right scale, and the views
are chosen so that the whole pattern stays inside the image.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from calib import project_points  # noqa: E402
from calib.gui.picking import detect_grid_corners  # noqa: E402


@pytest.fixture(scope="module")
def app():
    """The Qt application, shared by the tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def render_pattern(X, om, T, fc, cc, width=1280, height=960, fill=0.35):
    """Render a calibration pattern as a set of squares.

    Parameters
    ----------
    X : numpy.ndarray, shape (3, N)
        The pattern coordinates.
    om : numpy.ndarray
        The rotation of the pattern.
    T : numpy.ndarray
        The translation of the pattern.
    fc, cc : numpy.ndarray
        The camera parameters.
    width, height : int
        The size of the image.
    fill : float
        The fraction of the cell that the square covers.

    Returns
    -------
    numpy.ndarray
        The rendered image, and the projections.
    """
    proj = project_points(X, om, T, fc, cc, np.zeros(5))
    image = np.zeros((height, width), dtype=np.uint8)

    # The size of a square is derived from the projected distance between
    # two neighbouring corners, so that the squares keep their size in the
    # image whatever the pose.
    step = float(np.linalg.norm(proj[:, 1] - proj[:, 0]))
    half = max(int(round(fill * step / 2)), 2)

    for i in range(proj.shape[1]):
        x, y = int(round(proj[0, i])), int(round(proj[1, i]))
        if half + 1 <= x < width - half - 1 and half + 1 <= y < height - half - 1:
            image[y - half : y + half + 1, x - half : x + half + 1] = 255

    return image, proj


def make_views(n_views=8, nx=13, ny=9, spacing=30.0, seed=0):
    """Render a set of views of a planar pattern that fits in the frame."""
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    X = np.vstack(
        [
            (xs.ravel() * spacing).astype(float),
            (ys.ravel() * spacing).astype(float),
            np.zeros(nx * ny),
        ]
    )

    fc = np.array([1200.0, 1180.0])
    cc = np.array([640.0, 480.0])

    rng = np.random.default_rng(seed)
    views = []

    while len(views) < n_views:
        om = rng.normal(scale=0.18, size=3)
        T = np.array(
            [
                rng.uniform(-40, 40),
                rng.uniform(-40, 40),
                rng.uniform(700, 850),
            ]
        )
        image, proj = render_pattern(X, om, T, fc, cc)

        # The pattern must be entirely inside the image.
        if proj[0].min() < 20 or proj[0].max() > 1260:
            continue
        if proj[1].min() < 20 or proj[1].max() > 940:
            continue

        views.append((image, proj))

    return X, fc, cc, views


def test_detector_on_realistic_views():
    """The detector finds the corners of every rendered view."""
    X, fc, cc, views = make_views()

    for index, (image, proj) in enumerate(views):
        corners = detect_grid_corners(image, 13, 9)

        assert corners is not None, f"view {index}: no corners detected"
        assert corners.shape == (2, 117), f"view {index}: wrong shape"

        # Every detected corner must be close to a true corner.
        distance = np.linalg.norm(
            proj.reshape(2, -1, 1) - corners.reshape(2, 1, -1), axis=0
        ).min(axis=0)
        assert distance.max() < 3.0, f"view {index}: {distance.max():.2f} px"


def test_gui_end_to_end(app, tmp_path):
    """The whole workflow runs and recovers the true parameters."""
    from calib.gui import CalibrationWindow

    X, fc, cc, views = make_views(n_views=6, seed=5)

    window = CalibrationWindow()
    window.show()

    paths = []
    for index, (image, _) in enumerate(views):
        path = tmp_path / f"view{index:02d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)

    window.session.add_images(paths)
    window._refresh_table()

    # Detect the corners of every view.
    for row in range(len(window.session.views)):
        window.image_table.setCurrentCell(row, 0)
        window.on_auto_detect()

    n_corners = {v.corners.shape[1] for v in window.session.views}
    assert n_corners == {117}, f"unexpected corner counts: {n_corners}"

    # Run the calibration and wait for the worker thread.
    window.on_run_calibration()
    assert window._thread is not None

    # The worker runs in its own thread, so the event loop must be pumped
    # until it reports its result.
    deadline = time.monotonic() + 180.0
    while window.session.result is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()

    result = window.session.result
    assert result is not None, "the calibration produced no result"

    # The true parameters must be recovered.
    assert np.allclose(result.fc, fc, rtol=5e-3), result.fc
    # The principal point is recovered to a few pixels, which is the
    # accuracy of the synthetic pattern (the squares are rendered with a
    # rounding to the nearest pixel).
    assert np.allclose(result.cc, cc, atol=6.0), result.cc
    assert result.rmse < 1.0, result.rmse

    # The results tab must have been filled in.
    assert "Focal length" in window.results_text.toPlainText()

    # The error table must list every view.
    assert window.error_table.rowCount() == len(views)

    window.close()
