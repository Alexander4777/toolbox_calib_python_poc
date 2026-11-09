"""Tests for the intrinsic parameters, the distortion model and the images."""

from __future__ import annotations

import numpy as np
import pytest

from calib import (
    camera_matrix,
    comp_distortion,
    comp_distortion_oulu,
    distort_pixels,
    distort_points,
    normalize,
    rectify_image,
    undistort_points,
)
from calib.intrinsics import _col


def test_camera_matrix_with_skew():
    """The skew term is fc(0) * alpha."""
    K = camera_matrix([1000.0, 900.0], [320.0, 240.0], 0.02)
    assert K[0, 1] == pytest.approx(20.0)
    assert K[0, 0] == 1000.0
    assert K[1, 1] == 900.0


def test_normalize_without_distortion():
    """Without distortion the normalization is a simple scaling."""
    x = np.array([[100.0, 200.0], [50.0, 60.0]])
    f = np.array([1000.0, 1000.0])
    c = np.array([320.0, 240.0])

    xn = normalize(x, f, c, np.zeros(5), 0.0)

    assert np.allclose(xn[0], (x[0] - c[0]) / f[0])
    assert np.allclose(xn[1], (x[1] - c[1]) / f[1])


def test_normalize_removes_skew():
    """The skew is undone by the normalization."""
    x = np.array([[100.0], [50.0]])
    f = np.array([1000.0, 1000.0])
    c = np.array([0.0, 0.0])
    alpha = 0.1

    xn = normalize(x, f, c, np.zeros(5), alpha)

    assert np.allclose(xn[0], (100.0 - alpha * 50.0) / 1000.0)
    assert np.allclose(xn[1], 50.0 / 1000.0)


def test_distort_undistort_round_trip():
    """Distorting and undistorting gives the original points back."""
    f = np.array([600.0, 600.0])
    c = np.array([320.0, 240.0])
    k = np.array([-0.25, 0.05, 0.001, -0.002, 0.0])

    xs = np.arange(40, 600, 40.0)
    ys = np.arange(40, 440, 40.0)
    grid = np.vstack(
        [np.repeat(xs, len(ys)), np.tile(ys, len(xs))]
    )

    distorted = distort_pixels(grid, f, c, k)
    recovered = undistort_points(distorted, f, c, k)

    assert np.allclose(recovered, grid, atol=1e-8)


def test_distort_points_matches_distort_pixels():
    """The two distortion entry points agree."""
    f = np.array([600.0, 600.0])
    c = np.array([320.0, 240.0])
    k = np.array([-0.2, 0.03, 0.001, -0.001, 0.0])

    pixels = np.array([[100.0, 400.0], [80.0, 300.0]])

    xn = normalize(pixels, f, c, np.zeros(5), 0.0)
    a = distort_points(xn, f, c, k)
    b = distort_pixels(pixels, f, c, k)

    assert np.allclose(a, b)


def test_comp_distortion_simple_radial():
    """The simple radial compensation inverts the radial distortion."""
    x = np.array([[0.1, 0.2], [0.3, -0.4]])
    k = 0.5

    # The forward model.
    r2 = x[0] ** 2 + x[1] ** 2
    distorted = x * (1.0 + k * r2).reshape(1, -1)

    recovered = comp_distortion(distorted, k)

    # The simple compensation is a second order approximation, so the
    # result is close but not exact.
    assert np.allclose(recovered, x, atol=1e-2)


def test_comp_distortion_oulu_is_the_inverse():
    """The Oulu compensation inverts the full distortion model.

    The compensation works on *normalized* coordinates, so the distorted
    points are normalized first.
    """
    f = np.array([600.0, 600.0])
    c = np.array([320.0, 240.0])
    k = np.array([-0.25, 0.05, 0.001, -0.002, 0.0])

    xs = np.arange(60, 580, 40.0)
    ys = np.arange(60, 420, 40.0)
    grid = np.vstack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])

    # The ideal normalized coordinates and their distorted counterparts.
    xn = normalize(grid, f, c, np.zeros(5), 0.0)

    from calib.geometry import apply_distortion

    xd = apply_distortion(xn, k)

    recovered = comp_distortion_oulu(xd, k)

    assert np.allclose(recovered, xn, atol=1e-8)


def test_rectify_image_shape_and_border():
    """The rectified image has the same shape, and uncovered pixels are white."""
    img = np.zeros((48, 64), dtype=float)
    img[20:28, 30:34] = 255.0

    # A strong barrel distortion pushes the source position of the
    # corners outside the image: those pixels must be left white.
    f = np.array([60.0, 60.0])
    c = np.array([32.0, 24.0])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    out = rectify_image(img, None, f, c, k, 0.0)

    assert out.shape == img.shape

    # With a large positive radial distortion the corners are pushed
    # outside of the source image.
    k2 = np.array([0.6, 0.0, 0.0, 0.0, 0.0])
    out2 = rectify_image(img, None, f, c, k2, 0.0)

    assert out2.shape == img.shape
    assert out2[0, 0] == 255.0


def test_rectify_image_removes_distortion():
    """A distorted grid becomes a regular grid after rectification."""
    f = np.array([600.0, 600.0])
    c = np.array([320.0, 240.0])
    k = np.array([-0.25, 0.05, 0.0, 0.0, 0.0])

    h, w = 480, 640
    img = np.zeros((h, w), dtype=float)

    xs = np.arange(40, w - 40, 80.0)
    ys = np.arange(40, h - 40, 80.0)
    grid = np.vstack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))])

    distorted = distort_pixels(grid, f, c, k)
    for i in range(distorted.shape[1]):
        xi, yi = int(round(distorted[0, i])), int(round(distorted[1, i]))
        if 0 <= xi < w and 0 <= yi < h:
            img[yi, xi] = 255.0

    out = rectify_image(img, None, f, c, k, 0.0)

    # The rectified dots must be close to the ideal grid positions.
    yy, xx = np.nonzero(out > 100)
    assert xx.size > 0

    detected = np.vstack([xx.astype(float), yy.astype(float)])

    # Each detected dot must be close to one of the ideal positions.
    dist = np.linalg.norm(
        grid[:, :, None] - detected[:, None, :], axis=0
    )
    assert np.all(dist.min(axis=0) < 6.0)
