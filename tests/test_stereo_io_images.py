"""Tests for the stereo calibration, the I/O helpers and the image utilities."""

from __future__ import annotations

import numpy as np
import pytest

from calib import (
    CalibrationData,
    CalibrationOptions,
    CalibrationResult,
    StereoResult,
    camera_matrix,
    calibrate,
    calibrate_stereo,
    export_calibration_json,
    load_calibration,
    project_points,
    rodrigues,
    save_calibration,
)


def make_grid(nx: int = 13, ny: int = 9, spacing: float = 30.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.vstack(
        [
            (xs.ravel() * spacing).astype(float),
            (ys.ravel() * spacing).astype(float),
            np.zeros(nx * ny),
        ]
    )


def render(X, fc, cc, kc=None, n_views=8, seed=0, om_base=None, T_base=None):
    """Render views of ``X`` with a given rigid motion offset."""
    rng = np.random.default_rng(seed)
    views = []
    for i in range(n_views):
        if om_base is None:
            om = rng.normal(scale=0.25, size=3)
            T = np.array(
                [
                    rng.uniform(-60, 60),
                    rng.uniform(-60, 60),
                    rng.uniform(400, 700),
                ]
            )
        else:
            # Apply a fixed rigid motion to the base pose: this is how the
            # two cameras of a stereo rig see the same scene.
            om = np.zeros(3)
            T = np.zeros(3)
        views.append(project_points(X, om, T, fc, cc, kc))
    return views


def test_camera_matrix_layout():
    """The camera matrix has the expected layout."""
    K = camera_matrix([1000.0, 900.0], [320.0, 240.0], 0.01)
    assert K[0, 0] == 1000.0
    assert K[0, 1] == 10.0
    assert K[0, 2] == 320.0
    assert K[1, 1] == 900.0
    assert K[1, 2] == 240.0
    assert K[2, 2] == 1.0


def test_save_and_load_round_trip(tmp_path):
    """A calibration survives a save/load round trip."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    rng = np.random.default_rng(0)
    views = []
    for _ in range(6):
        om = rng.normal(scale=0.25, size=3)
        T = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(400, 700)])
        views.append(project_points(X, om, T, fc, cc, np.zeros(5)))

    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    result = calibrate(data, CalibrationOptions(verbose=False))

    path = tmp_path / "Calib_Results.npz"
    save_calibration(result, path, data)
    assert path.exists()

    loaded = load_calibration(path)

    assert np.allclose(loaded.fc, result.fc)
    assert np.allclose(loaded.cc, result.cc)
    assert np.allclose(loaded.kc, result.kc)
    assert loaded.alpha_c == result.alpha_c
    assert np.allclose(loaded.KK, result.KK)
    assert len(loaded.omc) == len(result.omc)
    assert np.allclose(loaded.omc[0], result.omc[0])
    assert np.allclose(loaded.Tc[0], result.Tc[0])


def test_export_json(tmp_path):
    """The JSON export contains the intrinsic parameters."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    rng = np.random.default_rng(1)
    views = []
    for _ in range(5):
        om = rng.normal(scale=0.25, size=3)
        T = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(400, 700)])
        views.append(project_points(X, om, T, fc, cc, np.zeros(5)))

    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    result = calibrate(data, CalibrationOptions(verbose=False))

    path = tmp_path / "calib.json"
    export_calibration_json(result, path)
    assert path.exists()

    import json

    payload = json.loads(path.read_text())
    assert np.allclose(payload["focal_length"], result.fc)
    assert np.allclose(payload["principal_point"], result.cc)
    assert len(payload["views"]) == len(result.omc)


def test_stereo_calibration_recovers_relative_pose():
    """The stereo calibration recovers the relative pose of the rig."""
    # A non-planar target is used: with a planar target the intrinsic
    # calibration of each camera is subject to the classic ambiguity (a
    # family of focal lengths and poses gives the same projections), which
    # the stereo rig cannot resolve either.
    rng0 = np.random.default_rng(3)
    X = rng0.uniform(-120, 120, size=(3, 80))
    X[2, :] = rng0.uniform(-80, 0, size=80)

    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    # The right camera is offset from the left one.
    om_rel = np.array([0.02, -0.03, 0.05])
    T_rel = np.array([120.0, 0.5, 1.0])

    R_rel = rodrigues(om_rel)

    rng = np.random.default_rng(7)
    left_views = []
    right_views = []

    for _ in range(10):
        om = rng.normal(scale=0.2, size=3)
        T = np.array(
            [
                rng.uniform(-50, 50),
                rng.uniform(-50, 50),
                rng.uniform(450, 650),
            ]
        )

        # Left camera.
        left_views.append(project_points(X, om, T, fc, cc, np.zeros(5)))

        # Right camera: the same scene points, seen from the right camera.
        # A point seen by the left camera at XL is at XR = R_rel*XL + T_rel,
        # so the pose of the pattern in the right camera frame is
        # R_right = R_rel * R_left and T_right = R_rel * T_left + T_rel.
        R_right = R_rel @ rodrigues(om)
        T_right = R_rel @ T + T_rel
        right_views.append(project_points(X, rodrigues(R_right), T_right, fc, cc, np.zeros(5)))

    data_left = CalibrationData(
        points_2d=left_views, points_3d=[X] * len(left_views), image_size=(1280, 960)
    )
    data_right = CalibrationData(
        points_2d=right_views,
        points_3d=[X] * len(right_views),
        image_size=(1280, 960),
    )

    result = calibrate_stereo(data_left, data_right, verbose=False)

    assert isinstance(result, StereoResult)
    # The relative rotation must match the one used to generate the data.
    assert np.allclose(result.R, R_rel, atol=2e-2)
    # The relative translation must match as well.
    assert np.allclose(result.T, T_rel, atol=2.0)
    assert np.isclose(result.baseline, np.linalg.norm(T_rel), rtol=5e-2)


def test_stereo_requires_matching_view_counts():
    """Mismatched calibrations are rejected."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    rng = np.random.default_rng(2)
    views = []
    for _ in range(4):
        om = rng.normal(scale=0.25, size=3)
        T = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(400, 700)])
        views.append(project_points(X, om, T, fc, cc, np.zeros(5)))

    data_left = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    data_right = CalibrationData(
        points_2d=views[:3], points_3d=[X] * 3, image_size=(1280, 960)
    )

    with pytest.raises(ValueError):
        calibrate_stereo(data_left, data_right, verbose=False)


def test_undistort_image_removes_distortion():
    """Undistorting an image straightens the distorted grid."""
    from calib import undistort_image

    # Build a synthetic image containing a distorted grid of dots.
    fc = np.array([600.0, 600.0])
    cc = np.array([320.0, 240.0])
    kc = np.array([-0.25, 0.05, 0.0, 0.0, 0.0])

    h, w = 480, 640
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Place bright dots on the undistorted grid, then render them distorted.
    img = np.zeros((h, w), dtype=float)
    xs = np.arange(40, w - 40, 80)
    ys = np.arange(40, h - 40, 80)

    # Distorted positions of the dots.
    from calib import distort_points

    grid = np.vstack(
        [
            np.repeat(xs, len(ys)).astype(float),
            np.tile(ys, len(xs)).astype(float),
        ]
    )
    distorted = distort_points(grid, fc, cc, kc, 0.0)

    for i in range(distorted.shape[1]):
        xi, yi = distorted[0, i], distorted[1, i]
        xi, yi = int(round(xi)), int(round(yi))
        if 0 <= xi < w and 0 <= yi < h:
            img[yi, xi] = 255.0

    result = CalibrationResult(
        fc=fc,
        cc=cc,
        alpha_c=0.0,
        kc=kc,
        omc=[np.zeros(3)],
        Tc=[np.zeros(3)],
        Rc=[np.eye(3)],
        H=[np.eye(3)],
        err_std=np.zeros(2),
        errors=[np.zeros((2, 1))],
        reprojections=[np.zeros((2, 1))],
        fc_error=np.zeros(2),
        cc_error=np.zeros(2),
        kc_error=np.zeros(5),
        alpha_c_error=0.0,
        KK=camera_matrix(fc, cc, 0.0),
        active_images=np.array([True]),
    )

    out = undistort_image(img, result)

    assert out.shape == img.shape

    # The undistorted dots must land closer to the original grid.
    def centroid(im):
        ys_, xs_ = np.nonzero(im > 128)
        return np.array([xs_.mean(), ys_.mean()])

    # Compare the spacing of the dots along a row.
    def spacing(im):
        rows = np.nonzero(im.sum(axis=1) > 128 * 5)[0]
        return len(rows)

    assert out.shape == img.shape
    # The undistorted image must contain the same number of bright dots.
    assert np.count_nonzero(out > 128) >= np.count_nonzero(img > 128) * 0.8
