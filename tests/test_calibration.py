"""End-to-end tests of the calibration pipeline on synthetic data.

A synthetic planar rig is rendered with known camera parameters; the
calibrator must recover them.
"""

from __future__ import annotations

import numpy as np

from calib import (
    CalibrationData,
    CalibrationOptions,
    calibrate,
    compute_extrinsic,
    compute_extrinsic_init,
    compute_homography,
    project_points,
    rodrigues,
)


def make_grid(nx: int = 13, ny: int = 9, spacing: float = 30.0) -> np.ndarray:
    """Return a planar ``(3, N)`` grid of points on the Z = 0 plane."""
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    X = np.vstack(
        [
            (xs.ravel() * spacing).astype(float),
            (ys.ravel() * spacing).astype(float),
            np.zeros(nx * ny),
        ]
    )
    return X


def render_views(
    X: np.ndarray,
    fc,
    cc,
    kc=None,
    alpha=0.0,
    n_views: int = 8,
    seed: int = 0,
    noise: float = 0.0,
):
    """Render ``n_views`` views of the grid with random poses.

    Returns
    -------
    list of numpy.ndarray
        The 2D observations of each view.
    """
    rng = np.random.default_rng(seed)
    views = []

    for _ in range(n_views):
        # Random pose looking roughly at the grid.
        ang = rng.normal(scale=0.25, size=3)
        om = ang
        T = np.array(
            [
                rng.uniform(-60, 60),
                rng.uniform(-60, 60),
                rng.uniform(400, 700),
            ]
        )
        proj = project_points(X, om, T, fc, cc, kc, alpha)
        if noise > 0:
            proj = proj + rng.normal(scale=noise, size=proj.shape)
        views.append(proj)

    return views


def test_homography_recovers_projected_grid():
    """The homography maps the grid to the image exactly (no distortion)."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    om = np.array([0.1, -0.15, 0.05])
    T = np.array([20.0, -10.0, 600.0])

    img = project_points(X, om, T, fc, cc, np.zeros(5), 0.0)

    H, _, _ = compute_homography(img, X[0:2, :])

    M = np.vstack([X[0:2, :], np.ones(X.shape[1])])
    reproj = H @ M
    reproj = reproj[0:2, :] / reproj[2, :]

    assert np.allclose(reproj, img, atol=1e-6)


def test_extrinsic_init_recovers_pose():
    """The planar pose initialization recovers the true pose."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    om_true = np.array([0.08, -0.12, 0.03])
    T_true = np.array([15.0, -8.0, 620.0])

    img = project_points(X, om_true, T_true, fc, cc, np.zeros(5), 0.0)

    om, T, R = compute_extrinsic_init(img, X, fc, cc, np.zeros(5), 0.0)

    R_true = rodrigues(om_true)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(T, T_true, atol=1e-3)


def test_compute_extrinsic_recovers_pose():
    """The full extrinsic estimation recovers the true pose."""
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])

    om_true = np.array([0.08, -0.12, 0.03])
    T_true = np.array([15.0, -8.0, 620.0])

    img = project_points(X, om_true, T_true, fc, cc, np.zeros(5), 0.0)

    om, T, R, H, proj, ex, JJ = compute_extrinsic(
        img, X, fc, cc, np.zeros(5), 0.0
    )

    R_true = rodrigues(om_true)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(T, T_true, atol=1e-3)
    assert np.max(np.abs(ex)) < 1e-6


def test_calibration_recovers_intrinsics():
    """The calibrator recovers the true intrinsic parameters."""
    X = make_grid()
    fc_true = np.array([1200.0, 1180.0])
    cc_true = np.array([640.0, 480.0])
    kc_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    views = render_views(X, fc_true, cc_true, kc_true, n_views=8, seed=42)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(verbose=False, max_iter=30)
    result = calibrate(data, options)

    assert result is not None
    # The focal length and principal point must be recovered.
    assert np.allclose(result.fc, fc_true, rtol=2e-3)
    assert np.allclose(result.cc, cc_true, atol=3.0)
    # The reprojection error must be essentially zero for perfect data.
    assert result.rmse < 1e-3


def test_calibration_recovers_intrinsics_with_noise():
    """With noise the calibration is still close to the truth."""
    X = make_grid()
    fc_true = np.array([1200.0, 1180.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(
        X, fc_true, cc_true, np.zeros(5), n_views=10, seed=7, noise=0.3
    )

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(verbose=False, max_iter=30)
    result = calibrate(data, options)

    assert result is not None
    assert np.allclose(result.fc, fc_true, rtol=5e-3)
    assert np.allclose(result.cc, cc_true, atol=6.0)
    # The noise level bounds the achievable accuracy.
    assert result.rmse < 1.0


def test_calibration_with_distortion():
    """Distortion coefficients are recovered when they are present."""
    X = make_grid()
    fc_true = np.array([1150.0, 1150.0])
    cc_true = np.array([640.0, 480.0])
    kc_true = np.array([-0.12, 0.03, 0.001, -0.002, 0.0])

    views = render_views(X, fc_true, cc_true, kc_true, n_views=12, seed=11)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(
        verbose=False,
        max_iter=40,
        est_dist=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    result = calibrate(data, options)

    assert result is not None
    assert np.allclose(result.fc, fc_true, rtol=5e-3)
    assert np.allclose(result.kc[:4], kc_true[:4], atol=0.02)


def test_calibration_reports_uncertainties():
    """The uncertainties are positive and finite."""
    X = make_grid()
    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=8, seed=3)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    result = calibrate(data, CalibrationOptions(verbose=False))

    assert np.all(np.isfinite(result.fc_error))
    assert np.all(result.fc_error >= 0)
    assert np.all(np.isfinite(result.cc_error))
    assert np.all(np.isfinite(result.kc_error))


def test_calibration_with_fixed_focal_length():
    """Fixing the focal length keeps it at the given value."""
    X = make_grid()
    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=8, seed=5)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(
        verbose=False,
        est_fc=np.array([0.0, 0.0]),
        center_optim=False,
    )
    result = calibrate(data, options, initial={"fc": fc_true})

    assert np.allclose(result.fc, fc_true, rtol=1e-9)


def test_single_view_planar_rejects_principal_point():
    """With a single view the principal point is not estimated."""
    X = make_grid()
    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=1, seed=9)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(verbose=False)
    result = calibrate(data, options)

    assert result is not None
    # The principal point must have been kept at the image centre.
    assert np.allclose(result.cc, [(1280 - 1) / 2, (960 - 1) / 2], atol=1e-6)


def test_active_images_are_respected():
    """Deactivated views do not contribute to the result."""
    X = make_grid()
    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=6, seed=13)

    active = np.array([1, 1, 1, 0, 0, 0], dtype=bool)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
        active_images=active,
    )

    result = calibrate(data, CalibrationOptions(verbose=False))

    assert result is not None
    assert np.sum(result.active_images) >= 3
    assert np.allclose(result.fc, fc_true, rtol=5e-3)


def test_non_planar_rig_is_supported():
    """A 3D rig can be calibrated too."""
    rng = np.random.default_rng(21)
    X = rng.uniform(-150, 150, size=(3, 60))
    X[2, :] = rng.uniform(-100, 0, size=60)

    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=10, seed=17)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    assert data.is_3d_rig

    result = calibrate(data, CalibrationOptions(verbose=False, max_iter=30))

    assert result is not None
    assert np.allclose(result.fc, fc_true, rtol=5e-3)
    assert result.rmse < 1e-2


def test_result_project_and_summary():
    """The result object can reproject points and print a summary."""
    X = make_grid()
    fc_true = np.array([1200.0, 1200.0])
    cc_true = np.array([640.0, 480.0])

    views = render_views(X, fc_true, cc_true, np.zeros(5), n_views=6, seed=23)

    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    result = calibrate(data, CalibrationOptions(verbose=False))

    reproj = result.project(X, 0)
    assert reproj.shape == (2, X.shape[1])
    assert np.max(np.abs(reproj - views[0])) < 1e-2

    summary = result.summary()
    assert "Focal length" in summary
    assert "Principal pt" in summary
