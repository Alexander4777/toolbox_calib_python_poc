#!/usr/bin/env python3
"""Validation script: compare the Python port against reference values.

This script exercises the calibration pipeline on synthetic data with
*known* answers and prints the recovered parameters next to the ground
truth, so the numerical behaviour can be inspected at a glance.

Run it with::

    python validate.py
"""

from __future__ import annotations

import numpy as np

from calib import (
    CalibrationData,
    CalibrationOptions,
    calibrate,
    compute_extrinsic,
    compute_homography,
    project_points,
    rodrigues,
)


def make_grid(nx=13, ny=9, spacing=30.0):
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.vstack(
        [
            (xs.ravel() * spacing).astype(float),
            (ys.ravel() * spacing).astype(float),
            np.zeros(nx * ny),
        ]
    )


def render(X, fc, cc, kc=None, alpha=0.0, n_views=10, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n_views):
        om = rng.normal(scale=0.25, size=3)
        T = np.array(
            [
                rng.uniform(-60, 60),
                rng.uniform(-60, 60),
                rng.uniform(400, 700),
            ]
        )
        proj = project_points(X, om, T, fc, cc, kc, alpha)
        if noise:
            proj = proj + rng.normal(scale=noise, size=proj.shape)
        views.append(proj)
    return views


def report(title, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {title}")
    if detail:
        print(f"       {detail}")


def main():
    print("=" * 72)
    print("Bouguet Camera Calibration Toolbox - Python port validation")
    print("=" * 72)

    # ------------------------------------------------------------------
    print("\n1. Rodrigues round trip")
    om = np.array([0.3, -0.2, 1.1])
    R = rodrigues(om)
    ok = np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    report("rotation matrix is orthonormal with det = 1", ok)

    om_back = rodrigues(R)
    ok = np.allclose(rodrigues(om_back), R, atol=1e-9)
    report("matrix -> vector -> matrix round trip", ok)

    # ------------------------------------------------------------------
    print("\n2. Homography on a planar grid")
    X = make_grid()
    fc = np.array([1200.0, 1200.0])
    cc = np.array([640.0, 480.0])
    om_t = np.array([0.1, -0.15, 0.05])
    T_t = np.array([20.0, -10.0, 600.0])
    img = project_points(X, om_t, T_t, fc, cc, np.zeros(5), 0.0)
    H, _, _ = compute_homography(img, X[0:2, :])
    M = np.vstack([X[0:2, :], np.ones(X.shape[1])])
    r = H @ M
    r = r[0:2] / r[2]
    err = np.max(np.abs(r - img))
    report("homography reproduces the projection", err < 1e-6, f"max error {err:.2e} px")

    # ------------------------------------------------------------------
    print("\n3. Extrinsic recovery (planar target)")
    om_e, T_e, R_e, H_e, proj, ex, JJ = compute_extrinsic(
        img, X, fc, cc, np.zeros(5), 0.0
    )
    ok = np.allclose(R_e, rodrigues(om_t), atol=1e-6)
    report("rotation recovered", ok, f"max diff {np.max(np.abs(R_e - rodrigues(om_t))):.2e}")
    ok = np.allclose(T_e, T_t, atol=1e-2)
    report("translation recovered", ok, f"T = {np.round(T_e, 4)}")
    ok = np.max(np.abs(ex)) < 1e-6
    report("reprojection error is zero", ok, f"max {np.max(np.abs(ex)):.2e} px")

    # ------------------------------------------------------------------
    print("\n4. Full calibration - planar target, no noise")
    fc_t = np.array([1200.0, 1180.0])
    cc_t = np.array([640.0, 480.0])
    views = render(X, fc_t, cc_t, np.zeros(5), n_views=10, seed=42)
    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    res = calibrate(data, CalibrationOptions(verbose=False))
    report(
        "focal length recovered",
        np.allclose(res.fc, fc_t, rtol=2e-3),
        f"fc = {np.round(res.fc, 4)}  (true {fc_t})",
    )
    report(
        "principal point recovered",
        np.allclose(res.cc, cc_t, atol=1.0),
        f"cc = {np.round(res.cc, 4)}  (true {cc_t})",
    )
    report("reprojection error ~ 0", res.rmse < 0.05, f"rmse = {res.rmse:.5f} px")

    # ------------------------------------------------------------------
    print("\n5. Full calibration - with lens distortion")
    kc_t = np.array([-0.12, 0.03, 0.001, -0.002, 0.0])
    views = render(X, fc_t, cc_t, kc_t, n_views=14, seed=11)
    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    res = calibrate(
        data,
        CalibrationOptions(verbose=False, max_iter=40, est_dist=np.array([1, 1, 1, 1, 0])),
    )
    report(
        "focal length recovered",
        np.allclose(res.fc, fc_t, rtol=5e-3),
        f"fc = {np.round(res.fc, 4)}  (true {fc_t})",
    )
    report(
        "distortion recovered",
        np.allclose(res.kc[:4], kc_t[:4], atol=0.02),
        f"kc = {np.round(res.kc[:4], 5)}  (true {kc_t[:4]})",
    )
    report("reprojection error ~ 0", res.rmse < 0.1, f"rmse = {res.rmse:.5f} px")

    # ------------------------------------------------------------------
    print("\n6. Full calibration - noisy observations")
    views = render(X, fc_t, cc_t, np.zeros(5), n_views=12, seed=7, noise=0.3)
    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    res = calibrate(data, CalibrationOptions(verbose=False))
    report(
        "focal length recovered within 0.5%",
        np.allclose(res.fc, fc_t, rtol=5e-3),
        f"fc = {np.round(res.fc, 4)}  (true {fc_t})",
    )
    report("rmse below the noise level", res.rmse < 0.6, f"rmse = {res.rmse:.5f} px")

    # ------------------------------------------------------------------
    print("\n7. Full calibration - 3D (non planar) target")
    rng = np.random.default_rng(21)
    X3 = rng.uniform(-150, 150, size=(3, 60))
    X3[2, :] = rng.uniform(-100, 0, size=60)
    views = render(X3, fc_t, cc_t, np.zeros(5), n_views=10, seed=17)
    data = CalibrationData(
        points_2d=views, points_3d=[X3] * len(views), image_size=(1280, 960)
    )
    res = calibrate(data, CalibrationOptions(verbose=False, max_iter=40))
    report(
        "focal length recovered",
        np.allclose(res.fc, fc_t, rtol=5e-3),
        f"fc = {np.round(res.fc, 4)}  (true {fc_t})",
    )
    report(
        "principal point recovered",
        np.allclose(res.cc, cc_t, atol=3.0),
        f"cc = {np.round(res.cc, 4)}  (true {cc_t})",
    )
    report("reprojection error ~ 0", res.rmse < 0.5, f"rmse = {res.rmse:.5f} px")

    # ------------------------------------------------------------------
    print("\n8. Skew estimation")
    alpha_t = 0.02
    views = render(X, fc_t, cc_t, np.zeros(5), alpha=alpha_t, n_views=12, seed=31)
    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    res = calibrate(
        data, CalibrationOptions(verbose=False, max_iter=40, est_alpha=True)
    )
    report(
        "skew recovered",
        abs(res.alpha_c - alpha_t) < 5e-3,
        f"alpha_c = {res.alpha_c:.6f}  (true {alpha_t})",
    )

    # ------------------------------------------------------------------
    print("\n9. Focal length kept fixed")
    views = render(X, fc_t, cc_t, np.zeros(5), n_views=8, seed=5)
    data = CalibrationData(
        points_2d=views, points_3d=[X] * len(views), image_size=(1280, 960)
    )
    res = calibrate(
        data,
        CalibrationOptions(verbose=False, est_fc=np.array([0.0, 0.0]), center_optim=False),
        initial={"fc": fc_t},
    )
    report(
        "focal length unchanged",
        np.allclose(res.fc, fc_t, rtol=1e-9),
        f"fc = {np.round(res.fc, 6)}",
    )

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
