"""Stereo calibration: relative pose between two calibrated cameras.

Port of ``go_calib_stereo.m`` and the underlying helpers.

Definition of the extrinsic parameters
--------------------------------------
A point ``P`` with coordinates ``XL`` in the left camera frame and ``XR`` in
the right camera frame is related by

``XR = R @ XL + T``

where ``R = rodrigues(om)``.  ``om`` and ``T`` describe the position of the
right camera with respect to the left one.

Workflow
--------
Both cameras must first be calibrated separately (see
:func:`calib.calibration.calibrate`), on the same views and with the
calibration pattern at the same 3D locations.  The relative pose is then
initialized from those two calibrations and refined jointly with the
intrinsic parameters of the two cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CalibrationData, CalibrationOptions
from .extrinsics import compose_motion
from .geometry import (
    flatten_points,
    project_points,
    project_points_with_jacobians,
    rodrigues,
)
from .intrinsics import camera_matrix

__all__ = ["StereoResult", "calibrate_stereo"]


@dataclass
class StereoResult:
    """Outcome of a stereo calibration.

    Attributes
    ----------
    om : numpy.ndarray, shape (3,)
        Rotation vector of the right camera relative to the left one.
    T : numpy.ndarray, shape (3,)
        Translation of the right camera relative to the left one.
    R : numpy.ndarray, shape (3, 3)
        The corresponding rotation matrix.
    fc_left, cc_left, kc_left : numpy.ndarray
        Intrinsic parameters of the left camera.
    fc_right, cc_right, kc_right : numpy.ndarray
        Intrinsic parameters of the right camera.
    alpha_c_left, alpha_c_right : float
        Skew of each camera.
    KK_left, KK_right : numpy.ndarray
        The camera matrices.
    baseline : float
        Distance between the two optical centres.
    om_error : numpy.ndarray, shape (3,)
        Uncertainty of the rotation vector.
    T_error : numpy.ndarray, shape (3,)
        Uncertainty of the translation.
    active_images : numpy.ndarray
        Boolean mask of the image pairs that were used.
    """

    om: np.ndarray
    T: np.ndarray
    R: np.ndarray
    fc_left: np.ndarray
    cc_left: np.ndarray
    kc_left: np.ndarray
    alpha_c_left: float
    fc_right: np.ndarray
    cc_right: np.ndarray
    kc_right: np.ndarray
    alpha_c_right: float
    KK_left: np.ndarray
    KK_right: np.ndarray
    baseline: float
    om_error: np.ndarray
    T_error: np.ndarray
    active_images: np.ndarray
    omc_left: list[np.ndarray]
    Tc_left: list[np.ndarray]

    def summary(self) -> str:
        """Return a human readable summary of the stereo calibration."""
        return "\n".join(
            [
                "Stereo calibration results:",
                f"  Left  camera: fc = {np.round(self.fc_left, 5)}"
                f"  cc = {np.round(self.cc_left, 5)}",
                f"  Right camera: fc = {np.round(self.fc_right, 5)}"
                f"  cc = {np.round(self.cc_right, 5)}",
                f"  Rotation vector:    om = {np.round(self.om, 5)}",
                f"  Translation vector:  T  = {np.round(self.T, 5)}",
                f"  Baseline:           {self.baseline:.5f}",
                f"  Pairs used:         {int(np.sum(self.active_images))}"
                f"/{len(self.active_images)}",
            ]
        )


def _relative_pose_from_poses(res_left, res_right, active):
    """Estimate the relative pose of the rig from the per-view poses.

    For every view, the pose of the pattern in the left camera frame
    (``R_left X + T_left``) and in the right camera frame
    (``R_right X + T_right``) are known.  The relative motion

    ``XR = R_rel XL + T_rel``

    must therefore satisfy ``R_rel = R_right R_left^T`` and
    ``T_rel = R_right T_left + T_right``; the estimates are averaged over
    the views, which is robust to the noise of the individual calibrations.

    Parameters
    ----------
    res_left, res_right : CalibrationResult
        The calibrations of the two cameras.
    active : numpy.ndarray
        Boolean mask of the image pairs to use.

    Returns
    -------
    tuple or None
        The ``(om, T)`` relative pose, or ``None`` when it cannot be
        computed.
    """
    R_acc = []
    T_acc = []

    for kk in np.flatnonzero(active):
        Rl = res_left.Rc[kk]
        Tl = res_left.Tc[kk]
        Rr = res_right.Rc[kk]
        Tr = res_right.Tc[kk]

        if not (
            np.all(np.isfinite(Rl))
            and np.all(np.isfinite(Tl))
            and np.all(np.isfinite(Rr))
            and np.all(np.isfinite(Tr))
        ):
            continue

        R_acc.append(Rr @ Rl.T)
        T_acc.append(Rr @ Tl + Tr)

    if not R_acc:
        return None

    # The rotations are averaged in their matrix form (the rotation
    # vectors cannot be averaged directly because of the angle wrapping).
    R_mean = R_acc[0].copy()
    for k in range(1, len(R_acc)):
        R_mean = R_mean @ R_acc[k].T

    # Project back onto the rotation group.
    U, _, Vt = np.linalg.svd(R_mean)
    R_mean = U @ Vt.T

    T_mean = np.mean(T_acc, axis=0)

    return rodrigues(R_mean), T_mean


def _build_param(
    fc_left, cc_left, alpha_c_left, kc_left,
    fc_right, cc_right, alpha_c_right, kc_right,
    om, T, res_left, n_ima,
):
    """Assemble the stereo parameter vector.

    Parameters
    ----------
    fc_left, cc_left, kc_left, fc_right, cc_right, kc_right : numpy.ndarray
        The intrinsic parameters of the two cameras.
    alpha_c_left, alpha_c_right : float
        The skew of the two cameras.
    om, T : numpy.ndarray
        The relative pose of the rig.
    res_left : CalibrationResult
        The calibration of the left camera, providing the per-view poses.
    n_ima : int
        The number of views.

    Returns
    -------
    numpy.ndarray
        The parameter vector.
    """
    param = np.concatenate(
        [
            fc_left,
            cc_left,
            [alpha_c_left],
            kc_left,
            fc_right,
            cc_right,
            [alpha_c_right],
            kc_right,
            om,
            T,
        ]
    )

    for kk in range(n_ima):
        om_kk = res_left.omc[kk]
        T_kk = res_left.Tc[kk]
        if not (np.all(np.isfinite(om_kk)) and np.all(np.isfinite(T_kk))):
            # The views that were rejected during the left calibration have
            # a NaN pose; they are reset to zero and their parameters are
            # not estimated (the selection mask excludes them).
            om_kk = np.zeros(3)
            T_kk = np.zeros(3)
        param = np.concatenate([param, om_kk, T_kk])

    return param


def _stereo_cost(
    data_left, data_right, param, active, options_left, options_right
) -> float:
    """Sum of squares of the stereo reprojection error.

    Parameters
    ----------
    data_left, data_right : CalibrationData
        The observations of the two cameras.
    param : numpy.ndarray
        The candidate parameter vector.
    active : numpy.ndarray
        Boolean mask of the image pairs to include.
    options_left, options_right : CalibrationOptions
        The calibration options.

    Returns
    -------
    float
        The total squared error, or ``inf`` if the projection failed.
    """
    fc_left = param[0:2]
    cc_left = param[2:4]
    alpha_c_left = param[4]
    kc_left = param[5:10]
    fc_right = param[10:12]
    cc_right = param[12:14]
    alpha_c_right = param[14]
    kc_right = param[15:20]
    om = param[20:23]
    T = param[23:26]

    total = 0.0

    for kk in np.flatnonzero(active):
        omc = param[26 + 6 * kk : 26 + 6 * kk + 3]
        Tc = param[26 + 6 * kk + 3 : 26 + 6 * kk + 6]
        if not (np.all(np.isfinite(omc)) and np.all(np.isfinite(Tc))):
            return np.inf

        try:
            xl = project_points(
                data_left.points_3d[kk], omc, Tc, fc_left, cc_left, kc_left,
                alpha_c_left,
            )
            omr, Tr, *_ = compose_motion(omc, Tc, om, T)
            xr = project_points(
                data_right.points_3d[kk], omr, Tr, fc_right, cc_right,
                kc_right, alpha_c_right,
            )
        except (ValueError, FloatingPointError):
            return np.inf

        if not (np.all(np.isfinite(xl)) and np.all(np.isfinite(xr))):
            return np.inf

        total += float(np.sum((data_left.points_2d[kk] - xl) ** 2))
        total += float(np.sum((data_right.points_2d[kk] - xr) ** 2))

    return total


def calibrate_stereo(
    data_left: CalibrationData,
    data_right: CalibrationData,
    options_left: CalibrationOptions | None = None,
    options_right: CalibrationOptions | None = None,
    initial_left: dict | None = None,
    initial_right: dict | None = None,
    max_iter: int = 100,
    threshold: float = 50.0,
    verbose: bool = True,
) -> StereoResult:
    """Calibrate a stereo rig.

    Both cameras must have been calibrated on the same views, with the
    calibration pattern at the same 3D locations, so that the 3D points of
    the two calibrations correspond.

    Parameters
    ----------
    data_left : CalibrationData
        The observations of the left camera.
    data_right : CalibrationData
        The observations of the right camera.
    options_left, options_right : CalibrationOptions, optional
        The options used for each camera.
    initial_left, initial_right : dict, optional
        Initial guesses for the intrinsic parameters.
    max_iter : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Maximum reprojection error (in pixels) for a pair of images to be
        considered consistent.  Pairs above it are dropped.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    StereoResult
        The stereo calibration.

    Notes
    -----
    Port of ``go_calib_stereo.m``.  The intrinsic parameters of the two
    cameras and the relative pose are estimated jointly.

    The two cameras are calibrated separately first: this gives the
    initial relative pose and the initial intrinsic parameters.  The joint
    optimization then refines them all together.
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    if options_left is None:
        options_left = CalibrationOptions(verbose=False)
    if options_right is None:
        options_right = CalibrationOptions(verbose=False)

    if data_left.n_ima != data_right.n_ima:
        raise ValueError(
            "the two calibrations must use the same number of views "
            f"({data_left.n_ima} != {data_right.n_ima})"
        )

    # The individual camera calibrations give the starting point.
    from .calibration import calibrate

    log("Calibrating the left camera...")
    res_left = calibrate(data_left, options_left, initial_left)
    if res_left is None:
        raise RuntimeError("the left camera calibration failed")

    log("Calibrating the right camera...")
    res_right = calibrate(data_right, options_right, initial_right)
    if res_right is None:
        raise RuntimeError("the right camera calibration failed")

    fc_left = res_left.fc.copy()
    cc_left = res_left.cc.copy()
    alpha_c_left = res_left.alpha_c
    kc_left = res_left.kc.copy()

    fc_right = res_right.fc.copy()
    cc_right = res_right.cc.copy()
    alpha_c_right = res_right.alpha_c
    kc_right = res_right.kc.copy()

    n_ima = data_left.n_ima

    active = (
        data_left.active_images
        & data_right.active_images
        & res_left.active_images
        & res_right.active_images
    )

    # The relative pose is read off the per-view poses of the two
    # individual calibrations.
    from_poses = _relative_pose_from_poses(res_left, res_right, active)
    if from_poses is not None:
        om, T = from_poses
    else:
        om = np.zeros(3)
        T = np.zeros(3)

    param = _build_param(
        fc_left, cc_left, alpha_c_left, kc_left,
        fc_right, cc_right, alpha_c_right, kc_right,
        om, T, res_left, n_ima,
    )

    # Which variables are estimated.
    est_fc_left = options_left.est_fc.copy()
    est_fc_right = options_right.est_fc.copy()
    if not options_left.est_aspect_ratio:
        est_fc_left[1] = 0.0
    if not options_right.est_aspect_ratio:
        est_fc_right[1] = 0.0

    def selection_mask(active_mask):
        return np.concatenate(
            [
                est_fc_left,
                [options_left.center_optim] * 2,
                [options_left.est_alpha],
                options_left.est_dist,
                est_fc_right,
                [options_right.center_optim] * 2,
                [options_right.est_alpha],
                options_right.est_dist,
                np.ones(6),
                np.repeat(active_mask.astype(float), 6),
            ]
        )

    sel = selection_mask(active)
    ind_Jac = np.flatnonzero(sel)

    log(
        f"\nMain stereo calibration optimization procedure - "
        f"Number of pairs of images: {int(np.sum(active))}"
    )
    log("Gradient descent iterations: ")

    change = 1.0
    iter_ = 0
    damping = 1e-4
    J2 = np.zeros((0, 0))
    e = np.zeros(0)
    errors_per_view = []
    used_views = []

    while change > 1e-11 and iter_ < max_iter:
        log(f"{iter_ + 1}...")

        J = []
        residuals = []
        errors_per_view = []
        used_views = []

        for kk in range(n_ima):
            if not active[kk]:
                continue

            X_kk = data_left.points_3d[kk]
            xl_kk = data_left.points_2d[kk]
            xr_kk = data_right.points_2d[kk]

            Nckk = X_kk.shape[1]

            # Left camera.
            omckk = param[26 + 6 * kk : 26 + 6 * kk + 3]
            Tckk = param[26 + 6 * kk + 3 : 26 + 6 * kk + 6]

            xl, dxldom, dxldT, dxldf, dxldc, dxldk, dxldalpha = (
                project_points_with_jacobians(
                    X_kk, omckk, Tckk, fc_left, cc_left, kc_left, alpha_c_left
                )
            )

            ekk = np.zeros(4 * Nckk)
            ekk[0 : 2 * Nckk] = flatten_points(xl_kk - xl)

            # The right camera sees the same points through the relative
            # pose of the rig.
            (
                omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT,
                dTrdomckk, dTrdTckk, dTrdom, dTrdT,
            ) = compose_motion(omckk, Tckk, om, T)

            xr, dxrdomr, dxrdTr, dxrdfr, dxrdcr, dxrdkr, dxrdalphar = (
                project_points_with_jacobians(
                    X_kk, omr, Tr, fc_right, cc_right, kc_right, alpha_c_right
                )
            )

            ekk[2 * Nckk :] = flatten_points(xr_kk - xr)

            dxrdom = dxrdomr @ domrdom + dxrdTr @ dTrdom
            dxrdT = dxrdomr @ domrdT + dxrdTr @ dTrdT
            dxrdomckk = dxrdomr @ domrdomckk + dxrdTr @ dTrdomckk
            dxrdTckk = dxrdomr @ domrdTckk + dxrdTr @ dTrdTckk

            Jkk = np.zeros((4 * Nckk, 26 + 6 * n_ima))

            # Left camera intrinsics.
            Jkk[0 : 2 * Nckk, 0:2] = dxldf
            Jkk[0 : 2 * Nckk, 2:4] = dxldc
            Jkk[0 : 2 * Nckk, 4] = dxldalpha[:, 0]
            Jkk[0 : 2 * Nckk, 5:10] = dxldk

            # Left camera pose.
            Jkk[0 : 2 * Nckk, 26 + 6 * kk : 26 + 6 * kk + 3] = dxldom
            Jkk[0 : 2 * Nckk, 26 + 6 * kk + 3 : 26 + 6 * kk + 6] = dxldT

            # Right camera intrinsics.
            Jkk[2 * Nckk :, 10:12] = dxrdfr
            Jkk[2 * Nckk :, 12:14] = dxrdcr
            Jkk[2 * Nckk :, 14] = dxrdalphar[:, 0]
            Jkk[2 * Nckk :, 15:20] = dxrdkr

            # Relative pose.
            Jkk[2 * Nckk :, 20:23] = dxrdom
            Jkk[2 * Nckk :, 23:26] = dxrdT

            # Right camera pose (through the relative motion).
            Jkk[2 * Nckk :, 26 + 6 * kk : 26 + 6 * kk + 3] = dxrdomckk
            Jkk[2 * Nckk :, 26 + 6 * kk + 3 : 26 + 6 * kk + 6] = dxrdTckk

            errors_per_view.append(float(np.max(np.abs(ekk))))
            J.append(Jkk)
            residuals.append(ekk)
            used_views.append(kk)

        if not J:
            break

        J = np.vstack(J)
        e = np.concatenate(residuals)

        # Recompute the selection with the updated active mask.
        ind_Jac = np.flatnonzero(selection_mask(active))

        J = J[:, ind_Jac]
        J2 = J.T @ J

        # Levenberg-Marquardt damping, scaled per parameter: the focal
        # length (~1000), the principal point (~600), the distortion
        # coefficients (~0.01) and the translations (~500) live on very
        # different scales, so a single damping constant would be
        # negligible for some of them and overwhelming for others.
        scale = np.maximum(np.abs(param[ind_Jac]), 1e-6)
        ridge = damping * np.diag(scale * scale)

        try:
            param_update = np.linalg.solve(J2 + ridge, J.T @ e)
        except np.linalg.LinAlgError:
            log("\nSingular normal equations: stopping.")
            break

        # Adaptive damping: trust the model more when the step reduces the
        # error, and damp it more when it does not.
        cost = float(e @ e)
        param_try = param.copy()
        param_try[ind_Jac] += param_update
        cost_try = _stereo_cost(
            data_left, data_right, param_try, active, options_left, options_right
        )
        if np.isfinite(cost_try) and cost_try < cost:
            damping = max(damping * 0.1, 1e-8)
        else:
            damping = min(damping * 10.0, 1e6)

        param = param_try

        # Keep the parameters in a physically meaningful range: a stereo
        # rig has a small relative rotation and a baseline comparable to
        # the distance to the scene.  Without these bounds a bad step can
        # send the parameters to absurd values.
        param[20:23] = np.clip(param[20:23], -np.pi / 2, np.pi / 2)
        param[23:26] = np.clip(param[23:26], -1e5, 1e5)
        for kk in range(n_ima):
            sl = slice(26 + 6 * kk, 26 + 6 * kk + 6)
            param[sl] = np.clip(param[sl], -1e5, 1e5)

        fc_left = param[0:2]
        cc_left = param[2:4]
        alpha_c_left = param[4]
        kc_left = param[5:10]
        fc_right = param[10:12]
        cc_right = param[12:14]
        alpha_c_right = param[14]
        kc_right = param[15:20]

        if not options_left.est_aspect_ratio:
            fc_left[1] = fc_left[0]
        if not options_right.est_aspect_ratio:
            fc_right[1] = fc_right[0]

        om_old = om.copy()
        T_old = T.copy()

        om = param[20:23]
        T = param[23:26]

        change = np.linalg.norm(
            np.concatenate([T, om]) - np.concatenate([T_old, om_old])
        ) / max(np.linalg.norm(np.concatenate([T, om])), 1e-12)

        iter_ += 1

    log("done\n")

    # --- Consistency of the image pairs ----------------------------------
    #
    # A pair of images is inconsistent when the calibration pattern was
    # moved between the left and the right shot: the two views then show
    # the pattern at two different places in space and cannot be used
    # together.  Such a pair stands out by a much larger reprojection
    # error than the other pairs.
    if errors_per_view:
        med = float(np.median(errors_per_view))
        limit = max(threshold, 3.0 * med)
        inconsistent = [
            kk for kk, err in zip(used_views, errors_per_view) if err > limit
        ]
        if inconsistent:
            log(
                "Disabling the following inconsistent views (the left and "
                "right images do not show the pattern at the same place): "
                + ", ".join(str(k) for k in inconsistent)
            )
            for kk in inconsistent:
                active[kk] = False

    # --- Uncertainties ---------------------------------------------------
    sigma_x = float(np.std(e)) if np.size(e) > 0 else float("nan")

    try:
        J2_inv = np.linalg.inv(J2)
    except np.linalg.LinAlgError:
        J2_inv = np.full_like(J2, np.nan)

    param_error = np.zeros(26 + 6 * n_ima)
    if param_error[ind_Jac].size == np.diag(J2_inv).size:
        param_error[ind_Jac] = (
            3.0 * np.sqrt(np.abs(np.diag(J2_inv))) * sigma_x
        )

    om_error = param_error[20:23]
    T_error = param_error[23:26]

    KK_left = camera_matrix(fc_left, cc_left, alpha_c_left)
    KK_right = camera_matrix(fc_right, cc_right, alpha_c_right)

    R = rodrigues(om)

    omc_left = []
    Tc_left = []
    for kk in range(n_ima):
        omc_left.append(param[26 + 6 * kk : 26 + 6 * kk + 3])
        Tc_left.append(param[26 + 6 * kk + 3 : 26 + 6 * kk + 6])

    result = StereoResult(
        om=om.copy(),
        T=T.copy(),
        R=R,
        fc_left=fc_left.copy(),
        cc_left=cc_left.copy(),
        kc_left=kc_left.copy(),
        alpha_c_left=float(alpha_c_left),
        fc_right=fc_right.copy(),
        cc_right=cc_right.copy(),
        kc_right=kc_right.copy(),
        alpha_c_right=float(alpha_c_right),
        KK_left=KK_left,
        KK_right=KK_right,
        baseline=float(np.linalg.norm(T)),
        om_error=om_error,
        T_error=T_error,
        active_images=active,
        omc_left=omc_left,
        Tc_left=Tc_left,
    )

    if verbose:
        print()
        print(result.summary())
        print()

    return result
