"""Extrinsic parameter estimation for a single view.

Ports of ``compute_extrinsic_init.m``, ``compute_extrinsic_refine.m``,
``compute_extrinsic.m`` and ``compose_motion.m``.
"""

from __future__ import annotations

import numpy as np

from .geometry import (
    _rotation_matrix_to_vector,
    flatten_points,
    project_points,
    project_points_with_jacobians,
    rigid_motion,
    rodrigues,
    rodrigues_jacobian,
    safe_svd,
)
from .homography import compute_homography
from .intrinsics import camera_matrix, normalize

__all__ = [
    "planarity_ratio",
    "is_planar",
    "compute_extrinsic_init",
    "compute_extrinsic_refine",
    "compute_extrinsic",
    "compose_motion",
]


def planarity_ratio(X: np.ndarray) -> float:
    """Ratio of the smallest to the second smallest singular value of ``X``.

    A value close to zero means the points are (nearly) coplanar.

    Parameters
    ----------
    X : array_like, shape (3, N)
        3D points.

    Returns
    -------
    float
        The planarity ratio ``S[2, 2] / S[1, 1]`` in 0-based indexing, i.e.
        the third over the second singular value.

    Notes
    -----
    Port of the planarity test used by ``is3D.m`` and
    ``compute_extrinsic_init.m``.
    """
    X = np.asarray(X, dtype=float)
    X_mean = X.mean(axis=1, keepdims=True)
    Y = X - X_mean
    YY = Y @ Y.T
    S = safe_svd(YY, compute_uv=False)
    if S[1] == 0:
        return 0.0
    return float(S[2] / S[1])


def is_planar(X: np.ndarray, threshold: float = 1e-3) -> bool:
    """Return ``True`` when the 3D points are (nearly) coplanar.

    Notes
    -----
    Port of ``is3D.m``.
    """
    return planarity_ratio(X) <= threshold


def compute_extrinsic_init(x, X, f, c, k=None, alpha=0.0):
    """Estimate the pose of a structure from its image projection.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Feature locations in the image.
    X : array_like, shape (3, N)
        Corresponding 3D grid coordinates.
    f : array_like, shape (2,)
        Focal length in pixels.
    c : array_like, shape (2,)
        Principal point in pixels.
    k : array_like, optional
        Distortion coefficients.
    alpha : float, optional
        Skew coefficient.

    Returns
    -------
    om : numpy.ndarray, shape (3,)
        Rotation vector.
    T : numpy.ndarray, shape (3,)
        Translation vector.
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.

    Notes
    -----
    Port of ``compute_extrinsic_init.m``.  Planar structures are handled with
    a homography, non-planar ones with the DLT.
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.shape[0] != 3:
        raise ValueError(f"X must have shape (3, N), got {X.shape}")

    xn = normalize(x, f, c, k, alpha)
    Np = xn.shape[1]

    # --- Planarity test -------------------------------------------------
    X_mean = X.mean(axis=1, keepdims=True)
    Y = X - X_mean
    YY = Y @ Y.T
    S = safe_svd(YY, compute_uv=False)
    r = S[2] / S[1] if S[1] != 0 else 0.0

    if r < 1e-3 or Np < 5:
        # Planar structure: bring the plane to Z = 0, then use a homography.
        _, _, Vt = safe_svd(YY)
        R_transform = Vt.T

        if np.linalg.norm(R_transform[0:2, 2]) < 1e-6:
            R_transform = np.eye(3)

        if np.linalg.det(R_transform) < 0:
            R_transform = -R_transform

        T_transform = -(R_transform @ X_mean)

        X_new = R_transform @ X + T_transform

        H = compute_homography(xn, X_new[0:2, :])[0]

        sc = np.mean([np.linalg.norm(H[:, 0]), np.linalg.norm(H[:, 1])])
        if sc == 0:
            raise ValueError("degenerate homography in extrinsic initialization")
        H = H / sc

        u1 = H[:, 0]
        u1 = u1 / np.linalg.norm(u1)
        u2 = H[:, 1] - np.dot(u1, H[:, 1]) * u1
        n2 = np.linalg.norm(u2)
        if n2 == 0:
            raise ValueError("degenerate homography in extrinsic initialization")
        u2 = u2 / n2
        u3 = np.cross(u1, u2)

        RRR = np.column_stack([u1, u2, u3])
        om = rodrigues(RRR)
        R = rodrigues(om)
        T = H[:, 2]

        # Fold the plane transform back into the pose.
        T = T + R @ T_transform.reshape(3)
        R = R @ R_transform
        om = rodrigues(R)
        R = rodrigues(om)

    else:
        # Non-planar structure: DLT on the normalized coordinates.
        #
        # The 12 unknowns are the column-major flattening of the 3x4
        # camera matrix P = [R | t], i.e.
        #
        #   [R00, R10, R20, R01, R11, R21, R02, R12, R22, tx, ty, tz]
        #
        # and the constraints are
        #
        #   xn_x * (R2 . X + tz) = R0 . X + tx
        #   xn_y * (R2 . X + tz) = R1 . X + ty
        J = np.zeros((2 * Np, 12))

        xX = X * xn[0].reshape(1, -1)
        yX = X * xn[1].reshape(1, -1)

        J[0::2, [0, 3, 6]] = -X.T
        J[1::2, [1, 4, 7]] = X.T
        J[0::2, [2, 5, 8]] = xX.T
        J[1::2, [2, 5, 8]] = -yX.T
        J[0::2, 11] = xn[0]
        J[1::2, 11] = -xn[1]
        J[0::2, 9] = -1.0
        J[1::2, 10] = 1.0

        JJ = J.T @ J
        _, _, Vt = safe_svd(JJ)

        # MATLAB: RR = reshape(V(1:9,12), 3, 3)
        #
        # V(1:9,12) is a column vector and MATLAB's reshape fills
        # column-major, so the rotation matrix is built with
        # ``order='F'`` here (the transpose of numpy's default).
        RR = Vt[-1, 0:9].reshape(3, 3, order="F")

        # Extract the rotation from the DLT solution.
        #
        # The DLT returns a *scaled* rotation matrix, so the singular value
        # decomposition is degenerate (its three singular values are equal)
        # and the factors U and V are not unique: numpy (like MATLAB) then
        # returns an arbitrary pair whose product is not the rotation.
        #
        # The polar decomposition gives the right answer in every case:
        # for a matrix A with det(A) > 0, the closest rotation matrix is
        # A (A^T A)^(-1/2), and for A = s * R this is exactly R.
        nrm = np.linalg.norm(RR)
        if nrm <= 0:
            raise ValueError(
                "degenerate camera matrix in extrinsic initialization"
            )

        A = RR / nrm

        # Polar decomposition: the closest rotation matrix to A is
        # A (A^T A)^(-1/2).  For A = s * R this is exactly R, and unlike
        # the SVD it is well defined even when the singular values are
        # all equal (which is the case here).
        AtA = A.T @ A
        w, Vh = np.linalg.eigh(AtA)
        w = np.clip(w, 1e-300, None)
        inv_sqrt = Vh @ np.diag(1.0 / np.sqrt(w)) @ Vh.T
        R = A @ inv_sqrt

        if np.linalg.det(R) < 0:
            # The reflection component must be removed: flip the direction
            # associated with the smallest singular value.
            U, S, Vt2 = safe_svd(A)
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt2.T

        R = rodrigues(rodrigues(R))  # re-orthonormalize

        sc = np.linalg.norm(Vt[-1, 0:9]) / np.linalg.norm(R.reshape(-1))
        T = Vt[-1, 9:12] / sc

        om = rodrigues(R)
        R = rodrigues(om)

    # The DLT of a 3D target has a projective ambiguity: for some views
    # the recovered pose puts the structure behind the camera.  The
    # alternative solution (the pose of the same structure seen from the
    # opposite side) is tried, and the one that keeps the structure in
    # front of the camera is kept.
    if not _structure_in_front(X, om, T):
        alt_om, alt_T = _flip_pose(X, om, T)
        if alt_om is not None and _structure_in_front(X, alt_om, alt_T):
            om = alt_om
            T = alt_T
            R = rodrigues(om)

    return om, T, R


def _structure_in_front(X: np.ndarray, om, T) -> bool:
    """Return ``True`` when the structure is in front of the camera."""
    Y = rigid_motion(X, om, T)
    return bool(np.all(Y[2] > 0))


def _flip_pose(X: np.ndarray, om, T):
    """Return the alternative pose of a structure seen from behind.

    The alternative solution of the DLT is the pose obtained by rotating
    the structure by 180 degrees around an axis and translating it
    accordingly.  It is computed by reflecting the structure through the
    camera centre.
    """
    X = np.asarray(X, dtype=float)
    om = np.asarray(om, dtype=float).reshape(-1)
    T = np.asarray(T, dtype=float).reshape(-1)

    R = rodrigues(om)

    # The structure is reflected through the plane z = 0 of the camera
    # frame, which is the classic ambiguity of the DLT.
    Xc = R @ X + T.reshape(3, 1)
    Xc_flipped = Xc.copy()
    Xc_flipped[2, :] = -Xc_flipped[2, :]

    # The reflected structure is mapped back to the world frame.
    R_new = R.copy()
    R_new[2, :] = -R_new[2, :]
    T_new = T.copy()
    T_new[2] = -T_new[2]

    om_new = rodrigues(R_new)

    if not np.all(np.isfinite(om_new)):
        return None, None

    return om_new, T_new


def compute_extrinsic_refine(
    om_init,
    T_init,
    x,
    X,
    f,
    c,
    k=None,
    alpha=0.0,
    max_iter: int = 20,
    thresh_cond: float = np.inf,
):
    """Refine the pose of a structure by minimizing the reprojection error.

    Parameters
    ----------
    om_init : array_like, shape (3,)
        Initial rotation vector.
    T_init : array_like, shape (3,)
        Initial translation vector.
    x : array_like, shape (2, N)
        Observed feature locations.
    X : array_like, shape (3, N)
        Corresponding 3D points.
    f : array_like, shape (2,)
        Focal length in pixels.
    c : array_like, shape (2,)
        Principal point in pixels.
    k : array_like, optional
        Distortion coefficients.
    alpha : float, optional
        Skew coefficient.
    max_iter : int, optional
        Maximum number of iterations.
    thresh_cond : float, optional
        Conditioning threshold.  Views whose Jacobian condition number
        exceeds it are considered degenerate.

    Returns
    -------
    om : numpy.ndarray, shape (3,)
        Refined rotation vector.
    T : numpy.ndarray, shape (3,)
        Refined translation vector.
    R : numpy.ndarray, shape (3, 3)
        Refined rotation matrix.
    JJ : numpy.ndarray, shape (2*N, 6)
        The final Jacobian of the projections w.r.t. ``(om, T)``.

    Notes
    -----
    Port of ``compute_extrinsic_refine.m``.
    """
    x = np.asarray(x, dtype=float)
    X = np.asarray(X, dtype=float)

    om = np.asarray(om_init, dtype=float).reshape(-1).copy()
    T = np.asarray(T_init, dtype=float).reshape(-1).copy()

    param = np.concatenate([om, T])
    change = 1.0
    iter_ = 0
    JJ = np.zeros((2 * X.shape[1], 6))
    err_prev = np.inf

    while change > 1e-10 and iter_ < max_iter:
        _, dxdom, dxdT, _, _, _, _ = project_points_with_jacobians(
            X, om, T, f, c, k, alpha
        )
        proj = project_points(X, om, T, f, c, k, alpha)
        ex = flatten_points(x - proj)

        JJ = np.hstack([dxdom, dxdT])

        cond_JJ = np.linalg.cond(JJ)
        if not np.isfinite(cond_JJ) or cond_JJ > thresh_cond:
            change = 0.0
        else:
            JJ2 = JJ.T @ JJ
            try:
                param_innov = np.linalg.solve(JJ2, JJ.T @ ex)
            except np.linalg.LinAlgError:
                break
            param_up = param + param_innov
            norm_up = np.linalg.norm(param_up)

            # Guard against a diverging refinement: with a poor initial
            # guess (e.g. a wrong focal length) the Gauss-Newton step can
            # blow the pose up to astronomic values, which would then
            # poison the whole optimization.  Such a step is rejected and
            # the pose is left where it was.
            if not np.all(np.isfinite(param_up)) or norm_up > 1e12:
                break

            # Only accept a step that actually decreases the reprojection
            # error.  Gauss-Newton is not guaranteed to descend, and for a
            # badly initialized view an unchecked step can send the pose
            # far away from the solution.
            om_up = param_up[0:3]
            T_up = param_up[3:6]
            try:
                proj_up = project_points(X, om_up, T_up, f, c, k, alpha)
            except (ValueError, FloatingPointError):
                break
            err_up = float(np.sum((x - proj_up) ** 2))

            if not np.isfinite(err_up) or err_up > err_prev:
                break

            err_prev = err_up
            change = np.linalg.norm(param_innov) / norm_up if norm_up > 0 else 0.0
            param = param_up

            om = param[0:3]
            T = param[3:6]
            iter_ += 1

    R = rodrigues(om)
    return om, T, R, JJ


def compute_extrinsic(
    x,
    X,
    f,
    c,
    k=None,
    alpha=0.0,
    max_iter: int = 20,
    thresh_cond: float = np.inf,
):
    """Estimate the pose of a structure and its reprojection error.

    Returns
    -------
    om : numpy.ndarray, shape (3,)
        Rotation vector.
    T : numpy.ndarray, shape (3,)
        Translation vector.
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    H : numpy.ndarray, shape (3, 3)
        Homography between the grid and the image plane, in pixels.
    x : numpy.ndarray, shape (2, N)
        The reprojected points.
    ex : numpy.ndarray, shape (2, N)
        The reprojection error ``observed - reprojected``.
    JJ : numpy.ndarray, shape (2*N, 6)
        The final Jacobian.

    Notes
    -----
    Port of ``compute_extrinsic.m``.
    """
    om, T, R = compute_extrinsic_init(x, X, f, c, k, alpha)
    om, T, R, JJ = compute_extrinsic_refine(
        om, T, x, X, f, c, k, alpha, max_iter, thresh_cond
    )

    H = np.column_stack([R[:, 0:2], T])
    proj = project_points(X, om, T, f, c, k, alpha)
    ex = np.asarray(x, dtype=float) - proj

    K = camera_matrix(f, c, alpha)
    H = K @ H

    return om, T, R, H, proj, ex, JJ


def compose_motion(om1, T1, om2, T2):
    """Compose two rigid motions.

    Parameters
    ----------
    om1, T1 : array_like
        First motion (rotation vector and translation).
    om2, T2 : array_like
        Second motion.

    Returns
    -------
    om : numpy.ndarray, shape (3,)
        Rotation vector of the composed motion.
    T : numpy.ndarray, shape (3,)
        Translation of the composed motion.
    dom1dom, dom1dT, dom2dom, dom2dT : numpy.ndarray
        Jacobians of the composed rotation w.r.t. the inputs.
    dTrdom, dTrdT : numpy.ndarray
        Jacobians of the composed translation w.r.t. the inputs.

    Notes
    -----
    Port of ``compose_motion.m``.  The composed motion maps a point ``X`` to
    ``R2 * (R1 * X + T1) + T2``.

    The rotation Jacobians follow the chain rule used by the toolbox:

    ``dom3/dom1 = dom3/dR3 * dR3/dR1 * dR1/dom1``

    where ``dR3/dR1`` and ``dR3/dR2`` are the "dAB" blocks of ``dAB.m``.
    """
    om1 = np.asarray(om1, dtype=float).reshape(-1)
    T1 = np.asarray(T1, dtype=float).reshape(-1)
    om2 = np.asarray(om2, dtype=float).reshape(-1)
    T2 = np.asarray(T2, dtype=float).reshape(-1)

    R1 = rodrigues(om1)
    R2 = rodrigues(om2)

    R3 = R2 @ R1
    T = R2 @ T1 + T2

    om3, dom3dR3 = _rotation_matrix_to_vector(R3, True)

    def skew(v):
        return np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
        )

    # dR3/dR1 and dR3/dR2 (the "dAB" blocks of dAB.m).
    #
    # R3(p, q) = sum_i R2(p, i) * R1(i, q), so
    #
    #   dR3(p, q) / dR1(i, j) = R2(p, i) * delta_qj
    #   dR3(p, q) / dR2(i, j) = delta_pi * R1(j, q)
    #
    # With the column-major flattening, R(a, b) sits at index a + 3*b.
    dR3dR1 = np.zeros((9, 9))
    dR3dR2 = np.zeros((9, 9))
    for p in range(3):
        for q in range(3):
            row = p + 3 * q
            for i in range(3):
                for j in range(3):
                    dR3dR1[row, i + 3 * j] = R2[p, i] * (1.0 if q == j else 0.0)
                    dR3dR2[row, i + 3 * j] = R1[j, q] * (1.0 if p == i else 0.0)

    # dR/dom for a rotation matrix and its rotation vector, in the same
    # (verified) convention as rodrigues_jacobian: the 9x3 matrix flattens
    # the derivative column-major.
    dR1dom1_f = rodrigues_jacobian(om1)
    dR2dom2_f = rodrigues_jacobian(om2)

    dom3dom1 = dom3dR3 @ dR3dR1 @ dR1dom1_f
    dom3dom2 = dom3dR3 @ dR3dR2 @ dR2dom2_f

    dom3dT1 = np.zeros((3, 3))
    dom3dT2 = np.zeros((3, 3))

    # Translation: T3 = R2 * T1 + T2
    #   dT3/dT1 = R2, dT3/dT2 = I
    #   dT3/dom2 = dT3/dR2 * dR2/dom2, with dT3/dR2 = -[T1]_x (in the
    #   column-major sense, the translation is rotated by R2).
    dT3dT1 = R2
    dT3dT2 = np.eye(3)

    # dT3/dR2: T3 = R2 T1, so perturbing R2(i, j) changes T3(p) by
    #
    #   dT3(p) / dR2(i, j) = delta_pi * T1(j)
    #
    # With the column-major flattening, R2(i, j) sits at index i + 3*j.
    dT3dR2 = np.zeros((3, 9))
    for i in range(3):
        for j in range(3):
            dT3dR2[i, i + 3 * j] = T1[j]

    dT3dom2 = dT3dR2 @ dR2dom2_f
    dT3dom1 = np.zeros((3, 3))

    return (
        om3,
        T,
        dom3dom1,
        dom3dT1,
        dom3dom2,
        dom3dT2,
        dT3dom1,
        dT3dT1,
        dT3dom2,
        dT3dT2,
    )
