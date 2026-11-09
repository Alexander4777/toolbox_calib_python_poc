"""Low-level geometry primitives: rotation vectors, rigid motion, projections.

This module is a faithful Python port of the corresponding MATLAB files of
Bouguet's Camera Calibration Toolbox:

===========================  =========================================
MATLAB                       Python
===========================  =========================================
``rodrigues.m``              :func:`rodrigues`
``rigid_motion.m``           :func:`rigid_motion`
``project_points2.m``        :func:`project_points`
``project_points.m``         :func:`project_points` (see notes)
``apply_distortion.m``       :func:`apply_distortion`
===========================  =========================================

Conventions
-----------
The toolbox uses *world* coordinates ``X`` of shape ``(3, N)`` and *camera*
coordinates given by ``Y = R @ X + T`` where ``R = rodrigues(om)``.  Pixel
coordinates are ``(2, N)`` arrays.  These conventions are preserved here so
that results can be cross-checked against the original MATLAB implementation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rodrigues",
    "rodrigues_jacobian",
    "rodrigues_matrix_jacobian",
    "rigid_motion",
    "apply_distortion",
    "project_points",
    "project_points_with_jacobians",
    "flatten_points",
    "safe_svd",
]


def safe_svd(a: np.ndarray, compute_uv: bool = True):
    """Compute the SVD of ``a``, tolerating non-finite entries.

    Parameters
    ----------
    a : numpy.ndarray
        The matrix to decompose.
    compute_uv : bool, optional
        Whether to compute the singular vectors.

    Returns
    -------
    numpy.ndarray or tuple
        The singular values, or the tuple ``(U, s, Vt)``.

    Notes
    -----
    A degenerate configuration can produce NaN or infinite entries (for
    instance when a view has been rejected and its parameters have been
    invalidated).  NumPy's SVD then fails with a confusing error, so the
    non-finite entries are replaced by zeros here and the caller checks
    the result.
    """
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        a = np.where(np.isfinite(a), a, 0.0)
    return np.linalg.svd(a, compute_uv=compute_uv)


def _as_column(v) -> np.ndarray:
    """Return ``v`` as a flat float array shaped like the MATLAB column vector."""
    arr = np.asarray(v, dtype=float).reshape(-1)
    return arr


def flatten_points(m: np.ndarray) -> np.ndarray:
    """Flatten a ``(2, N)`` array the way MATLAB's ``(:)`` operator does.

    MATLAB stores arrays in column-major order, so flattening a ``2 x N``
    matrix yields ``[x1, y1, x2, y2, ...]`` -- the x and y coordinates of
    each point interleaved.  This is the ordering used by every Jacobian
    returned by :func:`project_points_with_jacobians`, so residuals and
    Jacobians must be flattened the same way.

    Parameters
    ----------
    m : array_like, shape (2, N)
        The array to flatten.

    Returns
    -------
    numpy.ndarray, shape (2N,)
        The flattened array, in MATLAB column-major order.
    """
    return np.asarray(m, dtype=float).reshape(-1, order="F")


def _rotation_matrix_to_vector(R: np.ndarray, with_jacobian: bool = False):
    """Convert a 3x3 rotation matrix into a rotation vector.

    Parameters
    ----------
    R : numpy.ndarray, shape (3, 3)
        The rotation matrix.
    with_jacobian : bool, optional
        When ``True``, also return the Jacobian of the rotation vector with
        respect to the matrix entries, shape ``(3, 9)``.  The matrix is
        flattened in MATLAB (column-major) order.

    Returns
    -------
    om : numpy.ndarray, shape (3,)
        The rotation vector.
    dout : numpy.ndarray, shape (3, 9)
        Only when ``with_jacobian`` is ``True``.

    Notes
    -----
    The matrix is first projected onto the special orthogonal group with an
    SVD, then the angle is recovered from its trace.  The ``theta == pi``
    case (where the axis is ambiguous) is handled explicitly.
    """
    # Project the rotation matrix onto SO(3).
    U, _, Vt = safe_svd(R)
    R = U @ Vt

    tr = (np.trace(R) - 1.0) / 2.0
    tr = float(np.clip(tr, -1.0, 1.0))
    theta = float(np.arccos(tr))

    if np.sin(theta) >= 1e-5:
        vth = 1.0 / (2.0 * np.sin(theta))
        om1 = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
        om = vth * om1 * theta

        if not with_jacobian:
            return om

        # d(om)/dR
        #
        # om = theta/(2 sin theta) * om1, with
        # om1 = [R32-R23, R13-R31, R21-R12]
        # and theta obtained from the trace.
        #
        # d(theta)/dR = -1/sqrt(1-tr^2) * d(tr)/dR,  d(tr)/dR = [1,0,0, 0,1,0, 0,0,1]/2
        dtrdR = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]) / 2.0
        dthetadtr = -1.0 / np.sqrt(max(1.0 - tr**2, 1e-16))
        dthetadR = dthetadtr * dtrdR

        dvthdtheta = -vth * np.cos(theta) / np.sin(theta)

        # d(om1)/dR, with the matrix flattened column-major.
        dom1dR = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, -1, 0],
                [0, 0, -1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, -1, 0, 0, 0, 0, 0],
            ]
        )

        # om = vth(theta) * om1 * theta
        # d om / d R = (dvthdtheta * theta) * om1 * dthetadR
        #              + vth * (theta * dom1dR + om1 * dthetadR)
        dout = (
            (dvthdtheta * theta) * np.outer(om1, dthetadR)
            + vth * (theta * dom1dR + np.outer(om1, dthetadR))
        )

        return om, dout

    if tr > 0.0:
        # theta == 0
        if not with_jacobian:
            return np.zeros(3)
        return np.zeros(3), np.zeros((3, 9))

    # theta == pi: the rotation axis is ambiguous, recover it from the
    # symmetric part of the matrix.
    #
    # MATLAB: theta * (sqrt((diag(R)+1)/2) .* [1; 2*(R(1,2:3)>=0)'-1])
    axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
    # The first component is taken as positive; the other two follow the
    # sign of the corresponding off-diagonal entries.
    signs = np.ones(3)
    signs[1] = 1.0 if R[0, 1] >= 0 else -1.0
    signs[2] = 1.0 if R[0, 2] >= 0 else -1.0
    axis = axis * signs
    if axis[0] == 0:
        axis[0] = 1.0
    om = theta * axis / np.linalg.norm(axis)

    if not with_jacobian:
        return om
    # The Jacobian is undefined at theta == pi (the MATLAB original warns
    # about this too).
    return om, np.full((3, 9), np.nan)


def rodrigues_matrix_jacobian(R: np.ndarray) -> np.ndarray:
    """Derivative of the rotation vector with respect to the rotation matrix.

    Parameters
    ----------
    R : numpy.ndarray, shape (3, 3)
        A rotation matrix.

    Returns
    -------
    numpy.ndarray, shape (3, 9)
        The Jacobian ``d(om)/dR``, with ``R`` flattened in MATLAB
        (column-major) order.  The entries are ``NaN`` when the rotation
        angle is exactly ``pi``, where the axis is ambiguous.

    Notes
    -----
    This is the ``domdR`` branch of ``rodrigues.m``.
    """
    _, dout = _rotation_matrix_to_vector(np.asarray(R, dtype=float), True)
    return dout


def rodrigues(rot_vec) -> np.ndarray:
    """Convert between a rotation vector and a rotation matrix.

    Parameters
    ----------
    rot_vec : array_like, shape (3,) or (3, 3)
        Rotation vector (axis-angle representation).  Its norm is the
        rotation angle in radians.  A 3x3 rotation *matrix* is also
        accepted, in which case the corresponding rotation vector is
        returned.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` rotation matrix when a rotation vector is given, or a
        ``(3,)`` rotation vector when a rotation matrix is given.

    Notes
    -----
    Port of ``rodrigues.m`` (originally by Pietro Perona, Caltech).  The
    ``theta == 0`` case is handled explicitly to avoid a division by zero.
    """
    arr = np.asarray(rot_vec, dtype=float)

    if arr.ndim == 2 and arr.shape == (3, 3):
        return _rotation_matrix_to_vector(arr)

    v = _as_column(rot_vec)
    if v.size != 3:
        raise ValueError(f"rotation vector must have 3 elements, got {v.size}")

    theta = float(np.linalg.norm(v))
    if theta < np.finfo(float).eps:
        return np.eye(3)

    omega = v / theta
    wx, wy, wz = omega
    skew = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])

    alpha = np.cos(theta)
    beta = np.sin(theta)
    gamma = 1.0 - np.cos(theta)

    return alpha * np.eye(3) + beta * skew + gamma * np.outer(omega, omega)


def rodrigues_jacobian(rot_vec) -> np.ndarray:
    """Derivative of the rotation matrix with respect to the rotation vector.

    Parameters
    ----------
    rot_vec : array_like, shape (3,)
        Rotation vector.

    Returns
    -------
    numpy.ndarray, shape (9, 3)
        Jacobian ``dR/domega`` such that ``dR = J @ domega`` flattened in
        column-major (MATLAB) order.

    Notes
    -----
    Port of the Jacobian branch of ``rodrigues.m``.  The MATLAB version
    flattens with column-major ordering; :func:`rodrigues_jacobian` keeps
    that ordering so it can be fed directly into ``numpy.reshape(order='F')``.
    """
    v = _as_column(rot_vec)
    theta = float(np.linalg.norm(v))

    if theta < np.finfo(float).eps:
        # dR/domega at the identity: the derivative of the skew matrix.
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

    omega = v / theta
    w1, w2, w3 = omega
    skew = np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])
    A = np.outer(omega, omega)

    beta = np.sin(theta)
    gamma = 1.0 - np.cos(theta)

    # dR/dm1 where m1 = [alpha; beta; gamma; skew(:); A(:)] with
    # alpha = cos(theta).
    dRdm1 = np.zeros((9, 21))
    dRdm1[[0, 4, 8], 0] = 1.0
    dRdm1[:, 1] = skew.reshape(9, order="F")
    dRdm1[:, 3:12] = beta * np.eye(9)
    dRdm1[:, 2] = A.reshape(9, order="F")
    dRdm1[:, 12:21] = gamma * np.eye(9)

    # dm1/dm2 with m2 = [omega; theta]
    dm1dm2 = np.zeros((21, 4))
    dm1dm2[0, 3] = -np.sin(theta)
    dm1dm2[1, 3] = np.cos(theta)
    dm1dm2[2, 3] = np.sin(theta)
    dm1dm2[3:12, 0:3] = np.array(
        [
            [0, 0, 0, 0, 0, 1, 0, -1, 0],
            [0, 0, -1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0],
        ]
    ).T
    dm1dm2[12:21, 0] = [2 * w1, w2, w3, w2, 0, 0, w3, 0, 0]
    dm1dm2[12:21, 1] = [0, w1, 0, w1, 2 * w2, w3, 0, w3, 0]
    dm1dm2[12:21, 2] = [0, 0, w1, 0, 0, w2, w1, w2, 2 * w3]

    # dm2/dm3 with m3 = [in; theta]
    #
    # NOTE: the MATLAB original writes ``eye(3)/theta - in/theta^2``, but
    # there ``in`` is a 3x1 column, so MATLAB subtracts it from every *row*
    # of the 3x3 matrix (implicit broadcasting over the second dimension).
    # The mathematically correct derivative of ``omega = in/theta`` is
    #
    #   d(omega_i)/d(in_j) = delta_ij/theta - in_i*in_j/theta^3
    #
    # which is what is used here; it is verified against finite differences
    # in the test suite.
    dm2dm3 = np.zeros((4, 4))
    dm2dm3[0:3, 0:3] = np.eye(3) / theta - np.outer(v, v) / theta**3
    dm2dm3[3, 3] = 1.0

    # d(in)/d(in)
    dm3din = np.zeros((4, 3))
    dm3din[0:3, 0:3] = np.eye(3)
    dm3din[3, 0:3] = v / theta

    return dRdm1 @ dm1dm2 @ dm2dm3 @ dm3din


def rigid_motion(X, om, T) -> np.ndarray:
    """Apply the rigid motion ``Y = R @ X + T`` to a set of 3D points.

    Parameters
    ----------
    X : array_like, shape (3, N)
        3D points in the world reference frame.
    om : array_like, shape (3,)
        Rotation vector of the world-to-camera transform.
    T : array_like, shape (3,)
        Translation vector of the world-to-camera transform.

    Returns
    -------
    numpy.ndarray, shape (3, N)
        The points expressed in the camera reference frame.

    Notes
    -----
    Port of ``rigid_motion.m``.  The MATLAB version also returns the
    Jacobians ``dYdom`` and ``dYdT``; those are produced by
    :func:`rigid_motion_with_jacobians` instead.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != 3:
        raise ValueError(f"X must have shape (3, N), got {X.shape}")
    T = _as_column(T)
    R = rodrigues(om)
    return R @ X + T.reshape(3, 1)


def rigid_motion_with_jacobians(X, om, T):
    """Rigid motion together with its Jacobians.

    Returns
    -------
    Y : numpy.ndarray, shape (3, N)
        Points in the camera frame.
    dYdom : numpy.ndarray, shape (3*N, 3)
        Derivative of ``Y`` w.r.t. ``om`` (MATLAB row-major stacking of
        ``dY/dX(1)``, ``dY/dX(2)``, ``dY/dX(3)`` for every point).
    dYdT : numpy.ndarray, shape (3*N, 3)
        Derivative of ``Y`` w.r.t. ``T``.
    """
    X = np.asarray(X, dtype=float)
    T = _as_column(T)
    R = rodrigues(om)
    n = X.shape[1]
    Y = R @ X + T.reshape(3, 1)

    dRdom = rodrigues_jacobian(om)  # (9, 3), column-major flattening

    # dY/dR, with R flattened in MATLAB (column-major) order: the entry
    # R(i, j) sits at position i + 3*j.  Since
    #   Y(1, :) = R(1, 1)*X(1, :) + R(1, 2)*X(2, :) + R(1, 3)*X(3, :)
    # the block of R's first *row* (positions 0, 3, 6) multiplies X'.
    dYdR = np.zeros((3 * n, 9))
    for i in range(3):
        # Row i of R lives at column indices i, i+3, i+6.
        cols = np.arange(3) * 3 + i
        dYdR[i::3, cols] = X.T

    dYdom = dYdR @ dRdom

    dYdT = np.zeros((3 * n, 3))
    for i in range(3):
        dYdT[i::3, i] = 1.0

    return Y, dYdom, dYdT


def apply_distortion(x, k) -> np.ndarray:
    """Apply the radial + tangential distortion model to normalized points.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Undistorted (normalized) image plane coordinates.
    k : array_like
        Distortion coefficients.  Up to five values are used, in the order
        ``[k1, k2, p1, p2, k3]`` (radial 2nd, radial 4th, tangential x,
        tangential y, radial 6th order).  Shorter vectors are zero padded.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        The distorted (normalized) coordinates.

    Notes
    -----
    Port of ``apply_distortion.m``.  The tangential part is applied to the
    *already radially distorted* coordinates, matching the toolbox model.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"x must have shape (2, N), got {x.shape}")

    k = _as_column(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    k1, k2, p1, p2, k3 = k[:5]

    a = x[0]
    b = x[1]
    r2 = a * a + b * b
    r4 = r2 * r2
    r6 = r2 * r2 * r2

    # Radial distortion.
    cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd1 = x * cdist.reshape(1, -1)

    # Tangential distortion.
    a1 = 2.0 * a * b
    a2 = r2 + 2.0 * a * a
    a3 = r2 + 2.0 * b * b

    delta = np.vstack([p1 * a1 + p2 * a2, p1 * a3 + p2 * a1])

    return xd1 + delta


def _distortion_jacobian(x, k) -> np.ndarray:
    """Jacobian of :func:`apply_distortion` w.r.t. the distortion coefficients."""
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    k = _as_column(k)
    nk = k.size
    if nk < 5:
        k = np.concatenate([k, np.zeros(5 - nk)])

    k1, k2, p1, p2, k3 = k[:5]

    a = x[0]
    b = x[1]
    r2 = a * a + b * b
    r4 = r2 * r2
    r6 = r2 * r2 * r2

    dcdistdk = np.vstack([r2, r4, np.zeros(n), np.zeros(n), r6]).T  # (N, 5)

    dxd1dk = np.zeros((2 * n, 5))
    dxd1dk[0::2, :] = a.reshape(-1, 1) * dcdistdk
    dxd1dk[1::2, :] = b.reshape(-1, 1) * dcdistdk

    a1 = 2.0 * a * b
    a2 = r2 + 2.0 * a * a
    a3 = r2 + 2.0 * b * b

    ddelta_xdk = np.zeros((2 * n, 5))
    ddelta_xdk[0::2, 2] = a1
    ddelta_xdk[0::2, 3] = a2
    ddelta_xdk[1::2, 2] = a3
    ddelta_xdk[1::2, 3] = a1

    dxddk = dxd1dk + ddelta_xdk
    if nk < 5:
        dxddk = dxddk[:, :nk]
    return dxddk


def project_points(X, om, T, f, c, k=None, alpha=0.0) -> np.ndarray:
    """Project 3D points onto the image plane.

    Parameters
    ----------
    X : array_like, shape (3, N)
        3D points in the world reference frame.
    om : array_like, shape (3,)
        Rotation vector of the world-to-camera transform.
    T : array_like, shape (3,)
        Translation vector of the world-to-camera transform.
    f : array_like, shape (2,)
        Focal length in pixels ``[fx, fy]``.
    c : array_like, shape (2,)
        Principal point ``[cx, cy]`` in pixels.
    k : array_like, optional
        Distortion coefficients (up to 5).  Defaults to no distortion.
    alpha : float, optional
        Skew coefficient between the x and y pixel axes.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        The projected pixel coordinates.

    Notes
    -----
    Port of ``project_points2.m``.  The full model is

    ``a = x/z``, ``b = y/z``, ``r2 = a^2 + b^2``,

    ``xx = a*(1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*a*b + p2*(r2 + 2*a^2)``,
    ``yy = b*(1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*b^2) + 2*p2*a*b``,

    ``xp = fx*(xx + alpha*yy) + cx``, ``yp = fy*yy + cy``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != 3:
        raise ValueError(f"X must have shape (3, N), got {X.shape}")

    f = _as_column(f)
    c = _as_column(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    if f.size != 2:
        raise ValueError(f"f must have 1 or 2 elements, got {f.size}")
    if c.size != 2:
        raise ValueError(f"c must have 2 elements, got {c.size}")

    if k is None:
        k = np.zeros(5)
    k = _as_column(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    Y = rigid_motion(X, om, T)
    inv_z = 1.0 / Y[2]
    x = Y[0:2] * inv_z.reshape(1, -1)

    xd = apply_distortion(x, k)

    # Undo the skew, then apply the intrinsics.
    out = np.empty_like(xd)
    out[0] = f[0] * (xd[0] + alpha * xd[1]) + c[0]
    out[1] = f[1] * xd[1] + c[1]
    return out


def project_points_with_jacobians(X, om, T, f, c, k=None, alpha=0.0):
    """Project 3D points and return the analytic Jacobians.

    Parameters
    ----------
    X : array_like, shape (3, N)
        3D points in the world reference frame.
    om : array_like, shape (3,)
        Rotation vector.
    T : array_like, shape (3,)
        Translation vector.
    f : array_like, shape (2,)
        Focal length in pixels.
    c : array_like, shape (2,)
        Principal point in pixels.
    k : array_like, optional
        Distortion coefficients (up to 5).
    alpha : float, optional
        Skew coefficient.

    Returns
    -------
    xp : numpy.ndarray, shape (2, N)
        Projected pixel coordinates.
    dxdom : numpy.ndarray, shape (2*N, 3)
        Derivative of the projections w.r.t. ``om``.
    dxdT : numpy.ndarray, shape (2*N, 3)
        Derivative of the projections w.r.t. ``T``.
    dxdf : numpy.ndarray, shape (2*N, 2)
        Derivative of the projections w.r.t. ``f``.
    dxdc : numpy.ndarray, shape (2*N, 2)
        Derivative of the projections w.r.t. ``c``.
    dxdk : numpy.ndarray, shape (2*N, 5)
        Derivative of the projections w.r.t. the distortion coefficients.
    dxdalpha : numpy.ndarray, shape (2*N,)
        Derivative of the projections w.r.t. the skew coefficient.

    Notes
    -----
    Port of ``project_points2.m``.  All Jacobians are stacked in MATLAB
    row-major order: rows ``2i`` and ``2i+1`` hold the x and y derivative of
    point ``i``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != 3:
        raise ValueError(f"X must have shape (3, N), got {X.shape}")

    f = _as_column(f)
    c = _as_column(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    if f.size != 2:
        raise ValueError(f"f must have 1 or 2 elements, got {f.size}")
    if c.size != 2:
        raise ValueError(f"c must have 2 elements, got {c.size}")

    if k is None:
        k = np.zeros(5)
    k = _as_column(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    alpha = float(alpha)
    n = X.shape[1]

    Y, dYdom, dYdT = rigid_motion_with_jacobians(X, om, T)

    inv_z = 1.0 / Y[2]
    x = Y[0:2] * inv_z.reshape(1, -1)

    # d(x)/d(param) = dY/d(param) / z - x * d(z)/d(param) / z
    bb = (-x[0] * inv_z).reshape(-1, 1)
    cc = (-x[1] * inv_z).reshape(-1, 1)
    ones_inv = inv_z.reshape(-1, 1)

    dxdom = np.zeros((2 * n, 3))
    dxdom[0::2, :] = ones_inv * dYdom[0::3, :] + bb * dYdom[2::3, :]
    dxdom[1::2, :] = ones_inv * dYdom[1::3, :] + cc * dYdom[2::3, :]

    dxdT = np.zeros((2 * n, 3))
    dxdT[0::2, :] = ones_inv * dYdT[0::3, :] + bb * dYdT[2::3, :]
    dxdT[1::2, :] = ones_inv * dYdT[1::3, :] + cc * dYdT[2::3, :]

    # Radial distortion.
    a = x[0]
    b = x[1]
    r2 = a * a + b * b
    r4 = r2 * r2
    r6 = r2 * r2 * r2

    dr2dom = 2.0 * a.reshape(-1, 1) * dxdom[0::2, :] + 2.0 * b.reshape(-1, 1) * dxdom[1::2, :]
    dr2dT = 2.0 * a.reshape(-1, 1) * dxdT[0::2, :] + 2.0 * b.reshape(-1, 1) * dxdT[1::2, :]

    dr4dom = 2.0 * r2.reshape(-1, 1) * dr2dom
    dr4dT = 2.0 * r2.reshape(-1, 1) * dr2dT

    dr6dom = 3.0 * (r2 * r2).reshape(-1, 1) * dr2dom
    dr6dT = 3.0 * (r2 * r2).reshape(-1, 1) * dr2dT

    k1, k2, p1, p2, k3 = k[:5]

    cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    dcdistdom = k1 * dr2dom + k2 * dr4dom + k3 * dr6dom
    dcdistdT = k1 * dr2dT + k2 * dr4dT + k3 * dr6dT

    xd1 = x * cdist.reshape(1, -1)

    # The radial factor is scaled per coordinate row.  MATLAB's
    # reshape([cdist; cdist], 2*n, 1) flattens the 2xN matrix
    # column-major, giving [c1, c1, c2, c2, ...]: each point's factor
    # appears twice in a row.
    coeff = np.repeat(cdist, 2).reshape(-1, 1)

    dxd1dom = np.zeros((2 * n, 3))
    dxd1dom[0::2, :] = a.reshape(-1, 1) * dcdistdom
    dxd1dom[1::2, :] = b.reshape(-1, 1) * dcdistdom
    dxd1dom += coeff * dxdom

    dxd1dT = np.zeros((2 * n, 3))
    dxd1dT[0::2, :] = a.reshape(-1, 1) * dcdistdT
    dxd1dT[1::2, :] = b.reshape(-1, 1) * dcdistdT
    dxd1dT += coeff * dxdT

    dxd1dk = np.zeros((2 * n, 5))
    dcdistdk = np.vstack([r2, r4, np.zeros(n), np.zeros(n), r6]).T
    dxd1dk[0::2, :] = a.reshape(-1, 1) * dcdistdk
    dxd1dk[1::2, :] = b.reshape(-1, 1) * dcdistdk

    # Tangential distortion.
    a1 = 2.0 * a * b
    a2 = r2 + 2.0 * a * a
    a3 = r2 + 2.0 * b * b

    delta = np.vstack([p1 * a1 + p2 * a2, p1 * a3 + p2 * a1])

    aa = (2.0 * p1 * b + 6.0 * p2 * a).reshape(-1, 1)
    bb_ = (2.0 * p1 * a + 2.0 * p2 * b).reshape(-1, 1)
    cc_ = (6.0 * p1 * b + 2.0 * p2 * a).reshape(-1, 1)

    ddelta_xdom = np.zeros((2 * n, 3))
    ddelta_xdom[0::2, :] = aa * dxdom[0::2, :] + bb_ * dxdom[1::2, :]
    ddelta_xdom[1::2, :] = bb_ * dxdom[0::2, :] + cc_ * dxdom[1::2, :]

    ddelta_xdT = np.zeros((2 * n, 3))
    ddelta_xdT[0::2, :] = aa * dxdT[0::2, :] + bb_ * dxdT[1::2, :]
    ddelta_xdT[1::2, :] = bb_ * dxdT[0::2, :] + cc_ * dxdT[1::2, :]

    ddelta_xdk = np.zeros((2 * n, 5))
    ddelta_xdk[0::2, 2] = a1
    ddelta_xdk[0::2, 3] = a2
    ddelta_xdk[1::2, 2] = a3
    ddelta_xdk[1::2, 3] = a1

    xd2 = xd1 + delta
    dxd2dom = dxd1dom + ddelta_xdom
    dxd2dT = dxd1dT + ddelta_xdT
    dxd2dk = dxd1dk + ddelta_xdk

    # Skew.
    xd3 = np.vstack([xd2[0] + alpha * xd2[1], xd2[1]])

    dxd3dom = np.zeros((2 * n, 3))
    dxd3dom[0::2, :] = dxd2dom[0::2, :] + alpha * dxd2dom[1::2, :]
    dxd3dom[1::2, :] = dxd2dom[1::2, :]

    dxd3dT = np.zeros((2 * n, 3))
    dxd3dT[0::2, :] = dxd2dT[0::2, :] + alpha * dxd2dT[1::2, :]
    dxd3dT[1::2, :] = dxd2dT[1::2, :]

    dxd3dk = np.zeros((2 * n, 5))
    dxd3dk[0::2, :] = dxd2dk[0::2, :] + alpha * dxd2dk[1::2, :]
    dxd3dk[1::2, :] = dxd2dk[1::2, :]

    dxd3dalpha = np.zeros((2 * n,))
    dxd3dalpha[0::2] = xd2[1]

    # Pixels.
    fcol = f.reshape(2, 1)
    xp = fcol * xd3 + c.reshape(2, 1)

    # Scale each point's derivatives by the corresponding focal length.
    # The rows are interleaved (x, y, x, y, ...), so the focal length
    # pattern is [fx, fy, fx, fy, ...] -- this is what MATLAB's
    # reshape(f*ones(1, n), 2*n, 1) produces for a 2xN matrix.
    coeff2 = np.tile(f, n).reshape(-1, 1)

    dxpdom = coeff2 * dxd3dom
    dxpdT = coeff2 * dxd3dT
    dxpdk = coeff2 * dxd3dk
    dxpdalpha = (coeff2[:, 0] * dxd3dalpha).reshape(-1, 1)

    dxpdf = np.zeros((2 * n, 2))
    dxpdf[0::2, 0] = xd3[0]
    dxpdf[1::2, 1] = xd3[1]

    dxpdc = np.zeros((2 * n, 2))
    dxpdc[0::2, 0] = 1.0
    dxpdc[1::2, 1] = 1.0

    return xp, dxpdom, dxpdT, dxpdf, dxpdc, dxpdk, dxpdalpha
