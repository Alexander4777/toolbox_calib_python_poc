"""Intrinsic parameters, distortion compensation and image rectification.

Ports of the MATLAB files ``normalize.m``, ``comp_distortion.m``,
``comp_distortion_oulu.m``, ``apply_distortion2.m`` and ``rect.m``.
"""

from __future__ import annotations

import numpy as np

from .geometry import apply_distortion

__all__ = [
    "camera_matrix",
    "normalize",
    "comp_distortion",
    "comp_distortion_oulu",
    "undistort_points",
    "rectify_image",
    "distort_points",
]


def _col(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1)


def camera_matrix(f, c, alpha=0.0) -> np.ndarray:
    """Build the 3x3 camera matrix ``K``.

    Parameters
    ----------
    f : array_like, shape (2,)
        Focal length in pixels ``[fx, fy]``.
    c : array_like, shape (2,)
        Principal point ``[cx, cy]`` in pixels.
    alpha : float, optional
        Skew coefficient.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        ``K = [[fx, fx*alpha, cx], [0, fy, cy], [0, 0, 1]]``.
    """
    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    return np.array(
        [
            [f[0], f[0] * alpha, c[0]],
            [0.0, f[1], c[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def normalize(x, f, c, k=None, alpha=0.0) -> np.ndarray:
    """Convert pixel coordinates to normalized (undistorted) image coordinates.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Pixel coordinates.
    f : array_like, shape (2,)
        Focal length in pixels.
    c : array_like, shape (2,)
        Principal point in pixels.
    k : array_like, optional
        Distortion coefficients.  Defaults to no distortion.
    alpha : float, optional
        Skew coefficient.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        Normalized coordinates.

    Notes
    -----
    Port of ``normalize.m``: the principal point is subtracted, the skew is
    undone and the lens distortion is compensated for.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"x must have shape (2, N), got {x.shape}")

    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)

    x_distort = np.vstack([(x[0] - c[0]) / f[0], (x[1] - c[1]) / f[1]])
    x_distort[0] = x_distort[0] - alpha * x_distort[1]

    if k is None:
        k = np.zeros(5)
    k = _col(k)

    if np.linalg.norm(k) == 0:
        return x_distort

    return comp_distortion_oulu(x_distort, k)


def comp_distortion(x_dist, k2) -> np.ndarray:
    """Compensate the (simple) radial distortion.

    Parameters
    ----------
    x_dist : array_like, shape (2, N)
        Normalized image points *without* distortion compensation.
    k2 : float or array_like
        The single radial distortion coefficient.  When several coefficients
        are given the full model is used instead.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        Compensated coordinates.

    Notes
    -----
    Port of ``comp_distortion.m``.  The simple model performs a single
    second-order correction.
    """
    x_dist = np.asarray(x_dist, dtype=float)
    k2 = _col(k2)

    if k2.size > 1:
        return comp_distortion_oulu(x_dist, k2)

    k = float(k2[0])
    radius_2 = x_dist[0] ** 2 + x_dist[1] ** 2
    radial_distortion = 1.0 + k * radius_2
    radius_2_comp = radius_2 / radial_distortion
    radial_distortion = 1.0 + k * radius_2_comp
    return x_dist / radial_distortion.reshape(1, -1)


def comp_distortion_oulu(xd, k, iterations: int = 20) -> np.ndarray:
    """Compensate radial *and* tangential distortion (Oulu model).

    Parameters
    ----------
    xd : array_like, shape (2, N)
        Distorted (normalized) point coordinates.
    k : array_like
        Distortion coefficients ``[k1, k2, p1, p2, k3]``.  Shorter vectors are
        zero padded.
    iterations : int, optional
        Number of fixed-point iterations.  The MATLAB original uses 20.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        Undistorted (normalized) point coordinates.

    Notes
    -----
    Port of ``comp_distortion_oulu.m``.  The inversion is done iteratively:
    the distorted point is assumed to be ``x = xd - delta(x)`` scaled by the
    radial factor, and the iteration refines the estimate of the
    undistorted point.
    """
    xd = np.asarray(xd, dtype=float)
    if xd.ndim != 2 or xd.shape[0] != 2:
        raise ValueError(f"xd must have shape (2, N), got {xd.shape}")

    k = _col(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    if k.size == 1:
        return comp_distortion(xd, k)

    k1, k2, p1, p2, k3 = k[:5]

    x = xd.copy()
    for _ in range(iterations):
        r2 = x[0] ** 2 + x[1] ** 2
        k_radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        delta = np.vstack(
            [
                2.0 * p1 * x[0] * x[1] + p2 * (r2 + 2.0 * x[0] ** 2),
                p1 * (r2 + 2.0 * x[1] ** 2) + 2.0 * p2 * x[0] * x[1],
            ]
        )
        x = (xd - delta) / k_radial.reshape(1, -1)

    return x


def distort_points(x, f, c, k=None, alpha=0.0) -> np.ndarray:
    """Apply the forward distortion model to normalized points.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Undistorted *normalized* coordinates.  Use
        :func:`normalize` to convert pixel coordinates to this form.
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
    numpy.ndarray, shape (2, N)
        The distorted pixel coordinates.

    See Also
    --------
    distort_pixels : The same operation, directly on pixel coordinates.
    """
    x = np.asarray(x, dtype=float)
    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    if k is None:
        k = np.zeros(5)
    k = _col(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    xd = apply_distortion(x, k)
    out = np.empty_like(xd)
    out[0] = f[0] * (xd[0] + alpha * xd[1]) + c[0]
    out[1] = f[1] * xd[1] + c[1]
    return out


def distort_pixels(x, f, c, k=None, alpha=0.0) -> np.ndarray:
    """Distort pixel coordinates.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Undistorted pixel coordinates.
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
    numpy.ndarray, shape (2, N)
        The distorted pixel coordinates.

    See Also
    --------
    distort_points : The same operation, on normalized coordinates.
    """
    x = np.asarray(x, dtype=float)
    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    if k is None:
        k = np.zeros(5)
    k = _col(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    # The ideal (undistorted) normalized coordinates.
    xn = np.vstack([(x[0] - c[0]) / f[0], (x[1] - c[1]) / f[1]])
    xn[0] = xn[0] - alpha * xn[1]

    return distort_points(xn, f, c, k, alpha)


def undistort_points(x, f, c, k=None, alpha=0.0) -> np.ndarray:
    """Map distorted pixel coordinates to undistorted pixel coordinates.

    Parameters
    ----------
    x : array_like, shape (2, N)
        Distorted pixel coordinates.
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
    numpy.ndarray, shape (2, N)
        The undistorted pixel coordinates.
    """
    xn = normalize(x, f, c, k, alpha)
    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)
    out = np.empty_like(xn)
    out[0] = f[0] * (xn[0] + alpha * xn[1]) + c[0]
    out[1] = f[1] * xn[1] + c[1]
    return out


def rectify_image(
    image: np.ndarray,
    R: np.ndarray | None = None,
    f=None,
    c=None,
    k=None,
    alpha: float = 0.0,
    K_new: np.ndarray | None = None,
) -> np.ndarray:
    """Rectify (undistort) an image.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image, shape ``(rows, cols)``.  Colour images are accepted
        and rectified channel by channel.
    R : numpy.ndarray, optional
        Rotation applied to the viewing rays, shape ``(3, 3)``.  Defaults to
        the identity.
    f : array_like, optional
        Focal length of the *source* camera in pixels.
    c : array_like, optional
        Principal point of the *source* camera in pixels.
    k : array_like, optional
        Distortion coefficients of the source camera.
    alpha : float, optional
        Skew coefficient of the source camera.
    K_new : numpy.ndarray, optional
        Intrinsic matrix of the *destination* camera.  Defaults to ``f``/``c``
        with no skew.

    Returns
    -------
    numpy.ndarray
        The rectified image, same shape as the input.

    Notes
    -----
    Port of ``rect.m``.  The destination image is sampled with bilinear
    interpolation; pixels whose source falls outside the image are set to
    255 (white), matching the MATLAB original.

    The MATLAB version is a *pull* map: for every destination pixel it
    computes the corresponding source location.  That behaviour is
    reproduced here.
    """
    image = np.asarray(image)
    squeeze_color = False
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
            squeeze_color = False
        else:
            squeeze_color = True

    if image.ndim != 2:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

    if R is None:
        R = np.eye(3)
    R = np.asarray(R, dtype=float)

    if f is None or c is None:
        raise ValueError("f and c are required to rectify an image")

    f = _col(f)
    c = _col(c)
    if f.size == 1:
        f = np.repeat(f, 2)

    if k is None:
        k = np.zeros(5)
    k = _col(k)
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    if K_new is None:
        K_new = camera_matrix(f, c, alpha)
    K_new = np.asarray(K_new, dtype=float)

    nr, nc = image.shape
    Irec = np.full((nr, nc), 255.0)

    # Destination pixel grid (1-based, like MATLAB).
    px, py = np.meshgrid(np.arange(1, nc + 1), np.arange(1, nr + 1))
    px = px.reshape(-1).astype(float)
    py = py.reshape(-1).astype(float)

    rays = np.linalg.inv(K_new) @ np.vstack([px - 1.0, py - 1.0, np.ones_like(px)])
    rays2 = R.T @ rays

    with np.errstate(divide="ignore", invalid="ignore"):
        x = rays2[0:2] / rays2[2]

    xd = apply_distortion(x, k)

    px2 = f[0] * (xd[0] + alpha * xd[1]) + c[0]
    py2 = f[1] * xd[1] + c[1]

    px_0 = np.floor(px2)
    py_0 = np.floor(py2)

    good = (
        (px_0 >= 0)
        & (px_0 <= nc - 2)
        & (py_0 >= 0)
        & (py_0 <= nr - 2)
        & np.isfinite(px_0)
        & np.isfinite(py_0)
    )

    if not np.any(good):
        return Irec.reshape(nr, nc)

    px2 = px2[good]
    py2 = py2[good]
    px_0 = px_0[good].astype(int)
    py_0 = py_0[good].astype(int)

    alpha_x = px2 - px_0
    alpha_y = py2 - py_0

    a1 = (1 - alpha_y) * (1 - alpha_x)
    a2 = (1 - alpha_y) * alpha_x
    a3 = alpha_y * (1 - alpha_x)
    a4 = alpha_y * alpha_x

    src = image.astype(float)

    def gather(rows, cols):
        if src.ndim == 2:
            return src[rows, cols]
        return src[rows, cols, :]

    lu = gather(py_0, px_0)
    ru = gather(py_0, px_0 + 1)
    ld = gather(py_0 + 1, px_0)
    rd = gather(py_0 + 1, px_0 + 1)

    # The weights are per pixel; for a colour image they must be broadcast
    # over the channel axis.
    if src.ndim == 2:
        weights = (a1, a2, a3, a4)
    else:
        shape = (-1,) + (1,) * (src.ndim - 1)
        weights = (
            a1.reshape(shape),
            a2.reshape(shape),
            a3.reshape(shape),
            a4.reshape(shape),
        )

    values = weights[0] * lu
    values = values + weights[1] * ru
    values = values + weights[2] * ld
    values = values + weights[3] * rd

    # Destination index of the good pixels.
    dest_x = px[good].astype(int) - 1
    dest_y = py[good].astype(int) - 1

    if src.ndim == 2:
        Irec[dest_y, dest_x] = values
    else:
        for ch in range(src.shape[2]):
            Irec[dest_y, dest_x, ch] = values[:, ch]

    if squeeze_color:
        return Irec
    return Irec
