"""Planar homography estimation.

Port of ``compute_homography.m``.

Conventions
-----------
The toolbox maps *plane* coordinates onto *image* coordinates: ``m ~ H @ M``.
The homography is stored row-major (i.e. ``H[i, j] == h[3 * i + j]``), which
matches MATLAB's ``reshape(hh, 3, 3)'`` on a column vector.
"""

from __future__ import annotations

import numpy as np

from .geometry import safe_svd

__all__ = ["compute_homography"]


def _homogenize(p: np.ndarray) -> np.ndarray:
    """Return ``p`` as homogeneous coordinates, shape ``(3, N)``."""
    p = np.asarray(p, dtype=float)
    if p.shape[0] == 3:
        return p.copy()
    if p.shape[0] == 2:
        return np.vstack([p, np.ones((1, p.shape[1]))])
    raise ValueError(f"expected a (2, N) or (3, N) array, got {p.shape}")


def compute_homography(m, M, refine: bool = True, iterations: int = 10):
    """Compute the planar homography mapping ``M`` onto ``m``.

    Parameters
    ----------
    m : array_like, shape (2, N) or (3, N)
        Image plane coordinates.
    M : array_like, shape (2, N) or (3, N)
        Coordinates on the plane.
    refine : bool, optional
        When more than four points are available, refine the solution by
        minimizing the reprojection error.  Defaults to ``True``.
    iterations : int, optional
        Number of refinement iterations.

    Returns
    -------
    H : numpy.ndarray, shape (3, 3)
        The homography, normalized so that ``H[2, 2] == 1``.
    Hnorm : numpy.ndarray, shape (3, 3)
        The normalization matrix applied to the image points.
    inv_Hnorm : numpy.ndarray, shape (3, 3)
        Its inverse.

    Notes
    -----
    Port of ``compute_homography.m``.  The initial estimate comes from the
    normalized DLT and is then polished with a Gauss-Newton refinement on the
    reprojection error.
    """
    m = _homogenize(m)
    M = _homogenize(M)

    Np = m.shape[1]

    # Work with the affine (2 x N) parts; the homogeneous scale of the
    # inputs is irrelevant.
    m = m / m[2]
    M = M / M[2]

    # Affine normalization of the image points (centroid + mean absolute
    # deviation).
    ax = m[0]
    ay = m[1]
    mxx = ax.mean()
    myy = ay.mean()
    ax = ax - mxx
    ay = ay - myy

    scxx = np.mean(np.abs(ax))
    scyy = np.mean(np.abs(ay))

    if scxx == 0 or scyy == 0:
        raise ValueError("degenerate point distribution: cannot normalize")

    Hnorm = np.array(
        [
            [1.0 / scxx, 0.0, -mxx / scxx],
            [0.0, 1.0 / scyy, -myy / scyy],
            [0.0, 0.0, 1.0],
        ]
    )
    inv_Hnorm = np.array([[scxx, 0.0, mxx], [0.0, scyy, myy], [0.0, 0.0, 1.0]])

    mn = Hnorm @ m

    # Build the DLT system.  The rows encode, for each point,
    #
    #   [X, Y, 1, 0, 0, 0, -m_x*X, -m_x*Y, -m_x]
    #   [0, 0, 0, X, Y, 1, -m_y*X, -m_y*Y, -m_y]
    #
    # whose null vector holds the row-major flattening of the homography.
    L = np.zeros((2 * Np, 9))
    L[0::2, 0:3] = M.T
    L[1::2, 3:6] = M.T
    L[0::2, 6:9] = -(mn[0].reshape(1, -1) * M).T
    L[1::2, 6:9] = -(mn[1].reshape(1, -1) * M).T

    if Np > 4:
        L = L.T @ L

    _, _, Vt = safe_svd(L)
    hh = Vt[-1, :]
    hh = hh / hh[8]

    # MATLAB: Hrem = reshape(hh, 3, 3)'  ->  h is the row-major flattening.
    H = inv_Hnorm @ hh.reshape(3, 3)

    if refine and Np > 4:
        # Refine the eight free parameters (h33 is fixed to 1) by
        # Gauss-Newton on the reprojection error.
        hhv = H.reshape(9)[:8].copy()

        for _ in range(iterations):
            mrep = H @ M

            J = np.zeros((2 * Np, 8))
            MMM = M / mrep[2].reshape(1, -1)

            J[0::2, 0:3] = -MMM.T
            J[1::2, 3:6] = -MMM.T

            mrep = mrep / mrep[2].reshape(1, -1)

            m_err = (m[0:2] - mrep[0:2]).reshape(-1, order="F")

            MMM2 = mrep[0].reshape(1, -1) * MMM
            MMM3 = mrep[1].reshape(1, -1) * MMM

            J[0::2, 6:8] = MMM2[0:2].T
            J[1::2, 6:8] = MMM3[0:2].T

            hh_innov = np.linalg.solve(J.T @ J, J.T @ m_err)

            hhv = hhv - hh_innov
            H = np.concatenate([hhv, [1.0]]).reshape(3, 3)

    return H, Hnorm, inv_Hnorm
