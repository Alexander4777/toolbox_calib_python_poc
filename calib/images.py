"""Image utilities: undistortion and reprojection of images.

These helpers build on the calibration results and use OpenCV for the image
resampling when it is available, falling back to the pure NumPy
implementation of :func:`calib.intrinsics.rectify_image` otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import CalibrationResult
from .intrinsics import camera_matrix, rectify_image

__all__ = [
    "undistort_image",
    "undistort_points_array",
    "save_image",
    "load_image",
    "has_opencv",
]


def has_opencv() -> bool:
    """Return ``True`` when OpenCV is available."""
    try:
        import cv2

        return cv2 is not None
    except ImportError:
        return False


def load_image(path: str | Path):
    """Load an image from disk as a NumPy array.

    Parameters
    ----------
    path : str or Path
        The image file.

    Returns
    -------
    numpy.ndarray
        The image.  Colour images are returned as ``(H, W, 3)`` in RGB
        order, grayscale images as ``(H, W)``.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    return img


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image to disk.

    Parameters
    ----------
    path : str or Path
        The destination file.  The format is taken from the extension.
    image : numpy.ndarray
        The image to save.

    Returns
    -------
    pathlib.Path
        The path of the file that was written.
    """
    import cv2

    path = Path(path)
    img = image

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), img):
        raise OSError(f"cannot write image: {path}")

    return path


def undistort_points_array(
    points: np.ndarray, result: CalibrationResult
) -> np.ndarray:
    """Undistort a set of pixel coordinates.

    Parameters
    ----------
    points : array_like, shape (2, N)
        The distorted pixel coordinates.
    result : CalibrationResult
        The calibration providing the intrinsic parameters.

    Returns
    -------
    numpy.ndarray, shape (2, N)
        The undistorted pixel coordinates.
    """
    from .intrinsics import undistort_points

    return undistort_points(
        points, result.fc, result.cc, result.kc, result.alpha_c
    )


def undistort_image(
    image: np.ndarray,
    result: CalibrationResult,
    new_focal: float | None = None,
    balance: float = 0.0,
) -> np.ndarray:
    """Undistort an image using a calibration result.

    Parameters
    ----------
    image : numpy.ndarray
        The distorted image.  Grayscale ``(H, W)`` and colour ``(H, W, 3)``
        images are supported.
    result : CalibrationResult
        The calibration providing the intrinsic parameters.
    new_focal : float, optional
        The focal length of the undistorted image.  By default the original
        focal length is kept, which crops the corners of the image.
    balance : float, optional
        When ``new_focal`` is given, the fraction of the original field of
        view that is kept.  ``0`` keeps the original field of view,
        ``1`` fills the frame completely.

    Returns
    -------
    numpy.ndarray
        The undistorted image, of the same type as the input.

    Notes
    -----
    With OpenCV available the undistortion is done with
    ``cv2.initUndistortRectifyMap``; otherwise the pure NumPy
    implementation of :func:`calib.intrinsics.rectify_image` is used.

    The two implementations differ slightly in the way the border pixels
    are filled in: OpenCV leaves them black, the NumPy version leaves them
    white (as the MATLAB toolbox does).
    """
    image = np.asarray(image)

    if not has_opencv():
        K_new = None
        if new_focal is not None:
            K_new = camera_matrix(
                [new_focal, new_focal], result.cc, result.alpha_c
            )
        return rectify_image(
            image,
            None,
            result.fc,
            result.cc,
            result.kc,
            result.alpha_c,
            K_new,
        )

    import cv2

    h, w = image.shape[:2]

    K = result.KK
    dist = np.array(
        [
            result.kc[0],
            result.kc[1],
            result.kc[2],
            result.kc[3],
            result.kc[4],
        ],
        dtype=float,
    )

    if new_focal is None:
        new_focal = float(np.mean(result.fc))

    # The new camera matrix: keep the principal point, use the requested
    # focal length and adjust it so the requested field of view is kept.
    alpha = result.alpha_c
    new_K = camera_matrix([new_focal, new_focal], result.cc, alpha)

    if balance > 0:
        # Find the scale that keeps ``balance`` of the original field of
        # view.
        corners = np.array(
            [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32
        )
        undistorted = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist)
        extent = np.max(np.abs(undistorted.reshape(-1, 2)), axis=0)
        factor = 1.0 / max(np.mean(extent) * balance, 1e-6)
        new_K[0, 0] *= factor
        new_K[1, 1] *= factor

    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, new_K, (w, h), cv2.CV_32FC1
    )

    if image.ndim == 2:
        out = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    else:
        out = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    return out
