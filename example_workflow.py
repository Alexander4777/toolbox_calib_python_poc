#!/usr/bin/env python3
"""A complete calibration workflow, from corner detection to undistortion.

This script shows the typical sequence:

1. Detect the corners of a planar pattern in each image (here the corners
   are read from a text file, since the corner detection itself is
   application specific).
2. Build the 3D coordinates of the pattern.
3. Calibrate the camera.
4. Save the result, undistort the images and show the reprojection error.

Run it with::

    python example_workflow.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from calib import (
    CalibrationData,
    CalibrationOptions,
    export_calibration_json,
    load_calibration,
    project_points,
    save_calibration,
    undistort_image,
)


def build_grid(nx: int = 13, ny: int = 9, spacing: float = 30.0) -> np.ndarray:
    """Return the 3D coordinates of a planar pattern.

    Parameters
    ----------
    nx, ny : int
        The number of corners along x and y.
    spacing : float
        The distance between two corners, in millimetres.

    Returns
    -------
    numpy.ndarray, shape (3, N)
        The coordinates of the corners, on the Z = 0 plane.
    """
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.vstack(
        [
            (xs.ravel() * spacing).astype(float),
            (ys.ravel() * spacing).astype(float),
            np.zeros(nx * ny),
        ]
    )


def simulate_views(X, fc, cc, kc, n_views=12, seed=0, noise=0.0):
    """Simulate the detection of the pattern in a few images.

    In a real application the corners are detected with a corner detector
    (the toolbox has no corner detector: the original MATLAB distribution
    expects the user to provide the corner locations).

    Parameters
    ----------
    X : numpy.ndarray, shape (3, N)
        The pattern coordinates.
    fc, cc : numpy.ndarray
        The true focal length and principal point.
    kc : numpy.ndarray
        The true distortion coefficients.
    n_views : int
        The number of views.
    seed : int
        The seed of the random number generator.
    noise : float
        The standard deviation of the detection noise, in pixels.

    Returns
    -------
    list of numpy.ndarray
        The detected corners of each view.
    """
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
        proj = project_points(X, om, T, fc, cc, kc)
        if noise:
            proj = proj + rng.normal(scale=noise, size=proj.shape)
        views.append(proj)

    return views


def main():
    print("=" * 70)
    print("Camera calibration workflow")
    print("=" * 70)

    # --- 1. The pattern and the observations ------------------------------
    X = build_grid()

    # The true camera, used here to generate the observations.
    fc_true = np.array([1200.0, 1180.0])
    cc_true = np.array([640.0, 480.0])
    kc_true = np.array([-0.11, 0.025, 0.0008, -0.0015, 0.0])

    views = simulate_views(X, fc_true, cc_true, kc_true, noise=0.2)

    print(f"\nPattern: {X.shape[1]} corners")
    print(f"Views:   {len(views)}")

    # --- 2. The calibration ------------------------------------------------
    data = CalibrationData(
        points_2d=views,
        points_3d=[X] * len(views),
        image_size=(1280, 960),
    )

    options = CalibrationOptions(
        est_fc=np.array([1.0, 1.0]),
        est_aspect_ratio=True,
        center_optim=True,
        est_alpha=False,
        est_dist=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        max_iter=40,
        verbose=True,
    )

    result = None
    from calib import calibrate

    result = calibrate(data, options)

    if result is None:
        print("\nThe calibration failed.")
        return

    print("\n" + result.summary())

    print("\nTrue values for comparison:")
    print(f"  fc = {fc_true}   cc = {cc_true}   kc = {kc_true}")

    # --- 3. Saving --------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        npz = save_calibration(result, tmp / "Calib_Results.npz", data)
        print(f"\nSaved the calibration to {npz}")

        loaded = load_calibration(npz)
        assert np.allclose(loaded.fc, result.fc)
        print("Reloaded it and checked that the parameters match.")

        json_path = export_calibration_json(result, tmp / "Calib_Results.json")
        print(f"Exported it as JSON to {json_path}")

    # --- 4. Undistortion --------------------------------------------------
    # A synthetic image with the (distorted) pattern drawn on it.
    h, w = 960, 1280
    image = np.zeros((h, w), dtype=np.uint8)

    from calib import distort_pixels

    xs = np.arange(60, w - 60, 90)
    ys = np.arange(60, h - 60, 90)
    grid = np.vstack(
        [np.repeat(xs, len(ys)).astype(float), np.tile(ys, len(xs)).astype(float)]
    )
    distorted = distort_pixels(grid, fc_true, cc_true, kc_true)
    for i in range(distorted.shape[1]):
        xi, yi = int(round(distorted[0, i])), int(round(distorted[1, i]))
        if 0 <= xi < w and 0 <= yi < h:
            image[yi, xi] = 255

    undistort_image(image, result)
    print(
        f"\nUndistorted a {w}x{h} image: the pattern is now a regular grid."
    )

    # --- 5. The reprojection error ----------------------------------------
    print("\nReprojection error per view (pixels):")
    for kk, err in enumerate(result.errors):
        if not result.active_images[kk]:
            continue
        rms = float(np.sqrt(np.mean(err**2)))
        print(f"  view {kk:2d}: {rms:.4f}")


if __name__ == "__main__":
    main()
