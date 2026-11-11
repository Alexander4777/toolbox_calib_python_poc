# Camera Calibration Toolbox — Python port

A Python port of Jean-Yves Bouguet's
[Camera Calibration Toolbox for Matlab](http://www.vision.caltech.edu/bouguetj/calib_doc/)
(Caltech / Intel Corporation, December 2003).

The port covers the **core of the calibration pipeline**: the camera model,
the projection and its Jacobians, the homography, the extrinsic estimation,
the global calibration optimization, the stereo calibration, and the image
utilities. The GUI, the image formats specific to the original toolbox
(PGM/PPM/RAS), the mosaics and the scanner-specific scripts are **not**
part of this port.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The package requires Python 3.10+ and depends on `numpy`, `scipy`,
`opencv-python-headless` and `matplotlib`. The graphical interface
additionally needs `PySide6`.

## The graphical interface

```bash
python -m calib.gui          # or: python run_gui.py
```

The interface follows the workflow of the original MATLAB toolbox, one tab
per step:

| Tab | What it does |
| --- | --- |
| **Images** | Load the images of the calibration sequence. The list shows the number of corners and a checkbox to suppress a view. |
| **Corners** | Pick the corners of the pattern. Click them in the image, or use **Auto-detect corners**. Right or middle drag pans, the wheel zooms, `F` fits the image, `Backspace` removes the last corner. |
| **Model** | The geometry of the pattern (number of corners, spacing, depth of the plane) and the parameters to estimate (focal length, principal point, skew, distortion coefficients, iterations, damping). |
| **Results** | The calibrated parameters with their uncertainties, the reprojection error of every view, and the undistortion of the current image. |

The calibration runs in a worker thread, so the interface stays responsive.
The results can be saved (`.npz`), reloaded and exported (JSON) from the
**File** and **Calibration** menus.

The automatic corner detector works on the usual calibration patterns (a
grid of dark squares on a light background, or the opposite). It finds the
blobs, separates the ones that touch, orders them like the pattern and
refines them to sub-pixel accuracy. When it fails, the corners can be
picked by hand.

## The Python API

```python
import numpy as np
from calib import CalibrationData, CalibrationOptions, calibrate

# A planar grid of 13x9 points, 30 mm apart, on the Z = 0 plane.
xs, ys = np.meshgrid(np.arange(13), np.arange(9))
X = np.vstack([
    (xs.ravel() * 30.0),
    (ys.ravel() * 30.0),
    np.zeros(xs.size),
])

# The detected corners of one view: a (2, N) array of pixel coordinates.
x = np.array([[612.3, 648.9, 685.2], [401.7, 402.1, 403.4]])

data = CalibrationData(
    points_2d=[x],                 # one (2, N) array per view
    points_3d=[X],                 # one (3, N) array per view
    image_size=(1280, 960),        # (width, height) in pixels
)

result = calibrate(data, CalibrationOptions(verbose=True))
print(result.fc)    # focal length
print(result.cc)    # principal point
print(result.kc)    # distortion coefficients
print(result.rmse)  # reprojection error, in pixels
```

`CalibrationResult` also exposes the per-view poses (`omc`, `Tc`, `Rc`),
the per-view homographies (`H`), the parameter uncertainties
(`fc_error`, `cc_error`, `kc_error`, `alpha_c_error`) and a `project()`
method to reproject points with a given view.

A complete script is available in `example_workflow.py`.

## The camera model

The toolbox uses the classical Bouguet model.  The world points `X` (a
`3 x N` matrix) are mapped to the camera frame by `Y = R X + T` with
`R = rodrigues(om)`, then projected:

```
a  = x / z                      b  = y / z
r2 = a^2 + b^2

xx = a*(1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*a*b + p2*(r2 + 2*a^2)
yy = b*(1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*b^2) + 2*p2*a*b

xp = fx*(xx + alpha*yy) + cx     yp = fy*yy + cy
```

The distortion coefficients are stored in the order `[k1, k2, p1, p2, k3]`
(radial 2nd, radial 4th, tangential x, tangential y, radial 6th).

## Module map

| Python module | MATLAB files it replaces |
| --- | --- |
| `calib.geometry` | `rodrigues.m`, `rigid_motion.m`, `project_points2.m`, `apply_distortion.m` |
| `calib.intrinsics` | `normalize.m`, `comp_distortion.m`, `comp_distortion_oulu.m`, `rect.m` |
| `calib.homography` | `compute_homography.m` |
| `calib.extrinsics` | `compute_extrinsic_init.m`, `compute_extrinsic_refine.m`, `compute_extrinsic.m`, `compose_motion.m` |
| `calib.calibration` | `go_calib_optim_iter.m`, `init_intrinsic_param.m`, `comp_ext_calib.m`, `comp_error_calib.m`, `extract_parameters.m`, `check_active_images.m` |
| `calib.stereo` | `go_calib_stereo.m` |
| `calib.io` | `saving_calib.m`, `loading_calib.m` |
| `calib.images` | `undistort_image.m` |
| `calib.gui` | The graphical interface (`calib_gui.m` and the corner picking of `click_calib.m`) |

## Notable differences from the MATLAB original

The port is faithful to the original algorithms, with a few deliberate
improvements where the MATLAB code is either broken or numerically fragile.
Each of them is marked with a comment in the source.

* **Levenberg-Marquardt damping.**  The toolbox performs an undamped
  Gauss-Newton step, which overshoots badly when the normal matrix is
  close to singular — the normal situation for a 3D target — and makes
  the optimization diverge.  The port adds a per-parameter damping term
  whose value is adapted during the optimization.  Set
  `CalibrationOptions.damping = 0` to recover the original behaviour.

* **Correct Jacobians.**  The MATLAB `rodrigues.m` computes the derivative
  of the rotation matrix with respect to the rotation vector with
  `eye(3)/theta - in/theta^2`, where `in` is a column vector: MATLAB
  subtracts it from every *row* of the 3x3 matrix, so the derivative is
  wrong.  The port uses the mathematically correct expression, verified
  against finite differences in the test suite.

* **Rotation extraction from the DLT.**  For a 3D target the DLT returns
  a *scaled* rotation matrix, so its three singular values are equal and
  the SVD is degenerate: `U @ V.T` does not recover the rotation.  The
  port uses the polar decomposition instead, which is well defined in
  that case.

* **Pose disambiguation.**  The DLT of a 3D target has a projective
  ambiguity that can place the structure behind the camera.  The port
  detects this and tries the alternative solution.

* **Initialization of a 3D rig.**  The toolbox initializes a 3D rig from
  a field-of-view guess, which is usually far from the truth.  The port
  searches a range of focal lengths and keeps the best calibration.

* **View reactivation.**  A view that is rejected early because it is
  ill-conditioned is tested again once the calibration has converged, and
  re-activated when it has become usable.

* **Error handling.**  The MATLAB code uses `eval` on dynamically built
  variable names; the port passes the observations explicitly through
  `CalibrationData`.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q      # the test suite
.venv/bin/python validate.py              # a report against known answers
```

The test suite checks the analytic Jacobians against finite differences
(the same self-tests as the original MATLAB files), the homography against
a known projection, and the full calibration on synthetic data with known
intrinsic parameters, both for planar and for 3D targets, with and without
lens distortion and with noise.

The tests of the interface (`tests/test_gui.py` and
`tests/test_gui_end_to_end.py`) run with the `offscreen` Qt platform, so
they need no display. The end-to-end test renders realistic views of a
calibration pattern, detects the corners automatically, runs the
calibration through the interface and checks that the true parameters are
recovered.

## License

The original toolbox is distributed under a 3-clause BSD license, which
this port follows; see the header of `calib/__init__.py` for the full text.

```
Copyright (c) December 2003, Jean-Yves Bouguet
All rights reserved.
```

The original toolbox is available from
<http://www.vision.caltech.edu/bouguetj/calib_doc/>.
