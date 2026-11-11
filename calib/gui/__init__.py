"""A PySide6 graphical interface for the camera calibration toolbox.

The interface follows the workflow of the original MATLAB toolbox:

1. Load the images of the calibration pattern.
2. Extract the corners of the pattern in each image.
3. Run the calibration.
4. Inspect the results, reprojection errors and undistorted images.

The heavy computations are run in a worker thread so that the interface
stays responsive.

Run it with::

    python -m calib.gui
"""

from __future__ import annotations

from .main_window import CalibrationWindow, run

__all__ = ["CalibrationWindow", "run"]
