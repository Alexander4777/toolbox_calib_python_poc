"""A Python port of Bouguet's Camera Calibration Toolbox.

This package provides the core of Jean-Yves Bouguet's Camera Calibration
Toolbox (Caltech, December 2003) rewritten in Python with NumPy/SciPy.

The original toolbox is distributed under the following terms; this port
follows the same model.

    Copyright (c) December 2003, Jean-Yves Bouguet
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Quick start
-----------
>>> import numpy as np
>>> from calib import CalibrationData, calibrate
>>> # 2D grid of 13x9 points spaced 30 mm apart
>>> grid = np.array([[i * 30.0 for j in range(9)] for i in range(13)])
>>> X = np.vstack([grid[0].reshape(-1), grid[1].reshape(-1), np.zeros(grid.shape[1])])
>>> # ... add noisy 2D observations and run `calibrate`.
"""

from __future__ import annotations

from .calibration import (
    CalibrationData,
    CalibrationOptions,
    CalibrationResult,
    calibrate,
    compute_reprojection_errors,
    init_intrinsic_param,
)
from .extrinsics import (
    compose_motion,
    compute_extrinsic,
    compute_extrinsic_init,
    compute_extrinsic_refine,
    is_planar,
    planarity_ratio,
)
from .geometry import (
    apply_distortion,
    flatten_points,
    project_points,
    project_points_with_jacobians,
    rigid_motion,
    rodrigues,
    rodrigues_jacobian,
)
from .homography import compute_homography
from .images import (
    has_opencv,
    load_image,
    save_image,
    undistort_image,
    undistort_points_array,
)
from .intrinsics import (
    camera_matrix,
    comp_distortion,
    comp_distortion_oulu,
    distort_pixels,
    distort_points,
    normalize,
    rectify_image,
    undistort_points,
)
from .io import (
    calibration_from_dict,
    calibration_to_dict,
    export_calibration_json,
    load_calibration,
    save_calibration,
)
from .stereo import StereoResult, calibrate_stereo

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # Geometry
    "rodrigues",
    "rodrigues_jacobian",
    "rigid_motion",
    "apply_distortion",
    "project_points",
    "project_points_with_jacobians",
    "flatten_points",
    # Intrinsics
    "camera_matrix",
    "normalize",
    "comp_distortion",
    "comp_distortion_oulu",
    "distort_points",
    "distort_pixels",
    "undistort_points",
    "rectify_image",
    # Homography
    "compute_homography",
    # Extrinsics
    "planarity_ratio",
    "is_planar",
    "compute_extrinsic_init",
    "compute_extrinsic_refine",
    "compute_extrinsic",
    "compose_motion",
    # Calibration
    "CalibrationData",
    "CalibrationOptions",
    "CalibrationResult",
    "calibrate",
    "compute_reprojection_errors",
    "init_intrinsic_param",
    # Stereo
    "StereoResult",
    "calibrate_stereo",
    # I/O
    "save_calibration",
    "load_calibration",
    "calibration_to_dict",
    "calibration_from_dict",
    "export_calibration_json",
    # Images
    "undistort_image",
    "undistort_points_array",
    "load_image",
    "save_image",
    "has_opencv",
]
