"""Saving and loading calibration results.

The MATLAB toolbox stores its results in a ``Calib_Results.mat`` file.
This module provides the equivalent in Python, using NumPy's ``.npz``
format, plus a JSON export for interoperability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .calibration import CalibrationResult

__all__ = [
    "save_calibration",
    "load_calibration",
    "calibration_to_dict",
    "calibration_from_dict",
    "export_calibration_json",
]


def calibration_to_dict(result: CalibrationResult) -> dict:
    """Convert a :class:`CalibrationResult` into a plain dictionary.

    Parameters
    ----------
    result : CalibrationResult
        The calibration to export.

    Returns
    -------
    dict
        A dictionary with the intrinsic parameters, the per-view poses and
        the reprojection statistics.
    """
    return {
        "fc": np.asarray(result.fc, dtype=float),
        "cc": np.asarray(result.cc, dtype=float),
        "alpha_c": float(result.alpha_c),
        "kc": np.asarray(result.kc, dtype=float),
        "KK": np.asarray(result.KK, dtype=float),
        "fc_error": np.asarray(result.fc_error, dtype=float),
        "cc_error": np.asarray(result.cc_error, dtype=float),
        "kc_error": np.asarray(result.kc_error, dtype=float),
        "alpha_c_error": float(result.alpha_c_error),
        "err_std": np.asarray(result.err_std, dtype=float),
        "omc": [np.asarray(o, dtype=float) for o in result.omc],
        "Tc": [np.asarray(t, dtype=float) for t in result.Tc],
        "Rc": [np.asarray(r, dtype=float) for r in result.Rc],
        "H": [np.asarray(h, dtype=float) for h in result.H],
        "active_images": np.asarray(result.active_images, dtype=bool),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
    }


def calibration_from_dict(d: dict) -> CalibrationResult:
    """Rebuild a :class:`CalibrationResult` from a dictionary.

    Parameters
    ----------
    d : dict
        A dictionary produced by :func:`calibration_to_dict`.

    Returns
    -------
    CalibrationResult
        The reconstructed calibration.
    """
    omc = list(d["omc"])
    Tc = list(d["Tc"])
    n = len(omc)

    # The per-view reprojections and errors cannot be recomputed without
    # the observations, so they are left empty.
    errors = [np.full((2, 1), np.nan) for _ in range(n)]
    reprojections = [np.full((2, 1), np.nan) for _ in range(n)]

    return CalibrationResult(
        fc=np.asarray(d["fc"], dtype=float),
        cc=np.asarray(d["cc"], dtype=float),
        alpha_c=float(d["alpha_c"]),
        kc=np.asarray(d["kc"], dtype=float),
        omc=omc,
        Tc=Tc,
        Rc=list(d["Rc"]),
        H=list(d["H"]),
        err_std=np.asarray(d["err_std"], dtype=float),
        errors=errors,
        reprojections=reprojections,
        fc_error=np.asarray(d["fc_error"], dtype=float),
        cc_error=np.asarray(d["cc_error"], dtype=float),
        kc_error=np.asarray(d["kc_error"], dtype=float),
        alpha_c_error=float(d["alpha_c_error"]),
        KK=np.asarray(d["KK"], dtype=float),
        active_images=np.asarray(d["active_images"], dtype=bool),
        n_iter=int(d.get("n_iter", 0)),
        converged=bool(d.get("converged", False)),
    )


def save_calibration(
    result: CalibrationResult,
    path: str | os.PathLike = "Calib_Results.npz",
    data=None,
) -> Path:
    """Save a calibration to disk.

    Parameters
    ----------
    result : CalibrationResult
        The calibration to save.
    path : str or os.PathLike
        Destination file.  A ``.npz`` archive is written.
    data : CalibrationData, optional
        The observations, saved alongside the result so the calibration can
        be reloaded and reused.

    Returns
    -------
    pathlib.Path
        The path of the file that was written.
    """
    path = Path(path)
    d = calibration_to_dict(result)

    arrays: dict[str, np.ndarray] = {
        "fc": d["fc"],
        "cc": d["cc"],
        "alpha_c": np.array(d["alpha_c"]),
        "kc": d["kc"],
        "KK": d["KK"],
        "fc_error": d["fc_error"],
        "cc_error": d["cc_error"],
        "kc_error": d["kc_error"],
        "alpha_c_error": np.array(d["alpha_c_error"]),
        "err_std": d["err_std"],
        "active_images": d["active_images"],
        "n_iter": np.array(d["n_iter"]),
        "converged": np.array(d["converged"]),
    }

    for i, (o, t, r, h) in enumerate(zip(d["omc"], d["Tc"], d["Rc"], d["H"])):
        arrays[f"omc_{i}"] = o
        arrays[f"Tc_{i}"] = t
        arrays[f"Rc_{i}"] = r
        arrays[f"H_{i}"] = h

    if data is not None:
        arrays["image_size"] = np.array(data.image_size)
        for i, (p2, p3) in enumerate(zip(data.points_2d, data.points_3d)):
            arrays[f"x_{i}"] = p2
            arrays[f"X_{i}"] = p3
        arrays["n_ima"] = np.array(data.n_ima)

    np.savez(path, **arrays)
    return path


def load_calibration(path: str | os.PathLike) -> CalibrationResult:
    """Load a calibration saved by :func:`save_calibration`.

    Parameters
    ----------
    path : str or os.PathLike
        The file to read.

    Returns
    -------
    CalibrationResult
        The loaded calibration.
    """
    with np.load(path, allow_pickle=False) as z:
        n_iter = int(z["n_iter"]) if "n_iter" in z else 0
        converged = bool(z["converged"]) if "converged" in z else False

        omc_keys = sorted(
            (k for k in z.files if k.startswith("omc_")),
            key=lambda k: int(k.split("_")[1]),
        )
        Tc_keys = sorted(
            (k for k in z.files if k.startswith("Tc_")),
            key=lambda k: int(k.split("_")[1]),
        )
        Rc_keys = sorted(
            (k for k in z.files if k.startswith("Rc_")),
            key=lambda k: int(k.split("_")[1]),
        )
        H_keys = sorted(
            (k for k in z.files if k.startswith("H_")),
            key=lambda k: int(k.split("_")[1]),
        )

        d = {
            "fc": z["fc"],
            "cc": z["cc"],
            "alpha_c": float(z["alpha_c"]),
            "kc": z["kc"],
            "KK": z["KK"],
            "fc_error": z["fc_error"],
            "cc_error": z["cc_error"],
            "kc_error": z["kc_error"],
            "alpha_c_error": float(z["alpha_c_error"]),
            "err_std": z["err_std"],
            "omc": [z[k] for k in omc_keys],
            "Tc": [z[k] for k in Tc_keys],
            "Rc": [z[k] for k in Rc_keys],
            "H": [z[k] for k in H_keys],
            "active_images": z["active_images"],
            "n_iter": n_iter,
            "converged": converged,
        }

    return calibration_from_dict(d)


def export_calibration_json(
    result: CalibrationResult, path: str | os.PathLike, indent: int = 2
) -> Path:
    """Export a calibration as JSON.

    Parameters
    ----------
    result : CalibrationResult
        The calibration to export.
    path : str or os.PathLike
        Destination file.
    indent : int, optional
        Indentation of the JSON output.

    Returns
    -------
    pathlib.Path
        The path of the file that was written.
    """
    path = Path(path)

    payload = {
        "focal_length": result.fc.tolist(),
        "principal_point": result.cc.tolist(),
        "skew": result.alpha_c,
        "distortion": result.kc.tolist(),
        "camera_matrix": result.KK.tolist(),
        "uncertainty": {
            "focal_length": result.fc_error.tolist(),
            "principal_point": result.cc_error.tolist(),
            "distortion": result.kc_error.tolist(),
            "skew": result.alpha_c_error,
        },
        "pixel_error_std": result.err_std.tolist(),
        "rmse": result.rmse,
        "views": [
            {
                "index": i,
                "active": bool(result.active_images[i]),
                "rotation_vector": result.omc[i].tolist(),
                "rotation_matrix": result.Rc[i].tolist(),
                "translation": result.Tc[i].tolist(),
                "homography": result.H[i].tolist(),
            }
            for i in range(len(result.omc))
        ],
    }

    path.write_text(json.dumps(payload, indent=indent))
    return path
