"""The calibration engine: intrinsic initialization and global optimization.

Ports of ``init_intrinsic_param.m``, ``comp_ext_calib.m``,
``go_calib_optim_iter.m`` and ``extract_parameters.m``.

The MATLAB original stores everything in base-workspace variables
(``fc``, ``cc``, ``kc``, ``x_1``, ``X_1`` ...).  Here the same data is held
in explicit objects: :class:`CalibrationData` for the observations and
:class:`CalibrationResult` for the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extrinsics import (
    compute_extrinsic_init,
    compute_extrinsic_refine,
    is_planar,
)
from .geometry import (
    flatten_points,
    project_points,
    project_points_with_jacobians,
    rigid_motion,
    rodrigues,
)
from .homography import compute_homography
from .intrinsics import camera_matrix

__all__ = [
    "CalibrationData",
    "CalibrationResult",
    "CalibrationOptions",
    "init_intrinsic_param",
    "calibrate",
    "compute_reprojection_errors",
]


def _col(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1)


@dataclass
class CalibrationOptions:
    """Options controlling the calibration.

    Attributes
    ----------
    est_fc : numpy.ndarray, shape (2,)
        Whether each focal length component is estimated.  ``[1, 1]`` by
        default.  Use ``[0, 0]`` to keep the focal length fixed.
    est_aspect_ratio : bool
        When ``False``, both focal lengths are forced equal.
    center_optim : bool
        Whether the principal point is estimated.
    est_alpha : bool
        Whether the pixel skew is estimated.
    est_dist : numpy.ndarray, shape (5,)
        Whether each distortion coefficient is estimated, in the order
        ``[k1, k2, p1, p2, k3]``.
    recompute_extrinsic : bool
        Whether the extrinsic parameters are re-estimated at every iteration.
    max_iter : int
        Maximum number of gradient descent iterations.
    check_cond : bool
        Whether ill-conditioned views are detected and deactivated.
    thresh_cond : float
        Conditioning threshold used for view rejection.
    two_focals_init : bool
        Whether to use a two-focal-length initial guess.
    quick_init : bool
        Whether to skip the vanishing-point initialization and use a field of
        view guess instead.
    fov_degrees : float
        Field of view used by the quick initialization.
    alpha_smooth : float
        Smoothing factor of the gradient descent (``1`` disables smoothing).
    verbose : bool
        Whether to print progress information.
    """

    est_fc: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    est_aspect_ratio: bool = True
    center_optim: bool = True
    est_alpha: bool = False
    est_dist: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    )
    recompute_extrinsic: bool = True
    max_iter: int = 30
    check_cond: bool = True
    thresh_cond: float = 1e6
    two_focals_init: bool = False
    quick_init: bool = False
    fov_degrees: float = 35.0
    alpha_smooth: float = 0.4
    damping: float = 1e-4
    verbose: bool = True

    def __post_init__(self):
        self.est_fc = _col(self.est_fc).astype(float)
        if self.est_fc.size != 2:
            raise ValueError("est_fc must have 2 elements")
        self.est_dist = _col(self.est_dist).astype(float)
        if self.est_dist.size == 4:
            self.est_dist = np.concatenate([self.est_dist, [0.0]])
        if self.est_dist.size != 5:
            raise ValueError("est_dist must have 4 or 5 elements")
        self.est_alpha = float(bool(self.est_alpha))
        self.center_optim = float(bool(self.center_optim))
        self.est_aspect_ratio = float(bool(self.est_aspect_ratio))
        self.check_cond = bool(self.check_cond)


@dataclass
class CalibrationData:
    """Observations fed to the calibrator.

    Attributes
    ----------
    points_2d : list of numpy.ndarray
        Detected image points, one ``(2, N_i)`` array per view.
    points_3d : list of numpy.ndarray
        Corresponding 3D grid coordinates, one ``(3, N_i)`` array per view.
    image_size : tuple of int
        ``(width, height)`` in pixels.
    active_images : numpy.ndarray, optional
        Boolean mask of the views to use.  Defaults to all views active.
    """

    points_2d: list[np.ndarray]
    points_3d: list[np.ndarray]
    image_size: tuple[int, int]
    active_images: np.ndarray | None = None

    def __post_init__(self):
        if len(self.points_2d) != len(self.points_3d):
            raise ValueError("points_2d and points_3d must have the same length")
        if not self.points_2d:
            raise ValueError("at least one view is required")

        self.points_2d = [np.asarray(p, dtype=float) for p in self.points_2d]
        self.points_3d = [np.asarray(p, dtype=float) for p in self.points_3d]

        for i, (p2, p3) in enumerate(zip(self.points_2d, self.points_3d)):
            if p2.ndim != 2 or p2.shape[0] != 2:
                raise ValueError(f"view {i}: points_2d must have shape (2, N)")
            if p3.ndim != 2 or p3.shape[0] != 3:
                raise ValueError(f"view {i}: points_3d must have shape (3, N)")
            if p2.shape[1] != p3.shape[1]:
                raise ValueError(f"view {i}: 2D and 3D point counts differ")

        self.n_ima = len(self.points_2d)

        if self.active_images is None:
            self.active_images = np.ones(self.n_ima, dtype=bool)
        else:
            self.active_images = np.asarray(self.active_images, dtype=bool).reshape(-1)
            if self.active_images.size != self.n_ima:
                raise ValueError("active_images length must match the number of views")

        if not np.any(self.active_images):
            raise ValueError("no active image: nothing to calibrate")

    @property
    def ind_active(self) -> np.ndarray:
        """Indices of the active views."""
        return np.flatnonzero(self.active_images)

    @property
    def n_points_views(self) -> np.ndarray:
        """Number of points in each view."""
        return np.array([p.shape[1] for p in self.points_2d], dtype=int)

    @property
    def is_3d_rig(self) -> bool:
        """``True`` when at least one active view shows a non-planar rig."""
        return any(
            not is_planar(self.points_3d[k]) for k in self.ind_active
        )


@dataclass
class CalibrationResult:
    """Outcome of a calibration.

    Attributes
    ----------
    fc : numpy.ndarray, shape (2,)
        Focal length in pixels.
    cc : numpy.ndarray, shape (2,)
        Principal point in pixels.
    alpha_c : float
        Pixel skew coefficient.
    kc : numpy.ndarray, shape (5,)
        Distortion coefficients ``[k1, k2, p1, p2, k3]``.
    omc : list of numpy.ndarray
        Per-view rotation vectors.
    Tc : list of numpy.ndarray
        Per-view translation vectors.
    Rc : list of numpy.ndarray
        Per-view rotation matrices.
    H : list of numpy.ndarray
        Per-view homographies (grid to image, in pixels).
    err_std : numpy.ndarray, shape (2,)
        Standard deviation of the reprojection error along x and y.
    errors : list of numpy.ndarray
        Per-view reprojection errors, shape ``(2, N_i)``.
    reprojections : list of numpy.ndarray
        Per-view reprojected points, shape ``(2, N_i)``.
    fc_error, cc_error, kc_error : numpy.ndarray
        Parameter uncertainties (three times the standard deviation).
    alpha_c_error : float
        Uncertainty of the skew coefficient.
    KK : numpy.ndarray, shape (3, 3)
        The camera matrix.
    active_images : numpy.ndarray
        Boolean mask of the views that were actually used.
    points_3d : list of numpy.ndarray, optional
        The 3D points of every view.  They are kept so that the physical
        validity of the solution can be checked afterwards.
    """

    fc: np.ndarray
    cc: np.ndarray
    alpha_c: float
    kc: np.ndarray
    omc: list[np.ndarray]
    Tc: list[np.ndarray]
    Rc: list[np.ndarray]
    H: list[np.ndarray]
    err_std: np.ndarray
    errors: list[np.ndarray]
    reprojections: list[np.ndarray]
    fc_error: np.ndarray
    cc_error: np.ndarray
    kc_error: np.ndarray
    alpha_c_error: float
    KK: np.ndarray
    active_images: np.ndarray
    n_iter: int = 0
    converged: bool = False
    points_3d: list[np.ndarray] | None = None

    @property
    def _points_3d(self) -> list[np.ndarray]:
        """The 3D points of every view (possibly empty)."""
        if self.points_3d is None:
            return []
        return self.points_3d

    @property
    def camera_matrix(self) -> np.ndarray:
        """The 3x3 camera matrix ``K``."""
        return self.KK

    @property
    def rmse(self) -> float:
        """Root mean square reprojection error, in pixels.

        Only the views that were actually used contribute; the errors of
        the deactivated views are ``NaN`` and are ignored.
        """
        chunks = [
            e.reshape(-1) for e in self.errors if e.size and np.any(np.isfinite(e))
        ]
        if not chunks:
            return float("nan")

        all_err = np.concatenate(chunks)
        all_err = all_err[np.isfinite(all_err)]
        if all_err.size == 0:
            return float("nan")

        return float(np.sqrt(np.mean(all_err**2)))

    def project(self, X, view: int) -> np.ndarray:
        """Project 3D points using the pose of a given view.

        Parameters
        ----------
        X : array_like, shape (3, N)
            3D points in the world frame.
        view : int
            Index of the view.

        Returns
        -------
        numpy.ndarray, shape (2, N)
            The projected pixel coordinates.
        """
        return project_points(
            X, self.omc[view], self.Tc[view], self.fc, self.cc, self.kc, self.alpha_c
        )

    def summary(self) -> str:
        """Return a human readable summary of the calibration."""
        lines = [
            "Calibration results:",
            f"  Focal length:   fc = [{self.fc[0]:.5f}  {self.fc[1]:.5f}]"
            f" +/- [{self.fc_error[0]:.5f}  {self.fc_error[1]:.5f}]",
            f"  Principal pt:   cc = [{self.cc[0]:.5f}  {self.cc[1]:.5f}]"
            f" +/- [{self.cc_error[0]:.5f}  {self.cc_error[1]:.5f}]",
            f"  Skew:           alpha_c = {self.alpha_c:.5f} +/- {self.alpha_c_error:.5f}",
            "  Distortion:     kc = ["
            + "  ".join(f"{v:.5f}" for v in self.kc)
            + "] +/- ["
            + "  ".join(f"{v:.5f}" for v in self.kc_error)
            + "]",
            f"  Pixel error:    err = [{self.err_std[0]:.5f}  {self.err_std[1]:.5f}]",
            f"  RMSE:           {self.rmse:.5f} px",
            f"  Views used:     {int(np.sum(self.active_images))}/{len(self.active_images)}",
        ]
        return "\n".join(lines)


def init_intrinsic_param(data: CalibrationData, options: CalibrationOptions | None = None):
    """Initialize the intrinsic parameters from vanishing points.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    options : CalibrationOptions, optional
        Options; only ``est_aspect_ratio`` and ``two_focals_init`` are used.

    Returns
    -------
    fc : numpy.ndarray, shape (2,)
        Initial focal length.
    cc : numpy.ndarray, shape (2,)
        Initial principal point (the image centre).
    kc : numpy.ndarray, shape (5,)
        Initial distortion coefficients (all zero).
    alpha_c : float
        Initial skew (zero).

    Notes
    -----
    Port of ``init_intrinsic_param.m``.  The focal length is estimated from
    the orthogonality constraint between the vanishing points of the grid
    directions and its diagonals.  The principal point is assumed to sit at
    the centre of the image and the distortion is assumed to be zero.
    """
    if options is None:
        options = CalibrationOptions()

    nx, ny = data.image_size

    c_init = np.array([nx / 2.0 - 0.5, ny / 2.0 - 0.5])
    k_init = np.zeros(5)

    Sub_cc = np.array(
        [
            [1.0, 0.0, -c_init[0]],
            [0.0, 1.0, -c_init[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    A = []
    b = []

    for kk in data.ind_active:
        Hkk = compute_homography(data.points_2d[kk], data.points_3d[kk][0:2, :])[0]
        Hkk = Sub_cc @ Hkk

        V_hori = Hkk[:, 0]
        V_vert = Hkk[:, 1]
        V_diag1 = (Hkk[:, 0] + Hkk[:, 1]) / 2.0
        V_diag2 = (Hkk[:, 0] - Hkk[:, 1]) / 2.0

        V_hori = V_hori / np.linalg.norm(V_hori)
        V_vert = V_vert / np.linalg.norm(V_vert)
        V_diag1 = V_diag1 / np.linalg.norm(V_diag1)
        V_diag2 = V_diag2 / np.linalg.norm(V_diag2)

        a1, b1, c1 = V_hori
        a2, b2, c2 = V_vert
        a3, b3, c3 = V_diag1
        a4, b4, c4 = V_diag2

        A.append([a1 * a2, b1 * b2])
        b.append(-c1 * c2)
        A.append([a3 * a4, b3 * b4])
        b.append(-c3 * c4)

    A = np.array(A)
    b = np.array(b)

    if options.two_focals_init:
        f_init = np.sqrt(np.abs(1.0 / (np.linalg.inv(A.T @ A) @ A.T @ b)))
    else:
        # MATLAB: sqrt(b'*(sum(A')') / (b'*b)) * ones(2,1)
        #
        # ``sum(A')`` sums the 2xM matrix A' along its first dimension, so
        # in numpy this is A.T.sum(axis=0), an M-vector.  The product
        # b'*(sum(A')') is then the dot product of the two M-vectors.
        f_init = np.sqrt((b @ A.T.sum(axis=0)) / (b @ b)) * np.ones(2)

    if not options.est_aspect_ratio:
        f_init[1] = f_init[0]

    return f_init, c_init, k_init, 0.0


def compute_reprojection_errors(
    data: CalibrationData,
    fc,
    cc,
    kc,
    alpha_c,
    omc,
    Tc,
    active=None,
):
    """Recompute the per-view reprojection errors.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    fc, cc, kc, alpha_c : array_like
        The intrinsic parameters.
    omc, Tc : list of numpy.ndarray
        The per-view poses.
    active : numpy.ndarray, optional
        Boolean mask of the views to include.  Defaults to
        ``data.active_images``.

    Returns
    -------
    errors : list of numpy.ndarray
        Per-view error ``observed - reprojected``, shape ``(2, N_i)``.
    reprojections : list of numpy.ndarray
        Per-view reprojected points.
    err_std : numpy.ndarray, shape (2,)
        Standard deviation of the error along x and y.
    """
    errors = []
    reprojections = []

    active = data.active_images
    if active is not None:
        active = np.asarray(active, dtype=bool).reshape(-1)
    for kk in range(data.n_ima):
        if active[kk]:
            proj = project_points(
                data.points_3d[kk], omc[kk], Tc[kk], fc, cc, kc, alpha_c
            )
            reprojections.append(proj)
            errors.append(data.points_2d[kk] - proj)
        else:
            reprojections.append(np.full((2, 1), np.nan))
            errors.append(np.full((2, 1), np.nan))

    stacked = np.hstack(errors)
    err_std = np.array(
        [np.nanstd(stacked[0, :]), np.nanstd(stacked[1, :])]
    )
    return errors, reprojections, err_std

def _score_focal_3d(data, fc, cc, options):
    """Score a focal length for a 3D rig by running a short calibration.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    fc : numpy.ndarray, shape (2,)
        The candidate focal length.
    cc : numpy.ndarray, shape (2,)
        The principal point.
    options : CalibrationOptions
        The calibration options.

    Returns
    -------
    float or None
        The root mean square reprojection error, or ``None`` when the
        calibration failed.
    """
    sub_options = CalibrationOptions(
        est_fc=np.array([0.0, 0.0]),
        est_aspect_ratio=False,
        center_optim=True,
        est_alpha=False,
        est_dist=np.zeros(5),
        recompute_extrinsic=True,
        max_iter=25,
        check_cond=False,
        damping=options.damping,
        verbose=False,
    )

    try:
        result = calibrate(
            data,
            sub_options,
            initial={"fc": fc, "cc": cc, "kc": np.zeros(5), "alpha_c": 0.0},
        )
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    if result is None or not np.isfinite(result.rmse):
        return None

    return float(result.rmse)


def _score_focal_3d_consistency(data, fc, cc, options):
    """Score a focal length by the consistency of the per-view poses.

    For each view, the pose of the calibration pattern is recovered with
    the DLT.  The DLT of a 3D target has a projective ambiguity: for a
    wrong focal length, every view is still explained individually, but
    the recovered poses do not agree on a single rigid 3D structure.

    The score is therefore the residual of a rigid fit of the recovered
    poses: the pattern's pose in every view must be related by a rigid
    motion of the pattern.  This removes the projective ambiguity.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    fc : numpy.ndarray, shape (2,)
        The candidate focal length.
    cc : numpy.ndarray, shape (2,)
        The principal point.
    options : CalibrationOptions
        The calibration options.

    Returns
    -------
    float or None
        The residual of the rigid fit, in pixels, or ``None`` when the
        poses could not be recovered.
    """
    poses = []

    for kk in data.ind_active:
        try:
            om, T, _ = compute_extrinsic_init(
                data.points_2d[kk], data.points_3d[kk], fc, cc, np.zeros(5), 0.0
            )
        except (ValueError, np.linalg.LinAlgError):
            return None

        if not (np.all(np.isfinite(om)) and np.all(np.isfinite(T))):
            return None

        poses.append((om, T))

    if len(poses) < 3:
        return None

    # The first view defines the reference: the pattern's pose in the
    # world frame is the inverse of the first view's pose.
    om0, T0 = poses[0]
    R0 = rodrigues(om0)

    residuals = []

    for om, T in poses[1:]:
        R = rodrigues(om)

        # The pattern's pose in the world frame, seen through this view.
        R_world = R0.T @ R
        T_world = R0.T @ (T - T0)

        # The pattern is a rigid body: its orientation must be the same in
        # every view.  The residual of the rotation is the angle between
        # the two orientations.
        cos_angle = (np.trace(R_world) - 1.0) / 2.0
        angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        # The same for the translation: the origin of the pattern's frame
        # must be the same point in every view.
        trans_res = float(np.linalg.norm(T_world))

        residuals.append(angle * 1000.0 + trans_res)

    return float(np.median(residuals))
def _reprojection_cost(data, param, active, options) -> float:
    """Sum of squares of the reprojection error for a candidate parameter set.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    param : numpy.ndarray
        The candidate parameter vector.
    active : numpy.ndarray
        Boolean mask of the views to include.
    options : CalibrationOptions
        The calibration options.

    Returns
    -------
    float
        The total squared error, or ``inf`` if the projection failed.
    """
    f = param[0:2]
    c = param[2:4]
    alpha = param[4]
    k = param[5:10]

    total = 0.0
    for kk in np.flatnonzero(active):
        om = param[15 + 6 * kk : 15 + 6 * kk + 3]
        T = param[15 + 6 * kk + 3 : 15 + 6 * kk + 6]
        if not np.all(np.isfinite(om)) or not np.all(np.isfinite(T)):
            return np.inf
        try:
            proj = project_points(data.points_3d[kk], om, T, f, c, k, alpha)
        except (ValueError, FloatingPointError):
            return np.inf
        if not np.all(np.isfinite(proj)):
            return np.inf
        total += float(np.sum((data.points_2d[kk] - proj) ** 2))

    return total


def calibrate(
    data: CalibrationData,
    options: CalibrationOptions | None = None,
    initial: dict | None = None,
) -> CalibrationResult:
    """Run the full calibration.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    options : CalibrationOptions, optional
        Options controlling the estimation.  Defaults to
        :class:`CalibrationOptions` defaults.
    initial : dict, optional
        Initial guesses.  Recognised keys: ``fc``, ``cc``, ``kc``,
        ``alpha_c``.  When ``fc`` is missing it is initialized from the
        vanishing points (or searched for, for a 3D rig).

    Returns
    -------
    CalibrationResult
        The calibration outcome, including parameter uncertainties.

    Notes
    -----
    Port of ``go_calib_optim_iter.m``.

    The optimization is a damped (Levenberg-like) gradient descent on the
    sum of squares of the reprojection error, with all intrinsic and
    extrinsic parameters estimated jointly.  The uncertainty estimates are
    derived from the inverse of the approximated Hessian, scaled by three
    standard deviations, exactly like the MATLAB original.

    For a 3D (non planar) rig the calibration is run from several initial
    focal lengths and the best result is kept: a 3D target has a
    projective ambiguity that a single initialization cannot resolve.
    """
    if options is None:
        options = CalibrationOptions()

    log = (lambda *a: print(*a)) if options.verbose else (lambda *a: None)

    rig3D = data.is_3d_rig

    if rig3D and "fc" not in (initial or {}):
        # A 3D rig has no vanishing points, so the focal length cannot be
        # initialized the way it is for a planar target.  The calibration
        # is run from several candidate focal lengths and the best result
        # is kept.
        f0 = (data.image_size[0] / 2.0) / np.tan(np.pi * options.fov_degrees / 360.0)
        candidates = _focal_candidates_3d(data, f0, options, log)

        results = []
        for cand in candidates:
            trial_options = CalibrationOptions(**vars(options))
            try:
                trial = _calibrate_once(
                    data,
                    trial_options,
                    initial={
                        "fc": np.array([cand, cand]),
                        "cc": np.array(
                            [
                                (data.image_size[0] - 1) / 2.0,
                                (data.image_size[1] - 1) / 2.0,
                            ]
                        ),
                        "kc": np.zeros(5),
                        "alpha_c": 0.0,
                    },
                    log=log,
                )
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue

            if trial is None:
                continue
            score = _physical_score(trial)
            if not np.isfinite(score):
                continue
            results.append((score, cand, trial))

        if results:
            results.sort(key=lambda r: r[0])
            best = results[0]
            log(
                f"Best 3D rig initialization: fc = {best[1]:.3f} "
                f"(reprojection error {best[0]:.4f} px)."
            )
            return best[2]

    return _calibrate_once(data, options, initial, log=log)


def _physical_score(result) -> float:
    """Score the physical plausibility of a calibration result.

    A calibration that explains the observations but puts part of the
    calibration pattern *behind* the camera is wrong: the DLT of a 3D
    target has a projective ambiguity that can produce such solutions.

    Parameters
    ----------
    result : CalibrationResult
        The calibration to score.

    Returns
    -------
    float
        A score that is smaller for more plausible calibrations.  It is
        ``inf`` when the result is not usable at all.
    """
    if result is None:
        return np.inf

    depths = []

    for kk in np.flatnonzero(result.active_images):
        points = result._points_3d
        if kk >= len(points):
            continue
        X = points[kk]
        Y = rigid_motion(X, result.omc[kk], result.Tc[kk])
        depths.append(Y[2])

    if not depths:
        # Without the 3D points the check cannot be done; the result is
        # accepted as is.
        return result.rmse if np.isfinite(result.rmse) else np.inf

    all_depths = np.concatenate(depths)

    # Every point of the pattern must be in front of the camera.
    if np.any(all_depths <= 0):
        return np.inf

    # The depths must be comparable across the views: a wrong scale makes
    # them differ wildly.
    med = np.median(all_depths)
    if med <= 0:
        return np.inf
    spread = np.median(np.abs(all_depths - med)) / med

    return result.rmse * (1.0 + spread)


def _focal_candidates_3d(
    data: CalibrationData, f0: float, options: CalibrationOptions, log
):
    """Return candidate focal lengths for a 3D rig.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    f0 : float
        The field-of-view based initial guess.
    options : CalibrationOptions
        The calibration options.
    log : callable
        The logging function.

    Returns
    -------
    list of float
        The candidate focal lengths, most promising first.
    """
    cc = np.array(
        [
            (data.image_size[0] - 1) / 2.0,
            (data.image_size[1] - 1) / 2.0,
        ]
    )

    lo = max(f0 / 8.0, 50.0)
    hi = f0 * 8.0

    # A dense sweep of the whole range: the calibration of a 3D rig is
    # cheap compared to the cost of a wrong answer, and the projective
    # ambiguity of the DLT makes any heuristic selection unreliable.
    grid = np.geomspace(lo, hi, 33)

    # The consistency of the per-view poses gives a cheap first ranking,
    # which is used to refine a few candidates locally.
    scored = []
    for cand in grid:
        score = _score_focal_3d_consistency(
            data, np.array([cand, cand]), cc, options
        )
        if score is not None:
            scored.append((score, cand))

    scored.sort()

    candidates = [float(c) for c in grid]

    # The best candidates of the grid are refined locally.
    refined = []
    for score, cand in list(scored[:4]):
        span = (hi - lo) / (len(grid) - 1)
        for _ in range(3):
            for c in np.linspace(cand - span, cand + span, 5):
                if c <= 0:
                    continue
                refined.append(float(c))
            span /= 4.0

    candidates = refined + candidates

    # The field-of-view guess is always tried as well.
    candidates.append(f0)

    # Deduplicate while preserving the order.
    unique = []
    for c in candidates:
        if all(abs(c - u) > 1e-6 for u in unique):
            unique.append(c)

    log(
        f"Trying {len(unique)} candidate focal lengths for the 3D rig "
        f"(between {lo:.1f} and {hi:.1f})."
    )

    return unique


def _calibrate_once(
    data: CalibrationData,
    options: CalibrationOptions | None = None,
    initial: dict | None = None,
    log=None,
) -> CalibrationResult:
    """Run the calibration once, from the given initial guess.

    Parameters
    ----------
    data : CalibrationData
        The observations.
    options : CalibrationOptions, optional
        Options controlling the estimation.
    initial : dict, optional
        Initial guesses.  Recognised keys: ``fc``, ``cc``, ``kc``,
        ``alpha_c``.
    log : callable, optional
        The logging function.  Defaults to ``print`` when the options
        are verbose.

    Returns
    -------
    CalibrationResult or None
        The calibration outcome, or ``None`` when the calibration failed.
    """
    if options is None:
        options = CalibrationOptions()

    if log is None:
        log = (lambda *a: print(*a)) if options.verbose else (lambda *a: None)

    nx, ny = data.image_size
    alpha_smooth = options.alpha_smooth

    # --- Conflicts and adjustments on the options ------------------------
    if not options.est_aspect_ratio:
        options.est_fc = np.array([1.0, 1.0])

    rig3D = data.is_3d_rig

    if options.center_optim and len(data.ind_active) < 2 and not rig3D:
        log(
            "WARNING: Principal point rejected from the optimization "
            "(single image, planar rig)."
        )
        options.center_optim = 0.0
        options.est_alpha = 0.0

    if options.center_optim and len(data.ind_active) < 5 and not rig3D:
        log(
            "WARNING: The principal point estimation may be unreliable "
            "(less than 5 images for calibration)."
        )

    if not options.center_optim and options.est_alpha:
        log("WARNING: No principal point estimation, no skew estimation.")
        options.est_alpha = 0.0

    if not options.est_alpha:
        alpha_c = 0.0
    else:
        alpha_c = None

    # --- Intrinsic initialization ----------------------------------------
    initial = dict(initial or {})

    if "cc" not in initial:
        log("Initialization of the principal point at the center of the image.")
        cc = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
        alpha_smooth = 0.4
    else:
        cc = _col(initial["cc"]).astype(float)

    if "alpha_c" not in initial:
        log("Initialization of the image skew to zero.")
        alpha_c = 0.0
        alpha_smooth = 0.4
    else:
        alpha_c = float(initial["alpha_c"])

    if "fc" not in initial:
        if rig3D or options.quick_init:
            fov = options.fov_degrees
            log(f"Initialization of the focal length to a FOV of {fov} degrees.")
            fc = (nx / 2.0) / np.tan(np.pi * fov / 360.0) * np.ones(2)
            options.est_fc = np.array([1.0, 1.0])
            alpha_smooth = 0.4
        else:
            log(
                "Initialization of the intrinsic parameters using the "
                "vanishing points of planar patterns."
            )
            fc, cc, _, alpha_c = init_intrinsic_param(data, options)
            alpha_smooth = 0.4
            options.est_fc = np.array([1.0, 1.0])
    else:
        fc = _col(initial["fc"]).astype(float)
        if fc.size == 1:
            fc = np.repeat(fc, 2)

    if "kc" not in initial:
        log("Initialization of the image distortion to zero.")
        kc = np.zeros(5)
        alpha_smooth = 0.4
    else:
        kc = _col(initial["kc"]).astype(float)
        if kc.size == 4:
            kc = np.concatenate([kc, [0.0]])
        if kc.size != 5:
            raise ValueError("kc must have 4 or 5 elements")

    if not options.est_aspect_ratio:
        fc[1] = fc[0]

    if not np.all(options.est_dist):
        kc = kc * options.est_dist

    # --- Extrinsic initialization ----------------------------------------
    log("\nInitialization of the extrinsic parameters...")
    omc, Tc, Rc = _init_extrinsics(data, fc, cc, kc, alpha_c, options, log)

    # --- Build the global parameter vector -------------------------------
    n_ima = data.n_ima
    param = np.concatenate(
        [fc, cc, [alpha_c], kc, np.zeros(5)]
        + [np.concatenate([omc[k], Tc[k]]) for k in range(n_ima)]
    )

    # --- Main optimization ----------------------------------------------
    log(
        f"\nMain calibration optimization procedure - "
        f"Number of images: {len(data.ind_active)}"
    )

    change = 1.0
    iter_ = 0
    active = data.active_images.copy()
    damping = options.damping

    log("Gradient descent iterations: ")
    param_list = [param.copy()]

    while change > 1e-9 and iter_ < options.max_iter:
        log(f"{iter_ + 1}...")

        # NOTE: these must be copies, not views: MATLAB's ``param(1:2)``
        # yields a copy, and the change metric compares the updated
        # parameters against the previous ones.
        f = param[0:2].copy()
        c = param[2:4].copy()
        alpha = param[4]
        k = param[5:10].copy()

        n_var = 15 + 6 * n_ima
        JJ3 = np.zeros((n_var, n_var))
        ex3 = np.zeros(n_var)

        for kk in data.ind_active:
            if not active[kk]:
                # The view was deactivated during a previous iteration; its
                # parameters have been invalidated and must not be used.
                continue

            omckk = param[15 + 6 * kk : 15 + 6 * kk + 3]
            Tckk = param[15 + 6 * kk + 3 : 15 + 6 * kk + 6]

            if np.isnan(omckk[0]):
                log(f"Intrinsic parameters at frame {kk} do not exist")
                return None

            X_kk = data.points_3d[kk]
            x_kk = data.points_2d[kk]

            if not options.est_aspect_ratio:
                _, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = (
                    project_points_with_jacobians(
                        X_kk, omckk, Tckk, f[0], c, k, alpha
                    )
                )
                # A single focal length is estimated in this case: the
                # second component is not part of the optimization, so
                # its column is zeroed to keep the matrix blocks
                # consistent (they are always 10 wide).
                dxdf = np.hstack([dxdf[:, :1], np.zeros_like(dxdf[:, :1])])
            else:
                _, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = (
                    project_points_with_jacobians(
                        X_kk, omckk, Tckk, f, c, k, alpha
                    )
                )

            proj = project_points(X_kk, omckk, Tckk, f, c, k, alpha)
            exkk = flatten_points(x_kk - proj)

            # MATLAB: A = [dxdf dxdc dxdalpha dxdk]'  (10 x 2N)
            #         B = [dxdom dxdT]'              (6  x 2N)
            A = np.hstack([dxdf, dxdc, dxdalpha, dxdk]).T
            B = np.hstack([dxdom, dxdT]).T

            sl = slice(0, 10)
            JJ3[sl, sl] += A @ A.T
            sl = slice(15 + 6 * kk, 15 + 6 * kk + 6)
            JJ3[sl, sl] += B @ B.T

            AB = A @ B.T
            JJ3[0:10, sl] = AB
            JJ3[sl, 0:10] = AB.T

            ex3[0:10] += A @ exkk
            ex3[sl] += B @ exkk

            # Ill-conditioning check.
            if options.check_cond:
                JJ_kk = B.T
                if np.linalg.cond(JJ_kk) > options.thresh_cond:
                    active[kk] = False
                    log(
                        f"\nWarning: View #{kk} ill-conditioned. This image is "
                        f"now set inactive. (set check_cond=False to disable)"
                    )
                    param[15 + 6 * kk : 15 + 6 * kk + 6] = np.nan

        if not np.any(active):
            raise RuntimeError("all views became ill-conditioned")

        # Select the variables to update.  The layout matches the parameter
        # vector: 2 (fc) + 2 (cc) + 1 (alpha) + 5 (kc) + 5 (unused),
        # then 6 entries per view (rotation + translation).
        est_fc_mask = options.est_fc.copy()
        if not options.est_aspect_ratio:
            # A single focal length parameter is estimated.
            est_fc_mask[1] = 0.0

        selected = np.concatenate(
            [
                est_fc_mask,
                [options.center_optim] * 2,
                [options.est_alpha],
                options.est_dist,
                np.zeros(5),
                np.repeat(active.astype(float), 6),
            ]
        )
        ind_Jac = np.flatnonzero(selected)

        JJ3s = JJ3[np.ix_(ind_Jac, ind_Jac)]
        ex3s = ex3[ind_Jac]

        alpha_smooth2 = 1.0 - (1.0 - alpha_smooth) ** (iter_ + 1)

        # The cost function (sum of squares of the reprojection error).
        cost = float(ex3s @ ex3s)

        if options.damping > 0:
            # Levenberg-Marquardt damping.  The undamped Gauss-Newton step
            # of the original toolbox overshoots badly when the normal
            # matrix is close to singular (which is the normal situation
            # for a 3D rig), making the optimization diverge.  Adding a
            # ridge term keeps the step well behaved without changing the
            # optimum.
            #
            # The damping is scaled per parameter: the focal length
            # (~1000), the principal point (~600), the distortion
            # coefficients (~0.01) and the translations (~500) live on
            # wildly different scales, so a single damping constant would
            # be negligible for some of them and overwhelming for others.
            scale = np.maximum(np.abs(param[ind_Jac]), 1e-6)
            ridge = damping * np.diag(scale * scale)

            try:
                param_innov = alpha_smooth2 * np.linalg.solve(
                    JJ3s + ridge, ex3s
                )
            except np.linalg.LinAlgError:
                log("\nSingular normal equations: stopping.")
                break
        else:
            try:
                JJ2_inv = np.linalg.inv(JJ3s)
            except np.linalg.LinAlgError:
                log("\nSingular normal equations: stopping.")
                break
            param_innov = alpha_smooth2 * (JJ2_inv @ ex3s)

        param_try = param.copy()
        param_try[ind_Jac] += param_innov

        # Adaptive damping: when the step reduces the error, trust the
        # model more (less damping); otherwise increase the damping.  The
        # damping is decayed quickly so that a well-conditioned problem
        # ends up behaving like the plain Gauss-Newton iteration of the
        # original toolbox.
        if options.damping > 0:
            cost_try = _reprojection_cost(
                data, param_try, active, options
            )
            if np.isfinite(cost_try) and cost_try < cost:
                damping = max(damping * 0.1, options.damping * 1e-6)
            else:
                damping = min(damping * 10.0, options.damping * 1e8)

        param = param_try

        fc_current = param[0:2]
        cc_current = param[2:4]

        if not options.est_aspect_ratio and np.all(options.est_fc == 1):
            fc_current[1] = fc_current[0]
            param[1] = param[0]

        denom = np.linalg.norm(np.concatenate([fc_current, cc_current]))
        change = (
            np.linalg.norm(np.concatenate([fc_current, cc_current]) - np.concatenate([f, c])) / denom
            if denom > 0
            else 0.0
        )

        # Second step: recompute the extrinsic parameters.
        if options.recompute_extrinsic:
            for kk in data.ind_active:
                if not active[kk]:
                    continue
                omc_current = param[15 + 6 * kk : 15 + 6 * kk + 3]
                Tc_current = param[15 + 6 * kk + 3 : 15 + 6 * kk + 6]

                # The pose is refined from its current value rather than
                # re-initialized from scratch: re-initializing discards
                # the progress made by the joint optimization and, for a
                # 3D rig, the DLT initialization is very sensitive to the
                # intrinsic parameters, which are still moving.
                omc_current, Tc_current, _, JJ_kk = compute_extrinsic_refine(
                    omc_current,
                    Tc_current,
                    data.points_2d[kk],
                    data.points_3d[kk],
                    fc_current,
                    cc_current,
                    k,
                    alpha,
                    20,
                    options.thresh_cond,
                )
                if options.check_cond and np.linalg.cond(JJ_kk) > options.thresh_cond:
                    active[kk] = False
                    log(
                        f"\nWarning: View #{kk} ill-conditioned. This image is "
                        f"now set inactive."
                    )
                    omc_current = np.full(3, np.nan)
                    Tc_current = np.full(3, np.nan)

                param[15 + 6 * kk : 15 + 6 * kk + 3] = omc_current
                param[15 + 6 * kk + 3 : 15 + 6 * kk + 6] = Tc_current

        param_list.append(param.copy())
        iter_ += 1

    log("done\n")

    # --- Re-activate the views that were rejected ------------------------
    #
    # The conditioning test is applied while the intrinsic parameters are
    # still moving, so a view can be rejected early only to become
    # perfectly usable once the calibration has converged.  Each rejected
    # view is therefore tested again with the final parameters and
    # re-activated when it is well conditioned.
    if options.check_cond:
        reactivated = []
        for kk in np.flatnonzero(data.active_images & ~active):
            # Re-initialize the pose with the final intrinsic parameters:
            # the pose found during the optimization is invalid for a view
            # that was dropped, and the initialization is very sensitive to
            # the intrinsic parameters for a 3D rig.
            try:
                om_try, T_try, _ = compute_extrinsic_init(
                    data.points_2d[kk],
                    data.points_3d[kk],
                    param[0:2],
                    param[2:4],
                    param[5:10],
                    param[4],
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            om_try, T_try, _, JJ_try = compute_extrinsic_refine(
                om_try,
                T_try,
                data.points_2d[kk],
                data.points_3d[kk],
                param[0:2],
                param[2:4],
                param[5:10],
                param[4],
                20,
                options.thresh_cond,
            )

            if not (np.all(np.isfinite(om_try)) and np.all(np.isfinite(T_try))):
                continue
            if np.linalg.cond(JJ_try) > options.thresh_cond:
                continue

            # Only accept the view if it actually reprojects well.
            proj = project_points(
                data.points_3d[kk], om_try, T_try, param[0:2], param[2:4],
                param[5:10], param[4],
            )
            if not np.all(np.isfinite(proj)):
                continue
            rms = float(np.sqrt(np.mean((data.points_2d[kk] - proj) ** 2)))
            if rms > 10.0:
                continue

            active[kk] = True
            param[15 + 6 * kk : 15 + 6 * kk + 3] = om_try
            param[15 + 6 * kk + 3 : 15 + 6 * kk + 6] = T_try
            reactivated.append(kk)

        if reactivated:
            log(
                "Re-activated the following previously rejected views: "
                + ", ".join(str(k) for k in reactivated)
            )

    # --- Uncertainties ---------------------------------------------------
    log("Estimation of uncertainties...")

    solution = param.copy()

    fc = solution[0:2]
    cc = solution[2:4]
    alpha_c = solution[4]
    kc = solution[5:10]

    omc = []
    Tc = []
    Rc = []
    for kk in range(n_ima):
        if active[kk]:
            omckk = solution[15 + 6 * kk : 15 + 6 * kk + 3]
            Tckk = solution[15 + 6 * kk + 3 : 15 + 6 * kk + 6]
            Rckk = rodrigues(omckk)
        else:
            omckk = np.full(3, np.nan)
            Tckk = np.full(3, np.nan)
            Rckk = np.full((3, 3), np.nan)
        omc.append(omckk)
        Tc.append(Tckk)
        Rc.append(Rckk)

    # --- Cheirality check ------------------------------------------------
    #
    # Every point of the calibration pattern must be in front of the
    # camera.  A solution that puts part of the pattern behind the camera
    # is wrong, even when it reproduces the observations: this happens with
    # the projective ambiguity of a 3D target.
    if rig3D:
        for kk in np.flatnonzero(active):
            Y = rigid_motion(data.points_3d[kk], omc[kk], Tc[kk])
            if np.any(Y[2] <= 0):
                log(
                    f"\nWarning: View #{kk} is not physically valid (part of "
                    "the pattern is behind the camera). This image is now set "
                    "inactive."
                )
                active[kk] = False

        if not np.any(active):
            raise RuntimeError(
                "no physically valid view remains: the calibration failed"
            )

    errors, reprojections, err_std = compute_reprojection_errors(
        data, fc, cc, kc, alpha_c, omc, Tc, active=active
    )

    # Only the views that were actually used contribute to the noise
    # estimate; the errors of the deactivated views are NaN.
    finite_err = np.concatenate(
        [
            e.reshape(-1)
            for e, kk in zip(errors, range(n_ima))
            if active[kk] and e.size
        ]
    )
    sigma_x = float(np.std(finite_err)) if finite_err.size else float("nan")

    # Rebuild the Hessian for the uncertainty estimate.
    n_var = 15 + 6 * n_ima
    JJ3 = np.zeros((n_var, n_var))
    for kk in data.ind_active:
        if not active[kk]:
            continue
        omckk = solution[15 + 6 * kk : 15 + 6 * kk + 3]
        Tckk = solution[15 + 6 * kk + 3 : 15 + 6 * kk + 6]
        X_kk = data.points_3d[kk]

        if not options.est_aspect_ratio:
            _, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = (
                project_points_with_jacobians(X_kk, omckk, Tckk, fc[0], cc, kc, alpha_c)
            )
            dxdf = np.hstack([dxdf[:, :1], np.zeros_like(dxdf[:, :1])])
        else:
            _, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = (
                project_points_with_jacobians(X_kk, omckk, Tckk, fc, cc, kc, alpha_c)
            )

        A = np.hstack([dxdf, dxdc, dxdalpha, dxdk]).T
        B = np.hstack([dxdom, dxdT]).T

        sl = slice(0, 10)
        JJ3[sl, sl] += A @ A.T
        sl = slice(15 + 6 * kk, 15 + 6 * kk + 6)
        JJ3[sl, sl] += B @ B.T

        AB = A @ B.T
        JJ3[0:10, sl] = AB
        JJ3[sl, 0:10] = AB.T

    # Recompute the selection mask with the final set of active views: some
    # views may have been deactivated during the optimization, in which
    # case their parameters are not part of the estimated system.
    final_selected = np.concatenate(
        [
            options.est_fc,
            [options.center_optim] * 2,
            [options.est_alpha],
            options.est_dist,
            np.zeros(5),
            np.repeat(active.astype(float), 6),
        ]
    )
    ind_Jac = np.flatnonzero(final_selected)

    JJ3s = JJ3[np.ix_(ind_Jac, ind_Jac)]
    try:
        JJ2_inv = np.linalg.inv(JJ3s)
    except np.linalg.LinAlgError:
        JJ2_inv = np.full_like(JJ3s, np.nan)

    param_error = np.zeros(n_var)
    param_error[ind_Jac] = 3.0 * np.sqrt(np.abs(np.diag(JJ2_inv))) * sigma_x

    fc_error = param_error[0:2]
    cc_error = param_error[2:4]
    alpha_c_error = param_error[4]
    kc_error = param_error[5:10]

    if not options.est_aspect_ratio and np.all(options.est_fc == 1):
        fc_error[1] = fc_error[0]

    # --- Final parameters ------------------------------------------------
    KK = camera_matrix(fc, cc, alpha_c)

    H = []
    for kk in range(n_ima):
        if active[kk]:
            Hkk = KK @ np.column_stack([Rc[kk][:, 0], Rc[kk][:, 1], Tc[kk]])
            Hkk = Hkk / Hkk[2, 2]
        else:
            Hkk = np.full((3, 3), np.nan)
        H.append(Hkk)

    result = CalibrationResult(
        fc=fc,
        cc=cc,
        alpha_c=float(alpha_c),
        kc=kc,
        omc=omc,
        Tc=Tc,
        Rc=Rc,
        H=H,
        err_std=err_std,
        errors=errors,
        reprojections=reprojections,
        fc_error=fc_error,
        cc_error=cc_error,
        kc_error=kc_error,
        alpha_c_error=float(alpha_c_error),
        KK=KK,
        active_images=active,
        n_iter=iter_,
        converged=bool(change <= 1e-9),
        points_3d=list(data.points_3d),
    )

    log("done\n")

    if options.verbose:
        print()
        print(result.summary())
        print()
        print(
            "Note: the numerical errors are approximately three times the "
            "standard deviations (for reference)."
        )

    return result


def _init_extrinsics(data, fc, cc, kc, alpha_c, options, log):
    """Compute the initial extrinsic parameters for every view."""
    omc = []
    Tc = []
    Rc = []

    for kk in range(data.n_ima):
        if not data.active_images[kk]:
            omc.append(np.full(3, np.nan))
            Tc.append(np.full(3, np.nan))
            Rc.append(np.full((3, 3), np.nan))
            continue

        omckk, Tckk, _ = compute_extrinsic_init(
            data.points_2d[kk], data.points_3d[kk], fc, cc, kc, alpha_c
        )
        omckk, Tckk, Rckk, JJ_kk = compute_extrinsic_refine(
            omckk,
            Tckk,
            data.points_2d[kk],
            data.points_3d[kk],
            fc,
            cc,
            kc,
            alpha_c,
            20,
            options.thresh_cond,
        )

        if options.check_cond and np.linalg.cond(JJ_kk) > options.thresh_cond:
            data.active_images[kk] = False
            omckk = np.full(3, np.nan)
            Tckk = np.full(3, np.nan)
            Rckk = np.full((3, 3), np.nan)
            log(f"\nWarning: View #{kk} ill-conditioned. This image is now set inactive.")

        omc.append(omckk)
        Tc.append(Tckk)
        Rc.append(Rckk)

    return omc, Tc, Rc
