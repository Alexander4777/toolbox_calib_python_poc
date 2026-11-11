"""The state of a calibration session.

The model holds everything the interface manipulates: the loaded images,
the corners picked in each of them, the options of the calibration and the
results.  Keeping it separate from the widgets makes the logic testable
without a display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..calibration import CalibrationData, CalibrationOptions, CalibrationResult
from ..images import load_image

__all__ = ["ViewState", "CalibrationSession"]


@dataclass
class ViewState:
    """One image of the calibration sequence.

    Attributes
    ----------
    path : pathlib.Path
        The file the image was read from.
    image : numpy.ndarray
        The image.
    corners : numpy.ndarray
        The detected corners, shape ``(2, N)``.  Empty when no corner has
        been picked yet.
    active : bool
        Whether the view takes part in the calibration.
    """

    path: Path
    image: np.ndarray
    corners: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    active: bool = True

    @property
    def name(self) -> str:
        """The file name of the image."""
        return self.path.name

    @property
    def size(self) -> tuple[int, int]:
        """The size of the image, ``(width, height)``."""
        return self.image.shape[1], self.image.shape[0]

    @property
    def has_corners(self) -> bool:
        """``True`` when corners have been picked for this view."""
        return self.corners.shape[1] > 0


class CalibrationSession:
    """Everything the interface manipulates.

    Parameters
    ----------
    parent : object, optional
        Kept for symmetry with the Qt objects; unused.
    """

    def __init__(self):
        self.views: list[ViewState] = []
        self.result: CalibrationResult | None = None
        self.data: CalibrationData | None = None

        # The pattern: a regular grid of corners on a plane.
        self.grid_nx = 13
        self.grid_ny = 9
        self.grid_spacing = 30.0
        self.grid_z = 0.0

        # The options of the calibration.
        self.options = CalibrationOptions(verbose=False)

        self.dirty = True

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def add_images(self, paths) -> list[ViewState]:
        """Load images and append them to the session.

        Parameters
        ----------
        paths : iterable of str or pathlib.Path
            The image files.

        Returns
        -------
        list of ViewState
            The views that were added.  Files that could not be read are
            skipped.
        """
        added = []

        for path in paths:
            path = Path(path)
            try:
                image = load_image(path)
            except (FileNotFoundError, OSError):
                continue

            view = ViewState(path=path, image=image)
            self.views.append(view)
            added.append(view)

        if added:
            self.dirty = True

        return added

    def remove_view(self, index: int) -> None:
        """Remove a view from the session.

        Parameters
        ----------
        index : int
            The index of the view.
        """
        if 0 <= index < len(self.views):
            del self.views[index]
            self.dirty = True

    def clear(self) -> None:
        """Remove every view and every result."""
        self.views.clear()
        self.result = None
        self.data = None
        self.dirty = True

    # ------------------------------------------------------------------
    # The pattern
    # ------------------------------------------------------------------
    def grid_points(self) -> np.ndarray:
        """Return the 3D coordinates of the pattern.

        Returns
        -------
        numpy.ndarray, shape (3, N)
            The coordinates of the ``grid_nx`` by ``grid_ny`` corners, on
            the plane ``Z = grid_z``.
        """
        xs, ys = np.meshgrid(
            np.arange(self.grid_nx, dtype=float),
            np.arange(self.grid_ny, dtype=float),
        )
        return np.vstack(
            [
                (xs.ravel() * self.grid_spacing).ravel(),
                (ys.ravel() * self.grid_spacing).ravel(),
                np.full(xs.size, self.grid_z),
            ]
        )

    @property
    def n_corners(self) -> int:
        """The number of corners of the pattern."""
        return self.grid_nx * self.grid_ny

    # ------------------------------------------------------------------
    # The calibration data
    # ------------------------------------------------------------------
    def build_data(self) -> CalibrationData:
        """Assemble the observations into a :class:`CalibrationData`.

        Returns
        -------
        CalibrationData
            The observations of the views that have corners.

        Raises
        ------
        ValueError
            When no view has corners, or when the views do not share the
            same image size.
        """
        usable = [v for v in self.views if v.has_corners]

        if not usable:
            raise ValueError(
                "no corners have been picked yet: extract the grid corners "
                "in at least one image"
            )

        sizes = {v.size for v in usable}
        if len(sizes) > 1:
            raise ValueError(
                "the images do not all have the same size: "
                + ", ".join(f"{w}x{h}" for w, h in sorted(sizes))
            )

        n_corners = {v.corners.shape[1] for v in usable}
        if len(n_corners) > 1:
            raise ValueError(
                "the views do not all have the same number of corners: "
                + ", ".join(str(n) for n in sorted(n_corners))
            )

        grid = self.grid_points()

        points_2d = []
        points_3d = []
        active = []

        # Only the views that have corners take part in the calibration;
        # the active mask must have the same length as the point lists.
        for view in self.views:
            if not view.has_corners:
                continue
            points_2d.append(view.corners)
            points_3d.append(grid)
            active.append(view.active)

        return CalibrationData(
            points_2d=points_2d,
            points_3d=points_3d,
            image_size=usable[0].size,
            active_images=np.array(active, dtype=bool),
        )

    # ------------------------------------------------------------------
    # The results
    # ------------------------------------------------------------------
    def set_result(self, result: CalibrationResult, data: CalibrationData) -> None:
        """Store the outcome of a calibration."""
        self.result = result
        self.data = data
        self.dirty = False

    def reproject(self, view_index: int) -> np.ndarray | None:
        """Reproject the corners of a view with the calibration result.

        Parameters
        ----------
        view_index : int
            The index of the view.

        Returns
        -------
        numpy.ndarray, shape (2, N) or None
            The reprojected corners, or ``None`` when there is no result.
        """
        if self.result is None or self.data is None:
            return None

        # Find the index of the view inside the calibration data.
        data_index = None
        count = 0
        for i, view in enumerate(self.views):
            if not view.has_corners:
                continue
            if i == view_index:
                data_index = count
                break
            count += 1

        if data_index is None:
            return None

        return self.result.project(self.data.points_3d[data_index], data_index)

    def error_per_view(self) -> list[tuple[str, float]]:
        """Return the reprojection error of every view.

        Returns
        -------
        list of tuple
            The name of the view and its root mean square error.
        """
        if self.result is None:
            return []

        out = []
        count = 0
        for view in self.views:
            if not view.has_corners:
                continue
            if count < len(self.result.errors):
                err = self.result.errors[count]
                rms = float(np.sqrt(np.nanmean(err**2))) if err.size else float("nan")
                out.append((view.name, rms))
            count += 1

        return out
