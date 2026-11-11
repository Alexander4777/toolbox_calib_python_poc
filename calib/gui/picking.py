"""Corner picking on an image.

The original toolbox has no corner detector: the user is expected to
provide the corner locations.  This module provides the interactive
picking used by the GUI, plus a few automatic detectors that work on
synthetic patterns (a regular grid of dots or squares), which is what the
GUI's "auto-detect" button uses.

The interactive picker is a Qt widget that shows an image and lets the
user click the corners of the calibration pattern in order.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CornerPicker",
    "detect_grid_corners",
    "order_corners",
]


def order_corners(points: np.ndarray, nx: int | None = None, ny: int | None = None):
    """Order a set of 2D points as a grid (top-left first, then row by row).

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        The points to order.
    nx, ny : int, optional
        The number of corners along x and y.  When they are given, the
        points are grouped into rows of exactly ``nx`` points, which is
        much more reliable than guessing the rows.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        The ordered points.

    Notes
    -----
    The two directions of the grid are found with a robust search (see
    :func:`_find_grid_direction`), which works for a tilted pattern too;
    the principal axes of the points would be wrong in that case.

    The row that comes first is the one whose points are the highest in
    the image, and the points of a row are sorted from the left to the
    right.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 4:
        return points

    centre = points.mean(axis=0)
    centred = points - centre

    # The two directions of the grid.  The number of rows and columns is
    # known when the pattern is a grid, and guessed otherwise.
    if nx is not None and ny is not None:
        axis_u = _find_grid_direction(centred, nx)
        axis_v = _find_grid_direction(centred, ny)
    else:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        axis_u = Vt[0]
        axis_v = Vt[1]

    if axis_u is None or axis_v is None:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        axis_u = Vt[0]
        axis_v = Vt[1]

    # The two directions are swapped so that ``axis_u`` is the one that
    # goes from the left to the right of the image and ``axis_v`` the one
    # that goes from the top to the bottom.  The grid of the pattern is
    # usually aligned with the image, even when it is tilted.
    if abs(axis_u[0]) < abs(axis_v[0]):
        axis_u, axis_v = axis_v, axis_u

    u = centred @ axis_u
    v = centred @ axis_v

    if nx is not None and ny is not None and n == nx * ny:
        rows = _split_into_equal_groups(v, ny)
    else:
        rows = _split_into_rows(v)

    ordered = []
    for row in rows:
        # Sort the row along u.
        ordered.append(row[np.argsort(u[row])])

    # The rows are put in the order of the image: the top row first.
    # The comparison uses the vertical coordinate of the image, which is
    # unambiguous, unlike the projected axes of a tilted pattern.
    rows = sorted(rows, key=lambda r: -np.mean(points[r, 1]))

    # The points of a row are sorted from the left to the right, which is
    # the direction of the increasing image x.
    ordered = []
    for row in rows:
        ordered.append(row[np.argsort(points[row, 0])])

    # Finally, the whole grid is transposed when needed, so that the
    # points come out in the order of the pattern: the first row of the
    # pattern is the top row of the image, and the points of a row go from
    # the left to the right.
    if nx is not None and ny is not None and n == nx * ny:
        grid = np.array(ordered).reshape(ny, nx)
        if np.mean(points[grid[0], 1]) > np.mean(points[grid[-1], 1]):
            grid = grid[::-1]
        return points[grid.reshape(-1)]

    return points[np.concatenate(ordered)]


def _split_into_equal_groups(values: np.ndarray, count: int) -> list[np.ndarray]:
    """Split the indices of the sorted values into groups of equal size.

    Parameters
    ----------
    values : numpy.ndarray
        The values to sort and split.
    count : int
        The number of groups.

    Returns
    -------
    list of numpy.ndarray
        The indices of each group.
    """
    order = np.argsort(values)
    if count <= 0 or values.size % count != 0:
        return _split_into_rows(values)

    per_group = values.size // count
    return [order[i * per_group : (i + 1) * per_group] for i in range(count)]


def _split_into_rows(v: np.ndarray) -> list[np.ndarray]:
    """Split the indices of the sorted values into rows.

    Parameters
    ----------
    v : numpy.ndarray
        The values to sort and split.

    Returns
    -------
    list of numpy.ndarray
        The indices of each row.
    """
    order = np.argsort(v)
    v_sorted = v[order]

    if v_sorted.size < 2:
        return [order]

    gaps = np.diff(v_sorted)

    # A large gap separates two rows.
    threshold = 0.5 * np.median(gaps)
    breaks = np.flatnonzero(gaps > threshold) + 1

    rows = []
    start = 0
    for end in breaks:
        rows.append(order[start:end])
        start = end
    rows.append(order[start:])

    return [r for r in rows if r.size]


def detect_grid_corners(
    image: np.ndarray,
    nx: int,
    ny: int,
    subpixel: bool = True,
) -> np.ndarray | None:
    """Detect the corners of a regular grid pattern in an image.

    Parameters
    ----------
    image : numpy.ndarray
        The grayscale image.
    nx, ny : int
        The number of corners along x and y.
    subpixel : bool, optional
        Whether to refine the corner positions to sub-pixel accuracy.

    Returns
    -------
    numpy.ndarray, shape (ny * nx, 2) or None
        The detected corners, ordered as a grid, or ``None`` when the
        pattern could not be found.

    Notes
    -----
    A simple and robust detector for the patterns used in practice: the
    image is thresholded, the connected components of the resulting blobs
    are taken as the corners, and their centroids are refined to
    sub-pixel accuracy with the intensity gradient.

    This is not a general purpose corner detector: it works on high
    contrast patterns (dark squares on a light background, or the
    opposite) where the corners are well separated.
    """
    gray = _to_gray(image)
    if gray is None:
        return None

    expected = nx * ny

    # The pattern may be dark squares on a light background or the
    # opposite.  The blobs are the minority class in both cases, so the
    # polarity whose foreground is the smaller one is used.
    best = None
    best_score = None
    best_binary = None

    for invert in (True, False):
        binary = _threshold_blobs(gray, invert)
        if binary is None:
            continue

        # When the pattern is seen at an angle, the squares touch each
        # other and merge.  The watershed splits them again, which restores
        # one blob per corner.
        separated = _separate_blobs(binary, expected)
        if separated is not None:
            binary = separated

        candidates = _blob_candidates(binary, expected)
        if candidates is None:
            if best_binary is None:
                best_binary = binary
            continue

        # Prefer the polarity that gives the most blobs of a similar size.
        areas = np.array([c[1] for c in candidates])
        score = (
            len(candidates),
            -float(np.std(areas) / max(np.median(areas), 1e-9)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = candidates
            best_binary = binary

    if best is None:
        # The squares of the pattern touch each other, so the blobs cannot
        # be used: the corners are detected directly in the image instead.
        lattice = _detect_corners_by_features(gray, nx, ny)
        if lattice is None:
            return None

        if subpixel:
            lattice = _refine_subpixel(gray, lattice.T).T

        return lattice

    points = np.array([c[2] for c in best], dtype=float)

    if subpixel:
        points = _refine_subpixel(gray, points)

    # The corners are returned as a (2, N) array, like everywhere else in
    # the toolbox.
    return order_corners(points, nx, ny).T


def _detect_corners_by_features(gray: np.ndarray, nx: int, ny: int):
    """Detect the corners of a grid pattern with a corner detector.

    This is the fallback used when the squares of the pattern touch each
    other, which makes the blob approach unusable.  The corners are found
    with the Shi-Tomasi detector, and the grid is recovered by fitting a
    lattice to them.

    Parameters
    ----------
    gray : numpy.ndarray
        The grayscale image.
    nx, ny : int
        The number of corners along x and y.

    Returns
    -------
    numpy.ndarray or None
        The corners as a ``(2, N)`` array, or ``None`` when they could not
        be found.
    """
    import cv2

    features = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=8 * nx * ny,
        qualityLevel=0.005,
        minDistance=4.0,
        useHarrisDetector=False,
    )

    if features is None:
        return None

    points = features.reshape(-1, 2).astype(float)

    return _snap_to_lattice(points, nx, ny)


def _separate_blobs(binary: np.ndarray, expected: int):
    """Split the merged blobs of a pattern into individual ones.

    When the calibration pattern is seen at an angle, the squares touch
    each other and the thresholded image has fewer blobs than the pattern
    has corners.  The blobs are separated with a watershed, using the
    distance transform as the marker of the different basins.

    Parameters
    ----------
    binary : numpy.ndarray
        The binary image.
    expected : int
        The expected number of corners.

    Returns
    -------
    numpy.ndarray or None
        The separated binary image, or ``None`` when the separation was
        not needed or not possible.
    """
    import cv2

    count, _, _, _ = cv2.connectedComponentsWithStats(binary)
    if count - 1 >= expected:
        # The blobs are already separated.
        return None

    # The distance transform gives the "height" of the blobs; its local
    # maxima are the centres of the individual squares, even when the
    # squares touch each other.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    if distance.max() <= 0:
        return None

    # The seeds are the points that are far enough from the background.
    seeds = None
    for fraction in (0.4, 0.3, 0.2, 0.15, 0.1):
        _, candidate = cv2.threshold(
            distance, fraction * float(distance.max()), 255, cv2.THRESH_BINARY
        )
        candidate = candidate.astype(np.uint8)
        n_seeds, _ = cv2.connectedComponents(candidate)
        if n_seeds - 1 >= expected:
            seeds = candidate
            break

    if seeds is None:
        return None

    # The watershed grows the seeds over the whole binary image, which
    # splits the merged blobs.
    markers = np.zeros((binary.shape[0], binary.shape[1]), dtype=np.int32)
    markers[seeds > 0] = 1

    # Every seed gets its own label.
    n_seeds, labels = cv2.connectedComponents(seeds)
    markers = labels.astype(np.int32)

    # The watershed needs the background to be marked as unknown.
    markers[binary == 0] = -1

    cv2.watershed(binary * 255, markers)

    # The foreground regions keep their seed labels.
    separated = np.zeros(binary.shape, dtype=np.uint8)
    separated[markers > 0] = 255

    return separated


def _threshold_blobs(gray: np.ndarray, invert: bool) -> np.ndarray | None:
    """Threshold an image so that the pattern blobs are the foreground.

    Parameters
    ----------
    gray : numpy.ndarray
        The grayscale image.
    invert : bool
        Whether the blobs are darker than the background.

    Returns
    -------
    numpy.ndarray or None
        The binary image, or ``None`` when the threshold is degenerate.
    """
    import cv2

    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)

    fraction = _foreground_fraction(binary)

    # Otsu's method degenerates when the histogram has a single strong
    # mode (a sparse pattern on a uniform background): the threshold then
    # falls on one side of the whole distribution.  In that case the
    # midpoint between the darkest and the brightest intensity is used
    # instead.
    if fraction > 0.95 or fraction < 1e-4:
        lo = float(gray.min())
        hi = float(gray.max())
        if hi - lo < 20.0:
            return None
        level = (lo + hi) / 2.0
        _, binary = cv2.threshold(gray, level, 255, mode)

    # Close the small gaps that the threshold leaves inside the blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    fraction = _foreground_fraction(binary)
    if fraction > 0.95 or fraction < 1e-4:
        return None

    return binary


def _foreground_fraction(binary: np.ndarray) -> float:
    """Return the fraction of the pixels that are set.

    Notes
    -----
    ``binary.mean()`` on a ``uint8`` array returns the mean of the
    *values* (0 to 255), not the fraction of the pixels that are set, so
    the count is used explicitly.
    """
    return float(np.count_nonzero(binary)) / binary.size


def _blob_candidates(binary: np.ndarray, expected: int):
    """Extract the blob candidates of a binary image.

    Parameters
    ----------
    binary : numpy.ndarray
        The binary image.
    expected : int
        The number of corners the pattern has.

    Returns
    -------
    list or None
        The candidates, sorted by how close their area is to the median
        one, or ``None`` when there are not enough blobs.
    """
    import cv2

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        candidates.append((i, area, centroids[i]))

    if len(candidates) < expected:
        return None

    # The pattern corners are all of a similar size, so the blobs whose
    # area is closest to the median are the corners.
    areas = np.array([c[1] for c in candidates])
    median_area = float(np.median(areas))

    similar = [
        c for c in candidates if 0.3 * median_area <= c[1] <= 3.0 * median_area
    ]
    if len(similar) < expected:
        similar = candidates

    similar.sort(key=lambda c: abs(c[1] - median_area))

    return similar[:expected]


def _detect_grid_by_corners(binary: np.ndarray, nx: int, ny: int):
    """Detect the corners of a grid pattern from the corners of its blobs.

    When the pattern is viewed at an angle, adjacent squares touch and
    merge into bigger blobs, so the blob count no longer matches the
    number of corners.  The corners of the blobs are then detected with a
    Shi-Tomasi corner detector, and the grid is recovered by fitting a
    lattice to them.

    Parameters
    ----------
    binary : numpy.ndarray
        The binary image.
    nx, ny : int
        The number of corners along x and y.

    Returns
    -------
    numpy.ndarray or None
        The corners as a ``(2, N)`` array, or ``None`` when they could not
        be found.
    """
    import cv2

    # The corners of the blobs are good features: they are corners of a
    # high contrast region.
    features = cv2.goodFeaturesToTrack(
        binary,
        maxCorners=4 * nx * ny,
        qualityLevel=0.005,
        minDistance=2.0,
        useHarrisDetector=False,
    )

    if features is None:
        return None

    points = features.reshape(-1, 2).astype(float)

    # The grid is recovered by fitting a lattice to the points.
    return _snap_to_lattice(points, nx, ny)


def _snap_to_lattice(points: np.ndarray, nx: int, ny: int):
    """Fit a regular lattice to a set of points and return its nodes.

    The two directions of the grid are recovered by a RANSAC-like search:
    a candidate direction is a line through a pair of points, and the
    direction that leaves the most points on a few parallel lines wins.
    The points are then projected onto the two directions and clustered
    into levels, which gives the rows and the columns.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 2)
        The points.
    nx, ny : int
        The number of nodes along the two axes.

    Returns
    -------
    numpy.ndarray or None
        The nodes of the lattice as a ``(2, nx*ny)`` array, or ``None``
        when the fit failed.
    """
    if points.shape[0] < nx + ny:
        return None

    centre = points.mean(axis=0)
    centred = points - centre

    # The two directions of the grid.
    axis_u = _find_grid_direction(centred, nx)
    axis_v = _find_grid_direction(centred, ny)

    if axis_u is None or axis_v is None:
        return None

    # The two directions must be different enough.
    if abs(float(axis_u @ axis_v)) > 0.9:
        return None

    levels_u = _cluster_levels(centred @ axis_u, nx)
    levels_v = _cluster_levels(centred @ axis_v, ny)

    if levels_u is None or levels_v is None:
        return None

    rows = _refine_levels(centred, axis_v, levels_v)
    cols = _refine_levels(centred, axis_u, levels_u)

    if rows is None or cols is None:
        return None

    nodes = []
    for j in range(ny):
        for i in range(nx):
            uu = cols[i]
            vv = rows[j]
            nodes.append(centre + uu * axis_u + vv * axis_v)

    return np.array(nodes).T


def _find_grid_direction(centred: np.ndarray, count: int):
    """Find one of the two directions of a grid of points.

    The direction is the one along which the points form the fewest,
    most regularly spaced, parallel lines.

    Parameters
    ----------
    centred : numpy.ndarray, shape (N, 2)
        The points, centred.
    count : int
        The expected number of lines along that direction.

    Returns
    -------
    numpy.ndarray or None
        The unit direction, or ``None`` when it could not be found.
    """
    n = centred.shape[0]
    if n < 3:
        return None

    # The candidates: the directions through pairs of points that are
    # close to each other (the neighbours of the grid).
    best = None
    best_score = -np.inf

    # The pairs are taken from a random subset, so that the search stays
    # fast for a large number of points.
    rng = np.random.default_rng(0)
    n_try = min(400, n * 4)

    for _ in range(n_try):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue

        delta = centred[j] - centred[i]
        norm = np.linalg.norm(delta)
        if norm < 1e-6 or norm > 0.5 * np.ptp(centred, axis=0).max():
            continue

        direction = delta / norm

        # The score: how well the points fall on a few parallel lines.
        score = _line_score(centred, direction, count)
        if score > best_score:
            best_score = score
            best = direction

    if best is None:
        return None

    # The sign does not matter.
    return best / np.linalg.norm(best)


def _line_score(centred: np.ndarray, direction: np.ndarray, count: int) -> float:
    """Score a direction by how regularly the points fall on parallel lines.

    Parameters
    ----------
    centred : numpy.ndarray, shape (N, 2)
        The points, centred.
    direction : numpy.ndarray, shape (2,)
        The direction of the lines.
    count : int
        The expected number of lines.

    Returns
    -------
    float
        The score; the higher the better.
    """
    normal = np.array([-direction[1], direction[0]])

    projections = centred @ normal

    # A first estimate of the spacing between the lines.
    step = (projections.max() - projections.min()) / max(count - 1, 1)
    if step <= 0:
        return -np.inf

    # The distance of every point to the nearest line of the grid.
    index = np.round((projections - projections.min()) / step)
    residuals = projections - (projections.min() + index * step)

    # The score is the number of points that are close to a line, minus a
    # penalty for the spread of the residuals.
    inliers = np.count_nonzero(np.abs(residuals) < 0.35 * step)
    spread = float(np.std(residuals))

    return inliers - 0.5 * spread


def _cluster_levels(values: np.ndarray, count: int):
    """Cluster a set of projections into ``count`` levels.

    Parameters
    ----------
    values : numpy.ndarray
        The projections.
    count : int
        The expected number of levels.

    Returns
    -------
    numpy.ndarray or None
        The position of each level, or ``None`` when the clustering
        failed.
    """
    if values.size < count:
        return None

    ordered = np.sort(values)

    # A first estimate of the spacing: the span divided by the number of
    # gaps.
    step = (ordered[-1] - ordered[0]) / max(count - 1, 1)
    if step <= 0:
        return None

    # Assign every point to the nearest level of the initial grid.
    origin = ordered[0]
    index = np.round((ordered - origin) / step).astype(int)

    # Keep the levels that hold enough points; the others are artefacts.
    levels = []
    for i in range(index.max() + 1):
        members = ordered[index == i]
        if members.size >= 1:
            levels.append(float(members.mean()))

    if len(levels) < count:
        return None

    # Keep the ``count`` levels that are the most populated.
    if len(levels) > count:
        # The levels are trimmed symmetrically, so that the pattern stays
        # centred.
        extra = len(levels) - count
        keep = np.arange(len(levels))
        keep = keep[extra // 2 : len(levels) - (extra - extra // 2)]
        levels = [levels[i] for i in keep]

    return np.array(levels)


def _refine_levels(centred: np.ndarray, axis: np.ndarray, levels: np.ndarray):
    """Refine the position of the grid lines along an axis.

    The position of each line is the mean of the projection of the points
    that belong to it, which is more accurate than the level of the
    initial clustering.

    Parameters
    ----------
    centred : numpy.ndarray, shape (N, 2)
        The points, centred.
    axis : numpy.ndarray, shape (2,)
        The direction of the lines.
    levels : numpy.ndarray
        The initial position of the lines.

    Returns
    -------
    numpy.ndarray or None
        The refined positions.
    """
    projections = centred @ axis

    refined = []
    for level in levels:
        members = projections[np.abs(projections - level) < 0.5]
        if members.size == 0:
            refined.append(float(level))
        else:
            refined.append(float(members.mean()))

    return np.array(refined)


def _to_gray(image: np.ndarray) -> np.ndarray | None:
    """Convert an image to grayscale, or return ``None`` if it is empty."""
    if image is None:
        return None
    image = np.asarray(image)
    if image.size == 0:
        return None
    if image.ndim == 3:
        import cv2

        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        return image[:, :, 0]
    return image


def _refine_subpixel(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Refine the corner positions to sub-pixel accuracy.

    The refinement uses the gradient of the intensity: the corner is
    located where the intensity is extremal in both directions, which is
    found by solving a small linear system built from the second order
    derivatives.
    """
    import cv2

    out = points.copy()
    h, w = gray.shape[:2]

    for i, (x, y) in enumerate(points):
        xi, yi = int(round(x)), int(round(y))
        if not (2 <= xi < w - 3 and 2 <= yi < h - 3):
            continue

        window = gray[yi - 2 : yi + 3, xi - 2 : xi + 3].astype(float)

        # The derivatives of the intensity.
        gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        gxx = cv2.Sobel(window, cv2.CV_64F, 2, 0, ksize=3) / 8.0
        gyy = cv2.Sobel(window, cv2.CV_64F, 0, 2, ksize=3) / 8.0
        gxy = cv2.Sobel(window, cv2.CV_64F, 1, 1, ksize=3) / 8.0

        # The Harris-like refinement: minimize the quadratic form of the
        # gradient around the corner.
        try:
            A = np.array([[gxx[2, 2], gxy[2, 2]], [gxy[2, 2], gyy[2, 2]]])
            b = np.array([gx[2, 2], gy[2, 2]])
            delta = np.linalg.solve(A, -b)
        except np.linalg.LinAlgError:
            continue

        if not np.all(np.isfinite(delta)):
            continue

        # The refinement must stay in the neighbourhood of the corner.
        if np.linalg.norm(delta) > 2.0:
            continue

        out[i, 0] = x + delta[0]
        out[i, 1] = y + delta[1]

    return out


class CornerPicker:
    """State of an interactive corner picking session.

    This class holds the state of the picking; the Qt widget that
    displays the image and forwards the mouse events lives in
    :mod:`calib.gui.widgets`.

    Parameters
    ----------
    nx, ny : int
        The number of corners of the pattern along x and y.
    """

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.points: list[tuple[float, float]] = []

    @property
    def n_expected(self) -> int:
        """The number of corners the pattern has."""
        return self.nx * self.ny

    @property
    def is_complete(self) -> bool:
        """``True`` when every corner has been picked."""
        return len(self.points) >= self.n_expected

    def add(self, x: float, y: float) -> None:
        """Add a picked corner."""
        if not self.is_complete:
            self.points.append((float(x), float(y)))

    def undo(self) -> None:
        """Remove the last picked corner."""
        if self.points:
            self.points.pop()

    def clear(self) -> None:
        """Remove every picked corner."""
        self.points.clear()

    def as_array(self) -> np.ndarray:
        """Return the picked corners as a ``(2, N)`` array.

        Returns
        -------
        numpy.ndarray
            The corners, or an empty ``(2, 0)`` array when none were
            picked.
        """
        if not self.points:
            return np.zeros((2, 0))
        return np.array(self.points, dtype=float).T
