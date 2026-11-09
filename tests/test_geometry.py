"""Tests for the geometry primitives (rodrigues, rigid motion, projections).

The analytic Jacobians are verified against finite differences, mirroring the
self-tests embedded in the original MATLAB files.
"""

from __future__ import annotations

import numpy as np
import pytest

from calib.geometry import (
    apply_distortion,
    project_points,
    project_points_with_jacobians,
    rigid_motion,
    rigid_motion_with_jacobians,
    rodrigues,
    rodrigues_jacobian,
)


def test_rodrigues_identity():
    R = rodrigues([0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3))


def test_rodrigues_roundtrip():
    """A rotation vector maps to a proper rotation matrix."""
    om = np.array([0.3, -0.2, 1.1])
    R = rodrigues(om)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    # Rodrigues' formula with the unit rotation axis omega:
    #   R = I + sin(t) [omega]_x + (1-cos(t)) [omega]_x^2
    theta = np.linalg.norm(om)
    w = om / theta
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    expected = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    assert np.allclose(R, expected, atol=1e-12)


def test_rodrigues_rotation_angle():
    """The rotation angle equals the norm of the rotation vector."""
    om = np.array([0.1, 0.2, 0.3])
    R = rodrigues(om)
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    assert np.isclose(angle, np.linalg.norm(om))


def test_rodrigues_jacobian_matches_finite_difference():
    """dR/domega verified against a central finite difference."""
    om = np.array([0.3, -0.2, 1.1])
    J = rodrigues_jacobian(om)

    eps = 1e-7
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        R_plus = rodrigues(om + d)
        R_minus = rodrigues(om - d)
        num = (R_plus - R_minus) / (2 * eps)
        ana = J.reshape(3, 3, 3, order="F")[:, :, k]
        assert np.allclose(num, ana, atol=1e-6)


def test_rigid_motion_matches_manual():
    X = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    om = np.array([0.0, 0.0, np.pi / 2])
    T = np.array([1.0, 2.0, 3.0])
    Y = rigid_motion(X, om, T)
    R = rodrigues(om)
    assert np.allclose(Y, R @ X + T.reshape(3, 1))


def test_rigid_motion_jacobians():
    """dYdom and dYdT verified against finite differences."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 7))
    om = rng.normal(size=3)
    T = rng.normal(size=3)

    _, dYdom, dYdT = rigid_motion_with_jacobians(X, om, T)

    eps = 1e-7
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        Y_plus = rigid_motion(X, om + d, T)
        Y_minus = rigid_motion(X, om - d, T)
        num = (Y_plus - Y_minus).reshape(-1, order="F") / (2 * eps)
        # The Jacobian stacks dY/dX(1), dY/dX(2), dY/dX(3) per point.
        ana = np.zeros(3 * X.shape[1])
        for i in range(X.shape[1]):
            ana[3 * i : 3 * i + 3] = dYdom[3 * i : 3 * i + 3, k]
        assert np.allclose(num, ana, atol=1e-5)

        T_plus = rigid_motion(X, om, T + d)
        T_minus = rigid_motion(X, om, T - d)
        num_T = (T_plus - T_minus).reshape(-1, order="F") / (2 * eps)
        ana_T = np.zeros(3 * X.shape[1])
        for i in range(X.shape[1]):
            ana_T[3 * i : 3 * i + 3] = dYdT[3 * i : 3 * i + 3, k]
        assert np.allclose(num_T, ana_T, atol=1e-5)


def test_project_points_no_distortion_matches_pinhole():
    """With no distortion the projection is the plain pinhole model."""
    X = np.array([[0.1, 0.2], [-0.1, 0.3], [1.0, 2.0]])
    om = np.array([0.0, 0.0, 0.0])
    T = np.array([0.0, 0.0, 0.0])
    f = np.array([1000.0, 1000.0])
    c = np.array([320.0, 240.0])

    xp = project_points(X, om, T, f, c, np.zeros(5), 0.0)
    assert np.allclose(xp[0], 1000.0 * X[0] / X[2] + 320.0)
    assert np.allclose(xp[1], 1000.0 * X[1] / X[2] + 240.0)


def test_project_points_skew():
    """The skew mixes the y coordinate into x."""
    X = np.array([[0.1], [0.2], [1.0]])
    om = np.zeros(3)
    T = np.zeros(3)
    f = np.array([1000.0, 1000.0])
    c = np.array([320.0, 240.0])
    alpha = 0.01

    xp = project_points(X, om, T, f, c, np.zeros(5), alpha)
    assert np.isclose(xp[0, 0], 1000.0 * (0.1 + alpha * 0.2) + 320.0)


def test_project_points_jacobians_om():
    """The Jacobian w.r.t. om verified against finite differences."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 6)) + np.array([[0.0], [0.0], [5.0]])
    om = rng.normal(size=3) * 0.2
    T = rng.normal(size=3)
    f = np.array([900.0, 880.0])
    c = np.array([320.0, 240.0])
    k = np.array([0.1, -0.02, 0.005, 0.003, 0.0])
    alpha = 0.01

    xp, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = project_points_with_jacobians(
        X, om, T, f, c, k, alpha
    )

    eps = 1e-7
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        xp_p = project_points(X, om + d, T, f, c, k, alpha)
        xp_m = project_points(X, om - d, T, f, c, k, alpha)
        num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
        assert np.allclose(num, dxdom[:, i], atol=1e-4)


def test_project_points_jacobians_T():
    """The Jacobian w.r.t. T verified against finite differences."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 6)) + np.array([[0.0], [0.0], [5.0]])
    om = rng.normal(size=3) * 0.2
    T = rng.normal(size=3)
    f = np.array([900.0, 880.0])
    c = np.array([320.0, 240.0])
    k = np.array([0.1, -0.02, 0.005, 0.003, 0.0])

    _, _, dxdT, _, _, _, _ = project_points_with_jacobians(X, om, T, f, c, k, 0.0)

    eps = 1e-7
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        xp_p = project_points(X, om, T + d, f, c, k, 0.0)
        xp_m = project_points(X, om, T - d, f, c, k, 0.0)
        num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
        assert np.allclose(num, dxdT[:, i], atol=1e-4)


def test_project_points_jacobians_fc():
    """The Jacobians w.r.t. f, c, k and alpha verified against finite differences."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 5)) + np.array([[0.0], [0.0], [5.0]])
    om = rng.normal(size=3) * 0.2
    T = rng.normal(size=3)
    f = np.array([900.0, 880.0])
    c = np.array([320.0, 240.0])
    k = np.array([0.1, -0.02, 0.005, 0.003, 0.0])
    alpha = 0.01

    _, _, _, dxdf, dxdc, dxdk, dxdalpha = project_points_with_jacobians(
        X, om, T, f, c, k, alpha
    )

    eps = 1e-6

    for i in range(2):
        df = np.zeros(2)
        df[i] = eps
        xp_p = project_points(X, om, T, f + df, c, k, alpha)
        xp_m = project_points(X, om, T, f - df, c, k, alpha)
        num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
        assert np.allclose(num, dxdf[:, i], atol=1e-4)

        dc = np.zeros(2)
        dc[i] = eps
        xp_p = project_points(X, om, T, f, c + dc, k, alpha)
        xp_m = project_points(X, om, T, f, c - dc, k, alpha)
        num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
        assert np.allclose(num, dxdc[:, i], atol=1e-4)

    for i in range(5):
        dk = np.zeros(5)
        dk[i] = eps
        xp_p = project_points(X, om, T, f, c, k + dk, alpha)
        xp_m = project_points(X, om, T, f, c, k - dk, alpha)
        num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
        assert np.allclose(num, dxdk[:, i], atol=1e-4)

    xp_p = project_points(X, om, T, f, c, k, alpha + eps)
    xp_m = project_points(X, om, T, f, c, k, alpha - eps)
    num = ((xp_p - xp_m) / (2 * eps)).reshape(-1, order="F")
    assert np.allclose(num, dxdalpha[:, 0], atol=1e-4)


def test_apply_distortion_radial_only():
    """With p1 = p2 = 0 the model reduces to pure radial distortion."""
    x = np.array([[0.1, 0.2], [0.3, -0.4]])
    k = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    xd = apply_distortion(x, k)
    r2 = x[0] ** 2 + x[1] ** 2
    expected = x * (1.0 + 0.5 * r2).reshape(1, -1)
    assert np.allclose(xd, expected)


def test_apply_distortion_pads_short_vectors():
    """A short coefficient vector is zero padded."""
    x = np.array([[0.1], [0.2]])
    a = apply_distortion(x, np.array([0.5]))
    b = apply_distortion(x, np.array([0.5, 0.0, 0.0, 0.0, 0.0]))
    assert np.allclose(a, b)


@pytest.mark.parametrize("shape", [(2, 5), (4, 5)])
def test_project_points_shape_validation(shape):
    with pytest.raises(ValueError):
        project_points(np.zeros(shape), np.zeros(3), np.zeros(3), [1, 1], [0, 0])
