import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from e2l_common.geometry import (
    average_rotations,
    compose,
    delta_action,
    interpolate_poses,
    inv_T,
    make_T,
    matrix_to_rotvec,
    rotvec_to_matrix,
)

RNG = np.random.default_rng(0)


def random_T(n: int | None = None) -> np.ndarray:
    size = 1 if n is None else n
    T = make_T(Rotation.random(size, random_state=RNG).as_matrix(), RNG.normal(size=(size, 3)))
    return T[0] if n is None else T


def test_make_T_layout():
    R = rotvec_to_matrix(np.array([0, 0, np.pi / 2]))
    T = make_T(R, [1, 2, 3])
    assert T.shape == (4, 4) and T.dtype == np.float64
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    # p_a = T_a_b @ p_b: x axis of b maps to y axis of a, offset by t
    np.testing.assert_allclose(T @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)


def test_inverse_round_trip():
    T = random_T(10)
    np.testing.assert_allclose(
        compose(T, inv_T(T)), np.broadcast_to(np.eye(4), T.shape), atol=1e-12
    )
    np.testing.assert_allclose(inv_T(inv_T(T)), T, atol=1e-12)
    np.testing.assert_allclose(inv_T(T), np.linalg.inv(T), atol=1e-12)


def test_compose_chain_frames():
    T_a_b, T_b_c = random_T(), random_T()
    p_c = np.array([0.1, -0.2, 0.3, 1.0])
    np.testing.assert_allclose(compose(T_a_b, T_b_c) @ p_c, T_a_b @ (T_b_c @ p_c), atol=1e-12)


def test_rotvec_round_trip():
    rv = Rotation.random(20, random_state=RNG).as_rotvec()
    np.testing.assert_allclose(matrix_to_rotvec(rotvec_to_matrix(rv)), rv, atol=1e-10)
    assert rotvec_to_matrix(rv[0]).shape == (3, 3)


def test_interpolate_endpoints_and_midpoint():
    T0 = make_T(np.eye(3), [0, 0, 0])
    T1 = make_T(rotvec_to_matrix(np.array([0, 0, 1.0])), [1, 2, 0])
    out = interpolate_poses([0.0, 1.0], np.stack([T0, T1]), [0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(out[0], T0, atol=1e-12)
    np.testing.assert_allclose(out[2], T1, atol=1e-12)
    np.testing.assert_allclose(out[3], T1, atol=1e-12)  # clamped
    np.testing.assert_allclose(out[1, :3, 3], [0.5, 1.0, 0])
    np.testing.assert_allclose(matrix_to_rotvec(out[1, :3, :3]), [0, 0, 0.5], atol=1e-12)


def test_average_rotations_symmetric():
    Rs = rotvec_to_matrix(np.array([[0, 0, 0.2], [0, 0, -0.2]]))
    np.testing.assert_allclose(average_rotations(Rs), np.eye(3), atol=1e-12)


def test_delta_action_reconstructs_next_pose():
    T_prev, T_next = random_T(), random_T()
    d = delta_action(T_prev, T_next)
    assert d.shape == (6,)
    np.testing.assert_allclose(T_prev[:3, 3] + d[:3], T_next[:3, 3], atol=1e-12)
    np.testing.assert_allclose(rotvec_to_matrix(d[3:]) @ T_prev[:3, :3], T_next[:3, :3], atol=1e-10)


def test_delta_action_identity_and_batch():
    T = random_T(5)
    np.testing.assert_allclose(delta_action(T, T), np.zeros((5, 6)), atol=1e-12)


def test_compose_requires_args():
    with pytest.raises(ValueError):
        compose()
