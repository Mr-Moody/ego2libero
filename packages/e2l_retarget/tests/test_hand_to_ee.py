import numpy as np

from e2l_common.config import Pose, RetargetConfig
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_retarget.hand_to_ee import hand_to_ee, top_down_rotations

R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])  # scripted demo's grip-site rotation
# Hand axes in the object frame: fingers (x) along +y, palm normal (z) down, so the hand's y
# (across the knuckles) lies along the object's x.
R_HAND = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def test_top_down_from_horizontal_closing_axis():
    R = top_down_rotations(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(R[0], R_TOP_DOWN, atol=1e-12)
    np.testing.assert_allclose(R[1][:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    for r in R:
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)
        np.testing.assert_allclose(r[:, 2], [0.0, 0.0, -1.0])


def test_top_down_keeps_x_continuous_and_handles_vertical_axis():
    closing = np.array([[-1.0, 0.1, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    R = top_down_rotations(closing)
    assert np.isfinite(R).all()
    x = R[:, :, 0]
    assert x[0] @ [1.0, 0.0, 0.0] > 0  # first frame flipped towards object x
    assert all(x[i] @ x[i - 1] >= 0 for i in range(1, len(x)))  # no half-turn jumps
    np.testing.assert_allclose(x[2], x[1])  # near-vertical axis: previous x reused


def test_hand_to_ee_applies_offset_in_hand_frame_and_snaps_rotation():
    tilt = rotvec_to_matrix(np.array([0.3, 0.0, 0.0]))  # roll about object x keeps hand y on x
    R = tilt @ R_HAND
    T_obj_hand = make_T(R, np.array([0.05, 0.0, 0.15]))[None]
    cfg = RetargetConfig(T_hand_ee=Pose(xyz=(0.09, 0.0, 0.02)), closing_axis=(0.0, 1.0, 0.0))
    T = hand_to_ee(T_obj_hand, cfg)
    np.testing.assert_allclose(T[0, :3, 3], T_obj_hand[0, :3, 3] + R @ [0.09, 0.0, 0.02])
    np.testing.assert_allclose(T[0, :3, :3], R_TOP_DOWN, atol=1e-12)


def test_hand_to_ee_without_canonicalisation_composes_offset():
    T_obj_hand = make_T(R_HAND, np.array([0.0, 0.1, 0.2]))[None]
    off = Pose(xyz=(0.0, 0.0, 0.1), rotvec=(0.0, 0.0, 0.5))
    T = hand_to_ee(T_obj_hand, RetargetConfig(T_hand_ee=off, canonicalise_top_down=False))
    np.testing.assert_allclose(T[0], T_obj_hand[0] @ off.T)
