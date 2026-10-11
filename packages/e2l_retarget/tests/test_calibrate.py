import numpy as np
import pytest

from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_common.schemas import DemoTrajectory
from e2l_retarget.calibrate import estimate_grip

THUMB = np.array([0.08, -0.01, -0.02])  # hand frame, metres
INDEX = np.array([0.10, 0.01, -0.02])


def _demo(n: int = 20) -> DemoTrajectory:
    rng = np.random.default_rng(0)
    T = make_T(
        rotvec_to_matrix(rng.normal(scale=0.5, size=(n, 3))), rng.normal(scale=0.1, size=(n, 3))
    )
    kp = np.zeros((n, 21, 3))
    kp[:, 4] = np.einsum("nij,j->ni", T[:, :3, :3], THUMB) + T[:, :3, 3]
    kp[:, 8] = np.einsum("nij,j->ni", T[:, :3, :3], INDEX) + T[:, :3, 3]
    return DemoTrajectory(
        demo_id="20261011_test_000",
        fps=30.0,
        object_names=["bowl"],
        t=np.arange(n) / 30.0,
        T_table_hand=T,
        hand_valid=np.ones(n, bool),
        keypoints_table=kp,
        aperture=np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1),
        T_table_obj=np.tile(np.eye(4), (n, 1, 1, 1)),
        obj_valid=np.ones((n, 1), bool),
    )


def test_estimate_grip_recovers_pinch_point_and_axis_in_hand_frame():
    demo = _demo()
    closed = np.zeros(20, bool)
    closed[5:15] = True
    xyz, axis, n = estimate_grip(demo, closed)
    np.testing.assert_allclose(xyz, (THUMB + INDEX) / 2, atol=1e-9)
    want = (INDEX - THUMB) / np.linalg.norm(INDEX - THUMB)
    np.testing.assert_allclose(axis, want, atol=1e-9)
    assert n == 10


def test_estimate_grip_skips_invalid_and_nan_frames():
    demo = _demo()
    demo.hand_valid[5] = False
    demo.keypoints_table[6, 8] = np.nan
    demo.keypoints_table[7, 4] = 99.0  # garbage, but frame 7 is open so it is not used
    closed = np.zeros(20, bool)
    closed[[5, 6, 8, 9]] = True
    xyz, _, n = estimate_grip(demo, closed)
    assert n == 2
    np.testing.assert_allclose(xyz, (THUMB + INDEX) / 2, atol=1e-9)


def test_estimate_grip_without_closed_frames_raises():
    with pytest.raises(ValueError, match="20261011_test_000"):
        estimate_grip(_demo(), np.zeros(20, bool))
