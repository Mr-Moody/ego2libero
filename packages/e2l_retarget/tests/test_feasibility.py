import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_retarget.feasibility import check_segment

CFG = RetargetConfig()
FPS = 30.0


def _line(n: int = 10) -> np.ndarray:
    T = np.tile(make_T(np.diag([1.0, -1.0, -1.0]), np.array([0.05, 0.0, 0.1])), (n, 1, 1))
    T[:, 2, 3] = np.linspace(0.15, 0.05, n)  # 0.1 m over 9 frames = 0.33 m/s
    return T


def test_smooth_approach_is_feasible():
    assert check_segment(_line(), np.arange(10) / FPS, CFG).all()


def test_fast_jump_and_spin_are_flagged_at_the_frame_they_arrive():
    T = _line()
    T[5:, 0, 3] += 0.1  # 3 m/s into frame 5
    T[8:, :3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, 1.0])) @ T[8, :3, :3]  # 30 rad/s
    ok = check_segment(T, np.arange(10) / FPS, CFG)
    assert not ok[5] and not ok[8]
    assert ok[[0, 1, 2, 3, 4, 6, 7, 9]].all()


def test_below_table_and_far_from_object_are_flagged():
    T = _line(3)
    T[1, 2, 3] = -0.05
    T[2, :3, 3] = [0.7, 0.0, 0.1]
    ok = check_segment(T, np.arange(3) / 0.5, CFG)  # slow clock: only the position limits bite
    np.testing.assert_array_equal(ok, [True, False, False])


def test_check_segment_single_frame():
    assert check_segment(_line(1), np.zeros(1), CFG).tolist() == [True]
