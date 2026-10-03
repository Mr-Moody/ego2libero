import numpy as np
import pytest

from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_sim.control import POS_SCALE, ROT_SCALE, ee_pose, pose_error, target_to_action, track

R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # LIBERO home grip site


def test_zero_error_gives_zero_motion():
    T = make_T(R_DOWN, [0.1, 0.0, 1.0])
    a = target_to_action(T, T, gripper=1.0)
    assert a.dtype == np.float32 and a.shape == (7,)
    np.testing.assert_allclose(a, [0, 0, 0, 0, 0, 0, 1])


def test_small_error_is_scaled_like_osc_pose():
    T = make_T(R_DOWN, [0.1, 0.0, 1.0])
    rot = rotvec_to_matrix(np.array([0.0, 0.0, 0.1]))
    T_goal = make_T(rot @ R_DOWN, [0.11, -0.02, 1.0])
    a = target_to_action(T, T_goal, gripper=-1.0)
    np.testing.assert_allclose(a[:3], np.array([0.01, -0.02, 0.0]) / POS_SCALE, atol=1e-6)
    np.testing.assert_allclose(a[3:6], np.array([0.0, 0.0, 0.1]) / ROT_SCALE, atol=1e-6)
    np.testing.assert_allclose(target_to_action(T, T_goal, -1.0, gain=2.0)[:6], 2 * a[:6], 1e-6)


def test_large_error_saturates_without_changing_direction():
    T = make_T(R_DOWN, [0.0, 0.0, 1.0])
    a = target_to_action(T, make_T(R_DOWN, [0.4, 0.2, 1.0]), gripper=5.0)
    np.testing.assert_allclose(a[:3], [1.0, 0.5, 0.0], atol=1e-6)
    assert a[6] == 1.0


class LaggingArm:
    """Kinematic stand-in for LiberoEnv: the grip site covers `alpha` of each commanded delta,
    and the episode succeeds once it is within 1 cm of `goal`."""

    def __init__(self, T0: np.ndarray, goal: np.ndarray, alpha: float = 0.2):
        self.T, self.goal, self.alpha, self.steps = T0.copy(), goal, alpha, 0

    def obs(self) -> dict:
        return {"robot_state": {"eef": {"pos": self.T[:3, 3].copy(), "mat": self.T[:3, :3]}}}

    def step(self, action):
        self.steps += 1
        dpos = self.alpha * POS_SCALE * action[:3]
        drot = rotvec_to_matrix(self.alpha * ROT_SCALE * action[3:6])
        self.T = make_T(drot @ self.T[:3, :3], self.T[:3, 3] + dpos)
        success = np.linalg.norm(self.T[:3, 3] - self.goal) < 0.01
        return self.obs(), 0.0, success, False, {"is_success": success, "done": False}


def line_targets(p0, p1, n):
    return make_T(R_DOWN, np.linspace(p0, p1, n))


@pytest.mark.parametrize("gain,max_err", [(1.0, 0.03), (3.0, 0.01)])
def test_track_follows_a_moving_target(gain, max_err):
    targets = line_targets([0.0, 0.0, 1.0], [0.1, 0.0, 0.95], 40)
    arm = LaggingArm(targets[0], goal=np.array([9.0, 9.0, 9.0]))
    result = track(arm, targets, np.ones(40), 1, obs=arm.obs(), gain=gain)
    assert len(result.actions) == len(result.observations) == 40
    assert pose_error(ee_pose(result.final_obs), targets[-1])[0] < max_err


def test_track_waits_for_convergence_and_stops_on_success():
    targets = line_targets([0.0, 0.0, 1.0], [0.05, 0.0, 1.0], 2)
    arm = LaggingArm(targets[0], goal=targets[-1][:3, 3])
    result = track(arm, targets, -np.ones(2), 50, obs=arm.obs(), gain=3.0)
    assert result.success and arm.steps < 50
    assert len(result.actions) == arm.steps


def test_track_respects_step_budget():
    targets = line_targets([0.0, 0.0, 1.0], [0.3, 0.0, 1.0], 30)
    arm = LaggingArm(targets[0], goal=np.array([9.0, 9.0, 9.0]))
    assert len(track(arm, targets, np.ones(30), 5, obs=arm.obs(), max_steps=12).actions) == 12
