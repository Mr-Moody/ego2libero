import importlib.util
import os

import numpy as np
import pytest

from e2l_common.config import Pose, RetargetConfig, SimConfig, load_config
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_common.paths import repo_root
from e2l_common.schemas import HandSegment, HandSegments, RobotSegments
from e2l_retarget.hand_to_ee import retarget_segments
from e2l_sim.libero_setup import libero_available

# Hand axes in the object frame: fingers along +y, palm down, knuckle axis (hand y) along x.
R_HAND = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
CFG = RetargetConfig(T_hand_ee=Pose(xyz=(0.09, 0.0, 0.02)), closing_axis=(0.0, 1.0, 0.0))


def _scripted() -> RobotSegments:
    path = repo_root() / "scripts" / "scripted_segments.py"
    spec = importlib.util.spec_from_file_location("scripted_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scripted("20261011_retarget_000")


def _as_hand(robot: RobotSegments) -> HandSegments:
    """Hand segments a person could have shown: the hand rolls up to 0.3 rad about the object x
    while its knuckle axis stays along x, and its grip point follows the scripted gripper."""
    off = np.asarray(CFG.T_hand_ee.xyz)
    hand = []
    for s in robot.segments:
        m = len(s.T_obj_ee)
        roll = np.zeros((m, 3))
        roll[:, 0] = np.linspace(-0.3, 0.3, m)
        R = rotvec_to_matrix(roll) @ R_HAND
        p = s.T_obj_ee[:, :3, 3] - np.einsum("nij,j->ni", R, off)
        hand.append(
            HandSegment(
                ref_object=s.ref_object,
                t_start=0.0,
                t_end=(m - 1) / robot.fps,
                T_obj_hand=make_T(R, p),
                grip=(s.gripper > 0).astype(np.int8),
            )
        )
    return HandSegments(demo_id=robot.demo_id, fps=robot.fps, segments=hand)


def test_retarget_recovers_scripted_gripper_poses():
    scripted = _scripted()
    robot = retarget_segments(_as_hand(scripted), CFG)
    for got, want in zip(robot.segments, scripted.segments, strict=True):
        assert got.ref_object == want.ref_object
        np.testing.assert_allclose(got.T_obj_ee, want.T_obj_ee, atol=1e-9)
        np.testing.assert_array_equal(got.gripper, want.gripper)
        assert got.feasible.all()


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")
def test_retargeted_hand_demo_replays_successfully():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.replay import replay

    robot = retarget_segments(_as_hand(_scripted()), CFG)
    episode = replay(robot, load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig))
    assert episode.success
