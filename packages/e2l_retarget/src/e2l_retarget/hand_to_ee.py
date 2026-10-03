"""Hand segments to Panda end-effector segments."""

import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.schemas import HandSegments, RobotSegments
from e2l_common.stub import not_implemented


def hand_to_ee(T_obj_hand: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """T_obj_ee (M,4,4) = T_obj_hand (M,4,4) @ cfg.T_hand_ee.T, optionally canonicalised so the
    gripper approach axis points down (-z of the table frame) for top-down grasps."""
    not_implemented("e2l_retarget.hand_to_ee.hand_to_ee")


def retarget_segments(segments: HandSegments, cfg: RetargetConfig) -> RobotSegments:
    """Map every HandSegment to a RobotSegment: poses via `hand_to_ee`, grip {0,1} to robosuite
    gripper {-1,+1}, and `feasible` from `feasibility.check_segment`."""
    not_implemented("e2l_retarget.hand_to_ee.retarget_segments")
