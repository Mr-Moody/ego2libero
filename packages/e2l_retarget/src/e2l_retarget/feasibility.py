"""Workspace, speed and reachability checks for retargeted segments."""

import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.stub import not_implemented


def check_segment(T_table_ee: np.ndarray, t: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """Per-frame feasibility (M,) bool for end-effector poses in the table frame (M,4,4) at
    times t (M,): inside `cfg.workspace_min/max`, under the linear/angular speed limits, and
    reachable by the Panda (IK through `e2l_sim`)."""
    not_implemented("e2l_retarget.feasibility.check_segment")
