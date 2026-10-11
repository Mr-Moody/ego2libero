"""Workspace and speed checks for retargeted segments."""

import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.geometry import matrix_to_rotvec


def check_segment(T_obj_ee: np.ndarray, t: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """Per-frame feasibility (M,) bool for gripper poses in the reference object's frame
    (M,4,4) at times t (M,) seconds: not below `cfg.min_height` (object frames sit on the table,
    z up), within `cfg.max_object_distance` of the object, and reached from the previous frame
    within `cfg.max_linear_speed` (m/s) and `cfg.max_angular_speed` (rad/s). IK reachability
    is left to replay."""
    p = T_obj_ee[:, :3, 3]
    ok = (p[:, 2] >= cfg.min_height) & (np.linalg.norm(p, axis=1) <= cfg.max_object_distance)
    if len(p) > 1:
        dt = np.diff(t)
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
        R = T_obj_ee[:, :3, :3]
        dR = np.swapaxes(R[:-1], 1, 2) @ R[1:]
        w = np.linalg.norm(matrix_to_rotvec(dR), axis=1) / dt
        ok[1:] &= (v <= cfg.max_linear_speed) & (w <= cfg.max_angular_speed)
    return ok
