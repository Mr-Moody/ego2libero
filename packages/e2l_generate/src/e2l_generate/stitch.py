"""Transit motions between transformed segments."""

import numpy as np

from e2l_common.stub import not_implemented


def transit(T_sim_ee_from: np.ndarray, T_sim_ee_to: np.ndarray, steps: int) -> np.ndarray:
    """Interpolated poses (steps,4,4) from one end-effector pose to another (linear position,
    slerp rotation), excluding the start and including the end."""
    not_implemented("e2l_generate.stitch.transit")


def stitch(segments: list[np.ndarray], grippers: list[np.ndarray], steps: int):
    """Join transformed segments (each (M_i,4,4) T_sim_ee) with transits, returning the full
    target trajectory (T,4,4) and gripper commands (T,) in robosuite convention."""
    not_implemented("e2l_generate.stitch.stitch")
