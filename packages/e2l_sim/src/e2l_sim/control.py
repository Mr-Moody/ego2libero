"""Absolute end-effector targets to LIBERO 7-D delta actions."""

import numpy as np

from e2l_common.stub import not_implemented


def target_to_action(T_sim_ee: np.ndarray, T_sim_ee_target: np.ndarray, gripper: float):
    """7-D action (dpos (3), drot axis-angle (3), gripper (1)) moving the end effector from its
    current pose towards the target, scaled and clipped like the LIBERO datasets (OSC_POSE,
    +/-1 normalised). gripper: -1 open, +1 closed."""
    not_implemented("e2l_sim.control.target_to_action")


def track(env, T_sim_ee_targets: np.ndarray, grippers: np.ndarray, max_steps_per_target: int):
    """Closed-loop tracking of a target sequence (T,4,4); returns the executed actions and
    observations."""
    not_implemented("e2l_sim.control.track")
