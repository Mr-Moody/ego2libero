"""Place objects in the simulator from table-frame poses."""

import numpy as np

from e2l_common.stub import not_implemented


def set_object_poses(env, T_sim_obj: dict[str, np.ndarray]) -> None:
    """Teleport each named object's free joint to T_sim_obj (4,4); quaternions are converted to
    `quat_wxyz` only at the MuJoCo call site."""
    not_implemented("e2l_sim.scene.set_object_poses")


def table_to_sim(T_table_x: np.ndarray, T_sim_table: np.ndarray) -> np.ndarray:
    """T_sim_x = T_sim_table @ T_table_x for (...,4,4) poses."""
    not_implemented("e2l_sim.scene.table_to_sim")
