"""Place objects in the simulator from table-frame poses."""

import numpy as np
from scipy.spatial.transform import Rotation

from e2l_common.geometry import make_T


def _problem_env(env):
    """LiberoEnv -> OffScreenRenderEnv -> LIBERO problem env (holds sim and objects_dict)."""
    return env._env.env


def object_names(env) -> list[str]:
    """The task's movable objects (free-jointed bodies), in LIBERO order."""
    return list(_problem_env(env).objects_dict)


def object_poses(env) -> dict[str, np.ndarray]:
    """T_sim_obj (4,4) of every movable object: its MuJoCo body frame in the world."""
    inner = _problem_env(env)
    poses = {}
    for name in inner.objects_dict:
        body = inner.obj_body_id[name]
        quat_wxyz = inner.sim.data.body_xquat[body]
        R = Rotation.from_quat(quat_wxyz, scalar_first=True).as_matrix()
        poses[name] = make_T(R, inner.sim.data.body_xpos[body])
    return poses


def set_object_poses(env, T_sim_obj: dict[str, np.ndarray]) -> None:
    """Teleport each named object's free joint to T_sim_obj (4,4); quaternions are converted to
    `quat_wxyz` only at the MuJoCo call site."""
    inner = _problem_env(env)
    for name, T in T_sim_obj.items():
        if name not in inner.objects_dict:
            raise KeyError(f"{name!r} is not a movable object; have {list(inner.objects_dict)}")
        joint = inner.objects_dict[name].joints[-1]
        quat_wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
        inner.sim.data.set_joint_qpos(joint, np.concatenate([T[:3, 3], quat_wxyz]))
        inner.sim.data.set_joint_qvel(joint, np.zeros(6))
    inner.sim.forward()


def table_to_sim(T_table_x: np.ndarray, T_sim_table: np.ndarray) -> np.ndarray:
    """T_sim_x = T_sim_table @ T_table_x for (...,4,4) poses."""
    return np.asarray(T_sim_table, dtype=np.float64) @ np.asarray(T_table_x, dtype=np.float64)
