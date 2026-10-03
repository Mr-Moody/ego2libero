"""Absolute end-effector targets to LIBERO 7-D delta actions.

LIBERO drives the Panda with robosuite's OSC_POSE controller in delta mode: an action in [-1, 1]
is scaled by `output_max` (0.05 m, 0.5 rad) and added to the current grip-site pose in the world
(`sim`) frame, rotations as R_goal = R(drot) @ R_current. The LIBERO datasets store actions in
that normalised form, so we produce them the same way."""

from dataclasses import dataclass, field

import numpy as np

from e2l_common.geometry import delta_action, make_T

POS_SCALE = 0.05  # m per unit action (OSC_POSE output_max)
ROT_SCALE = 0.5  # rad per unit action
GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0


def ee_pose(obs: dict) -> np.ndarray:
    """T_sim_ee (4,4) of the grip site from a LiberoEnv observation."""
    eef = obs["robot_state"]["eef"]
    return make_T(eef["mat"], eef["pos"])


def _limit(v: np.ndarray) -> np.ndarray:
    """Shrink v uniformly so every component lies in [-1, 1], keeping its direction."""
    peak = np.abs(v).max()
    return v / peak if peak > 1.0 else v


def target_to_action(
    T_sim_ee: np.ndarray, T_sim_ee_target: np.ndarray, gripper: float, gain: float = 1.0
):
    """7-D action (dpos (3), drot axis-angle (3), gripper (1)) moving the end effector from its
    current pose towards the target, scaled and clipped like the LIBERO datasets (OSC_POSE,
    +/-1 normalised). gripper: -1 open, +1 closed.

    `gain` multiplies the pose error: OSC only covers a fraction of a commanded delta within one
    control step, so closed-loop tracking needs gain > 1 to keep up with a moving target."""
    d = gain * delta_action(T_sim_ee, T_sim_ee_target)
    action = np.concatenate(
        [_limit(d[:3] / POS_SCALE), _limit(d[3:] / ROT_SCALE), [np.clip(gripper, -1.0, 1.0)]]
    )
    return action.astype(np.float32)


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """(translation error m, rotation error rad) between two poses in the same frame."""
    d = delta_action(T_a, T_b)
    return float(np.linalg.norm(d[:3])), float(np.linalg.norm(d[3:]))


@dataclass
class TrackResult:
    """What `track` executed: obs[i] was observed before actions[i]; `final_obs` after the last."""

    observations: list[dict] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    success: bool = False
    final_obs: dict | None = None


def track(
    env,
    T_sim_ee_targets: np.ndarray,
    grippers: np.ndarray,
    max_steps_per_target: int,
    obs: dict | None = None,
    gain: float = 1.0,
    pos_tol: float = 0.005,
    rot_tol: float = 0.05,
    max_steps: int | None = None,
    stop_on_success: bool = True,
) -> TrackResult:
    """Closed-loop tracking of a target sequence (T,4,4); returns the executed actions and
    observations.

    Each target gets one env step, plus up to `max_steps_per_target - 1` more while the grip
    site is further than `pos_tol` / `rot_tol` from it. `obs` is the current observation (the
    one returned by the previous reset or step); `gain` goes to `target_to_action`. Stops early
    on task success (unless `stop_on_success` is false) or after `max_steps` env steps."""
    if obs is None:
        raise ValueError("track needs the current observation (from env.reset or env.step)")
    result = TrackResult(final_obs=obs)
    for T_target, grip in zip(T_sim_ee_targets, grippers, strict=True):
        for k in range(max_steps_per_target):
            T_now = ee_pose(obs)
            if k > 0:
                pos_err, rot_err = pose_error(T_now, T_target)
                if pos_err <= pos_tol and rot_err <= rot_tol:
                    break
            if max_steps is not None and len(result.actions) >= max_steps:
                return result
            action = target_to_action(T_now, T_target, grip, gain)
            result.observations.append(obs)
            result.actions.append(action)
            obs, _, terminated, _, info = env.step(action)
            result.final_obs = obs
            result.success |= bool(info.get("is_success", False))
            if (terminated and stop_on_success) or info.get("done", False):
                return result
    return result
