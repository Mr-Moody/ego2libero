"""Execute RobotSegments in LIBERO and record a SimEpisode."""

import numpy as np

from e2l_common.config import SimConfig
from e2l_common.geometry import interpolate_poses
from e2l_common.schemas import RobotSegment, RobotSegments, SimEpisode
from e2l_sim.control import GRIPPER_OPEN, ee_pose, pose_error, track
from e2l_sim.env import make_env
from e2l_sim.scene import object_poses, set_object_poses

SETTLE_STEPS = 10  # no-op steps after teleporting objects, as LiberoEnv does after reset
TRANSIT_ROT_SPEED = 1.5  # rad/s


def libero_state(obs: dict) -> np.ndarray:
    """8-D state as in the LIBERO datasets: eef pos (3), eef axis-angle (3), gripper qpos (2).

    The axis-angle follows robosuite's `quat2axisangle` (angle = 2 acos w, not wrapped to pi),
    which is what LeRobot's LIBERO processor computes at evaluation time."""
    eef = obs["robot_state"]["eef"]
    quat_xyzw = np.asarray(eef["quat"], dtype=np.float64)
    w = np.clip(quat_xyzw[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    axis_angle = np.zeros(3) if np.isclose(den, 0.0) else quat_xyzw[:3] * 2.0 * np.arccos(w) / den
    return np.concatenate([eef["pos"], axis_angle, obs["robot_state"]["gripper"]["qpos"]])


def resample_segment(seg: RobotSegment, fps: float, control_freq: float):
    """T_obj_ee (M,4,4) and gripper (M,) at the demo fps -> the control rate."""
    t = np.arange(len(seg.T_obj_ee)) / fps
    t_query = np.arange(0.0, t[-1] + 1e-9, 1.0 / control_freq)
    T = interpolate_poses(t, seg.T_obj_ee, t_query)
    idx = np.minimum(np.round(t_query * fps).astype(int), len(seg.gripper) - 1)
    return T, np.asarray(seg.gripper, dtype=np.float64)[idx]


def transit(T_from: np.ndarray, T_to: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Straight-line (slerp in rotation) targets from T_from to T_to at `cfg.transit_speed`;
    excludes T_from, includes T_to. (K,4,4) with K >= 1."""
    dist, angle = pose_error(T_from, T_to)
    dt = 1.0 / cfg.control_freq
    k = max(1, int(np.ceil(max(dist / (cfg.transit_speed * dt), angle / (TRANSIT_ROT_SPEED * dt)))))
    return interpolate_poses(
        np.array([0.0, 1.0]), np.stack([T_from, T_to]), np.arange(1, k + 1) / k
    )


def hold_at_grip_changes(T: np.ndarray, grip: np.ndarray, steps: int):
    """Repeat the target `steps` extra times wherever the gripper command flips, so the
    fingers finish closing (or opening) before the arm moves on."""
    flips = np.flatnonzero(np.diff(np.sign(grip)) != 0) + 1
    reps = np.ones(len(T), dtype=int)
    reps[flips] += steps
    return np.repeat(T, reps, axis=0), np.repeat(grip, reps)


def sim_object(name: str, cfg: SimConfig) -> str:
    return cfg.object_map.get(name, name)


def replay(
    segments: RobotSegments,
    cfg: SimConfig,
    T_sim_obj: dict[str, np.ndarray] | None = None,
    episode_index: int = 0,
    stop_on_success: bool = True,
) -> SimEpisode:
    """Place objects at their nominal sim poses, express each segment in the sim frame via its
    reference object, track it with `control.track`, and record images, state, actions and the
    task success flag.

    Nominal poses come from the LIBERO init state `episode_index`; `T_sim_obj` (LIBERO object
    name -> (4,4)) overrides some or all of them. Each segment is anchored to its reference
    object's pose at the moment the segment starts, as in MimicGen, and is reached from the
    current grip-site pose by a straight `transit`."""
    env = make_env(cfg, episode_index=episode_index)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        if T_sim_obj:
            set_object_poses(env, T_sim_obj)
            for _ in range(SETTLE_STEPS):
                obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, GRIPPER_OPEN], dtype=np.float32))
        initial_poses = object_poses(env)

        observations, actions, success = [], [], False
        gripper = GRIPPER_OPEN
        for seg in segments.segments:
            budget = cfg.max_steps - len(actions)
            if budget <= 0 or (success and stop_on_success):
                break
            name = sim_object(seg.ref_object, cfg)
            T_obj_ee, grip = resample_segment(seg, segments.fps, cfg.control_freq)
            T_sim_ee = object_poses(env)[name] @ T_obj_ee
            T_lead = transit(ee_pose(obs), T_sim_ee[0], cfg)
            targets = np.concatenate([T_lead, T_sim_ee])
            grips = np.concatenate([np.full(len(T_lead), gripper), grip])
            targets, grips = hold_at_grip_changes(targets, grips, cfg.gripper_settle_steps)
            result = track(
                env,
                targets,
                grips,
                cfg.max_steps_per_target,
                obs=obs,
                gain=cfg.tracking_gain,
                max_steps=budget,
                stop_on_success=stop_on_success,
            )
            observations += result.observations
            actions += result.actions
            success |= result.success
            obs, gripper = result.final_obs, float(grips[-1])
    finally:
        env.close()

    if not actions:
        raise ValueError(f"{segments.demo_id}: no segments to replay")
    return SimEpisode(
        task=env.task_description,
        success=success,
        source_demo_ids=[segments.demo_id],
        images={
            cam: np.stack([o["pixels"][cam] for o in observations]).astype(np.uint8)
            for cam in observations[0]["pixels"]
        },
        state=np.stack([libero_state(o) for o in observations]),
        actions=np.stack(actions).astype(np.float64),
        object_poses=initial_poses,
    )
