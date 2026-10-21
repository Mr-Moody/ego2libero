"""Execute RobotSegments in LIBERO and record a SimEpisode."""

import numpy as np

from e2l_common.config import SimConfig
from e2l_common.geometry import (
    interpolate_poses,
    inv_T,
    make_T,
    matrix_to_rotvec,
    rotvec_to_matrix,
)
from e2l_common.schemas import RobotSegment, RobotSegments, SimEpisode
from e2l_sim.control import GRIPPER_CLOSED, GRIPPER_OPEN, ee_pose, pose_error, track
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
    """T_obj_ee (M,4,4), gripper (M,) and T_obj_held (M,4,4, or None) at the demo fps -> the
    control rate."""
    t = np.arange(len(seg.T_obj_ee)) / fps
    t_query = np.arange(0.0, t[-1] + 1e-9, 1.0 / control_freq)
    if t[-1] - t_query[-1] > 1e-9:  # end on the last frame: it is often the grasp or release
        t_query = np.append(t_query, t[-1])
    T = interpolate_poses(t, seg.T_obj_ee, t_query)
    held = None if seg.T_obj_held is None else interpolate_poses(t, seg.T_obj_held, t_query)
    idx = np.minimum(np.round(t_query * fps).astype(int), len(seg.gripper) - 1)
    return T, np.asarray(seg.gripper, dtype=np.float64)[idx], held


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
    """Wherever the gripper command flips, hold the event's target `steps` times with the old
    command so the arm arrives first, then `steps` more with the new one so the fingers finish
    closing (or opening) before the arm moves on."""
    flips = np.flatnonzero(np.diff(np.sign(grip)) != 0) + 1
    reps = np.ones(len(T), dtype=int)
    reps[flips] += steps
    T_out, g_out = np.repeat(T, reps, axis=0), np.repeat(grip, reps)
    starts = np.cumsum(reps) - reps  # index of each input frame in the output
    arrive = np.repeat(starts[flips], steps)
    return (
        np.insert(T_out, arrive, T_out[arrive], axis=0),
        np.insert(g_out, arrive, g_out[arrive - 1]),
    )


def trim_approach(
    T_obj_ee: np.ndarray, grip: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drop a grasp segment's targets before its approach last enters `radius` of the grasp
    (its first closed target): far from the object the hand track is noisiest, and the transit
    into the segment covers that stretch in a straight line. Segments without a grasp, or
    starting closed, are returned unchanged."""
    closed = np.flatnonzero(np.asarray(grip) == GRIPPER_CLOSED)
    if len(closed) == 0 or closed[0] == 0:
        return T_obj_ee, grip
    g = closed[0]
    far = np.linalg.norm(T_obj_ee[:g, :3, 3] - T_obj_ee[g, :3, 3], axis=1) > radius
    start = int(np.flatnonzero(far)[-1]) + 1 if far.any() else 0
    return T_obj_ee[start:], grip[start:]


def snap_grasp(
    T_obj_ee: np.ndarray,
    grip: np.ndarray,
    radius: float,
    height: float,
    blend_steps: int,
    lift: float = 0.0,
    descend_steps: int = 0,
) -> np.ndarray:
    """Move a segment's grasp (its first closed target, if it starts open) onto the object's rim:
    the circle of `radius` at `height` in the object frame, at the human grasp's angle around
    it, top-down with the fingers closing radially (the half turn nearest the human yaw).

    The last `descend_steps` targets before the grasp drop straight onto it from `lift` above,
    so the fingers clear the object's wall; the `blend_steps` before those shift the human
    approach towards that pre-grasp pose with a smoothstep weight, so it is kept until the last
    moment. Later targets hold the grasp."""
    closed = np.flatnonzero(np.asarray(grip) == GRIPPER_CLOSED)
    if len(closed) == 0 or closed[0] == 0:
        return T_obj_ee
    g = closed[0]
    p, x = T_obj_ee[g, :3, 3], T_obj_ee[g, :3, 0]
    radial = np.array([p[0], p[1], 0.0]) / max(np.hypot(p[0], p[1]), 1e-9)
    radial = radial if radial @ x >= 0 else -radial
    down = np.array([0.0, 0.0, -1.0])
    sign = 1.0 if radial @ p >= 0 else -1.0  # the rim point on the human's side
    T_grasp = make_T(
        np.column_stack([radial, np.cross(down, radial), down]),
        sign * radius * radial + [0.0, 0.0, height],
    )
    out = T_obj_ee.copy()
    out[g:] = T_grasp
    d = max(0, g - descend_steps)
    T_pre = T_grasp.copy()
    if d < g:
        T_pre[2, 3] += lift
        for i in range(d, g):
            out[i] = T_grasp
            out[i, :3, 3] = T_pre[:3, 3] + (i - d) / (g - d) * (T_grasp[:3, 3] - T_pre[:3, 3])
    dp = T_pre[:3, 3] - T_obj_ee[d, :3, 3]
    dr = matrix_to_rotvec(T_pre[:3, :3] @ T_obj_ee[d, :3, :3].T)
    for i in range(max(0, d - blend_steps), d):
        u = (i - (d - blend_steps)) / blend_steps
        w = u * u * (3.0 - 2.0 * u)
        out[i, :3, :3] = rotvec_to_matrix(w * dr) @ T_obj_ee[i, :3, :3]
        out[i, :3, 3] = T_obj_ee[i, :3, 3] + w * dp
    return out


def carry_lift(n: int, lift: float, ramp_steps: int) -> np.ndarray:
    """(n,) heights added to a carry that ends in its set-down at index n - 1: `lift` until the
    last `ramp_steps` targets, then a linear ramp to zero, so the carried object clears the
    reference object's rim but is set down where the demo set it."""
    return lift * np.clip((n - 1 - np.arange(n)) / max(1, ramp_steps), 0.0, 1.0)


def carry_targets(T_ref_held: np.ndarray, T_held_ee: np.ndarray) -> np.ndarray:
    """End-effector targets (M,4,4) that move the held object along its demo path relative to the
    reference object, T_ref_held (M,4,4), while holding it as the robot actually does now,
    T_held_ee (4,4)."""
    return T_ref_held @ T_held_ee


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
    current grip-site pose by a straight `transit`. A segment that starts with the gripper
    closed carries the previous segment's object: the object follows its demo path (its own track,
    `T_obj_held`, else the hand path through the demo grasp) and the gripper holds it as measured
    in the sim at the segment's start (`carry_targets`). The object is first lifted straight up
    by `cfg.carry_lift` and carried that much higher than the demo until the set-down
    (`carry_lift`)."""
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
        prev: RobotSegment | None = None
        for seg in segments.segments:
            budget = cfg.max_steps - len(actions)
            if budget <= 0 or (success and stop_on_success):
                break
            name = sim_object(seg.ref_object, cfg)
            T_obj_ee, grip, T_obj_held = resample_segment(seg, segments.fps, cfg.control_freq)
            poses = object_poses(env)
            if name in cfg.rim_grasp:
                if cfg.approach_radius is not None:
                    T_obj_ee, grip = trim_approach(T_obj_ee, grip, cfg.approach_radius)
                T_obj_ee = snap_grasp(
                    T_obj_ee,
                    grip,
                    *cfg.rim_grasp[name],
                    blend_steps=max(1, round(cfg.grasp_blend_s * cfg.control_freq)),
                    lift=cfg.grasp_lift,
                    descend_steps=round(cfg.grasp_descend_s * cfg.control_freq),
                )
            if (
                prev is not None
                and seg.gripper[0] == GRIPPER_CLOSED
                and prev.ref_object != seg.ref_object
            ):
                held = poses[sim_object(prev.ref_object, cfg)]
                if T_obj_held is None:  # no object track: the hand path and the demo grasp
                    T_obj_held = T_obj_ee @ inv_T(prev.T_obj_ee[-1])
                T_obj_ee = carry_targets(T_obj_held, inv_T(held) @ ee_pose(obs))
                opened = np.flatnonzero(grip == GRIPPER_OPEN)
                n = int(opened[0]) + 1 if len(opened) else len(grip)
                ramp = round(cfg.carry_ramp_s * cfg.control_freq)
                T_obj_ee[:n, 2, 3] += carry_lift(n, cfg.carry_lift, ramp)
                T_up = ee_pose(obs)  # lift straight off the table before moving across it
                T_up[2, 3] += cfg.carry_lift
                T_lead = transit(ee_pose(obs), T_up, cfg)
            else:
                T_up, T_lead = ee_pose(obs), np.empty((0, 4, 4))
            prev = seg
            T_sim_ee = poses[name] @ T_obj_ee
            T_lead = np.concatenate([T_lead, transit(T_up, T_sim_ee[0], cfg)])
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
                pos_tol=cfg.track_pos_tol,
                rot_tol=cfg.track_rot_tol,
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
