"""Hand segments to Panda end-effector segments."""

import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.schemas import HandSegments, RobotSegment, RobotSegments
from e2l_retarget.feasibility import check_segment

GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0  # robosuite gripper convention

_UP = np.array([0.0, 0.0, 1.0])
_MIN_HORIZONTAL = 0.2  # closing axis with a shorter horizontal projection is ambiguous


def top_down_rotations(closing: np.ndarray, x_init: np.ndarray | None = None) -> np.ndarray:
    """Rotations (M,3,3) with z straight down and x along the horizontal projection of the
    closing direction (M,3) in the object frame (z up). A near-vertical closing direction reuses
    the previous x (object x before any). The gripper is symmetric under a half turn about z, so
    x is flipped to stay within 90 deg of the previous x (first frame: `x_init`, default object
    x)."""
    x_prev = np.array([1.0, 0.0, 0.0]) if x_init is None else np.asarray(x_init, np.float64)
    z = -_UP
    R = np.empty((len(closing), 3, 3))
    for i, a in enumerate(np.asarray(closing, dtype=np.float64)):
        h = a - (a @ _UP) * _UP
        n = np.linalg.norm(h)
        x = h / n if n >= _MIN_HORIZONTAL else x_prev
        if x @ x_prev < 0:
            x = -x
        R[i] = np.column_stack([x, np.cross(z, x), z])
        x_prev = x
    return R


def hand_to_ee(
    T_obj_hand: np.ndarray, cfg: RetargetConfig, x_init: np.ndarray | None = None
) -> np.ndarray:
    """T_obj_ee (M,4,4) = T_obj_hand (M,4,4) @ cfg.T_hand_ee.T: the grip point at a fixed offset
    in the hand frame. With `cfg.canonicalise_top_down` the rotation becomes a top-down grasp
    whose fingers close along the hand's `cfg.closing_axis`
    (see `top_down_rotations`, which `x_init` seeds)."""
    T_obj_ee = np.asarray(T_obj_hand, dtype=np.float64) @ cfg.T_hand_ee.T
    if cfg.canonicalise_top_down:
        axis = np.asarray(cfg.closing_axis, dtype=np.float64)
        closing = T_obj_hand[:, :3, :3] @ (axis / np.linalg.norm(axis))
        T_obj_ee[:, :3, :3] = top_down_rotations(closing, x_init)
    return T_obj_ee


def retarget_segments(segments: HandSegments, cfg: RetargetConfig) -> RobotSegments:
    """Map every HandSegment to a RobotSegment: poses via `hand_to_ee` (positions scaled by
    `cfg.object_scale` and shifted by `cfg.object_offset` of the segment's reference object, as
    is a carried object's path), grip {0,1} to robosuite gripper {-1,+1}, and `feasible` from
    `feasibility.check_segment` at the demo fps."""
    robot = []
    prev: tuple[np.ndarray, np.ndarray] | None = None  # last gripper x and hand R, old object frame
    for seg in segments.segments:
        x_init = None
        if prev is not None:  # carry the gripper yaw over the shared boundary hand pose
            R_new_old = seg.T_obj_hand[0, :3, :3] @ prev[1].T
            h = R_new_old @ prev[0]
            h[2] = 0.0
            if np.linalg.norm(h) >= _MIN_HORIZONTAL:
                x_init = h / np.linalg.norm(h)
        T_obj_ee = hand_to_ee(seg.T_obj_hand, cfg, x_init)
        prev = (T_obj_ee[-1, :3, 0], seg.T_obj_hand[-1, :3, :3])
        T_obj_held = None if seg.T_obj_held is None else seg.T_obj_held.copy()
        if seg.ref_object in cfg.object_scale:
            scale = np.asarray(cfg.object_scale[seg.ref_object])
            T_obj_ee[:, :3, 3] *= scale
            if T_obj_held is not None:
                T_obj_held[:, :3, 3] *= scale
        if seg.ref_object in cfg.object_offset:
            offset = np.asarray(cfg.object_offset[seg.ref_object])
            T_obj_ee[:, :3, 3] += offset
            if T_obj_held is not None:
                T_obj_held[:, :3, 3] += offset
        t = np.arange(len(T_obj_ee)) / segments.fps
        robot.append(
            RobotSegment(
                ref_object=seg.ref_object,
                T_obj_ee=T_obj_ee,
                gripper=np.where(seg.grip > 0, GRIPPER_CLOSED, GRIPPER_OPEN),
                feasible=check_segment(T_obj_ee, t, cfg),
                T_obj_held=T_obj_held,
            )
        )
    return RobotSegments(demo_id=segments.demo_id, fps=segments.fps, segments=robot)
