"""Object-anchored hand depth.

Monocular PnP places the hand well along its camera ray (the 2D detection is good) but its
depth swings by tens of centimetres as the fingers flex and foreshorten. While the hand carries
an object, the pinch point is rigidly attached to that object, whose ArUco pose is reliable, so
those frames fix the depth; free-motion frames drop depth spikes and take the ratio of the
nearest carry frames."""

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.assign import grasped_object

THUMB_TIP, INDEX_TIP = 4, 8


def carry_runs(grip: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) frame index of each closed-grip run."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(grip) > 0, 0]))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _pinch(demo: DemoTrajectory) -> np.ndarray:
    return demo.keypoints_table[:, [THUMB_TIP, INDEX_TIP]].mean(axis=1)


def depth_ratios(demo: DemoTrajectory, grip: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """(N,) corrected / raw pinch depth on carry frames, NaN elsewhere.

    For each closed-grip run the carried object is the one that moves while held
    (`assign.grasped_object`). The pinch-to-object offset (object frame) is the median over the
    run's frames where hand and object are both seen; each such frame's depth is then the one
    that brings the pinch closest to object pose @ offset along its ray from the camera centre
    `cam` (N,3)."""
    pinch = _pinch(demo)
    ray = pinch - cam
    depth = np.linalg.norm(ray, axis=1)
    ratio = np.full(len(demo.t), np.nan)
    for a, b in carry_runs(grip):
        k = demo.object_names.index(grasped_object(demo, a, b + 1))
        idx = np.arange(a, b + 1)
        idx = idx[demo.hand_valid[idx] & demo.obj_valid[idx, k]]
        if len(idx) < 3:
            continue
        T = demo.T_table_obj[idx, k]
        offset = np.median(np.einsum("nji,nj->ni", T[:, :3, :3], pinch[idx] - T[:, :3, 3]), 0)
        target = T[:, :3, 3] + T[:, :3, :3] @ offset
        unit = ray[idx] / depth[idx, None]
        ratio[idx] = np.einsum("ni,ni->n", target - cam[idx], unit) / depth[idx]
    return ratio


def _running_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)


def despike(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Replace samples further than `threshold` from the running median (`window` samples) by
    linear interpolation of the rest. The median only flags spikes: on a trend it lags."""
    keep = np.abs(x - _running_median(x, window)) <= threshold
    if keep.all() or not keep.any():
        return x.copy()
    i = np.arange(len(x))
    return np.interp(i, i[keep], x[keep])


def anchor_depth(demo: DemoTrajectory, grip: np.ndarray, cfg: SegmentConfig) -> DemoTrajectory:
    """Rescale every hand frame about the camera centre so its pinch depth is corrected.

    Carry frames take `depth_ratios`; other hand-valid frames take the raw depth with spikes
    removed (`despike`: window `cfg.depth_window_s`, threshold `cfg.depth_outlier_m`) times
    the ratio interpolated from the carry frames (held beyond the first and last). Keypoints
    and wrist move along their rays; the hand rotation is unchanged and the aperture is
    recomputed. Returns `demo` itself when anchoring is off, the camera pose is unknown, or no
    carry frame could be anchored."""
    if not cfg.anchor_depth or demo.cam_table is None or not demo.hand_valid.any():
        return demo
    seen = np.isfinite(demo.cam_table).all(axis=1)
    if not seen.any():
        return demo
    cam = np.where(seen[:, None], demo.cam_table, np.median(demo.cam_table[seen], axis=0))
    ratio = depth_ratios(demo, grip, cam)
    anchored = np.isfinite(ratio)
    if not anchored.any():
        return demo

    valid = np.flatnonzero(demo.hand_valid)
    depth = np.linalg.norm(_pinch(demo) - cam, axis=1)
    window = max(1, int(round(cfg.depth_window_s * demo.fps)))
    smooth = depth.copy()
    smooth[valid] = despike(depth[valid], window, cfg.depth_outlier_m)
    hold = np.interp(np.arange(len(depth)), np.flatnonzero(anchored), ratio[anchored])
    scale = np.where(anchored, ratio, smooth * hold / depth)
    scale[~demo.hand_valid] = 1.0

    kp = cam[:, None] + scale[:, None, None] * (demo.keypoints_table - cam[:, None])
    T_hand = demo.T_table_hand.copy()
    T_hand[valid, :3, 3] = cam[valid] + scale[valid, None] * (T_hand[valid, :3, 3] - cam[valid])
    return demo.model_copy(
        update={
            "keypoints_table": kp,
            "T_table_hand": T_hand,
            "aperture": np.linalg.norm(kp[:, THUMB_TIP] - kp[:, INDEX_TIP], axis=1),
        }
    )
