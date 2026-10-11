"""Per-demo perception: normalised video -> DemoTrajectory in the table frame."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.geometry import compose, inv_T, make_T
from e2l_common.schemas import DemoTrajectory
from e2l_perception.calibration import load_intrinsics
from e2l_perception.hand.detector import HandDetection
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.hand.pose import estimate_T_cam_hand, palm_pose
from e2l_perception.smoothing import smooth_tracks
from e2l_perception.video import VideoReader

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass
class FrameObservation:
    """Everything seen in one frame, in the camera frame. None / missing keys = not detected."""

    T_cam_table: np.ndarray | None
    T_cam_obj: dict[str, np.ndarray] = field(default_factory=dict)
    T_cam_hand: np.ndarray | None = None
    keypoints_cam: np.ndarray | None = None  # (21,3) metres
    detection: HandDetection | None = None


def _transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def observe_frames(
    video: Path, cfg: PerceptionConfig, capture: CaptureConfig
) -> Iterator[tuple[float, np.ndarray, FrameObservation]]:
    """Yield (t, BGR frame, observation) for every frame: board and object poses from ArUco,
    the demonstrating hand from MediaPipe + HandFilter, and its metric pose by PnP."""
    if cfg.mirrored:
        raise ValueError("mirrored frames break ArUco and the chirality of PnP; set mirrored=false")
    from e2l_perception.hand.detector import HandDetector
    from e2l_perception.hand.filter import HandFilter
    from e2l_perception.markers import MarkerTracker

    reader = VideoReader(video)
    intr = load_intrinsics(capture.intrinsics)
    if tuple(intr.size) != tuple(reader.size):
        raise ValueError(f"intrinsics are for {intr.size}, video {video} is {reader.size}")
    tracker = MarkerTracker(capture, intr.K, intr.dist)
    hand_filter = HandFilter(cfg.hand, cfg.max_wrist_jump_px)
    with HandDetector(
        cfg.hand_model,
        cfg.num_hands,
        cfg.mirrored,
        cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    ) as detector:
        for frame, t in reader:
            markers = tracker.observe(frame)
            detection = hand_filter.select(detector.detect_all(frame, t))
            obs = FrameObservation(markers.T_cam_table, markers.T_cam_obj, detection=detection)
            if detection is not None:
                world = detection.landmarks_3d.astype(np.float64)
                T_world_hand = palm_pose(world)
                obs.T_cam_hand = estimate_T_cam_hand(
                    world, detection.landmarks_2d, intr.K, intr.dist, T_world_hand
                )
                if obs.T_cam_hand is not None:
                    obs.keypoints_cam = _transform(
                        compose(obs.T_cam_hand, inv_T(T_world_hand)), world
                    )
            yield t, frame, obs


def fill_short_gaps(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Mark interior runs of False no longer than `max_gap` frames as True."""
    out = valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            out[a + 1 : b] = True
    return out


def assemble_trajectory(
    demo_id: str,
    fps: float,
    observations: list[FrameObservation],
    object_names: list[str],
    cfg: PerceptionConfig,
) -> DemoTrajectory:
    """Camera-frame observations -> table-frame DemoTrajectory. Hand frames need both the hand
    and the board; interior gaps up to `cfg.max_gap_s` are filled (Kalman+RTS if `cfg.smooth`,
    else linear interpolation). T_table_hand is the palm frame of the table-frame keypoints.
    Invalid entries are NaN."""
    N, K = len(observations), len(object_names)
    kp = np.full((N, 21, 3), np.nan)
    T_table_obj = np.full((N, K, 4, 4), np.nan)
    for i, obs in enumerate(observations):
        if obs.T_cam_table is None:
            continue
        T_table_cam = inv_T(obs.T_cam_table)
        if obs.keypoints_cam is not None:
            kp[i] = _transform(T_table_cam, obs.keypoints_cam)
        for k, name in enumerate(object_names):
            if name in obs.T_cam_obj:
                T_table_obj[i, k] = compose(T_table_cam, obs.T_cam_obj[name])
    seen = np.isfinite(kp).all(axis=(1, 2))
    hand_valid = fill_short_gaps(seen, int(np.floor(cfg.max_gap_s * fps + 1e-9)))

    flat = kp.reshape(N, -1)
    if cfg.smooth:
        flat = smooth_tracks(flat, seen, 1 / fps, cfg.smooth_accel_noise, cfg.smooth_meas_std)
    elif seen.any():
        frames = np.arange(N)
        flat = np.stack([np.interp(frames, frames[seen], col[seen]) for col in flat.T], axis=1)
    kp = flat.reshape(N, 21, 3)
    kp[~hand_valid] = np.nan

    T_table_hand = np.full((N, 4, 4), np.nan)
    for i in np.flatnonzero(hand_valid):
        T_table_hand[i] = make_T(compute_palm_frame(kp[i]), kp[i, 0])
    return DemoTrajectory(
        demo_id=demo_id,
        fps=fps,
        object_names=object_names,
        t=np.arange(N) / fps,
        T_table_hand=T_table_hand,
        hand_valid=hand_valid,
        keypoints_table=kp,
        aperture=np.linalg.norm(kp[:, THUMB_TIP] - kp[:, INDEX_TIP], axis=1),
        T_table_obj=T_table_obj,
        obj_valid=np.isfinite(T_table_obj).all(axis=(2, 3)),
    )


def run_demo(
    video: Path, demo_id: str, cfg: PerceptionConfig, capture: CaptureConfig
) -> DemoTrajectory:
    """Track a normalised video and assemble its table-frame DemoTrajectory."""
    observations = [obs for _, _, obs in observe_frames(video, cfg, capture)]
    fps = VideoReader(video).fps
    return assemble_trajectory(demo_id, fps, observations, list(capture.objects), cfg)
