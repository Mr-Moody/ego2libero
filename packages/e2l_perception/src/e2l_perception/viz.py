# Drawing helpers ported from Hand-Tracking-Fusion src/visualiser.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: operate on video frames (no live-window HUD hint); added pose-axis drawing.
"""Overlay videos and trajectory plots."""

from pathlib import Path

import cv2
import numpy as np

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.schemas import DemoTrajectory
from e2l_perception.calibration import load_intrinsics
from e2l_perception.hand.detector import HandDetection
from e2l_perception.hand.palm import HAND_CONNECTIONS
from e2l_perception.pipeline import FrameObservation, observe_frames

_CYAN = (0, 220, 255)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_GREY = (100, 100, 100)


def draw_detection(frame: np.ndarray, detection: HandDetection, color: tuple = _CYAN) -> None:
    pts = [(int(x), int(y)) for x, y in detection.landmarks_2d]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _WHITE, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        dot_color = color if detection.visible_mask[i] else _GREY
        cv2.circle(frame, (x, y), 5, dot_color, -1)
        cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)


def draw_bounding_box(frame: np.ndarray, detection: HandDetection) -> None:
    xs, ys = detection.landmarks_2d[:, 0], detection.landmarks_2d[:, 1]
    cv2.rectangle(
        frame,
        (int(xs.min()) - 10, int(ys.min()) - 10),
        (int(xs.max()) + 10, int(ys.max()) + 10),
        _GREEN,
        2,
    )


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA
        )


def draw_axes(
    frame: np.ndarray, T_cam_x: np.ndarray, K: np.ndarray, dist: np.ndarray, length: float = 0.05
) -> None:
    """Draw the x/y/z axes (red/green/blue) of a pose T_cam_x (4,4) into the frame."""
    rvec, _ = cv2.Rodrigues(T_cam_x[:3, :3])
    cv2.drawFrameAxes(frame, K, dist, rvec, T_cam_x[:3, 3], length)


def draw_observation(
    frame: np.ndarray, obs: FrameObservation, K: np.ndarray, dist: np.ndarray
) -> None:
    """Landmarks, palm axes, table axes (10 cm) and object axes (5 cm), plus a status HUD."""
    if obs.detection is not None:
        draw_detection(frame, obs.detection)
    if obs.T_cam_hand is not None:
        draw_axes(frame, obs.T_cam_hand, K, dist, 0.05)
    if obs.T_cam_table is not None:
        draw_axes(frame, obs.T_cam_table, K, dist, 0.10)
    for T_cam_obj in obs.T_cam_obj.values():
        draw_axes(frame, T_cam_obj, K, dist, 0.05)
    draw_hud(
        frame,
        [
            f"board {'ok' if obs.T_cam_table is not None else '--'}  "
            f"hand {'ok' if obs.T_cam_hand is not None else '--'}",
            "objects: " + (", ".join(sorted(obs.T_cam_obj)) or "--"),
        ],
    )


def render_overlay(video: Path, out: Path, cfg: PerceptionConfig, capture: CaptureConfig) -> Path:
    """Re-track `video` and write an mp4 with `draw_observation` overlays to `out`."""
    import imageio.v2 as imageio

    from e2l_perception.video import VideoReader

    intr = load_intrinsics(capture.intrinsics)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=VideoReader(video).fps, macro_block_size=1) as writer:
        for t, frame, obs in observe_frames(video, cfg, capture):
            draw_observation(frame, obs, intr.K, intr.dist)
            draw_hud(frame[-40:], [f"t={t:.2f}s"])  # slice is a view: draws in the bottom strip
            writer.append_data(frame[:, :, ::-1])
    return out


def plot_trajectory(demo: DemoTrajectory, out: Path) -> Path:
    """3D wrist path with object start positions, wrist xyz over time, and aperture; PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrist = demo.T_table_hand[:, :3, 3]
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, (1, 3), projection="3d")
    ax3d.plot(*wrist.T, lw=1.5, label="wrist")
    for k, name in enumerate(demo.object_names):
        seen = np.flatnonzero(demo.obj_valid[:, k])
        if seen.size:
            ax3d.scatter(*demo.T_table_obj[seen[0], k, :3, 3], s=40, label=name)
    ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"{demo.demo_id} (table frame)")
    ax3d.legend()
    ax_xyz = fig.add_subplot(2, 2, 2)
    for i, axis in enumerate("xyz"):
        ax_xyz.plot(demo.t, wrist[:, i], label=axis)
    ax_xyz.set(ylabel="wrist position (m)")
    ax_xyz.legend()
    ax_ap = fig.add_subplot(2, 2, 4, sharex=ax_xyz)
    ax_ap.plot(demo.t, demo.aperture)
    ax_ap.set(xlabel="t (s)", ylabel="aperture (m)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
