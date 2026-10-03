# Drawing helpers ported from Hand-Tracking-Fusion src/visualiser.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: operate on video frames (no live-window HUD hint); added pose-axis drawing.
"""Overlay videos and trajectory plots."""

from pathlib import Path

import cv2
import numpy as np

from e2l_common.stub import not_implemented
from e2l_perception.hand.detector import HandDetection
from e2l_perception.hand.palm import HAND_CONNECTIONS

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


def render_overlay(video: Path, demo_id: str, out: Path) -> Path:
    """Draw landmarks, palm axes and marker axes on the normalised video; write mp4 to `out`."""
    not_implemented("e2l_perception.viz.render_overlay")


def plot_trajectory(demo_id: str, out: Path) -> Path:
    """Plot table-frame wrist position and aperture over time; write a PNG to `out`."""
    not_implemented("e2l_perception.viz.plot_trajectory")
