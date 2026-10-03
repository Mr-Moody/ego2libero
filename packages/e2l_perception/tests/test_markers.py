"""Render markers with a known pose into a synthetic pinhole image and recover the pose."""

import cv2
import numpy as np
import pytest

from e2l_common.config import CaptureConfig, Pose, TableBoard, TaggedObject
from e2l_common.geometry import compose, inv_T, make_T, matrix_to_rotvec, rotvec_to_matrix
from e2l_perception.markers import MarkerTracker

W, H = 1280, 720
K = np.array([[1000.0, 0, W / 2], [0, 1000.0, H / 2], [0, 0, 1]])
DIST = np.zeros(5)

CAPTURE = CaptureConfig(
    aruco_dictionary="DICT_4X4_50",
    table_board=TableBoard(markers_x=4, markers_y=3, marker_length=0.04, marker_separation=0.01),
    objects={"bowl": TaggedObject(marker_id=20, marker_length=0.05,
                                  T_marker_obj=Pose(xyz=(0, 0, -0.03)))},
)  # fmt: skip


def render_plane(texture: np.ndarray, A: np.ndarray, T_cam_plane: np.ndarray) -> np.ndarray:
    """Warp a texture lying in the z=0 plane of `T_cam_plane` into the camera image. A maps
    texture pixel (u, v, 1) to plane coordinates (x, y, 1) in metres."""
    Rt = T_cam_plane[:3, [0, 1, 3]]
    H_img = K @ Rt @ A
    return cv2.warpPerspective(texture, H_img, (W, H), flags=cv2.INTER_AREA, borderValue=128)


def pixel_to_plane(scale: float, ox: float, oy: float, flip_y: bool) -> np.ndarray:
    """Texture pixel centres (u + 0.5) * scale - offset, optionally with y pointing up."""
    sy = -scale if flip_y else scale
    return np.array([[scale, 0, 0.5 * scale - ox], [0, sy, 0.5 * sy - oy], [0, 0, 1]])


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(matrix_to_rotvec(Ra.T @ Rb))))


def camera_pose(tilt_deg: float, yaw_deg: float, dist: float) -> np.ndarray:
    """T_cam_table for a camera looking at the table centre from `dist` metres."""
    R_down = rotvec_to_matrix(np.array([np.pi, 0, 0]))  # camera z = -table z
    R = rotvec_to_matrix(np.radians([tilt_deg, 0, yaw_deg])) @ R_down
    return make_T(R, [0.01, -0.02, dist])


FRONTAL = pytest.param(
    0,
    0,
    marks=pytest.mark.xfail(
        reason="a lone small marker seen head-on has ill-conditioned orientation: 0.1 px corner "
        "noise gives 3-10 deg error. Phone views of the table are oblique; constrain object "
        "z to the table normal if this matters.",
        strict=False,
    ),
)


@pytest.mark.parametrize("tilt,yaw", [(25, 30), (-35, -60), (45, 150), FRONTAL])
def test_single_marker_pose(tilt, yaw):
    tracker = MarkerTracker(CAPTURE, K, DIST)
    side_px = 600  # texture: 50 mm marker in a 100 mm white quiet zone
    tex = np.full((side_px, side_px), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(tracker.dictionary, 20, side_px // 2)
    tex[side_px // 4 : 3 * side_px // 4, side_px // 4 : 3 * side_px // 4] = marker
    scale = 0.1 / side_px
    T_cam_marker = camera_pose(tilt, yaw, 0.45)
    img = render_plane(tex, pixel_to_plane(scale, 0.05, -0.05, flip_y=True), T_cam_marker)

    obs = tracker.observe(img)
    assert "bowl" in obs.T_cam_obj
    T_expected = compose(T_cam_marker, CAPTURE.objects["bowl"].T_marker_obj.T)
    T_est = obs.T_cam_obj["bowl"]
    assert np.linalg.norm(T_est[:3, 3] - T_expected[:3, 3]) < 0.002
    assert angle_deg(T_est[:3, :3], T_expected[:3, :3]) < 1.0


@pytest.mark.parametrize("tilt,yaw", [(0, 0), (30, 20), (-40, 120)])
def test_table_board_pose(tilt, yaw):
    tracker = MarkerTracker(CAPTURE, K, DIST)
    ppm, margin = 4000, 200  # pixels per metre, white margin in px
    bw, bh = 0.19, 0.14
    tex = tracker.board.generateImage(
        (int(bw * ppm) + 2 * margin, int(bh * ppm) + 2 * margin), marginSize=margin
    )
    A = pixel_to_plane(1 / ppm, margin / ppm, margin / ppm, flip_y=False)
    T_cam_table = camera_pose(tilt, yaw, 0.6)
    T_cam_board = compose(T_cam_table, inv_T(tracker.T_board_table))
    img = render_plane(tex, A, T_cam_board)

    T_est = tracker.observe(img).T_cam_table
    assert T_est is not None
    assert np.linalg.norm(T_est[:3, 3] - T_cam_table[:3, 3]) < 0.002
    assert angle_deg(T_est[:3, :3], T_cam_table[:3, :3]) < 1.0
    # table z points up, towards the camera above it
    assert (inv_T(T_est) @ [0, 0, 0, 1])[2] > 0


def test_nothing_visible():
    tracker = MarkerTracker(CAPTURE, K, DIST)
    obs = tracker.observe(np.full((H, W, 3), 200, np.uint8))
    assert obs.T_cam_table is None and obs.T_cam_obj == {}
