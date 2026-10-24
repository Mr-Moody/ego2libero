import cv2
import numpy as np

from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_perception.hand.pose import (
    estimate_T_cam_hand,
    estimate_T_cam_world,
    palm_pose,
    scale_hand,
)

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
T_CAM_WORLD = make_T(rotvec_to_matrix(np.array([0.3, -0.2, 0.1])), np.array([0.05, -0.02, 0.45]))


def _project(T_cam_world: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    px, _ = cv2.projectPoints(pts, rvec, T_cam_world[:3, 3], K, DIST)
    return px.reshape(-1, 2)


def test_palm_pose_origin_is_wrist_and_rotation_is_proper(hand_pts):
    shifted = hand_pts + np.array([0.01, 0.02, 0.03])
    T = palm_pose(shifted)
    np.testing.assert_allclose(T[:3, 3], shifted[0])
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_recovers_known_pose_exactly(hand_pts):
    px = _project(T_CAM_WORLD, hand_pts)
    T_world_hand = palm_pose(hand_pts)
    T = estimate_T_cam_hand(hand_pts, px, K, DIST, T_world_hand)
    np.testing.assert_allclose(T, T_CAM_WORLD @ T_world_hand, atol=1e-5)


def test_pose_is_robust_to_pixel_noise(hand_pts):
    rng = np.random.default_rng(0)
    px = _project(T_CAM_WORLD, hand_pts) + rng.normal(scale=0.5, size=(21, 2))
    T = estimate_T_cam_world(hand_pts, px, K, DIST)
    assert np.linalg.norm(T[:3, 3] - T_CAM_WORLD[:3, 3]) < 0.01


def test_non_finite_input_returns_none(hand_pts):
    px = _project(T_CAM_WORLD, hand_pts)
    px[3] = np.nan
    assert estimate_T_cam_world(hand_pts, px, K, DIST) is None
    assert estimate_T_cam_hand(hand_pts, px, K, DIST, palm_pose(hand_pts)) is None


def test_hand_scale_corrects_depth_of_a_smaller_hand(hand_pts):
    # A real hand 0.85x MediaPipe's model: PnP on the unscaled model overestimates depth by
    # 1/0.85; scaling the world landmarks first recovers the true pose.
    real = 0.85 * hand_pts
    px = _project(T_CAM_WORLD, real)
    naive = estimate_T_cam_world(hand_pts, px, K, DIST)
    np.testing.assert_allclose(naive[:3, 3], T_CAM_WORLD[:3, 3] / 0.85, atol=1e-4)
    world = scale_hand(hand_pts, 0.85)
    np.testing.assert_allclose(estimate_T_cam_world(world, px, K, DIST), T_CAM_WORLD, atol=1e-5)
