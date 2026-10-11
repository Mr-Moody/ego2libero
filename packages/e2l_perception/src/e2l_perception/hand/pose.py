"""Recover the metric hand pose in the camera frame."""

import cv2
import numpy as np

from e2l_common.geometry import compose, make_T
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.markers import pnp_to_T


def palm_pose(world_landmarks: np.ndarray) -> np.ndarray:
    """T_world_hand (4,4): palm frame (`compute_palm_frame` axes) with its origin at the wrist
    landmark, expressed in MediaPipe's world-landmark frame. Input (21,3) metres."""
    pts = np.asarray(world_landmarks, dtype=np.float64)
    return make_T(compute_palm_frame(pts), pts[0])


def estimate_T_cam_world(
    world_landmarks: np.ndarray, image_landmarks_px: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray | None:
    """T_cam_world (4,4) from solvePnP (SQPNP) of the 21 world landmarks (21,3), metres, against
    their 2D detections (21,2), pixels. None if the input is non-finite or PnP fails or puts the
    hand behind the camera."""
    obj = np.asarray(world_landmarks, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_landmarks_px, dtype=np.float64).reshape(-1, 2)
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    T = pnp_to_T(rvec, tvec)
    return T if np.isfinite(T).all() and T[2, 3] > 0 else None


def estimate_T_cam_hand(
    world_landmarks: np.ndarray,
    image_landmarks_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_world_hand: np.ndarray,
) -> np.ndarray | None:
    """T_cam_hand (4,4) = T_cam_world @ T_world_hand, with `T_world_hand` from `palm_pose`.
    None if PnP fails."""
    T_cam_world = estimate_T_cam_world(world_landmarks, image_landmarks_px, K, dist)
    return None if T_cam_world is None else compose(T_cam_world, T_world_hand)
