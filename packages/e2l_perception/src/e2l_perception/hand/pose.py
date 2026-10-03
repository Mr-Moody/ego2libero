"""Recover the metric hand pose in the camera frame."""

import numpy as np

from e2l_common.stub import not_implemented


def estimate_T_cam_hand(
    world_landmarks: np.ndarray,
    image_landmarks_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_world_hand: np.ndarray,
) -> np.ndarray | None:
    """T_cam_hand (4,4) via solvePnP of MediaPipe's 21 metric world landmarks (21,3), metres,
    against their 2D detections (21,2), pixels. `T_world_hand` is the palm frame expressed in
    MediaPipe's world-landmark frame (from `palm.compute_palm_frame`). None if PnP fails."""
    not_implemented("e2l_perception.hand.pose.estimate_T_cam_hand")
