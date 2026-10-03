# Ported from Hand-Tracking-Fusion src/palm_utils.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: none beyond formatting.
"""Palm frame, rotation smoothing and bone-length helpers for (21,3) hand landmarks."""

import numpy as np

# MediaPipe hand landmark connectivity
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm arch
]

# All five MCP knuckles + wrist used for palm plane fitting
_PALM_PLANE_IDX = np.array([0, 5, 9, 13, 17])
# Middle MCP used for the x-axis reference (central, not an extremity)
_MIDDLE_MCP = 9


def compute_palm_frame(pts: np.ndarray) -> np.ndarray:
    """Compute a right-handed palm rotation matrix from (21,3) wrist-centred landmarks.

    Returns R (3x3) whose columns are [palm_x, palm_y, palm_z] in world space:
      palm_x  — wrist → middle MCP (central knuckle, more stable than index/pinky)
      palm_z  — palm normal, fitted to all five MCP joints via SVD
      palm_y  — palm_z × palm_x (roughly wrist → fingertips)

    Fitting palm_z to five points (instead of a cross product of two) means
    noise in any single landmark has much less effect on the palm orientation,
    which eliminates the jitter on index and pinky that was caused by using
    those landmarks as the sole frame anchors.

    To convert world → palm frame:  pts_palm = pts @ R
    To convert palm → world frame:  pts_world = pts_palm @ R.T
    """
    palm_pts = pts[_PALM_PLANE_IDX]
    centroid = palm_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(palm_pts - centroid, full_matrices=False)
    palm_z = Vt[-1]  # eigenvector of smallest variance = plane normal
    if np.linalg.norm(palm_z) < 1e-6:
        return np.eye(3)
    palm_z /= np.linalg.norm(palm_z)
    # SVD sign is arbitrary — use a cross product to pick the consistent direction
    ref_z = np.cross(pts[5] - pts[0], pts[17] - pts[0])
    if np.dot(palm_z, ref_z) < 0:
        palm_z = -palm_z

    # palm_x: wrist → middle MCP, projected onto the palm plane
    v_mid = pts[_MIDDLE_MCP] - pts[0]
    palm_x = v_mid - np.dot(v_mid, palm_z) * palm_z
    norm_x = np.linalg.norm(palm_x)
    if norm_x < 1e-6:
        return np.eye(3)
    palm_x /= norm_x

    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y) + 1e-8

    return np.column_stack([palm_x, palm_y, palm_z])


def to_palm_frame(pts: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Transform (21,3) wrist-centred world points into the palm frame."""
    return pts @ R  # equivalent to (R.T @ pts.T).T for orthonormal R


def from_palm_frame(pts_palm: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Transform (21,3) palm-frame points back to world frame."""
    return pts_palm @ R.T


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle in radians between two rotation matrices."""
    cos_theta = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def palm_depth_quality(pts_palm: np.ndarray) -> float:
    """Estimate how reliable the depth axis is in the palm-frame observation.

    In the palm frame, palm_z represents depth out of the palm plane.  A
    healthy detection has fingertips clearly displaced in palm_z from the
    knuckles.  When MediaPipe's depth collapses (hand face-on to camera), all
    palm_z values are near zero.

    Returns a score in [0, 1]: 1 = reliable depth, 0 = depth collapsed.
    """
    hand_scale = np.linalg.norm(pts_palm[9] - pts_palm[0]) + 1e-6
    z_std = float(np.std(pts_palm[:, 2]))
    # Empirically: good 3-D detections have z_std ≈ 15 % of hand scale.
    return float(np.clip(z_std / (hand_scale * 0.15), 0.0, 1.0))


def average_rotations(R_list: list[np.ndarray]) -> np.ndarray:
    """Geodesic mean of rotation matrices via SVD projection onto SO(3)."""
    R_sum = sum(R_list)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:  # reflection guard
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def smooth_rotation(R_prev: np.ndarray, R_new: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average on SO(3) via linear blend + SVD re-projection."""
    # Prevent z-axis from flipping discontinuously between frames
    if np.dot(R_new[:, 2], R_prev[:, 2]) < 0:
        R_new = R_new.copy()
        R_new[:, 2] *= -1
        R_new[:, 0] *= -1  # flip x too to keep det = +1

    R_blend = (1.0 - alpha) * R_prev + alpha * R_new
    U, _, Vt = np.linalg.svd(R_blend)
    R_s = U @ Vt
    if np.linalg.det(R_s) < 0:
        U[:, -1] *= -1
        R_s = U @ Vt
    return R_s


def compute_bone_lengths(pts: np.ndarray) -> np.ndarray:
    """Return Euclidean length of each bone in HAND_CONNECTIONS order."""
    return np.array(
        [np.linalg.norm(pts[child] - pts[parent]) for parent, child in HAND_CONNECTIONS]
    )


def enforce_bone_lengths(pts: np.ndarray, ref_lengths: np.ndarray) -> np.ndarray:
    """Project pts onto the manifold defined by the reference bone lengths.

    One forward pass from proximal to distal joints.  Keeps the parent fixed
    and repositions the child along the existing bone direction.
    """
    pts = pts.copy()
    for i, (parent, child) in enumerate(HAND_CONNECTIONS):
        vec = pts[child] - pts[parent]
        actual = np.linalg.norm(vec)
        if actual < 1e-8:
            continue
        pts[child] = pts[parent] + vec * (ref_lengths[i] / actual)
    return pts
