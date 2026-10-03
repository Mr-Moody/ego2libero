# Ported from Hand-Tracking-Fusion src/fusion.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: package-relative imports; single-camera path is the default. The stereo branch is
# kept but UNTESTED: the original main.py always passed [detection, None], so it never ran.
# Calls ekf.commit() at the end of each step so recorded runs can be RTS-smoothed.
"""Palm-frame EKF fusion of hand detections."""

from __future__ import annotations

import numpy as np

from e2l_perception.hand.detector import HandDetection
from e2l_perception.hand.ekf import JointHandEKF
from e2l_perception.hand.palm import (
    average_rotations,
    compute_bone_lengths,
    compute_palm_frame,
    enforce_bone_lengths,
    from_palm_frame,
    palm_depth_quality,
    rotation_angle,
    smooth_rotation,
    to_palm_frame,
)


class HandFusion:
    """Hand pose fusion with a palm-frame EKF.

    Architecture
    ------------
    Rather than tracking raw world-space joint positions (where global hand
    rotation shows up as large, depth-heavy movements that MediaPipe estimates
    poorly), the EKF operates in the *palm frame*:

      palm_x  — wrist → middle MCP
      palm_y  — palm_z × palm_x
      palm_z  — palm normal (out-of-palm)

    In this frame the finger *shape* (flexion / extension) is stable during
    global rotation, so the constant-position EKF prediction stays accurate.
    The palm orientation itself is tracked separately as a smoothed SO(3)
    rotation and applied at output time.

    Stereo (UNTESTED): if a stereo calibration dict is provided and both cameras
    see the hand, triangulation replaces MediaPipe's monocular depth estimate.
    """

    def __init__(self, fps: float = 30.0, calibration: dict | None = None, record: bool = False):
        self._ekf = JointHandEKF(dt=1.0 / fps, record=record)

        # Stereo triangulation (optional, untested)
        self._proj: tuple | None = None
        if calibration is not None:
            from e2l_perception.calibration import build_projection_matrices

            self._proj = build_projection_matrices(calibration)

        self._R_palm: np.ndarray | None = None  # smoothed palm orientation
        self._bone_lengths: np.ndarray | None = None  # set at first detection

    @property
    def is_initialised(self) -> bool:
        return self._ekf.initialised

    @property
    def ekf(self) -> JointHandEKF:
        return self._ekf

    @property
    def R_palm(self) -> np.ndarray | None:
        """Smoothed palm orientation (columns = palm axes in MediaPipe world coords)."""
        return self._R_palm

    def update(self, detections: list[HandDetection | None]) -> np.ndarray | None:
        """Fuse detections (one entry per camera, None if no hand detected).

        Returns (21, 3) fused joint positions in MediaPipe world space (wrist-centred),
        or None before the first detection.
        """
        visible = [d for d in detections if d is not None]

        if not visible:
            if self._ekf.initialised:
                self._ekf.freeze()
                return self._to_world(self._ekf.positions)
            return None

        # Obtain 3D landmark sets
        if (
            self._proj is not None
            and len(detections) >= 2
            and detections[0] is not None
            and detections[1] is not None
        ):
            lm3d_list = [self._triangulate(detections[0], detections[1])]
            mask_list = [detections[0].visible_mask & detections[1].visible_mask]
        else:
            lm3d_list = [d.landmarks_3d for d in visible]
            mask_list = [d.visible_mask for d in visible]

        # Compute palm frames and average orientation across cameras
        R_list = [compute_palm_frame(pts) for pts in lm3d_list]
        R_current = average_rotations(R_list) if len(R_list) > 1 else R_list[0]

        # Initialise EKF on first detection
        if not self._ekf.initialised:
            return self._reseed(R_current, lm3d_list)

        # Detect large orientation jumps. Any inter-frame rotation > 90° is
        # physically impossible for a hand at normal speed — it means MediaPipe
        # has flipped its palm frame estimate.  Re-seeding the EKF is cleaner
        # than trying to smooth through it (which would produce the squash).
        angle = rotation_angle(self._R_palm, R_current)
        if angle > np.pi / 2:
            return self._reseed(R_current, lm3d_list)

        # Adaptive rotation smoothing: track fast rotations more aggressively
        # so the palm frame doesn't lag behind a quickly flipping hand.
        # Update _R_palm BEFORE computing palm_obs so the EKF always operates
        # in the same smoothed frame that _to_world uses.
        adaptive_alpha = min(0.85, 0.15 + angle / (np.pi / 4) * 0.15)
        self._R_palm = smooth_rotation(self._R_palm, R_current, alpha=adaptive_alpha)

        # Transform observations into the smoothed palm frame
        palm_obs = [to_palm_frame(pts, self._R_palm) for pts in lm3d_list]

        self._ekf.predict()
        for pts_palm, mask in zip(palm_obs, mask_list, strict=True):
            # Scale up depth noise when MediaPipe's z estimate is unreliable
            # (hand flat-on to the camera).  The EKF then relies on its motion
            # model for depth and only trusts x/y from the observation.
            quality = palm_depth_quality(pts_palm)
            depth_noise_scale = float(np.exp(3.0 * (1.0 - quality)))  # 1x→20x
            self._ekf.update(pts_palm, mask, depth_noise_scale=depth_noise_scale)

        # Enforce bone lengths, write corrections back into EKF state
        positions = self._ekf.positions
        if self._bone_lengths is not None:
            positions = enforce_bone_lengths(positions, self._bone_lengths)
            for i in range(21):
                self._ekf.x[i * 6 : i * 6 + 3] = positions[i]
        self._ekf.commit()

        return self._to_world(positions)

    def _reseed(self, R_current: np.ndarray, lm3d_list: list[np.ndarray]) -> np.ndarray:
        self._R_palm = R_current.copy()
        palm_obs = [to_palm_frame(pts, self._R_palm) for pts in lm3d_list]
        self._ekf.init(palm_obs[0])
        if self._bone_lengths is None:
            self._bone_lengths = compute_bone_lengths(palm_obs[0])
        return self._to_world(self._ekf.positions)

    def _to_world(self, pts_palm: np.ndarray) -> np.ndarray:
        """Rotate palm-frame positions back to world space."""
        if self._R_palm is None:
            return pts_palm
        return from_palm_frame(pts_palm, self._R_palm)

    def _triangulate(self, det0: HandDetection, det1: HandDetection) -> np.ndarray:
        from e2l_perception.calibration import triangulate_landmarks

        P0, P1, K0, d0, K1, d1 = self._proj
        return triangulate_landmarks(det0.landmarks_2d, det1.landmarks_2d, P0, P1, K0, d0, K1, d1)
