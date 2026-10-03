# Ported from Hand-Tracking-Fusion src/joint_hand_ekf.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: optional per-step recording of predicted and filtered x, P and the transition
# matrix F so `e2l_perception.smoothing` can run a Rauch-Tung-Striebel backward pass.
"""Constant-velocity Kalman filter over 21 palm-frame joint positions."""

import numpy as np


def build_partial_H(vis_indices: np.ndarray, N: int = 21) -> np.ndarray:
    """Observation matrix for a subset of joints.

    Each visible joint contributes a 3-row block selecting its [x, y, z]
    from the 6-element (pos + vel) per-joint state.

    Returns: (3*k, 6*N) where k = len(vis_indices)
    """
    k = len(vis_indices)
    H = np.zeros((3 * k, 6 * N))
    for row, joint in enumerate(vis_indices):
        H[row * 3 : row * 3 + 3, joint * 6 : joint * 6 + 3] = np.eye(3)
    return H


class JointHandEKF:
    def __init__(self, N: int = 21, dt: float = 1 / 30, record: bool = False):
        self.N = N
        self.n = N * 6
        self.F = np.kron(np.eye(N), self._f_single(dt))

        # Reduced position process noise: in the palm frame, finger shapes are
        # stable during rotation, so we trust the prediction more.
        self.Q = np.kron(np.eye(N), np.diag([0.5] * 3 + [5.0] * 3))
        self.P = np.kron(np.eye(N), np.diag([50.0] * 3 + [20.0] * 3))
        self.x = np.zeros(self.n)
        self.initialised = False

        # RTS recording: one entry per predict() / commit(). P is (6N)^2 floats per step
        # (~127 kB for N=21), so only enable it for offline runs.
        self.record = record
        self._hist: dict[str, list[np.ndarray]] = {k: [] for k in ("x_pred", "P_pred", "x", "P")}

    def _f_single(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    @property
    def positions(self) -> np.ndarray:
        """Current estimated joint positions as (N, 3)."""
        return self.x.reshape(self.N, 6)[:, :3].copy()

    def init(self, pts3d: np.ndarray):
        """Initialise state from a (21, 3) point cloud."""
        for i in range(self.N):
            self.x[i * 6 : i * 6 + 3] = pts3d[i]
        self.initialised = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.record:
            self._hist["x_pred"].append(self.x.copy())
            self._hist["P_pred"].append(self.P.copy())

    def freeze(self):
        """Hold positions and zero velocities — called when no hand is visible."""
        self.x.reshape(self.N, 6)[:, 3:] = 0

    def update(self, pts3d: np.ndarray, visible_mask: np.ndarray, depth_noise_scale: float = 1.0):
        """Update with a partial observation.

        pts3d:             (21, 3) — palm-frame positions from one camera
        visible_mask:      (21,)  bool — which joints are reliably observed
        depth_noise_scale: multiplier on R_z; pass a large value (e.g. 20–50)
                           when MediaPipe depth is known to be unreliable
                           (hand face-on to camera).
        """
        vis = np.where(visible_mask)[0]
        if len(vis) == 0:
            return

        H = build_partial_H(vis, self.N)
        z = pts3d[vis].flatten()

        # Anisotropic measurement noise: x/y come from reliable 2-D projections;
        # z (palm-normal depth) is inherently noisier and can be further scaled
        # up by the caller when depth quality is low.
        R_diag = np.tile([2.0, 2.0, 6.0 * depth_noise_scale], len(vis))
        R_meas = np.diag(R_diag)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P

    def commit(self):
        """Record the filtered state after this step's predict + updates (if recording)."""
        if self.record:
            self._hist["x"].append(self.x.copy())
            self._hist["P"].append(self.P.copy())

    def history(self) -> dict[str, np.ndarray]:
        """Stacked records with a leading step axis: x_pred, P_pred (after predict) and x, P
        (after commit), plus the constant transition matrix F (n, n). Steps that only re-seed
        the filter record nothing, so a re-seed splits the run for smoothing purposes."""
        out = {k: np.array(v) for k, v in self._hist.items()}
        out["F"] = self.F.copy()
        return out
