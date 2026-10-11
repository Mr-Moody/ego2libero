"""Offline smoothing: constant-velocity Kalman filter forward, Rauch-Tung-Striebel backward."""

import numpy as np


def rts_smooth(
    x_filt: np.ndarray, P_filt: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backward RTS pass. Per step k: filtered x (N,n) and P (N,n,n), predicted x_pred[k] =
    F[k] @ x_filt[k-1] (N,n) and P_pred (N,n,n), transition F (N,n,n) or one constant (n,n).
    x may carry trailing track axes, (N,n,M), for M tracks sharing one covariance.
    Index 0 of the predictions is unused. Returns smoothed (x, P)."""
    N = len(x_filt)
    F = np.broadcast_to(F, (N, *P_filt.shape[1:]))
    xs, Ps = x_filt.astype(float).copy(), P_filt.astype(float).copy()
    for k in range(N - 2, -1, -1):
        C = np.linalg.solve(P_pred[k + 1], F[k + 1] @ P_filt[k]).T
        xs[k] = x_filt[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P_filt[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T
    return xs, Ps


def smooth_tracks(
    z: np.ndarray, valid: np.ndarray, dt: float, accel_noise: float, meas_std: float
) -> np.ndarray:
    """Smooth M scalar tracks z (N,M) sampled every dt seconds; rows where `valid` (N,) is False
    are missing and get filled. Model: constant velocity with white acceleration of spectral
    density `accel_noise` (m^2/s^3), measurement std `meas_std` (m). Rows before the first
    valid one stay NaN; rows after the last valid one are extrapolated (the caller masks them)."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    first = idx[0]
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = accel_noise * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    R = meas_std**2

    x = np.stack([z[first], np.zeros(z.shape[1])])  # (2, M)
    P = np.diag([R, 1.0])
    xf, Pf, xp, Pp = [], [], [], []
    for k in range(first, len(z)):
        if k > first:
            x, P = F @ x, F @ P @ F.T + Q
        xp.append(x)
        Pp.append(P)
        if valid[k]:
            gain = P @ H.T / (H @ P @ H.T + R)  # (2, 1)
            x = x + gain @ (z[k][None, :] - H @ x)
            P = (np.eye(2) - gain @ H) @ P
        xf.append(x)
        Pf.append(P)
    xs, _ = rts_smooth(np.array(xf), np.array(Pf), np.array(xp), np.array(Pp), F)
    out[first:] = xs[:, 0, :]
    return out
