"""Offline Rauch-Tung-Striebel smoothing over the forward EKF."""

import numpy as np

from e2l_common.stub import not_implemented


def rts_smooth(
    x_filt: np.ndarray, P_filt: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backward RTS pass. Inputs per step from `hand.ekf` recording: filtered x (N,n), P (N,n,n),
    predicted x (N,n), P (N,n,n) and transition Jacobians F (N,n,n). Returns smoothed (x, P)."""
    not_implemented("e2l_perception.smoothing.rts_smooth")
