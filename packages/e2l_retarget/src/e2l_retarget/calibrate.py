"""Measure the wrist-to-grip offset and the finger closing axis from a recorded demo."""

import numpy as np

from e2l_common.schemas import DemoTrajectory

THUMB_TIP, INDEX_TIP = 4, 8


def estimate_grip(demo: DemoTrajectory, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Over hand-valid frames where the grip is `closed` (N,) bool and both fingertips are
    finite: the median pinch point (thumb/index tip midpoint) and the normalised summed
    thumb-to-index direction, both in the hand frame, plus the number of frames used. These are
    `T_hand_ee.xyz` and `closing_axis` for retarget.yaml. ValueError if no frame qualifies."""
    tips = demo.keypoints_table[:, [THUMB_TIP, INDEX_TIP]]
    use = demo.hand_valid & closed & np.isfinite(tips).all(axis=(1, 2))
    if not use.any():
        raise ValueError(f"{demo.demo_id}: no hand-valid frame with the grip closed")
    R = demo.T_table_hand[use, :3, :3]
    origin = demo.T_table_hand[use, :3, 3]
    thumb, index = tips[use, 0], tips[use, 1]
    pinch = np.einsum("nji,nj->ni", R, (thumb + index) / 2 - origin)  # R^T (p - o)
    d = np.einsum("nji,nj->ni", R, index - thumb)
    axis = d.sum(axis=0)  # raw vectors: wider pinches weigh more, coincident tips add nothing
    return np.median(pinch, axis=0), axis / np.linalg.norm(axis), int(use.sum())
