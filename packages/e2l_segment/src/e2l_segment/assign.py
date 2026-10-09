"""Reference object for each segment."""

import numpy as np

from e2l_common.schemas import DemoTrajectory


def reference_object(
    demo: DemoTrajectory, event_index: int, exclude: frozenset[str] = frozenset()
) -> str:
    """Name of the object nearest the hand at the event that ends a segment.

    Distance is between `demo.T_table_hand[i, :3, 3]` and each valid `demo.T_table_obj[i, k, :3, 3]`
    (table frame, metres) at `event_index`, falling back to the nearest valid frame.
    Objects in `exclude` (e.g. the one being carried at a release) and objects never seen are
    skipped; ValueError if none is left."""
    hand = demo.T_table_hand[nearest_valid_index(demo.hand_valid, event_index), :3, 3]
    best, best_d = None, np.inf
    for k, name in enumerate(demo.object_names):
        if name in exclude or not demo.obj_valid[:, k].any():
            continue
        j = nearest_valid_index(demo.obj_valid[:, k], event_index)
        d = float(np.linalg.norm(demo.T_table_obj[j, k, :3, 3] - hand))
        if d < best_d:
            best, best_d = name, d
    if best is None:
        raise ValueError(f"{demo.demo_id}: no object is ever visible")
    return best


def nearest_valid_index(valid: np.ndarray, index: int) -> int:
    """Index of the valid frame closest to `index` in a boolean mask (N,).

    Ties go to the earlier frame; ValueError if none is valid."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("no valid frames")
    return int(idx[np.argmin(np.abs(idx - index))])
