"""Reference object for each segment."""

from typing import Literal

import numpy as np

from e2l_common.schemas import DemoTrajectory

Side = Literal["nearest", "before", "after"]


def reference_object(
    demo: DemoTrajectory,
    event_index: int,
    exclude: frozenset[str] = frozenset(),
    side: Side = "nearest",
) -> str:
    """Name of the object nearest the hand at the event that ends a segment.

    Distance is between the pinch point (mean of `demo.keypoints_table[i, [4, 8]]`, thumb and
    index tip; the wrist `T_table_hand[i, :3, 3]` if those are not finite) at the hand-valid
    frame nearest `event_index`, and each object's table-frame position at the valid frame chosen
    by `side` (see `nearest_valid_index`). Objects in `exclude` (e.g. the one being carried at a
    release) are skipped unless nothing else was ever seen, in which case they are used with
    their post-event pose (`side="after"`). Objects never seen are skipped; ValueError if none."""
    try:
        i = nearest_valid_index(demo.hand_valid, event_index)
    except ValueError:
        raise ValueError(f"{demo.demo_id}: hand is never valid") from None
    pinch = demo.keypoints_table[i, [4, 8]].mean(axis=0)
    hand = pinch if np.isfinite(pinch).all() else demo.T_table_hand[i, :3, 3]
    seen = [k for k in range(len(demo.object_names)) if demo.obj_valid[:, k].any()]
    if not seen:
        raise ValueError(f"{demo.demo_id}: no object is ever visible")
    kept = [k for k in seen if demo.object_names[k] not in exclude]
    if not kept:
        kept, side = seen, "after"
    dist = [
        np.linalg.norm(
            demo.T_table_obj[nearest_valid_index(demo.obj_valid[:, k], event_index, side), k, :3, 3]
            - hand
        )
        for k in kept
    ]
    return demo.object_names[kept[int(np.argmin(dist))]]


def nearest_valid_index(valid: np.ndarray, index: int, side: Side = "nearest") -> int:
    """Index of a valid frame near `index` in a boolean mask (N,).

    "nearest": closest, ties to the earlier frame. "before": last valid <= index. "after":
    first valid >= index. Each falls back to nearest when that side has none. ValueError if no
    frame is valid."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("no valid frames")
    if side == "before" and (b := idx[idx <= index]).size:
        return int(b[-1])
    if side == "after" and (a := idx[idx >= index]).size:
        return int(a[0])
    return int(idx[np.argmin(np.abs(idx - index))])
