"""Reference object for each segment."""

import numpy as np

from e2l_common.schemas import DemoTrajectory
from e2l_common.stub import not_implemented


def reference_object(demo: DemoTrajectory, event_index: int) -> str:
    """Name of the object nearest the hand at the event that ends a segment.

    Distance is between `demo.T_table_hand[i, :3, 3]` and each valid `demo.T_table_obj[i, k, :3, 3]`
    (table frame, metres) at `event_index`, falling back to the nearest valid frame."""
    not_implemented("e2l_segment.assign.reference_object")


def nearest_valid_index(valid: np.ndarray, index: int) -> int:
    """Index of the valid frame closest to `index` in a boolean mask (N,)."""
    not_implemented("e2l_segment.assign.nearest_valid_index")
