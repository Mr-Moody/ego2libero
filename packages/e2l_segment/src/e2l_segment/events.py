"""Gripper aperture to binary grip, and grasp/release events."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.stub import not_implemented


@dataclass(frozen=True)
class GripEvent:
    kind: Literal["grasp", "release"]
    index: int  # frame index in the DemoTrajectory
    t: float  # seconds


def aperture_to_grip(t: np.ndarray, aperture: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """Binarise aperture with hysteresis and minimum dwell time.

    Inputs: t (N,) seconds, aperture (N,) metres (NaN where the hand is invalid).
    Output: grip (N,) int8, 0 open / 1 closed. Closes below `cfg.close_threshold`, reopens
    above `cfg.open_threshold`; state changes shorter than `cfg.min_dwell_s` are discarded."""
    not_implemented("e2l_segment.events.aperture_to_grip")


def grip_events(t: np.ndarray, grip: np.ndarray) -> list[GripEvent]:
    """Grasp (0->1) and release (1->0) transitions of a binary grip (N,), in time order."""
    not_implemented("e2l_segment.events.grip_events")
