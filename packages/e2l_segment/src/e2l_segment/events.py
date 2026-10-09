"""Gripper aperture to binary grip, and grasp/release events."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from e2l_common.config import SegmentConfig


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
    grip = np.zeros(len(aperture), np.int8)
    state = 0
    for i, a in enumerate(aperture):
        if np.isfinite(a):
            if state == 0 and a < cfg.close_threshold:
                state = 1
            elif state == 1 and a > cfg.open_threshold:
                state = 0
        grip[i] = state
    return _drop_short_runs(t, grip, cfg.min_dwell_s)


def _runs(grip: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of constant-value runs."""
    edges = np.r_[0, np.flatnonzero(np.diff(grip)) + 1, len(grip)]
    return list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))


def _drop_short_runs(t: np.ndarray, grip: np.ndarray, min_dwell_s: float) -> np.ndarray:
    """Flip runs shorter than `min_dwell_s` (run length x median frame period), shortest first,
    until none remain or only one run is left."""
    if len(grip) < 2:
        return grip
    dt = float(np.median(np.diff(t)))
    grip = grip.copy()
    while len(runs := _runs(grip)) > 1:
        start, stop = min(runs, key=lambda r: r[1] - r[0])
        if (stop - start) * dt >= min_dwell_s:
            break
        grip[start:stop] = 1 - grip[start]
    return grip


def grip_events(t: np.ndarray, grip: np.ndarray) -> list[GripEvent]:
    """Grasp (0->1) and release (1->0) transitions of a binary grip (N,), in time order."""
    return [
        GripEvent("grasp" if grip[i] == 1 else "release", int(i), float(t[i]))
        for i in np.flatnonzero(np.diff(grip)) + 1
    ]
