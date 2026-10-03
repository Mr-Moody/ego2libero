"""Frame and video output."""

from pathlib import Path

import numpy as np

from e2l_common.stub import not_implemented


def save_video(frames: np.ndarray, path: Path, fps: int = 20) -> Path:
    """Write frames (T,H,W,3) uint8 RGB to an mp4."""
    not_implemented("e2l_sim.render.save_video")
