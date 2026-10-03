"""Frame and video output."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def upright(frames: np.ndarray) -> np.ndarray:
    """LIBERO renders upside down; flip H and W for viewing only (policies see raw images)."""
    return np.ascontiguousarray(np.asarray(frames)[..., ::-1, ::-1, :])


def save_video(frames: np.ndarray, path: Path, fps: int = 20) -> Path:
    """Write frames (T,H,W,3) uint8 RGB to an mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps, codec="libx264")
    return path
