"""Phone video normalisation and frame reading."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from e2l_common.stub import not_implemented


def normalise(src: Path, dst: Path, fps: float) -> Path:
    """Re-encode `src` to constant-frame-rate H.264 at `fps` with ffmpeg; returns `dst`."""
    not_implemented("e2l_perception.video.normalise")


class VideoReader:
    """Yields (frame BGR uint8 (H,W,3), t seconds) with t = frame index / fps."""

    def __init__(self, path: Path) -> None:
        not_implemented("e2l_perception.video.VideoReader.__init__")

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        not_implemented("e2l_perception.video.VideoReader.__iter__")
