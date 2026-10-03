"""Phone video normalisation and frame reading. Timestamps always come from frame index / fps of
the normalised constant-frame-rate video, never from wall-clock time."""

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def normalise(src: Path, dst: Path, fps: float, crf: int = 18) -> Path:
    """Re-encode `src` to constant-frame-rate H.264 (yuv420p, no audio) at `fps` with ffmpeg.

    ffmpeg applies the phone's rotation metadata, so frames come out upright; calibrate the
    camera from videos normalised the same way so intrinsics match."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (sudo apt install ffmpeg)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vf", f"fps={fps}", "-r", str(fps),  # fps filter + output rate: CFR on ffmpeg 4.4+
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-an", str(dst),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return dst


class VideoReader:
    """Iterate a video as (frame BGR uint8 (H,W,3), t seconds) with t = frame index / fps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        if self.fps <= 0:
            raise RuntimeError(f"{self.path} reports no frame rate; normalise it first")

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame, index / self.fps
                index += 1
        finally:
            cap.release()
