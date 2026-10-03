"""Calibrate phone intrinsics from a video of the printed checkerboard.

Film the checkerboard page from scripts/make_markers.py from many angles and distances, with
the same camera mode (lens, resolution, stabilisation off if possible) used for the demos.

    uv run python scripts/calibrate_phone.py data/raw/calib/phone_calib.mp4 --device phone
"""

import itertools
from pathlib import Path

import cv2
import numpy as np
import typer

from e2l_common.paths import DataPaths
from e2l_perception.calibration import BOARD_SIZE, SQUARE_M, calibrate_from_frames, find_corners
from e2l_perception.video import VideoReader, normalise


def main(
    video: Path,
    device: str = typer.Option("phone", help="Name of data/calib/<device>.json."),
    fps: float = typer.Option(30.0, help="Normalisation frame rate (match the demos)."),
    max_views: int = typer.Option(40, help="Checkerboard views used in the calibration."),
    cols: int = typer.Option(BOARD_SIZE[0], help="Inner corners per row."),
    rows: int = typer.Option(BOARD_SIZE[1], help="Inner corners per column."),
    square: float = typer.Option(SQUARE_M, help="Square side, metres."),
) -> None:
    paths = DataPaths.default()
    normalised = paths.root / "normalised" / "calib" / f"{video.stem}.mp4"
    normalise(video, normalised, fps)  # same rotation handling as the demos
    reader = VideoReader(normalised)

    # Scan a subsample for frames where the board is found, then keep an even spread.
    step = max(1, len(reader) // (max_views * 4))
    found = []
    for frame, _ in itertools.islice(reader, 0, None, step):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if find_corners(gray, (cols, rows)) is not None:
            found.append(gray)
    typer.echo(f"checkerboard found in {len(found)} of {len(reader) // step} scanned frames")
    keep = [
        found[i] for i in np.linspace(0, len(found) - 1, min(max_views, len(found))).astype(int)
    ]

    intr = calibrate_from_frames(keep, (cols, rows), square, min_views=min(10, max_views))
    out = intr.save(paths.calib(device))
    typer.echo(f"RMS reprojection error {intr.rms:.3f} px, size {intr.size}; wrote {out}")
    typer.echo(np.array2string(intr.K, precision=1))


if __name__ == "__main__":
    typer.run(main)
