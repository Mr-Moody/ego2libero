"""Camera intrinsics: loading and checkerboard calibration."""

from pathlib import Path

from e2l_common.stub import not_implemented


def load_intrinsics(path: Path):
    """Read data/calib/<device>.json -> (K (3,3), dist (n,), (width, height))."""
    not_implemented("e2l_perception.calibration.load_intrinsics")
