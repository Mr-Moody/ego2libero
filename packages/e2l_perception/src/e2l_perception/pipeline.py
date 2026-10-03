"""Per-demo perception: normalised video -> DemoTrajectory in the table frame."""

from pathlib import Path

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.schemas import DemoTrajectory
from e2l_common.stub import not_implemented


def run_demo(
    video: Path, demo_id: str, cfg: PerceptionConfig, capture: CaptureConfig
) -> DemoTrajectory:
    """Per frame: T_cam_table and T_cam_obj from markers, T_cam_hand and keypoints from the hand
    tracker; express everything in the table frame (T_table_x = inv(T_cam_table) @ T_cam_x),
    compute aperture (thumb tip to index tip, metres), then optionally RTS-smooth."""
    not_implemented("e2l_perception.pipeline.run_demo")
