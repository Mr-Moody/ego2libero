import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.viz import plot_segments


def test_plot_segments_writes_png(tmp_path):
    N = 90
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = np.linspace(0.1, -0.1, N)
    aperture = np.full(N, 0.08)
    aperture[30:60] = 0.02
    aperture[70:75] = np.nan
    T_obj = np.tile(np.eye(4), (N, 2, 1, 1))
    T_obj[:, 1, 0, 3] = -0.2  # a second object, since a release needs one other than the carried
    demo = DemoTrajectory(
        demo_id="20261008_test_000",
        fps=30.0,
        object_names=["bowl", "plate"],
        t=np.arange(N) / 30,
        T_table_hand=T,
        hand_valid=np.ones(N, bool),
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=T_obj,
        obj_valid=np.ones((N, 2), bool),
    )
    out = plot_segments(demo, SegmentConfig(), tmp_path / "seg.png")
    assert out.is_file() and out.stat().st_size > 1000
