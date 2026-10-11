import numpy as np

from e2l_common.geometry import make_T
from e2l_common.schemas import DemoTrajectory
from e2l_perception.pipeline import FrameObservation
from e2l_perception.viz import draw_observation, plot_trajectory

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def test_draw_observation_draws_hand_table_and_objects(make_detection):
    frame = np.zeros((480, 640, 3), np.uint8)
    T = make_T(t=np.array([0.0, 0.0, 0.5]))
    obs = FrameObservation(
        T_cam_table=T,
        T_cam_obj={"bowl": make_T(t=np.array([0.05, 0.0, 0.5]))},
        T_cam_hand=make_T(t=np.array([-0.05, 0.0, 0.5])),
        detection=make_detection("Right", (320.0, 240.0)),
    )
    draw_observation(frame, obs, K, np.zeros(5))
    assert frame.any()


def test_draw_observation_tolerates_empty_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_observation(frame, FrameObservation(T_cam_table=None), K, np.zeros(5))
    assert frame.any()  # HUD text still says nothing was detected


def test_plot_trajectory_writes_png_with_invalid_frames(tmp_path):
    N = 20
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = np.linspace(0, 0.2, N)
    T[5:8] = np.nan
    valid = np.ones(N, bool)
    valid[5:8] = False
    aperture = np.linspace(0.08, 0.02, N)
    aperture[5:8] = np.nan
    demo = DemoTrajectory(
        demo_id="20261008_test_000",
        fps=30.0,
        object_names=["bowl"],
        t=np.arange(N) / 30,
        T_table_hand=T,
        hand_valid=valid,
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=np.tile(np.eye(4), (N, 1, 1, 1)),
        obj_valid=np.ones((N, 1), bool),
    )
    out = plot_trajectory(demo, tmp_path / "plots" / "traj.png")
    assert out.is_file() and out.stat().st_size > 1000
