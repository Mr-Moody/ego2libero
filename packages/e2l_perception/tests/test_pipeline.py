import cv2
import numpy as np
import pytest

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.geometry import compose, inv_T, make_T, rotvec_to_matrix
from e2l_perception.calibration import Intrinsics
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.pipeline import (
    FrameObservation,
    assemble_trajectory,
    fill_short_gaps,
    observe_frames,
)

FPS = 30.0
T_CAM_TABLE = make_T(rotvec_to_matrix(np.array([2.2, 0.0, 0.0])), np.array([0.0, 0.1, 0.6]))
T_TABLE_BOWL = make_T(t=np.array([0.1, 0.0, 0.0]))


def _hand_table(i: int, hand_pts: np.ndarray) -> np.ndarray:
    """Keypoints in the table frame at frame i: the canonical hand moving 1 cm/frame along x."""
    return hand_pts + np.array([-0.2 + 0.01 * i, 0.05, 0.15])


def _obs(i: int, hand_pts: np.ndarray, hand=True, board=True, bowl=True) -> FrameObservation:
    kp_table = _hand_table(i, hand_pts)
    kp_cam = (T_CAM_TABLE[:3, :3] @ kp_table.T).T + T_CAM_TABLE[:3, 3]
    T_cam_hand = compose(T_CAM_TABLE, make_T(compute_palm_frame(kp_table), kp_table[0]))
    return FrameObservation(
        T_cam_table=T_CAM_TABLE if board else None,
        T_cam_obj={"bowl": compose(T_CAM_TABLE, T_TABLE_BOWL)} if bowl else {},
        T_cam_hand=T_cam_hand if hand else None,
        keypoints_cam=kp_cam if hand else None,
        detection=None,
    )


def _cfg(**kw) -> PerceptionConfig:
    return PerceptionConfig(smooth=False, **kw)


def test_fill_short_gaps_only_fills_interior_gaps_up_to_max():
    v = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], bool)
    out = fill_short_gaps(v, max_gap=2)
    np.testing.assert_array_equal(out, [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0])


def test_assemble_expresses_hand_and_objects_in_table_frame(hand_pts):
    obs = [_obs(i, hand_pts) for i in range(10)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl", "plate"], _cfg())
    assert demo.hand_valid.all()
    np.testing.assert_allclose(demo.t, np.arange(10) / FPS)
    np.testing.assert_allclose(demo.keypoints_table[4], _hand_table(4, hand_pts), atol=1e-9)
    np.testing.assert_allclose(demo.T_table_hand[4, :3, 3], _hand_table(4, hand_pts)[0], atol=1e-9)
    np.testing.assert_allclose(demo.aperture, np.linalg.norm(hand_pts[4] - hand_pts[8]), atol=1e-9)
    np.testing.assert_allclose(demo.T_table_obj[3, 0], T_TABLE_BOWL, atol=1e-9)
    assert demo.obj_valid[:, 0].all() and not demo.obj_valid[:, 1].any()
    assert np.isnan(demo.T_table_obj[:, 1]).all()


def test_short_gap_is_interpolated_long_gap_stays_invalid(hand_pts):
    hand = np.ones(40, bool)
    hand[10:14] = False  # 4 frames = 0.13 s <= 0.2 s
    hand[20:30] = False  # 10 frames = 0.33 s > 0.2 s
    obs = [_obs(i, hand_pts, hand=bool(hand[i])) for i in range(40)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], _cfg(max_gap_s=0.2))
    assert demo.hand_valid[10:14].all() and not demo.hand_valid[20:30].any()
    np.testing.assert_allclose(demo.keypoints_table[12], _hand_table(12, hand_pts), atol=1e-9)
    assert np.isnan(demo.keypoints_table[25]).all() and np.isnan(demo.aperture[25])
    assert np.isnan(demo.T_table_hand[25]).all()


def test_board_missing_invalidates_hand(hand_pts):
    obs = [_obs(i, hand_pts, board=i != 0) for i in range(5)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], _cfg())
    assert not demo.hand_valid[0] and demo.hand_valid[1:].all()
    assert not demo.obj_valid[0, 0]


def test_smoothing_keeps_linear_motion(hand_pts):
    obs = [_obs(i, hand_pts) for i in range(30)]
    cfg = PerceptionConfig(smooth=True)
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], cfg)
    truth = np.stack([_hand_table(i, hand_pts) for i in range(30)])
    assert np.abs(demo.keypoints_table - truth).max() < 0.005


def test_observe_frames_rejects_mismatched_intrinsics(tmp_path):
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    calib = Intrinsics(K=np.eye(3), dist=np.zeros(5), size=(1920, 1080)).save(tmp_path / "c.json")
    capture = CaptureConfig(intrinsics=calib)
    with pytest.raises(ValueError, match="1920"):
        next(observe_frames(video, PerceptionConfig(), capture))


def test_observe_frames_rejects_mirrored(tmp_path):
    with pytest.raises(ValueError, match="mirrored"):
        next(observe_frames(tmp_path / "x.mp4", PerceptionConfig(mirrored=True), CaptureConfig()))


def test_assemble_records_the_camera_centre_in_the_table_frame(hand_pts):
    obs = [_obs(i, hand_pts, board=i != 2) for i in range(5)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], _cfg())
    np.testing.assert_allclose(demo.cam_table[0], inv_T(T_CAM_TABLE)[:3, 3], atol=1e-9)
    assert np.isnan(demo.cam_table[2]).all()
