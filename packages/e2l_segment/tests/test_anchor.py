import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.anchor import anchor_depth, carry_runs, depth_ratios

FPS = 30.0
CAM = np.array([0.0, -0.45, 0.42])
BOWL = np.array([0.1, 0.1, 0.0])
PLATE = np.array([-0.15, 0.2, 0.0])
OFFSET = np.array([0.06, 0.0, 0.07])  # pinch point on the rim, bowl frame
N = 90


def _truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True pinch (N,3), bowl position (N,3) and grip (N,): approach, carry 30..59, retreat."""
    pinch = np.zeros((N, 3))
    pinch[:30] = np.linspace(BOWL + OFFSET + [0, -0.2, 0.15], BOWL + OFFSET, 30)
    pinch[30:60] = np.linspace(BOWL + OFFSET, PLATE + OFFSET + [0, 0, 0.02], 30)
    pinch[60:] = np.linspace(PLATE + OFFSET + [0, 0, 0.02], PLATE + [0, -0.1, 0.3], 30)
    bowl = np.tile(BOWL, (N, 1))
    bowl[30:60] = pinch[30:60] - OFFSET
    bowl[60:] = pinch[59] - OFFSET
    grip = np.zeros(N, np.int8)
    grip[30:60] = 1
    return pinch, bowl, grip


def _demo(depth_scale: np.ndarray, cam: bool = True) -> DemoTrajectory:
    """Every hand keypoint pushed along its camera ray by `depth_scale` (N,)."""
    pinch, bowl, _ = _truth()
    rest = np.array([[0.0, 0.0, 0.0], [0.0, -0.08, 0.02], [0.02, -0.05, 0.0]])  # wrist, knuckles
    kp = np.repeat(pinch[:, None], 21, axis=1)
    kp[:, :3] += rest
    kp[:, 4] += [0.0, 0.01, 0.0]
    kp[:, 8] -= [0.0, 0.01, 0.0]  # thumb and index tips 2 cm apart, centred on the pinch point
    kp = CAM + depth_scale[:, None, None] * (kp - CAM)
    T_hand = np.tile(np.eye(4), (N, 1, 1))
    T_hand[:, :3, 3] = kp[:, 0]
    T_obj = np.tile(np.eye(4), (N, 2, 1, 1))
    T_obj[:, 0, :3, 3] = bowl
    T_obj[:, 1, :3, 3] = PLATE
    return DemoTrajectory(
        demo_id="20261008_test_000",
        fps=FPS,
        object_names=["bowl", "plate"],
        t=np.arange(N) / FPS,
        T_table_hand=T_hand,
        hand_valid=np.ones(N, bool),
        keypoints_table=kp,
        aperture=np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1),
        T_table_obj=T_obj,
        obj_valid=np.ones((N, 2), bool),
        cam_table=np.tile(CAM, (N, 1)) if cam else None,
    )


def _pinch(demo: DemoTrajectory) -> np.ndarray:
    return demo.keypoints_table[:, [4, 8]].mean(axis=1)


def test_carry_runs_are_inclusive_closed_intervals():
    grip = np.array([0, 1, 1, 0, 0, 1, 1, 1], np.int8)
    assert carry_runs(grip) == [(1, 2), (5, 7)]


def test_carry_frames_land_at_a_constant_offset_from_the_carried_object():
    scale = np.full(N, 1.0)
    scale[40:46] = 1.4  # depth spike mid-carry
    _, bowl, grip = _truth()
    out = anchor_depth(_demo(scale), grip, SegmentConfig())
    before = _pinch(_demo(scale))[30:60] - bowl[30:60]
    after = _pinch(out)[30:60] - bowl[30:60]
    spread = np.linalg.norm(after - np.median(after, axis=0), axis=1)
    assert np.linalg.norm(before - np.median(before, axis=0), axis=1).max() > 0.1
    assert spread.max() < 0.01  # the spike is gone; the pinch rides the bowl


def test_ratios_are_nan_outside_carries():
    _, _, grip = _truth()
    demo = _demo(np.full(N, 1.2))
    r = depth_ratios(demo, grip, demo.cam_table)
    assert np.isfinite(r[30:60]).all() and np.isnan(r[:30]).all() and np.isnan(r[60:]).all()
    np.testing.assert_allclose(r[30:60], 1.0, atol=0.02)  # a constant bias is not a spike


def test_free_motion_takes_the_nearest_anchor_ratio_and_drops_short_spikes():
    scale = np.full(N, 1.2)
    scale[18:24] = 1.6  # 0.2 s spike on the approach, shorter than the median window
    _, _, grip = _truth()
    out = anchor_depth(_demo(scale), grip, SegmentConfig(depth_window_s=1.0))
    clean = _pinch(_demo(np.full(N, 1.2)))
    np.testing.assert_allclose(_pinch(out)[:30], clean[:30], atol=0.01)
    np.testing.assert_allclose(_pinch(out)[60:], clean[60:], atol=0.01)


def test_rescaling_moves_the_wrist_but_keeps_the_hand_rotation():
    _, _, grip = _truth()
    demo = _demo(np.full(N, 1.2))
    demo.T_table_hand[:, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    out = anchor_depth(demo, grip, SegmentConfig())
    np.testing.assert_allclose(out.T_table_hand[:, :3, :3], demo.T_table_hand[:, :3, :3])
    np.testing.assert_allclose(out.T_table_hand[:, :3, 3], out.keypoints_table[:, 0], atol=1e-9)
    np.testing.assert_allclose(
        out.aperture, np.linalg.norm(out.keypoints_table[:, 4] - out.keypoints_table[:, 8], axis=1)
    )


def test_without_camera_or_carry_the_demo_is_unchanged():
    _, _, grip = _truth()
    demo = _demo(np.full(N, 1.2), cam=False)
    assert anchor_depth(demo, grip, SegmentConfig()) is demo
    demo = _demo(np.full(N, 1.2))
    assert anchor_depth(demo, np.zeros(N, np.int8), SegmentConfig()) is demo
    assert anchor_depth(demo, grip, SegmentConfig(anchor_depth=False)) is demo
