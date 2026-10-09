import numpy as np
import pytest

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.assign import nearest_valid_index, reference_object
from e2l_segment.segment import segment_demo

FPS = 30.0
BOWL = np.array([0.1, 0.0, 0.0])
PLATE = np.array([-0.1, 0.1, 0.0])


def _demo(hand_valid: np.ndarray | None = None, plate_seen: bool = True) -> DemoTrajectory:
    """Approach the bowl (grasp at 30), carry it to the plate (release at 60), retreat to 89."""
    N = 90
    wrist = np.zeros((N, 3))
    wrist[:31] = np.linspace(BOWL + [0, 0, 0.2], BOWL + [0, 0, 0.05], 31)
    wrist[30:61] = np.linspace(BOWL + [0, 0, 0.05], PLATE + [0, 0, 0.08], 31)
    wrist[60:] = np.linspace(PLATE + [0, 0, 0.08], PLATE + [0, 0, 0.25], 30)
    T_hand = np.tile(np.eye(4), (N, 1, 1))
    T_hand[:, :3, 3] = wrist
    T_obj = np.tile(np.eye(4), (N, 2, 1, 1))
    T_obj[:, 0, :3, 3] = BOWL
    T_obj[30:61, 0, :3, 3] = wrist[30:61] - [0, 0, 0.05]  # bowl travels with the hand
    T_obj[61:, 0, :3, 3] = PLATE + [0, 0, 0.03]
    T_obj[:, 1, :3, 3] = PLATE
    obj_valid = np.ones((N, 2), bool)
    if not plate_seen:
        obj_valid[:, 1] = False
        T_obj[:, 1] = np.nan
    aperture = np.full(N, 0.08)
    aperture[30:60] = 0.02
    valid = np.ones(N, bool) if hand_valid is None else hand_valid
    T_hand[~valid] = np.nan
    aperture[~valid] = np.nan
    return DemoTrajectory(
        demo_id="20261008_test_000",
        fps=FPS,
        object_names=["bowl", "plate"],
        t=np.arange(N) / FPS,
        T_table_hand=T_hand,
        hand_valid=valid,
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=T_obj,
        obj_valid=obj_valid,
    )


def test_nearest_valid_index():
    v = np.array([0, 1, 0, 0, 1, 0], bool)
    assert nearest_valid_index(v, 0) == 1
    assert nearest_valid_index(v, 3) == 4
    assert nearest_valid_index(v, 2) == 1  # tie -> earlier
    with pytest.raises(ValueError):
        nearest_valid_index(np.zeros(3, bool), 1)


def test_reference_object_is_nearest_at_event_and_skips_unseen():
    demo = _demo()
    assert reference_object(demo, 30) == "bowl"
    assert reference_object(demo, 60) == "bowl"  # the carried bowl is nearest at release
    assert reference_object(demo, 60, exclude=frozenset({"bowl"})) == "plate"
    assert reference_object(_demo(plate_seen=False), 60) == "bowl"


def test_reference_object_without_any_object_raises():
    demo = _demo()
    demo.obj_valid[:] = False
    with pytest.raises(ValueError, match="20261008_test_000"):
        reference_object(demo, 30)


def test_segments_cut_at_events_relative_to_reference_object():
    segs = segment_demo(_demo(), SegmentConfig())
    assert segs.demo_id == "20261008_test_000" and segs.fps == FPS
    assert [s.ref_object for s in segs.segments] == ["bowl", "plate", "plate"]
    s0, s1, s2 = segs.segments
    assert len(s0.T_obj_hand) == 31 and np.isclose(s0.t_start, 0) and np.isclose(s0.t_end, 1.0)
    np.testing.assert_allclose(s0.T_obj_hand[-1, :3, 3], [0, 0, 0.05], atol=1e-9)
    assert s0.grip[-1] == 1 and not s0.grip[:-1].any()  # ends on the grasp frame
    assert s1.grip[:-1].all() and s1.grip[-1] == 0  # ends on the release frame
    np.testing.assert_allclose(s1.T_obj_hand[-1, :3, 3], [0, 0, 0.08], atol=1e-9)
    assert np.isclose(s2.t_start, 2.0) and np.isclose(s2.t_end, 89 / FPS)


def test_invalid_hand_frames_are_dropped_from_segments():
    valid = np.ones(90, bool)
    valid[10:20] = False
    segs = segment_demo(_demo(hand_valid=valid), SegmentConfig())
    assert len(segs.segments[0].T_obj_hand) == 21
    assert all(np.isfinite(s.T_obj_hand).all() for s in segs.segments)


def test_no_events_gives_one_segment_and_event_on_last_frame_adds_none():
    demo = _demo()
    demo.aperture[:] = 0.08
    segs = segment_demo(demo, SegmentConfig())
    assert len(segs.segments) == 1 and len(segs.segments[0].T_obj_hand) == 90

    demo = _demo()
    demo.aperture[:] = 0.08
    demo.aperture[60:] = 0.02  # grasp at 60, held to the end: no trailing release
    demo.aperture[89] = 0.02
    segs = segment_demo(demo, SegmentConfig())
    assert all(len(s.T_obj_hand) >= 2 for s in segs.segments)
