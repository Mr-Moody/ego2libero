import numpy as np
import pytest

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.assign import nearest_valid_index, reference_object
from e2l_segment.segment import largest_gap, segment_demo

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
    keypoints = np.zeros((N, 21, 3))
    keypoints[:, [4, 8]] = wrist[:, None]  # pinch point at the wrist
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
        keypoints_table=keypoints,
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
    ref = segment_demo(_demo(), SegmentConfig()).segments[0]
    s0 = segs.segments[0]
    assert len(s0.T_obj_hand) == 31 and np.isclose(s0.t_end - s0.t_start, 1.0)  # uniform in time
    np.testing.assert_allclose(s0.T_obj_hand, ref.T_obj_hand, atol=1e-9)  # filled on the line
    assert np.array_equal(s0.grip, ref.grip)
    assert all(np.isfinite(s.T_obj_hand).all() for s in segs.segments)
    assert np.isclose(largest_gap(_demo(hand_valid=valid), s0), 11 / FPS)


def test_edge_invalid_hand_frames_are_trimmed():
    valid = np.ones(90, bool)
    valid[:5] = False
    s0 = segment_demo(_demo(hand_valid=valid), SegmentConfig()).segments[0]
    assert len(s0.T_obj_hand) == 26 and np.isclose(s0.t_start, 5 / FPS)


def test_nearest_valid_index_sides():
    v = np.array([0, 1, 0, 0, 0, 1, 0], bool)
    assert nearest_valid_index(v, 3, "before") == 1
    assert nearest_valid_index(v, 3, "after") == 5
    assert nearest_valid_index(v, 1, "before") == 1 and nearest_valid_index(v, 1, "after") == 1
    assert nearest_valid_index(v, 0, "before") == 1  # nothing before: nearest
    assert nearest_valid_index(v, 6, "after") == 5  # nothing after: nearest
    with pytest.raises(ValueError):
        nearest_valid_index(np.zeros(3, bool), 1, "before")


def test_object_hidden_at_grasp_uses_pose_from_before():
    demo = _demo()
    demo.obj_valid[15:37, 0] = False
    demo.T_table_obj[15:37, 0] = np.nan
    segs = segment_demo(demo, SegmentConfig())
    np.testing.assert_allclose(segs.segments[0].T_obj_hand[-1, :3, 3], [0, 0, 0.05], atol=1e-9)


def test_single_object_demo_references_carried_object_after_release():
    for hidden in (False, True):
        demo = _demo(plate_seen=False)
        if hidden:
            demo.obj_valid[55:66, 0] = False
            demo.T_table_obj[55:66, 0] = np.nan
        segs = segment_demo(demo, SegmentConfig())
        assert [s.ref_object for s in segs.segments] == ["bowl"] * 3
        s1, s2 = segs.segments[1:]  # post-release pose: bowl rests 0.05 below the release hand
        np.testing.assert_allclose(s1.T_obj_hand[-1, :3, 3], [0, 0, 0.05], atol=1e-9)
        np.testing.assert_allclose(s2.T_obj_hand[0, :3, 3], [0, 0, 0.05], atol=1e-9)


def test_reference_object_all_excluded_falls_back_and_distinguishes_errors():
    demo = _demo(plate_seen=False)
    assert reference_object(demo, 60, exclude=frozenset({"bowl"})) == "bowl"
    demo.obj_valid[:] = False
    with pytest.raises(ValueError, match="no object is ever visible"):
        reference_object(demo, 60, exclude=frozenset({"bowl"}))


def test_never_valid_hand_error_names_demo():
    demo = _demo(hand_valid=np.zeros(90, bool))
    with pytest.raises(ValueError, match="20261008_test_000"):
        segment_demo(demo, SegmentConfig())
    with pytest.raises(ValueError, match="20261008_test_000"):
        reference_object(demo, 30)


def test_reference_uses_pinch_point_not_wrist():
    demo = _demo()
    demo.keypoints_table[30, [4, 8]] = PLATE  # fingers far from the wrist, at the plate
    assert reference_object(demo, 30) == "plate"
    demo.keypoints_table[30, 4] = np.nan  # non-finite pinch: fall back to the wrist
    assert reference_object(demo, 30) == "bowl"


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
