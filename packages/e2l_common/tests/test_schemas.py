import numpy as np
import pytest

from e2l_common.schemas import (
    DemoTrajectory,
    HandSegment,
    HandSegments,
    RobotSegment,
    RobotSegments,
    SimEpisode,
)


def demo(n: int = 5, k: int = 2) -> DemoTrajectory:
    return DemoTrajectory(
        demo_id="20261004_bowlplate_007",
        fps=30.0,
        object_names=["bowl", "plate"][:k],
        t=np.arange(n) / 30.0,
        T_table_hand=np.tile(np.eye(4), (n, 1, 1)),
        hand_valid=np.ones(n, bool),
        keypoints_table=np.zeros((n, 21, 3)),
        aperture=np.linspace(0.08, 0.02, n),
        T_table_obj=np.tile(np.eye(4), (n, k, 1, 1)),
        obj_valid=np.ones((n, k), bool),
    )


def test_demo_round_trip(tmp_path):
    d = demo()
    npz, js = d.save(tmp_path / "perception" / d.demo_id)
    assert npz.suffix == ".npz" and js.suffix == ".json"
    back = DemoTrajectory.load(tmp_path / "perception" / d.demo_id)
    assert back.object_names == d.object_names and back.fps == d.fps
    np.testing.assert_array_equal(back.aperture, d.aperture)
    np.testing.assert_array_equal(back.T_table_obj, d.T_table_obj)


def test_demo_rejects_mismatched_lengths():
    d = demo()
    with pytest.raises(ValueError, match="N="):
        DemoTrajectory(**{**d.__dict__, "aperture": np.zeros(4)})
    with pytest.raises(ValueError, match="expected"):
        DemoTrajectory(**{**d.__dict__, "keypoints_table": np.zeros((5, 20, 3))})
    with pytest.raises(ValueError, match="object_names"):
        DemoTrajectory(**{**d.__dict__, "object_names": ["bowl"]})


def test_segments_round_trip(tmp_path):
    hs = HandSegments(
        demo_id="20261004_bowlplate_007",
        fps=30.0,
        segments=[
            HandSegment(
                ref_object="bowl",
                t_start=0.0,
                t_end=1.0,
                T_obj_hand=np.tile(np.eye(4), (4, 1, 1)),
                grip=np.array([0, 0, 1, 1]),
            ),
            HandSegment(
                ref_object="plate",
                t_start=1.0,
                t_end=2.0,
                T_obj_hand=np.tile(np.eye(4), (3, 1, 1)),
                grip=np.array([1, 1, 0]),
            ),
        ],
    )
    hs.save(tmp_path / "s")
    back = HandSegments.load(tmp_path / "s")
    assert [s.ref_object for s in back.segments] == ["bowl", "plate"]
    np.testing.assert_array_equal(back.segments[1].grip, [1, 1, 0])

    rs = RobotSegments(
        demo_id="x",
        fps=20.0,
        segments=[
            RobotSegment(
                ref_object="bowl",
                T_obj_ee=np.tile(np.eye(4), (2, 1, 1)),
                gripper=np.array([-1.0, 1.0]),
                feasible=np.array([True, True]),
            )
        ],
    )
    rs.save(tmp_path / "r.npz")
    assert RobotSegments.load(tmp_path / "r").segments[0].gripper.tolist() == [-1.0, 1.0]


def test_sim_episode_round_trip_and_wrong_schema(tmp_path):
    ep = SimEpisode(
        task="put the bowl on the plate",
        success=True,
        source_demo_ids=["20261004_bowlplate_007"],
        images={
            "agentview": np.zeros((3, 8, 8, 3), np.uint8),
            "robot0_eye_in_hand": np.zeros((3, 8, 8, 3), np.uint8),
        },
        state=np.zeros((3, 8)),
        actions=np.zeros((3, 7)),
        object_poses={"bowl": np.eye(4)},
    )
    ep.save(tmp_path / "ep")
    back = SimEpisode.load(tmp_path / "ep")
    assert back.success and set(back.images) == {"agentview", "robot0_eye_in_hand"}
    with pytest.raises(ValueError, match="does not hold"):
        DemoTrajectory.load(tmp_path / "ep")
    with pytest.raises(ValueError, match="uint8"):
        SimEpisode(**{**ep.__dict__, "images": {"agentview": np.zeros((3, 8, 8, 3))}})
    with pytest.raises(ValueError, match="T="):
        SimEpisode(**{**ep.__dict__, "actions": np.zeros((2, 7))})
