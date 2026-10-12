import numpy as np
import pytest

from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegments
from e2l_sim.scripted import FPS, scripted_demo_id, scripted_segments, write_scripted

R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def _draws(n: int = 200) -> list[RobotSegments]:
    return [scripted_segments(scripted_demo_id(1), np.random.default_rng([7, i])) for i in range(n)]


def test_nominal_demo_matches_the_original_script():
    segs = scripted_segments(scripted_demo_id(0)).segments
    assert [s.ref_object for s in segs] == ["bowl", "plate", "plate"]
    assert [len(s.T_obj_ee) for s in segs] == [30, 30, 20]
    for s in segs:
        np.testing.assert_array_equal(
            s.T_obj_ee[:, :3, :3], np.broadcast_to(R_TOP_DOWN, (len(s.T_obj_ee), 3, 3))
        )
    np.testing.assert_allclose(segs[0].T_obj_ee[0, :3, 3], [0.05, 0.0, 0.145])
    np.testing.assert_allclose(segs[0].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.045])
    np.testing.assert_allclose(segs[1].T_obj_ee[0, :3, 3], [0.05, 0.0, 0.225])
    np.testing.assert_allclose(segs[1].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.075])
    np.testing.assert_allclose(segs[2].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.225])


def test_structure_is_preserved():
    for demo in [scripted_segments(scripted_demo_id(0)), *_draws(50)]:
        a, c, r = demo.segments
        assert [s.ref_object for s in demo.segments] == ["bowl", "plate", "plate"]
        assert (a.gripper[:-1] == -1).all() and a.gripper[-1] == 1  # grasp on the last sample
        assert (c.gripper[:-1] == 1).all() and c.gripper[-1] == -1  # release on the last sample
        assert (r.gripper == -1).all()
        assert all(s.feasible.all() and len(s.feasible) == len(s.T_obj_ee) for s in demo.segments)
        assert demo.fps == FPS


def test_jitter_is_seeded():
    a = scripted_segments("20261003_scripted_005", np.random.default_rng([0, 5]))
    b = scripted_segments("20261003_scripted_005", np.random.default_rng([0, 5]))
    c = scripted_segments("20261003_scripted_005", np.random.default_rng([1, 5]))
    for sa, sb in zip(a.segments, b.segments, strict=True):
        np.testing.assert_array_equal(sa.T_obj_ee, sb.T_obj_ee)
    assert not np.array_equal(a.segments[0].T_obj_ee[-1], c.segments[0].T_obj_ee[-1])


def test_jitter_stays_in_bounds():
    for demo in _draws():
        a, c, r = demo.segments
        grasp, place = a.T_obj_ee[-1, :3, 3], c.T_obj_ee[-1, :3, 3]
        angle = np.arctan2(grasp[1], grasp[0])
        assert abs(angle) <= 0.4 + 1e-9
        np.testing.assert_allclose(np.hypot(grasp[0], grasp[1]), 0.05)
        assert grasp[2] == pytest.approx(0.045)
        assert 0.08 - 1e-9 <= a.T_obj_ee[0, 2, 3] - grasp[2] <= 0.12 + 1e-9
        d = place - grasp
        assert np.abs(d[:2]).max() <= 0.01 + 1e-9
        assert 0.02 - 1e-9 <= d[2] <= 0.04 + 1e-9
        for seg, nominal in zip(demo.segments, (30, 30, 20), strict=True):
            n = len(seg.T_obj_ee)
            assert round(0.7 * nominal) - 1 <= n <= round(1.3 * nominal) + 1
            # grip-site z axis points straight down; fingers close radially at the grasp angle
            np.testing.assert_allclose(seg.T_obj_ee[:, :3, 2], np.tile([0, 0, -1.0], (n, 1)))
            np.testing.assert_allclose(seg.T_obj_ee[0, :2, 0], [np.cos(angle), np.sin(angle)])


def test_write_scripted_writes_nominal_first_and_jittered_rest(tmp_path):
    paths = DataPaths(tmp_path)
    npzs = write_scripted(paths, count=3, seed=0)
    assert [p.stem for p in npzs] == [f"20261003_scripted_{i:03d}" for i in range(3)]
    assert paths.demo_ids("robot_segments") == [p.stem for p in npzs]
    first = RobotSegments.load(npzs[0])
    np.testing.assert_array_equal(
        first.segments[0].T_obj_ee, scripted_segments(npzs[0].stem).segments[0].T_obj_ee
    )
    second = RobotSegments.load(npzs[1])
    assert not np.array_equal(second.segments[0].T_obj_ee[-1], first.segments[0].T_obj_ee[-1])
    again = write_scripted(DataPaths(tmp_path / "again"), count=3, seed=0)
    np.testing.assert_array_equal(
        RobotSegments.load(again[1]).segments[1].T_obj_ee, second.segments[1].T_obj_ee
    )


@pytest.mark.parametrize("count", [0, 1001])
def test_count_outside_range_is_rejected(tmp_path, count):
    with pytest.raises(ValueError, match="count"):
        write_scripted(DataPaths(tmp_path), count=count)
    assert not (tmp_path / "robot_segments").exists()


def test_demo_id_outside_range_is_rejected():
    with pytest.raises(ValueError):
        scripted_demo_id(1000)
