import os

import imageio.v3 as iio
import numpy as np
import pytest

from e2l_common.config import SimConfig, load_config
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_common.paths import repo_root
from e2l_common.schemas import RobotSegment, RobotSegments
from e2l_sim.control import pose_error
from e2l_sim.libero_setup import libero_available
from e2l_sim.render import save_video, upright
from e2l_sim.replay import hold_at_grip_changes, libero_state, resample_segment, transit
from e2l_sim.scene import table_to_sim


def _obs(quat_xyzw):
    eef = {"pos": np.array([0.1, 0.2, 0.3]), "quat": np.asarray(quat_xyzw, dtype=float)}
    return {"robot_state": {"eef": eef, "gripper": {"qpos": np.array([0.04, -0.04])}}}


def test_libero_state_matches_robosuite_quat2axisangle():
    # Gripper pointing down: 180 deg about x. w < 0 gives angle > pi, as robosuite does.
    np.testing.assert_allclose(libero_state(_obs([1, 0, 0, 0]))[3:6], [np.pi, 0, 0], atol=1e-9)
    s = np.sin(0.1)
    state = libero_state(_obs([s, 0, 0, -np.cos(0.1)]))
    np.testing.assert_allclose(state[3:6], [2 * (np.pi - 0.1), 0, 0], atol=1e-9)
    np.testing.assert_allclose(state[[0, 1, 2, 6, 7]], [0.1, 0.2, 0.3, 0.04, -0.04])
    np.testing.assert_allclose(libero_state(_obs([0, 0, 0, 1]))[3:6], 0.0)


def test_resample_segment_to_control_rate():
    T = make_T(np.eye(3), np.linspace([0, 0, 0], [0.1, 0, 0], 11))  # 1 s at 10 fps
    grip = np.r_[-np.ones(6), np.ones(5)]
    seg = RobotSegment(ref_object="bowl", T_obj_ee=T, gripper=grip, feasible=np.ones(11, bool))
    T20, g20 = resample_segment(seg, fps=10.0, control_freq=20.0)
    assert len(T20) == len(g20) == 21
    np.testing.assert_allclose(T20[1, :3, 3], [0.005, 0, 0])
    assert g20[0] == -1 and g20[-1] == 1 and set(np.unique(g20)) == {-1.0, 1.0}


def test_transit_speed_and_endpoints():
    cfg = SimConfig(transit_speed=0.25, control_freq=20)
    T0 = make_T(np.eye(3), [0, 0, 1.0])
    T1 = make_T(rotvec_to_matrix(np.array([0, 0, 0.1])), [0.1, 0, 1.0])
    path = transit(T0, T1, cfg)
    assert len(path) == 8  # 0.1 m at 0.25 m/s, 20 Hz -> 8 steps
    np.testing.assert_allclose(path[-1], T1, atol=1e-12)
    assert pose_error(T0, path[0])[0] == pytest.approx(0.1 / 8)
    assert len(transit(T0, T0, cfg)) == 1


def test_hold_at_grip_changes():
    T = make_T(np.eye(3), np.zeros((4, 3)))
    T_out, g_out = hold_at_grip_changes(T, np.array([-1.0, -1.0, 1.0, 1.0]), steps=3)
    assert len(T_out) == 7
    np.testing.assert_array_equal(g_out, [-1, -1, 1, 1, 1, 1, 1])


def test_table_to_sim_broadcasts():
    T_sim_table = make_T(np.eye(3), [0, 0, 0.9])
    T_table_x = make_T(np.eye(3), np.array([[0.1, 0, 0], [0, 0.2, 0]]))
    np.testing.assert_allclose(
        table_to_sim(T_table_x, T_sim_table)[:, :3, 3], [[0.1, 0, 0.9], [0, 0.2, 0.9]]
    )


def test_save_video_round_trip(tmp_path):
    frames = np.random.default_rng(0).integers(0, 255, (5, 32, 48, 3), dtype=np.uint8)
    path = save_video(upright(frames), tmp_path / "a" / "v.mp4", fps=10)
    assert iio.imread(path).shape == (5, 32, 48, 3)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")
def test_scripted_bowl_onto_plate_succeeds():
    """The hand-designed demo in scripts/scripted_segments.py solves the chosen task."""
    import importlib.util

    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.replay import replay

    path = repo_root() / "scripts" / "scripted_segments.py"
    spec = importlib.util.spec_from_file_location("scripted_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    segments: RobotSegments = module.scripted("20261003_scripted_000")

    cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    episode = replay(segments, cfg)
    assert episode.success
    assert len(episode.actions) < cfg.max_steps
    assert np.abs(episode.actions).max() <= 1.0
    assert set(episode.images) == {"image", "image2"}
