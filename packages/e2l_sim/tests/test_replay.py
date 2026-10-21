import os

import imageio.v3 as iio
import numpy as np
import pytest

from e2l_common.config import SimConfig, load_config
from e2l_common.geometry import inv_T, make_T, rotvec_to_matrix
from e2l_common.paths import repo_root
from e2l_common.schemas import RobotSegment, RobotSegments
from e2l_sim.control import pose_error
from e2l_sim.libero_setup import libero_available
from e2l_sim.render import save_video, upright
from e2l_sim.replay import (
    carry_lift,
    carry_targets,
    hold_at_grip_changes,
    keepout_yaw,
    libero_state,
    resample_segment,
    snap_grasp,
    transit,
    trim_approach,
    warp_carry,
)
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
    T20, g20, held = resample_segment(seg, fps=10.0, control_freq=20.0)
    assert held is None
    assert len(T20) == len(g20) == 21
    np.testing.assert_allclose(T20[1, :3, 3], [0.005, 0, 0])
    assert g20[0] == -1 and g20[-1] == 1 and set(np.unique(g20)) == {-1.0, 1.0}


def test_resample_segment_keeps_the_last_frame_off_the_control_grid():
    # 41 frames at 30 fps end at 4/3 s, between 20 Hz ticks; the grasp is the last frame only
    T = make_T(np.eye(3), np.linspace([0, 0, 0], [0.1, 0, 0], 41))
    grip = np.r_[-np.ones(40), 1.0]
    seg = RobotSegment(ref_object="bowl", T_obj_ee=T, gripper=grip, feasible=np.ones(41, bool))
    T20, g20, _ = resample_segment(seg, fps=30.0, control_freq=20.0)
    assert g20[-1] == 1 and g20[-2] == -1
    np.testing.assert_allclose(T20[-1], T[-1], atol=1e-12)


def test_transit_speed_and_endpoints():
    cfg = SimConfig(transit_speed=0.25, control_freq=20)
    T0 = make_T(np.eye(3), [0, 0, 1.0])
    T1 = make_T(rotvec_to_matrix(np.array([0, 0, 0.1])), [0.1, 0, 1.0])
    path = transit(T0, T1, cfg)
    assert len(path) == 8  # 0.1 m at 0.25 m/s, 20 Hz -> 8 steps
    np.testing.assert_allclose(path[-1], T1, atol=1e-12)
    assert pose_error(T0, path[0])[0] == pytest.approx(0.1 / 8)
    assert len(transit(T0, T0, cfg)) == 1


def test_carry_targets_move_the_held_object_along_its_demo_path():
    T_ref_held = np.stack([make_T(t=np.array([0.1 * i, 0.0, 0.2])) for i in range(3)])
    grasp = make_T(rotvec_to_matrix(np.array([np.pi, 0.0, 0.3])), np.array([0.04, 0.01, 0.03]))
    T = carry_targets(T_ref_held, grasp)
    # the object the robot holds ends up where the demo's object was, relative to the reference
    np.testing.assert_allclose(T @ inv_T(grasp), T_ref_held, atol=1e-12)


def test_resample_segment_resamples_the_held_object_path():
    T = make_T(np.eye(3), np.linspace([0, 0, 0], [0.1, 0, 0], 11))
    seg = RobotSegment(
        ref_object="plate",
        T_obj_ee=T,
        gripper=np.ones(11),
        feasible=np.ones(11, bool),
        T_obj_held=T @ make_T(t=np.array([0, 0, -0.05])),
    )
    T20, _, held = resample_segment(seg, fps=10.0, control_freq=20.0)
    np.testing.assert_allclose(held[:, :3, 3], T20[:, :3, 3] - [0, 0, 0.05], atol=1e-12)


def test_snap_grasp_blends_the_approach_onto_the_rim():
    # Human approach descends to a grasp 7 cm out at 90 deg, 2 cm too high, yawed 0.3 rad off.
    R = rotvec_to_matrix(np.array([np.pi, 0, 0])) @ rotvec_to_matrix(np.array([0, 0, 0.3]))
    T = make_T(R, np.linspace([0.0, 0.07, 0.2], [0.0, 0.07, 0.065], 20))
    grip = np.r_[-np.ones(17), np.ones(3)]
    out = snap_grasp(T, grip, radius=0.05, height=0.045, blend_steps=5)
    np.testing.assert_allclose(out[:12], T[:12])  # untouched before the blend window
    np.testing.assert_allclose(out[17:, :3, 3], np.tile([0.0, 0.05, 0.045], (3, 1)), atol=1e-12)
    np.testing.assert_allclose(out[17, :3, 2], [0, 0, -1], atol=1e-12)  # top-down
    assert abs(out[17, :3, 0] @ [0, 1, 0]) == pytest.approx(1.0)  # fingers close radially
    assert out[17, :3, 0] @ T[17, :3, 0] > 0  # the half turn nearest the human yaw
    step = np.linalg.norm(np.diff(out[10:18, :3, 3], axis=0), axis=1)
    assert step.max() < 0.02  # no jump into the grasp
    flip = snap_grasp(T, grip, 0.05, 0.045, 5, x_pref=-T[17, :3, 0])  # the other half turn
    np.testing.assert_allclose(flip[17, :3, 0], -out[17, :3, 0], atol=1e-12)
    np.testing.assert_allclose(flip[17:, :3, 3], out[17:, :3, 3], atol=1e-12)  # same grasp
    np.testing.assert_allclose(snap_grasp(T, -np.ones(20), 0.05, 0.045, 5), T)  # no grasp
    np.testing.assert_allclose(snap_grasp(T, np.ones(20), 0.05, 0.045, 5), T)  # a carry


def test_snap_grasp_descends_straight_onto_the_rim_from_above():
    R = rotvec_to_matrix(np.array([np.pi, 0, 0]))
    T = make_T(R, np.linspace([0.0, 0.2, 0.03], [0.0, 0.07, 0.03], 30))  # low, from the side
    grip = np.r_[-np.ones(25), np.ones(5)]
    out = snap_grasp(T, grip, 0.05, 0.045, blend_steps=6, lift=0.06, descend_steps=5)
    np.testing.assert_allclose(out[:14, :2, 3], T[:14, :2, 3])  # the human path, in plan view
    np.testing.assert_allclose(out[:20, 2, 3], 0.105)  # but never below the pre-grasp height
    np.testing.assert_allclose(out[20, :3, 3], [0.0, 0.05, 0.105], atol=1e-12)  # pre-grasp
    xy = out[20:26, :3, 3][:, :2]
    np.testing.assert_allclose(xy, np.tile([0.0, 0.05], (6, 1)), atol=1e-12)  # straight down
    assert np.all(np.diff(out[20:26, 2, 3]) < 0)
    np.testing.assert_allclose(out[25:, :3, 3], np.tile([0.0, 0.05, 0.045], (5, 1)), atol=1e-12)


def test_trim_approach_starts_where_the_path_last_enters_the_radius():
    x = np.r_[0.5, 0.05, 0.3, 0.2, 0.1, 0.05, 0.0]  # wanders out and back before the grasp
    T = make_T(np.eye(3), np.c_[x, np.zeros((7, 2))])
    grip = np.r_[-np.ones(6), 1.0]
    T_out, g_out = trim_approach(T, grip, radius=0.15)
    np.testing.assert_allclose(T_out[:, 0, 3], [0.1, 0.05, 0.0])
    np.testing.assert_allclose(g_out, [-1, -1, 1])
    T_out, g_out = trim_approach(T, -np.ones(7), radius=0.15)  # no grasp: unchanged
    assert len(T_out) == 7


def test_carry_lift_is_full_until_the_set_down_then_ramps_to_zero():
    z = carry_lift(10, lift=0.04, ramp_steps=4)
    np.testing.assert_allclose(z, [0.04] * 6 + [0.03, 0.02, 0.01, 0.0])
    np.testing.assert_allclose(carry_lift(3, lift=0.04, ramp_steps=4), [0.02, 0.01, 0.0])


def test_warp_carry_moves_the_start_and_keeps_the_set_down():
    path = make_T(np.eye(3), np.linspace([0.2, 0.2, 0.01], [0.0, 0.0, 0.01], 11))
    path = np.concatenate([path, path[-1:].repeat(3, axis=0)])  # held still after the set-down
    start = make_T(rotvec_to_matrix(np.array([0, 0, 0.4])), [-0.16, 0.0, 0.01])
    out = warp_carry(path, start, n=11)
    np.testing.assert_allclose(out[0], start, atol=1e-12)
    np.testing.assert_allclose(out[10:], path[10:], atol=1e-12)
    step = np.linalg.norm(np.diff(out[:11, :3, 3], axis=0), axis=1)
    assert step.max() < 0.06  # no jump: the shift fades along the way
    np.testing.assert_allclose(warp_carry(path, start, n=1), path)
    out = warp_carry(path, start, n=11, end_xy=[0.02, -0.01])  # centre the set-down
    np.testing.assert_allclose(out[0], start, atol=1e-12)
    np.testing.assert_allclose(out[10:, :3, 3], np.tile([0.02, -0.01, 0.01], (4, 1)), atol=1e-12)


def test_warp_carry_turns_the_object_by_end_yaw_at_the_set_down():
    path = make_T(np.eye(3), np.linspace([0.2, 0.0, 0.01], [0.0, 0.0, 0.01], 11))
    out = warp_carry(path, path[0], n=11, end_yaw=0.5)
    np.testing.assert_allclose(out[0], path[0], atol=1e-12)
    np.testing.assert_allclose(out[-1, :3, :3], rotvec_to_matrix(np.array([0, 0, 0.5])))
    np.testing.assert_allclose(out[:, :3, 3], path[:, :3, 3], atol=1e-12)


def test_keepout_yaw_turns_to_the_nearest_edge():
    assert keepout_yaw(np.radians(-80), (-95.0, 45.0)) == pytest.approx(np.radians(30))
    assert keepout_yaw(np.radians(-120), (-95.0, 45.0)) == pytest.approx(np.radians(-20))
    assert keepout_yaw(np.radians(170), (-170.0, 45.0)) == pytest.approx(np.radians(-25))
    assert keepout_yaw(np.radians(0), (-95.0, 45.0)) == 0.0
    assert keepout_yaw(1.0, None) == 0.0


def test_hold_at_grip_changes_arrives_open_then_closes():
    T = make_T(np.eye(3), np.arange(12.0).reshape(4, 3))
    T_out, g_out = hold_at_grip_changes(T, np.array([-1.0, -1.0, 1.0, 1.0]), steps=3)
    np.testing.assert_array_equal(g_out, [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(T_out[2:9], np.repeat(T[2:3], 7, axis=0))  # held at the event


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
    """The nominal scripted demo (e2l_sim.scripted) solves the chosen task."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.replay import replay
    from e2l_sim.scripted import scripted_segments

    segments: RobotSegments = scripted_segments("20261003_scripted_000")

    cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    episode = replay(segments, cfg)
    assert episode.success
    assert len(episode.actions) < cfg.max_steps
    assert np.abs(episode.actions).max() <= 1.0
    assert set(episode.images) == {"image", "image2"}
