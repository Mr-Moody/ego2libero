import os

import numpy as np
import pytest

from e2l_common.config import EvalConfig, SimConfig, WidePoseConfig, load_config
from e2l_common.geometry import make_T, matrix_to_rotvec
from e2l_common.paths import repo_root
from e2l_sim.libero_setup import libero_available
from e2l_sim.wide import placement_valid, sample_wide_placement, settled

NOMINAL = {
    "bowl": make_T(np.eye(3), [-0.05, 0.0, 0.9]),
    "plate": make_T(np.eye(3), [0.05, 0.15, 0.9]),
    "cheese": make_T(np.eye(3), [-0.2, -0.3, 0.9]),
    "bottle": make_T(np.eye(3), [-0.2, -0.25, 0.9]),  # 5 cm from cheese, neither moves
}
CFG = WidePoseConfig(table_xy=((-0.5, 0.5), (-0.6, 0.6)))


def test_placements_move_named_objects_beyond_the_generation_square():
    rng = np.random.default_rng(0)
    accepted = 0
    for _ in range(300):
        poses = sample_wide_placement(NOMINAL, ["plate", "bowl"], CFG, rng)
        if poses is None:
            continue
        accepted += 1
        assert set(poses) == {"bowl", "plate"}
        for name, T in poses.items():
            d = T[:3, 3] - NOMINAL[name][:3, 3]
            assert np.max(np.abs(d[:2])) > CFG.min_offset and np.linalg.norm(d[:2]) <= 0.15
            assert d[2] == 0.0
            assert CFG.min_yaw - 1e-9 <= abs(matrix_to_rotvec(T[:3, :3])[2]) <= CFG.max_yaw
    assert accepted > 100  # close unmoved cheese/bottle must not block every draw


def test_placement_is_seeded_and_order_independent():
    a = sample_wide_placement(NOMINAL, ["bowl", "plate"], CFG, np.random.default_rng(3))
    b = sample_wide_placement(NOMINAL, ["plate", "bowl"], CFG, np.random.default_rng(3))
    assert a is not None
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_placement_valid_checks_table_and_moved_pairs_only():
    others = {"cheese": NOMINAL["cheese"], "bottle": NOMINAL["bottle"]}
    ok = {"bowl": make_T(np.eye(3), [0.1, 0.1, 0.9])}
    assert placement_valid(ok, others, CFG)
    off_table = {"bowl": make_T(np.eye(3), [0.6, 0.1, 0.9])}
    assert not placement_valid(off_table, others, CFG)
    near_cheese = {"bowl": make_T(np.eye(3), [-0.2, -0.35, 0.9])}
    assert not placement_valid(near_cheese, others, CFG)
    two_close = {
        "bowl": make_T(np.eye(3), [0.1, 0.1, 0.9]),
        "plate": make_T(np.eye(3), [0.15, 0.1, 0.9]),
    }
    assert not placement_valid(two_close, others, CFG)


def test_unknown_object_is_named():
    with pytest.raises(KeyError, match="bowl_2"):
        sample_wide_placement(NOMINAL, ["bowl_2"], CFG, np.random.default_rng(0))


def test_settled_checks_xy_drift_and_height_change():
    target = {"bowl": make_T(np.eye(3), [0.1, 0.1, 0.9])}
    assert settled({"bowl": make_T(np.eye(3), [0.11, 0.1, 0.895])}, target, 0.02)
    assert not settled({"bowl": make_T(np.eye(3), [0.13, 0.1, 0.9])}, target, 0.02)
    assert not settled({"bowl": make_T(np.eye(3), [0.1, 0.1, 0.95])}, target, 0.02)  # on a fixture
    assert not settled({"bowl": make_T(np.eye(3), [0.1, 0.1, 0.5])}, target, 0.02)  # fell off


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")
def test_wide_env_reset_moves_task_objects_reproducibly():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.env import make_env
    from e2l_sim.scene import object_poses

    sim_cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    wide = load_config(repo_root() / "configs" / "eval.yaml", model=EvalConfig).wide
    names = sorted(sim_cfg.object_map.values())
    base = make_env(sim_cfg)
    try:
        base.reset(seed=11_000)
        nominal = object_poses(base)
    finally:
        base.close()
    poses = []
    for _ in range(2):
        env = make_env(sim_cfg, wide=wide)
        try:
            obs, _ = env.reset(seed=11_000)
            assert "pixels" in obs
            assert not env.placements[-1]["placement_failed"]
            poses.append(object_poses(env))
        finally:
            env.close()
    (x0, x1), (y0, y1) = wide.table_xy
    for n in names:
        d = poses[0][n][:2, 3] - nominal[n][:2, 3]
        assert np.max(np.abs(d)) > wide.min_offset - wide.settle_tol
        assert x0 <= poses[0][n][0, 3] <= x1 and y0 <= poses[0][n][1, 3] <= y1
        np.testing.assert_allclose(poses[0][n], poses[1][n], atol=1e-6)
