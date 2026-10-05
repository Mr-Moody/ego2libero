import os

import numpy as np
import pytest

from e2l_common.config import SimConfig, load_config
from e2l_common.paths import repo_root
from e2l_sim.libero_setup import libero_available

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets"),
]


def test_nominal_object_poses_follow_the_init_state():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.env import init_state_count
    from e2l_sim.scene import nominal_object_poses

    cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    assert init_state_count(cfg) == 50
    a = nominal_object_poses(cfg, 10)
    b = nominal_object_poses(cfg, 20)
    assert {"akita_black_bowl_1", "plate_1"} <= set(a)
    assert all(T.shape == (4, 4) for T in a.values())
    # Init states 10 and 20 place the plate ~2 cm apart (measured 2026-10-05).
    assert np.linalg.norm(a["plate_1"][:3, 3] - b["plate_1"][:3, 3]) > 0.005
    np.testing.assert_allclose(nominal_object_poses(cfg, 10)["plate_1"], a["plate_1"])
