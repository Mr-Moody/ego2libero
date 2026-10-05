import importlib.util
import os

import pytest

from e2l_common.config import GenerateConfig, SimConfig, load_config
from e2l_common.paths import DataPaths, repo_root
from e2l_common.schemas import SimEpisode
from e2l_sim.libero_setup import libero_available

pytestmark = [
    pytest.mark.sim,
    pytest.mark.slow,
    pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets"),
]
DEMO = "20261003_scripted_000"


def scripted_demo(root):
    path = repo_root() / "scripts" / "scripted_segments.py"
    spec = importlib.util.spec_from_file_location("scripted_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.scripted(DEMO).save(DataPaths(root).robot_segments(DEMO))


@pytest.mark.parametrize("workers", [1, 2])
def test_generate_scripted_demo(tmp_path, workers):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_generate.run import generate

    scripted_demo(tmp_path)
    sim_cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    cfg = GenerateConfig(episodes_per_demo=2, workers=workers, keep_failures=True)
    m = generate([DEMO], f"w{workers}", cfg, sim_cfg, tmp_path)

    assert m["demos"][DEMO]["attempts"] == 2
    assert {e["status"] for e in m["episodes"]} <= {"success", "failure"}, m["episodes"]
    for e in m["episodes"]:
        episode = SimEpisode.load(DataPaths(tmp_path).generated(f"w{workers}", e["stem"]))
        assert episode.success == (e["status"] == "success")
        assert set(episode.object_poses) >= {"akita_black_bowl_1", "plate_1"}
    assert [e["episode_index"] for e in m["episodes"]] == [0, 1]
