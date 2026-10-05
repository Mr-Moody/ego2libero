import json

import numpy as np
import pytest

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.geometry import make_T
from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegment, RobotSegments, SimEpisode
from e2l_generate import run

DEMO = "20261003_scripted_000"
SIM = SimConfig(object_map={"bowl": "akita_black_bowl_1", "plate": "plate_1"})
NOMINAL = {
    "akita_black_bowl_1": make_T(np.eye(3), [-0.1, 0.0, 0.9]),
    "plate_1": make_T(np.eye(3), [0.05, 0.0, 0.9]),
    "wine_bottle_1": make_T(np.eye(3), [0.0, 0.3, 0.9]),
}


def write_demo(root, refs=("bowl", "plate")):
    seg = [
        RobotSegment(
            ref_object=r,
            T_obj_ee=make_T(np.eye(3), np.zeros((3, 3))),
            gripper=-np.ones(3),
            feasible=np.ones(3, bool),
        )
        for r in refs
    ]
    RobotSegments(demo_id=DEMO, fps=20.0, segments=seg).save(DataPaths(root).robot_segments(DEMO))


@pytest.fixture
def fake_sim(monkeypatch):
    """Replace LIBERO: 3 init states; replay succeeds on even init states."""
    calls = []

    def fake_replay(segments, sim_cfg, T_sim_obj=None, episode_index=0):
        calls.append((episode_index, sorted(T_sim_obj)))
        return SimEpisode(
            task="put the bowl on the plate",
            success=episode_index % 2 == 0,
            source_demo_ids=[segments.demo_id],
            images={"image": np.zeros((2, 4, 4, 3), np.uint8)},
            state=np.zeros((2, 8)),
            actions=np.zeros((2, 7)),
            object_poses=T_sim_obj,
        )

    monkeypatch.setattr(run, "init_state_count", lambda cfg: 3)
    monkeypatch.setattr(run, "nominal_object_poses", lambda cfg, i: NOMINAL)
    monkeypatch.setattr(run, "replay", fake_replay)
    monkeypatch.setattr(run, "_segments_cache", {})
    monkeypatch.setattr(run, "_nominal_cache", {})
    return calls


def cfg(**kw):
    return GenerateConfig(**({"episodes_per_demo": 4, "workers": 1} | kw))


def test_inline_run_saves_successes_and_manifest(tmp_path, fake_sim):
    write_demo(tmp_path)
    m = run.generate([DEMO], "r", cfg(), SIM, tmp_path)
    # k -> init state k % 3 = 0, 1, 2, 0 -> success for k = 0, 2, 3
    assert [e["status"] for e in m["episodes"]] == ["success", "failure", "success", "success"]
    assert m["demos"][DEMO]["yield"] == 0.75
    run_dir = DataPaths(tmp_path).generated("r")
    assert sorted(p.stem for p in run_dir.glob("*.npz")) == [
        f"{DEMO}_0000",
        f"{DEMO}_0002",
        f"{DEMO}_0003",
    ]
    assert SimEpisode.load(run_dir / f"{DEMO}_0002").success
    assert json.loads((run_dir / "manifest.json").read_text()) == m
    assert m["mean_episode_mb"] is not None
    # Only the referenced objects, mapped to LIBERO names, are placed.
    assert {tuple(names) for _, names in fake_sim} == {("akita_black_bowl_1", "plate_1")}


def test_keep_failures_saves_failures(tmp_path, fake_sim):
    write_demo(tmp_path)
    m = run.generate([DEMO], "r", cfg(keep_failures=True), SIM, tmp_path)
    assert all(e["saved"] for e in m["episodes"])
    assert len(list(DataPaths(tmp_path).generated("r").glob("*.npz"))) == 4


def test_same_seed_same_placements(tmp_path, fake_sim):
    write_demo(tmp_path)
    run.generate([DEMO], "a", cfg(), SIM, tmp_path)
    run.generate([DEMO], "b", cfg(), SIM, tmp_path)
    a = SimEpisode.load(DataPaths(tmp_path).generated("a", f"{DEMO}_0002"))
    b = SimEpisode.load(DataPaths(tmp_path).generated("b", f"{DEMO}_0002"))
    for name in a.object_poses:
        np.testing.assert_array_equal(a.object_poses[name], b.object_poses[name])


def test_overwrite_replaces_stale_files(tmp_path, fake_sim):
    write_demo(tmp_path)
    run.generate([DEMO], "r", cfg(), SIM, tmp_path)
    stale = DataPaths(tmp_path).generated("r", "stale.npz")
    stale.write_bytes(b"")
    with pytest.raises(FileExistsError):
        run.generate([DEMO], "r", cfg(), SIM, tmp_path)
    run.generate([DEMO], "r", cfg(episodes_per_demo=1), SIM, tmp_path, overwrite=True)
    assert not stale.exists()
    assert len(list(DataPaths(tmp_path).generated("r").glob("*.npz"))) == 1


def test_missing_demo_fails_before_sim_work(tmp_path, fake_sim):
    with pytest.raises(FileNotFoundError, match="20261003_nope_000"):
        run.generate(["20261003_nope_000"], "r", cfg(), SIM, tmp_path)
    assert fake_sim == [] and not DataPaths(tmp_path).generated("r").exists()


def test_attempt_errors_are_recorded(tmp_path, fake_sim, monkeypatch):
    write_demo(tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "replay", boom)
    m = run.generate([DEMO], "r", cfg(), SIM, tmp_path)
    assert {e["status"] for e in m["episodes"]} == {"error"}
    assert m["episodes"][0]["error"] == "RuntimeError: boom"
    assert m["demos"][DEMO]["errors"] == 4 and m["demos"][DEMO]["yield"] == 0.0


def test_unknown_object_is_an_attempt_error(tmp_path, fake_sim):
    write_demo(tmp_path, refs=("mug",))
    m = run.generate([DEMO], "r", cfg(episodes_per_demo=1), SIM, tmp_path)
    assert m["episodes"][0]["status"] == "error" and "mug" in m["episodes"][0]["error"]


def test_placement_failed_is_recorded(tmp_path, fake_sim):
    write_demo(tmp_path)
    m = run.generate([DEMO], "r", cfg(min_separation=5.0, max_placement_tries=2), SIM, tmp_path)
    assert m["demos"][DEMO]["placement_failed"] == 4 and fake_sim == []


def test_no_jobs_writes_empty_manifest(tmp_path, monkeypatch):
    def no_libero(cfg):
        raise AssertionError("must not touch LIBERO")

    monkeypatch.setattr(run, "init_state_count", no_libero)
    m = run.generate([], "r", cfg(), SIM, tmp_path)
    assert m["demos"] == {} and DataPaths(tmp_path).manifest("r").exists()
