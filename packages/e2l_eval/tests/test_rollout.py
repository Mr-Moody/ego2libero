import json

from e2l_eval.rollout import RolloutResult, checkpoint_rename_map, write_results


def test_checkpoint_rename_map_reads_train_config(tmp_path):
    rename = {"observation.images.image": "observation.images.camera1"}
    (tmp_path / "train_config.json").write_text(json.dumps({"rename_map": rename}))
    assert checkpoint_rename_map(str(tmp_path)) == rename


def test_checkpoint_rename_map_empty_without_file_or_for_hub_id(tmp_path):
    assert checkpoint_rename_map(str(tmp_path)) == {}
    assert checkpoint_rename_map("HuggingFaceVLA/smolvla_libero") == {}
    (tmp_path / "train_config.json").write_text(json.dumps({"rename_map": None}))
    assert checkpoint_rename_map(str(tmp_path)) == {}


def test_write_results_records_run_setting_and_video_outcomes(tmp_path):
    result = RolloutResult(
        policy="ckpt",
        task="libero_goal/8",
        successes=[True, False, True],
        seeds=[11000, 11001, 11002],
        video_paths=["v/eval_episode_0.mp4", "v/eval_episode_1.mp4"],
        run="B",
        setting="wide",
        placements=[{"seed": 11000, "placement_failed": False, "T_sim_obj": {}}],
    )
    path = write_results(result, tmp_path)
    data = json.loads(path.read_text())
    assert path == tmp_path / "results.json"
    assert data["run"] == "B" and data["setting"] == "wide"
    assert data["video_successes"] == [True, False]
    assert data["placements"][0]["seed"] == 11000
    assert abs(data["success_rate"] - 2 / 3) < 1e-12
    lo, hi = data["wilson_95"]
    assert 0 < lo < 2 / 3 < hi < 1
