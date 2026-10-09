import json

from e2l_eval.rollout import checkpoint_rename_map


def test_checkpoint_rename_map_reads_train_config(tmp_path):
    rename = {"observation.images.image": "observation.images.camera1"}
    (tmp_path / "train_config.json").write_text(json.dumps({"rename_map": rename}))
    assert checkpoint_rename_map(str(tmp_path)) == rename


def test_checkpoint_rename_map_empty_without_file_or_for_hub_id(tmp_path):
    assert checkpoint_rename_map(str(tmp_path)) == {}
    assert checkpoint_rename_map("HuggingFaceVLA/smolvla_libero") == {}
    (tmp_path / "train_config.json").write_text(json.dumps({"rename_map": None}))
    assert checkpoint_rename_map(str(tmp_path)) == {}
