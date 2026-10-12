import json
from pathlib import Path

import pytest

from e2l_common.config import AblationConfig, EvalConfig, TrainConfig
from e2l_eval import ablation, hub
from e2l_eval.ablation import cell_label, eligible_demos, select_cells

REPLAY, GEN = "rep", "gen"


def _episode(path: Path, success: bool, sources: list[str]) -> None:
    """Sidecar-only SimEpisode stand-in: selection reads the JSON, never the npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".npz").write_bytes(b"")
    path.with_suffix(".json").write_text(
        json.dumps({"schema": "SimEpisode", "success": success, "source_demo_ids": sources})
    )


def _data(root: Path, n_demos: int = 6, failed=("d_003",), per_demo: int = 2) -> None:
    for i in range(n_demos):
        demo = f"d_{i:03d}"
        _episode(root / "generated" / REPLAY / demo, demo not in failed, [demo])
        for k in range(per_demo):
            _episode(root / "generated" / GEN / f"{demo}_{k:04d}", k == 0, [demo])


def _cfg(root: Path, **kw) -> AblationConfig:
    return AblationConfig(data_root=root, replay_run=REPLAY, generated_run=GEN, **kw)


def test_cell_labels():
    assert cell_label(5, False) == "n05_raw"
    assert cell_label(10, True) == "n10_aug"
    assert cell_label("all", True) == "all_aug"


def test_eligible_demos_are_successful_replays(tmp_path):
    _data(tmp_path)
    assert eligible_demos(tmp_path / "generated" / REPLAY) == [
        "d_000",
        "d_001",
        "d_002",
        "d_004",
        "d_005",
    ]


def test_cells_follow_config_order_with_nested_subsets(tmp_path):
    _data(tmp_path)
    cells = select_cells(_cfg(tmp_path, demo_counts=[2, 4, "all"]))
    assert [c.label for c in cells] == [
        "n02_raw",
        "n02_aug",
        "n04_raw",
        "n04_aug",
        "all_raw",
        "all_aug",
    ]
    by = {c.label: c for c in cells}
    assert set(by["n02_raw"].demo_ids) <= set(by["n04_raw"].demo_ids)
    assert by["n02_raw"].demo_ids == by["n02_aug"].demo_ids
    assert by["all_raw"].n_demos == 5 and "d_003" not in by["all_raw"].demo_ids


def test_raw_is_replays_and_aug_adds_only_own_successful_generated(tmp_path):
    _data(tmp_path)
    by = {c.label: c for c in select_cells(_cfg(tmp_path, demo_counts=[2]))}
    raw, aug = by["n02_raw"], by["n02_aug"]
    assert [p.stem for p in raw.episodes] == list(raw.demo_ids)
    assert all(p.parent.name == REPLAY for p in raw.episodes)
    extra = aug.episodes[len(raw.episodes) :]
    assert aug.episodes[: len(raw.episodes)] == raw.episodes
    assert sorted(p.stem for p in extra) == sorted(f"{d}_0000" for d in raw.demo_ids)


def test_subsets_depend_only_on_the_seed(tmp_path):
    _data(tmp_path)
    a = select_cells(_cfg(tmp_path, demo_counts=[3], subset_seed=1))[0].demo_ids
    b = select_cells(_cfg(tmp_path, demo_counts=[3], subset_seed=1))[0].demo_ids
    c = [select_cells(_cfg(tmp_path, demo_counts=[3], subset_seed=s))[0].demo_ids for s in range(8)]
    assert a == b and len(set(c)) > 1


def test_count_above_eligible_names_both_numbers(tmp_path):
    _data(tmp_path)
    with pytest.raises(ValueError, match=r"demo count 9 .* 5 demos"):
        select_cells(_cfg(tmp_path, demo_counts=[9]))


def test_no_successful_replays_is_an_error(tmp_path):
    _data(tmp_path, n_demos=2, failed=("d_000", "d_001"))
    with pytest.raises(ValueError, match="no successful replays"):
        select_cells(_cfg(tmp_path))


def test_augmented_cell_without_generated_episodes_is_an_error(tmp_path):
    _data(tmp_path, per_demo=0)
    with pytest.raises(ValueError, match="no successful generated episodes"):
        select_cells(_cfg(tmp_path, demo_counts=[2]))
    raw_only = select_cells(_cfg(tmp_path, demo_counts=[2], augmentation=[False]))
    assert [c.label for c in raw_only] == ["n02_raw"]


@pytest.fixture
def fake_hub(monkeypatch):
    """Offline Hub + export: records pushes; export writes only meta/info.json."""
    state = {"existing": set(), "pushed": [], "folders": [], "exports": []}
    monkeypatch.setattr(hub, "namespace", lambda configured: configured or "me")
    monkeypatch.setattr(hub, "repo_exists", lambda repo, kind="model": repo in state["existing"])
    monkeypatch.setattr(
        hub, "push_dataset", lambda root, repo, private: state["pushed"].append((repo, private))
    )
    monkeypatch.setattr(
        hub, "push_folder", lambda folder, repo, private: state["folders"].append((folder, repo))
    )

    def export(episodes, cfg, max_episodes=None):
        state["exports"].append((cfg.dataset_root, cfg.dataset_repo_id, len(episodes)))
        (cfg.dataset_root / "meta").mkdir(parents=True)
        info = {"total_episodes": len(episodes), "total_frames": 10 * len(episodes)}
        (cfg.dataset_root / "meta" / "info.json").write_text(json.dumps(info))
        return cfg.dataset_root

    monkeypatch.setattr(ablation, "export_dataset", export)
    return state


def _eval_cfg(root: Path, **kw) -> EvalConfig:
    return EvalConfig(ablation=_cfg(root, name="rehearsal", demo_counts=[2, "all"], **kw))


def test_prepare_exports_and_pushes_only_selected_cells(tmp_path, fake_hub):
    _data(tmp_path)
    plan_path = ablation.prepare(
        _eval_cfg(tmp_path, steps=123),
        TrainConfig(batch_size=8),
        ["all_raw", "all_aug"],
        plan_root=tmp_path / "plans",
    )
    plan = json.loads(plan_path.read_text())
    assert plan_path == tmp_path / "plans" / "rehearsal" / "plan.json"
    assert (plan["name"], plan["namespace"], plan["steps"], plan["batch_size"]) == (
        "rehearsal",
        "me",
        123,
        8,
    )
    assert plan["plan_repo"] == "me/e2l-rehearsal-plan"
    cells = {c["label"]: c for c in plan["cells"]}
    assert set(cells) == {"n02_raw", "n02_aug", "all_raw", "all_aug"}
    assert [c for c in cells if cells[c]["prepared"]] == ["all_raw", "all_aug"]
    assert cells["all_aug"]["dataset_repo"] == "me/e2l-rehearsal-all_aug"
    assert cells["all_aug"]["policy_repo"] == "me/e2l-rehearsal-all_aug-policy"
    assert cells["all_raw"]["episodes"] == 5 and cells["all_raw"]["frames"] == 50
    assert cells["n02_raw"]["frames"] is None
    assert [r for r, _ in fake_hub["pushed"]] == [
        "me/e2l-rehearsal-all_raw",
        "me/e2l-rehearsal-all_aug",
    ]
    assert all(private for _, private in fake_hub["pushed"])
    roots = [r for r, _, _ in fake_hub["exports"]]
    assert roots == [
        tmp_path / "lerobot" / "rehearsal-all_raw",
        tmp_path / "lerobot" / "rehearsal-all_aug",
    ]
    assert fake_hub["folders"] == [(plan_path.parent, "me/e2l-rehearsal-plan")]
    assert (plan_path.parent / "spark.sh").is_file()


def test_prepare_reuses_export_and_skips_existing_repo(tmp_path, fake_hub):
    _data(tmp_path)
    ablation.prepare(_eval_cfg(tmp_path), TrainConfig(), ["all_raw"], plan_root=tmp_path / "p")
    fake_hub["existing"].add("me/e2l-rehearsal-all_raw")
    plan_path = ablation.prepare(
        _eval_cfg(tmp_path), TrainConfig(), ["all_raw"], plan_root=tmp_path / "p"
    )
    assert len(fake_hub["exports"]) == 1 and len(fake_hub["pushed"]) == 1
    cells = {c["label"]: c for c in json.loads(plan_path.read_text())["cells"]}
    assert cells["all_raw"]["frames"] == 50


def test_prepare_rejects_unknown_cells(tmp_path, fake_hub):
    _data(tmp_path)
    with pytest.raises(ValueError, match=r"all_row.*all_raw"):
        ablation.prepare(_eval_cfg(tmp_path), TrainConfig(), ["all_row"], plan_root=tmp_path / "p")
    assert fake_hub["exports"] == [] and fake_hub["pushed"] == [] and fake_hub["folders"] == []


def test_spark_script_trains_each_prepared_cell_from_the_hub(tmp_path, fake_hub):
    _data(tmp_path)
    plan_path = ablation.prepare(
        _eval_cfg(tmp_path, steps=77),
        TrainConfig(),
        ["all_raw", "all_aug"],
        plan_root=tmp_path / "p",
    )
    script = (plan_path.parent / "spark.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert "uv run hf download me/e2l-rehearsal-plan --repo-type dataset" in script
    runs = [line for line in script.splitlines() if "e2l train finetune" in line]
    assert len(runs) == 2
    assert "dataset_repo_id=me/e2l-rehearsal-all_raw" in runs[0]
    assert "push_repo_id=me/e2l-rehearsal-all_raw-policy" in runs[0]
    assert "dataset_root=null" in runs[0] and "steps=77" in runs[0] and "private=true" in runs[0]
    assert "output_dir=outputs/train/rehearsal-all_aug" in runs[1]


def test_spark_script_skips_existing_output_dirs(tmp_path, fake_hub):
    import subprocess

    _data(tmp_path)
    plan_path = ablation.prepare(
        _eval_cfg(tmp_path), TrainConfig(), ["all_raw"], plan_root=tmp_path / "p"
    )
    script = (plan_path.parent / "spark.sh").read_text().replace("uv run e2l", "echo TRAIN")
    work = tmp_path / "spark"
    (work / "outputs" / "train" / "rehearsal-all_raw").mkdir(parents=True)
    out = subprocess.run(["bash", "-c", script], cwd=work, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "TRAIN" not in out.stdout and "skip rehearsal-all_raw" in out.stdout
