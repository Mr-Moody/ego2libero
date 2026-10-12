import json
from pathlib import Path

import pytest

from e2l_common.config import AblationConfig
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
