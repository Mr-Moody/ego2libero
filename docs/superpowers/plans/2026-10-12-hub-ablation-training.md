# Hub-mediated training and ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn replayed + generated episodes into Hub-pushed per-cell datasets, Spark-trained
SmolVLA checkpoints and evaluated results (`e2l eval ablate prepare | evaluate`), then run the
smoke chain and the rehearsal Run A / Run B through it.

**Architecture:** `e2l_eval.ablation` selects nested demo subsets per (count × augmentation)
cell, exports and pushes each cell's LeRobot dataset, and writes `plan.json` + `spark.sh`
(also pushed, as `<ns>/e2l-<plan>-plan`). The user runs `spark.sh` on the Spark: one
`e2l train finetune` per cell, training from the Hub dataset and pushing the checkpoint.
`evaluate` pulls each checkpoint and calls the existing `rollout` for `standard` and `wide`.
All Hub calls go through a thin `e2l_eval.hub` module so the rest is testable offline.

**Tech Stack:** Python 3.12, pydantic, typer, numpy, matplotlib (Agg), huggingface_hub
(`whoami`, `HfApi`, `snapshot_download`), LeRobot 0.6.1 (`LeRobotDataset.push_to_hub`,
`lerobot-train --policy.push_to_hub/repo_id/private`).

**Spec:** `docs/superpowers/specs/2026-10-12-hub-ablation-training-design.md`

## Global Constraints

- Work in `.worktrees/hub-ablation-training` on branch `feat/hub-ablation-training` (venv set
  up; `outputs` is a symlink to the main checkout's `outputs/`, never `git add` it).
  `unset PYTHONPATH VIRTUAL_ENV` before any `uv run`; `HF_HOME=/mnt/data/hf-cache` for every
  command that touches the Hub or checkpoints.
- Commits: `<type>(<scope>)?: <subject>`; scopes `common train eval configs repo`. **No
  `Co-Authored-By:`, `Claude-Session:` or "Generated with Claude Code" lines, ever.**
- `huggingface_hub` and LeRobot are imported lazily inside functions in `e2l_eval` (tests and
  `--help` stay offline). LIBERO only via `e2l_sim` / `rollout`.
- Repo ids: dataset `<ns>/e2l-<plan>-<cell>`, policy `<ns>/e2l-<plan>-<cell>-policy`, plan files
  `<ns>/e2l-<plan>-plan` (dataset type). All private unless `ablation.private=false`.
- Cell labels: `n05_raw`, `n05_aug`, `n10_raw`, …, `all_raw`, `all_aug`. Run A = `all_raw`,
  Run B = `all_aug`. Eval run label = `<plan>-<cell>` (folder `outputs/eval/<plan>-<cell>/`).
- Eligible demos = successful raw replays in `ablation.replay_run`; subsets nested via one
  `np.random.default_rng(subset_seed).permutation` of the sorted eligible ids.
- `aug` adds every successful generated episode whose `source_demo_ids` ⊆ the cell's demos.
- Every cell: same `ablation.steps`, batch size and seed from `configs/train/smolvla.yaml`.
- Nothing writes to `docs/figures/`, README or LICENSE (Track 3): reports go to
  `outputs/report`.
- ruff, line length 100; type hints; comment density like the surrounding files.

## Review Focus

1. Rerunning `prepare` after a partial run → reuses existing local exports and skips dataset
   repos already on the Hub instead of failing on `FileExistsError` or re-uploading 6 GB.
   Pinned by `test_prepare_reuses_export_and_skips_existing_repo` (Task 5).
2. Rerunning `spark.sh` after a crash mid-way → skips cells whose `outputs/train/<plan>-<cell>`
   exists, with a message, instead of dying on the first one. Pinned by
   `test_spark_script_skips_existing_output_dirs` (Task 5).
3. `--cells` naming a label not in the grid (typo `all_row`) → error listing the valid labels,
   before anything is exported or pushed. Pinned by `test_prepare_rejects_unknown_cells`
   (Task 5).
4. Augmented cell when the generated run is missing or has no successes for those demos →
   clear error at selection time, not a silent copy of the raw cell. Pinned by
   `test_augmented_cell_without_generated_episodes_is_an_error` (Task 4).
5. `evaluate` run before the Spark has finished some cells → those cells reported `pending`,
   finished ones evaluated, already-evaluated settings not rerun. Pinned by
   `test_evaluate_skips_done_and_pending_cells` (Task 6).

---

### Task 1: Config: `AblationConfig`, Hub fields on `TrainConfig`

**Files:**
- Modify: `packages/e2l_common/src/e2l_common/config.py` (imports, `__all__`, `TrainConfig`,
  new `AblationConfig` before `EvalConfig`, `EvalConfig`)
- Modify: `configs/eval.yaml` (replace `demo_counts` / `augmentation` lines)
- Modify: `configs/train/smolvla.yaml` (two new commented keys)
- Test: `packages/e2l_common/tests/test_config.py` (append)

**Interfaces:**
- Produces:
  - `AblationConfig` fields: `name: str = "real"`, `data_root: Path = Path("data")`,
    `replay_run: str = "replay"`, `generated_run: str = "run0"`,
    `demo_counts: list[int | Literal["all"]] = [5, 10, "all"]`,
    `augmentation: list[bool] = [False, True]`, `subset_seed: int = 0`, `steps: int = 20000`,
    `hub_namespace: str | None = None`, `private: bool = True`.
  - `EvalConfig.ablation: AblationConfig` (the old `demo_counts` / `augmentation` removed).
  - `TrainConfig.dataset_root: Path | None`, `TrainConfig.push_repo_id: str | None = None`,
    `TrainConfig.private: bool = True`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/e2l_common/tests/test_config.py` (keep its existing imports; add any of
`pytest`, `Path`, `ValidationError` that are missing at the top):

```python
from pydantic import ValidationError

from e2l_common.config import AblationConfig, EvalConfig, TrainConfig


def test_ablation_defaults_describe_the_real_run():
    ab = EvalConfig().ablation
    assert ab.demo_counts == [5, 10, "all"]
    assert ab.augmentation == [False, True]
    assert (ab.data_root, ab.replay_run, ab.generated_run) == (Path("data"), "replay", "run0")
    assert ab.private and ab.hub_namespace is None


def test_ablation_overrides_from_cli_strings(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text("ablation:\n  name: rehearsal\n")
    from e2l_common.config import load_config

    cfg = load_config(
        path,
        [
            "ablation.demo_counts=[3, all]",
            "ablation.data_root=data/rehearsal",
            "ablation.steps=200",
        ],
        model=EvalConfig,
    )
    assert cfg.ablation.name == "rehearsal"
    assert cfg.ablation.demo_counts == [3, "all"]
    assert cfg.ablation.data_root == Path("data/rehearsal")
    assert cfg.ablation.steps == 200


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "Has Space"},
        {"name": ""},
        {"demo_counts": []},
        {"demo_counts": [0]},
        {"demo_counts": ["most"]},
        {"augmentation": []},
        {"steps": 0},
    ],
)
def test_ablation_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        AblationConfig(**bad)


def test_train_config_hub_fields_default_off():
    cfg = TrainConfig()
    assert cfg.push_repo_id is None and cfg.private is True
    assert TrainConfig(dataset_root=None).dataset_root is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_common/tests/test_config.py -q`
Expected: collection error `ImportError: cannot import name 'AblationConfig'`.

- [ ] **Step 3: Implement**

In `config.py`: add `import re` with the stdlib imports; add `"AblationConfig"` to `__all__`
(alphabetical, first). Change `TrainConfig`:

```python
class TrainConfig(_Config):
    policy_path: str = "lerobot/smolvla_base"
    dataset_repo_id: str = "local/ego2libero"
    dataset_root: Path | None = Path("data/lerobot/ego2libero")  # None: dataset_repo_id on Hub
    output_dir: Path = Path("outputs/train/smolvla")
    steps: int = 20000
    batch_size: int = 32
    num_workers: int = 4
    save_freq: int = 5000
    wandb: bool = False
    push_repo_id: str | None = None  # push the final checkpoint to this Hub model repo
    private: bool = True  # Hub visibility of push_repo_id
    extra_args: list[str] = Field(default_factory=list)
```

Add before `EvalConfig`:

```python
class AblationConfig(_Config):
    """Demo count x augmentation sweep (`e2l eval ablate`); Run A / Run B are its `all` cells.
    Defaults describe the real run; the rehearsal overrides name, data_root and the run ids."""

    name: str = "real"  # plan id: outputs/ablation/<name>/, Hub repos e2l-<name>-<cell>
    data_root: Path = Path("data")  # E2L data root holding replay_run and generated_run
    replay_run: str = "replay"  # data/generated/<replay_run>: raw replays (Run A source)
    generated_run: str = "run0"  # data/generated/<generated_run>: multiplied episodes
    demo_counts: list[int | Literal["all"]] = Field(default_factory=lambda: [5, 10, "all"])
    augmentation: list[bool] = Field(default_factory=lambda: [False, True])
    subset_seed: int = 0  # one shuffle of the eligible demos; cell n takes the first n
    steps: int = 20000  # training steps, the same for every cell
    hub_namespace: str | None = None  # None: the logged-in Hugging Face user
    private: bool = True

    @model_validator(mode="after")
    def _check(self) -> "AblationConfig":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.name):
            raise ValueError(f"ablation.name {self.name!r}: lowercase letters, digits and '-'")
        if not self.demo_counts or not self.augmentation:
            raise ValueError("ablation needs at least one demo count and one augmentation flag")
        if any(c != "all" and c < 1 for c in self.demo_counts):
            raise ValueError("ablation.demo_counts must be positive or 'all'")
        if self.steps < 1:
            raise ValueError("ablation.steps must be at least 1")
        return self
```

In `EvalConfig`, replace the `demo_counts` and `augmentation` lines with:

```python
    ablation: AblationConfig = Field(default_factory=AblationConfig)
```

In `configs/eval.yaml`, replace the two lines `demo_counts: ...` and `augmentation: ...` with:

```yaml
ablation:                                    # e2l eval ablate prepare | evaluate
  name: real                                 # plan id: outputs/ablation/<name>/, Hub e2l-<name>-<cell>
  data_root: data                            # rehearsal: data/rehearsal
  replay_run: replay                         # raw replays (Run A source); rehearsal: rehearsal_replay
  generated_run: run0                        # multiplied episodes; rehearsal: rehearsal_gen
  demo_counts: [5, 10, all]                  # nested demo subsets; all = Run A / Run B
  augmentation: [false, true]                # raw replays only / plus generated episodes
  subset_seed: 0
  steps: 20000                               # per cell; set from the Spark smoke run's speed
  hub_namespace: null                        # null: the logged-in Hugging Face user
  private: true
```

In `configs/train/smolvla.yaml`, after the `wandb: false` line add:

```yaml
push_repo_id: null                           # Hub model repo for the final checkpoint (ablation sets it)
private: true
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_common -q`
Expected: all pass (the repo-config tests load the edited YAMLs; `"all"` parses as the
literal, `5` as int).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_common configs/eval.yaml configs/train/smolvla.yaml
git commit -m "feat(common): configure the ablation and Hub training fields"
```

---

### Task 2: `finetune` / `export` with Hub datasets and checkpoint push

**Files:**
- Modify: `packages/e2l_train/src/e2l_train/finetune.py`
- Modify: `packages/e2l_train/src/e2l_train/export.py:60-68` (guard `dataset_root is None`)
- Test: `packages/e2l_train/tests/test_finetune.py` (append), `packages/e2l_train/tests/test_export.py` (append)

**Interfaces:**
- Consumes: `TrainConfig.dataset_root: Path | None`, `push_repo_id`, `private` (Task 1).
- Produces: `build_command(cfg)` emitting `--policy.push_to_hub=true`,
  `--policy.repo_id=<id>`, `--policy.private=<true|false>` when `push_repo_id` is set (else
  `--policy.push_to_hub=false` as today), and no `--dataset.root` when `dataset_root is None`.
  `launch` skips the local-dataset check when `dataset_root is None`. `export_dataset` raises
  `ValueError` when `cfg.dataset_root is None`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/e2l_train/tests/test_finetune.py`:

```python
def test_build_command_trains_from_hub_and_pushes_checkpoint():
    cfg = TrainConfig(
        dataset_root=None,
        dataset_repo_id="me/e2l-x-all_raw",
        push_repo_id="me/e2l-x-all_raw-policy",
    )
    cmd = build_command(cfg)
    assert "--dataset.repo_id=me/e2l-x-all_raw" in cmd
    assert not any(a.startswith("--dataset.root") for a in cmd)
    assert "--policy.push_to_hub=true" in cmd
    assert "--policy.repo_id=me/e2l-x-all_raw-policy" in cmd
    assert "--policy.private=true" in cmd
    assert "--policy.push_to_hub=false" not in cmd


def test_build_command_without_push_is_unchanged():
    cmd = build_command(TrainConfig())
    assert "--policy.push_to_hub=false" in cmd
    assert not any(a.startswith(("--policy.repo_id", "--policy.private")) for a in cmd)
    assert "--dataset.root=data/lerobot/ego2libero" in cmd


def test_launch_from_hub_skips_local_dataset_check(tmp_path, monkeypatch):
    import subprocess

    from e2l_train import finetune

    calls = []
    monkeypatch.setattr(finetune.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
    cfg = TrainConfig(dataset_root=None, output_dir=tmp_path / "out")
    finetune.launch(cfg)
    assert calls and calls[0][0] == "/bin/lerobot-train"
```

Append to `packages/e2l_train/tests/test_export.py`:

```python
def test_export_needs_a_local_dataset_root(tmp_path):
    with pytest.raises(ValueError, match="dataset_root"):
        export_dataset([tmp_path / "x.npz"], TrainConfig(dataset_root=None))
```

(`pytest`, `TrainConfig` and `export_dataset` are already imported in `test_export.py`; check
the top of the file and add any that are missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_train -q`
Expected: the three new finetune tests and the export test FAIL (`--dataset.root=None` present,
no push flags, `FileNotFoundError` from launch, `AttributeError: 'NoneType' ... exists`).

- [ ] **Step 3: Implement**

`finetune.py` `build_command`:

```python
def build_command(cfg: TrainConfig) -> list[str]:
    """The full `lerobot-train` argv for a config (fine-tune from `cfg.policy_path`). Without
    `dataset_root` the dataset is `dataset_repo_id` on the Hub; with `push_repo_id` the final
    checkpoint is pushed there."""
    if cfg.push_repo_id:
        push = [
            "--policy.push_to_hub=true",
            f"--policy.repo_id={cfg.push_repo_id}",
            f"--policy.private={str(cfg.private).lower()}",
        ]
    else:
        push = ["--policy.push_to_hub=false"]
    root = [] if cfg.dataset_root is None else [f"--dataset.root={cfg.dataset_root}"]
    return [
        "lerobot-train",
        f"--policy.path={cfg.policy_path}",
        *push,
        f"--dataset.repo_id={cfg.dataset_repo_id}",
        *root,
        f"--output_dir={cfg.output_dir}",
        f"--job_name={cfg.output_dir.name}",
        f"--steps={cfg.steps}",
        f"--batch_size={cfg.batch_size}",
        f"--num_workers={cfg.num_workers}",
        f"--save_freq={cfg.save_freq}",
        f"--wandb.enable={str(cfg.wandb).lower()}",
        *cfg.extra_args,
    ]
```

In `launch`, change the dataset check to:

```python
    local = cfg.dataset_root
    if local is not None and not (local / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"no LeRobot dataset at {local}; run `e2l train export`")
```

In `export.py` `export_dataset`, first lines of the body (before the `LeRobotDataset` import):

```python
    if cfg.dataset_root is None:
        raise ValueError("export needs a local dataset_root to write to")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_train -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_train
git commit -m "feat(train): fine-tune from Hub datasets and push checkpoints"
```

---

### Task 3: `e2l_eval.hub`: thin Hub wrappers

**Files:**
- Create: `packages/e2l_eval/src/e2l_eval/hub.py`
- Test: `packages/e2l_eval/tests/test_hub.py`

**Interfaces:**
- Produces:
  - `namespace(configured: str | None) -> str` (configured, else `whoami()["name"]`)
  - `repo_exists(repo_id: str, repo_type: str = "model") -> bool`
  - `push_dataset(root: Path, repo_id: str, private: bool) -> None`
  - `push_folder(folder: Path, repo_id: str, private: bool) -> None` (dataset-type repo)
  - `download_policy(repo_id: str) -> Path` (local snapshot dir)

- [ ] **Step 1: Write the failing tests**

`packages/e2l_eval/tests/test_hub.py`:

```python
from pathlib import Path

import huggingface_hub

from e2l_eval import hub


def test_namespace_prefers_config_then_logged_in_user(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "whoami", lambda: {"name": "someone"})
    assert hub.namespace("team") == "team"
    assert hub.namespace(None) == "someone"


class _Api:
    calls: list = []

    def repo_exists(self, repo_id, repo_type=None):
        _Api.calls.append(("exists", repo_id, repo_type))
        return repo_id.endswith("yes")

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        _Api.calls.append(("create", repo_id, repo_type, private, exist_ok))

    def upload_folder(self, repo_id, folder_path, repo_type=None):
        _Api.calls.append(("upload", repo_id, folder_path, repo_type))


def test_repo_exists_and_push_folder_use_dataset_type(monkeypatch, tmp_path):
    _Api.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert hub.repo_exists("a/yes", "dataset") and not hub.repo_exists("a/no")
    hub.push_folder(tmp_path, "a/e2l-x-plan", private=True)
    assert ("exists", "a/yes", "dataset") in _Api.calls
    assert ("exists", "a/no", "model") in _Api.calls
    assert ("create", "a/e2l-x-plan", "dataset", True, True) in _Api.calls
    assert ("upload", "a/e2l-x-plan", str(tmp_path), "dataset") in _Api.calls


def test_download_policy_returns_snapshot_dir(monkeypatch, tmp_path):
    seen = {}

    def fake(repo_id, repo_type=None):
        seen.update(repo_id=repo_id, repo_type=repo_type)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert hub.download_policy("a/b-policy") == Path(tmp_path)
    assert seen == {"repo_id": "a/b-policy", "repo_type": "model"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_hub.py -q`
Expected: collection error `ImportError: cannot import name 'hub' from 'e2l_eval'`.

- [ ] **Step 3: Implement**

`packages/e2l_eval/src/e2l_eval/hub.py`:

```python
"""Hugging Face Hub transfers between the laptop and the DGX Spark. Imports are lazy so tests
and `--help` work offline; needs `hf auth login` (write token) for anything but namespace()."""

from pathlib import Path


def namespace(configured: str | None) -> str:
    """The configured Hub namespace, else the logged-in user."""
    if configured:
        return configured
    from huggingface_hub import whoami

    return whoami()["name"]


def repo_exists(repo_id: str, repo_type: str = "model") -> bool:
    from huggingface_hub import HfApi

    return HfApi().repo_exists(repo_id, repo_type=repo_type)


def push_dataset(root: Path, repo_id: str, private: bool) -> None:
    """Upload the LeRobot dataset at `root` (images, no videos) to `repo_id`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LeRobotDataset(repo_id, root=root).push_to_hub(private=private, push_videos=False)


def push_folder(folder: Path, repo_id: str, private: bool) -> None:
    """Upload a small folder (plan.json, spark.sh) as a dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="dataset")


def download_policy(repo_id: str) -> Path:
    """Local snapshot of a pushed checkpoint (config, weights, train_config, processors)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="model"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_hub.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval/src/e2l_eval/hub.py packages/e2l_eval/tests/test_hub.py
git commit -m "feat(eval): add Hub transfer helpers for Spark training"
```

---

### Task 4: Cell selection: nested demo subsets × augmentation

**Files:**
- Modify (rewrite): `packages/e2l_eval/src/e2l_eval/ablation.py` (stub replaced; Tasks 5–6 add
  to it)
- Test: `packages/e2l_eval/tests/test_ablation.py` (new)
- Modify: `packages/e2l_eval/tests/test_skeleton.py` (no stubs remain)

**Interfaces:**
- Consumes: `AblationConfig` (Task 1), `DataPaths(root).generated(run_id)`.
- Produces:
  - `@dataclass(frozen=True) class Cell: label: str; n_demos: int; augmented: bool;
    demo_ids: tuple[str, ...]; episodes: tuple[Path, ...]` (npz paths, replays first)
  - `cell_label(count: int | str, augmented: bool) -> str`
  - `eligible_demos(replay_dir: Path) -> list[str]`
  - `select_cells(cfg: AblationConfig) -> list[Cell]` (order: demo_counts outer,
    augmentation inner, as configured)

- [ ] **Step 1: Write the failing tests**

`packages/e2l_eval/tests/test_ablation.py`:

```python
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
```

In `packages/e2l_eval/tests/test_skeleton.py`, replace `test_stubs_raise_named_not_implemented`
with:

```python
def test_no_stubs_remain():
    assert check_stubs("e2l_eval") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_ablation.py packages/e2l_eval/tests/test_skeleton.py -q`
Expected: collection error importing `cell_label` from `e2l_eval.ablation`; skeleton test
FAILS (`1 == 0`, the stub still exists).

- [ ] **Step 3: Implement (replace the whole stub module)**

`packages/e2l_eval/src/e2l_eval/ablation.py`:

```python
"""Demo count x augmentation sweep, trained on the DGX Spark through the Hugging Face Hub.

`prepare` exports and pushes one LeRobot dataset per cell and writes plan.json + spark.sh; the
user runs spark.sh on the Spark (one `e2l train finetune` per cell, each pushing its checkpoint);
`evaluate` pulls every trained checkpoint and runs both eval settings. Run A and Run B are the
`all_raw` and `all_aug` cells."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e2l_common.config import AblationConfig
from e2l_common.paths import DataPaths


@dataclass(frozen=True)
class Cell:
    label: str
    n_demos: int
    augmented: bool
    demo_ids: tuple[str, ...]
    episodes: tuple[Path, ...]  # SimEpisode npz files: the demos' replays, then generated


def cell_label(count: int | str, augmented: bool) -> str:
    size = "all" if count == "all" else f"n{int(count):02d}"
    return f"{size}_{'aug' if augmented else 'raw'}"


def _sidecar(npz: Path) -> dict:
    return json.loads(npz.with_suffix(".json").read_text())


def eligible_demos(replay_dir: Path) -> list[str]:
    """Sorted demo ids whose raw replay in `replay_dir` succeeded (file stem = demo id)."""
    return sorted(p.stem for p in replay_dir.glob("*.npz") if _sidecar(p)["success"])


def select_cells(cfg: AblationConfig) -> list[Cell]:
    """Every (demo count, augmentation) cell in config order. Eligible demos are shuffled once
    with rng(subset_seed) and cell n takes the first n, so smaller subsets nest in larger ones.
    `aug` cells add every successful generated episode whose sources are all in the subset."""
    paths = DataPaths(cfg.data_root)
    replay_dir = paths.generated(cfg.replay_run)
    eligible = eligible_demos(replay_dir)
    if not eligible:
        raise ValueError(f"no successful replays in {replay_dir}")
    order = [eligible[i] for i in np.random.default_rng(cfg.subset_seed).permutation(len(eligible))]
    gen_dir = paths.generated(cfg.generated_run)
    generated = [
        (p, set(meta["source_demo_ids"]))
        for p in sorted(gen_dir.glob("*.npz"))
        if (meta := _sidecar(p))["success"]
    ]
    cells = []
    for count in cfg.demo_counts:
        n = len(order) if count == "all" else count
        if n > len(order):
            raise ValueError(
                f"demo count {count} exceeds the {len(order)} demos whose replay succeeded "
                f"in {replay_dir}"
            )
        demos = tuple(sorted(order[:n]))
        raw = tuple(replay_dir / f"{d}.npz" for d in demos)
        for augmented in cfg.augmentation:
            extra = tuple(p for p, src in generated if src <= set(demos)) if augmented else ()
            if augmented and not extra:
                raise ValueError(
                    f"no successful generated episodes for {cell_label(count, True)} in {gen_dir}"
                )
            cells.append(Cell(cell_label(count, augmented), n, augmented, demos, raw + extra))
    return cells
```

Note: `e2l_eval/cli.py` still imports `run_ablation` inside the old `ablate` command; Task 6
replaces that command. Until then `e2l eval ablate` would fail at call time only (the import is
inside the function), so `check_cli_help` keeps passing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval -q`
Expected: all pass (8 new ablation tests; skeleton `test_no_stubs_remain`; `test_cli_help`
still passes).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval/src/e2l_eval/ablation.py packages/e2l_eval/tests/test_ablation.py packages/e2l_eval/tests/test_skeleton.py
git commit -m "feat(eval): select nested demo subsets per ablation cell"
```

---

### Task 5: `prepare`: export, push, `plan.json` and `spark.sh`

**Files:**
- Modify: `packages/e2l_eval/src/e2l_eval/ablation.py` (append)
- Test: `packages/e2l_eval/tests/test_ablation.py` (append)

**Interfaces:**
- Consumes: `select_cells`, `Cell` (Task 4); `hub.namespace`, `hub.repo_exists`,
  `hub.push_dataset`, `hub.push_folder` (Task 3); `e2l_train.export.export_dataset(paths,
  cfg)`; `TrainConfig` (Task 1).
- Produces:
  - `PLAN_ROOT = Path("outputs/ablation")`
  - `repo_ids(namespace: str, plan: str, label: str) -> tuple[str, str]` (dataset, policy)
  - `prepare(cfg: EvalConfig, train_cfg: TrainConfig, only: list[str] | None = None,
    plan_root: Path = PLAN_ROOT) -> Path` (path of `plan.json`)
  - `spark_script(plan: dict) -> str`
  - `plan.json` keys: `name, created, namespace, plan_repo, steps, batch_size, ablation,
    cells[]`; each cell: `label, n_demos, augmented, demo_ids, episodes, frames,
    dataset_repo, policy_repo, prepared`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/e2l_eval/tests/test_ablation.py`:

```python
from e2l_common.config import EvalConfig, TrainConfig
from e2l_eval import ablation, hub


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_ablation.py -q`
Expected: the 5 new tests FAIL with `AttributeError: module 'e2l_eval.ablation' has no
attribute 'export_dataset'` (from the fixture) / `'prepare'`.

- [ ] **Step 3: Implement**

At the top of `ablation.py`, extend the imports:

```python
import json
import shlex
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from e2l_common.config import AblationConfig, EvalConfig, TrainConfig
from e2l_common.paths import DataPaths
from e2l_eval import hub
from e2l_train.export import export_dataset

PLAN_ROOT = Path("outputs/ablation")
```

(`e2l_train.export` imports LeRobot only inside `export_dataset`, so this stays offline.)
Append:

```python
def repo_ids(namespace: str, plan: str, label: str) -> tuple[str, str]:
    """(dataset repo, policy repo) for one cell."""
    base = f"{namespace}/e2l-{plan}-{label}"
    return base, f"{base}-policy"


def prepare(
    cfg: EvalConfig,
    train_cfg: TrainConfig,
    only: list[str] | None = None,
    plan_root: Path = PLAN_ROOT,
) -> Path:
    """Export and push the datasets of the selected cells (`only`, default all), then write and
    push plan.json + spark.sh. An existing local export is reused (delete it to rebuild) and a
    dataset repo already on the Hub is not pushed again. Returns the plan.json path."""
    ab = cfg.ablation
    cells = select_cells(ab)
    labels = [c.label for c in cells]
    chosen = labels if only is None else only
    unknown = [c for c in chosen if c not in labels]
    if unknown:
        raise ValueError(f"unknown cells {unknown}; this plan has {labels}")
    ns = hub.namespace(ab.hub_namespace)
    data = DataPaths(ab.data_root)
    rows = []
    for cell in cells:
        dataset_repo, policy_repo = repo_ids(ns, ab.name, cell.label)
        row = {
            "label": cell.label,
            "n_demos": cell.n_demos,
            "augmented": cell.augmented,
            "demo_ids": list(cell.demo_ids),
            "episodes": len(cell.episodes),
            "frames": None,
            "dataset_repo": dataset_repo,
            "policy_repo": policy_repo,
            "prepared": cell.label in chosen,
        }
        if row["prepared"]:
            root = data.root / "lerobot" / f"{ab.name}-{cell.label}"
            if not root.exists():
                update = {"dataset_root": root, "dataset_repo_id": dataset_repo}
                export_dataset(list(cell.episodes), train_cfg.model_copy(update=update))
            row["frames"] = json.loads((root / "meta" / "info.json").read_text())["total_frames"]
            if not hub.repo_exists(dataset_repo, "dataset"):
                hub.push_dataset(root, dataset_repo, ab.private)
        rows.append(row)
    plan = {
        "name": ab.name,
        "created": date.today().isoformat(),
        "namespace": ns,
        "plan_repo": f"{ns}/e2l-{ab.name}-plan",
        "steps": ab.steps,
        "batch_size": train_cfg.batch_size,
        "ablation": ab.model_dump(mode="json"),
        "cells": rows,
    }
    out = plan_root / ab.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "spark.sh").write_text(spark_script(plan))
    hub.push_folder(out, plan["plan_repo"], ab.private)
    return out / "plan.json"


def spark_script(plan: dict) -> str:
    """Bash for the DGX Spark: one `e2l train finetune` per prepared cell, training from the
    cell's Hub dataset and pushing the checkpoint; cells with an output dir are skipped."""
    name = plan["name"]
    private = str(plan["ablation"]["private"]).lower()
    lines = [
        "#!/usr/bin/env bash",
        f"# Ablation plan {name!r}, written by `e2l eval ablate prepare` on {plan['created']}.",
        "# On the DGX Spark, in a clone of ego2libero:",
        "#   git pull && uv sync --group train --group dev",
        f"#   uv run hf download {plan['plan_repo']} --repo-type dataset "
        f"--local-dir outputs/ablation/{name}",
        f"#   bash outputs/ablation/{name}/spark.sh",
        "set -euo pipefail",
        "unset PYTHONPATH",
    ]
    for row in plan["cells"]:
        if not row["prepared"]:
            continue
        run = f"{name}-{row['label']}"
        sets = {
            "dataset_root": "null",
            "dataset_repo_id": row["dataset_repo"],
            "output_dir": f"outputs/train/{run}",
            "steps": plan["steps"],
            "push_repo_id": row["policy_repo"],
            "private": private,
        }
        args = " ".join(f"--set {shlex.quote(f'{k}={v}')}" for k, v in sets.items())
        lines += [
            f"if [ -d outputs/train/{run} ]; then",
            f"  echo 'skip {run}: outputs/train/{run} exists (delete it to retrain)'",
            "else",
            f"  uv run e2l train finetune {args}",
            "fi",
        ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval/src/e2l_eval/ablation.py packages/e2l_eval/tests/test_ablation.py
git commit -m "feat(eval): export and push ablation cells with a Spark training script"
```

---

### Task 6: `evaluate` and the `e2l eval ablate` CLI

**Files:**
- Modify: `packages/e2l_eval/src/e2l_eval/ablation.py` (append `evaluate`)
- Modify: `packages/e2l_eval/src/e2l_eval/cli.py` (replace `ablate` command with a sub-app)
- Modify: `CLAUDE.md` (Commands section)
- Test: `packages/e2l_eval/tests/test_ablation.py` (append), `packages/e2l_eval/tests/test_cli.py` (append)

**Interfaces:**
- Consumes: plan.json (Task 5); `hub.repo_exists`, `hub.download_policy` (Task 3);
  `e2l_eval.rollout.rollout(policy_path, cfg, sim_cfg, out_dir, max_videos=, setting=, run=)`
  and `SETTINGS` (existing).
- Produces:
  - `evaluate(cfg: EvalConfig, sim_cfg: SimConfig, plan_path: Path, videos: int = 3) ->
    dict[str, str]`: label → `"done" | "evaluated" | "pending" | "not prepared"`.
  - CLI: `e2l eval ablate prepare [--cells a,b] [-c eval.yaml] [--train-config ...] [--set ...]`
    and `e2l eval ablate evaluate <plan> [--videos N] [-c ...] [--sim-config ...] [--set ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/e2l_eval/tests/test_ablation.py`:

```python
def _plan(tmp_path: Path) -> Path:
    plan = {
        "name": "rehearsal",
        "cells": [
            {"label": "all_raw", "prepared": True, "policy_repo": "me/raw-policy"},
            {"label": "all_aug", "prepared": True, "policy_repo": "me/aug-policy"},
            {"label": "n05_raw", "prepared": True, "policy_repo": "me/n05-policy"},
            {"label": "n05_aug", "prepared": False, "policy_repo": "me/x-policy"},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_evaluate_skips_done_and_pending_cells(tmp_path, monkeypatch):
    import e2l_eval.rollout as rollout_module
    from e2l_common.config import SimConfig

    cfg = EvalConfig(output_dir=tmp_path / "eval")
    done = cfg.output_dir / "rehearsal-all_raw"
    for s in ("standard", "wide"):
        (done / s).mkdir(parents=True)
        (done / s / "results.json").write_text("{}")
    (cfg.output_dir / "rehearsal-n05_raw" / "standard").mkdir(parents=True)
    (cfg.output_dir / "rehearsal-n05_raw" / "standard" / "results.json").write_text("{}")
    monkeypatch.setattr(hub, "repo_exists", lambda repo, kind="model": repo != "me/aug-policy")
    monkeypatch.setattr(hub, "download_policy", lambda repo: tmp_path / "snap" / repo)
    calls = []
    monkeypatch.setattr(
        rollout_module,
        "rollout",
        lambda policy, c, s, out, max_videos, setting, run: calls.append(
            (policy, out, max_videos, setting, run)
        ),
    )
    status = ablation.evaluate(cfg, SimConfig(), _plan(tmp_path), videos=2)
    assert status == {
        "all_raw": "done",
        "all_aug": "pending",
        "n05_raw": "evaluated",
        "n05_aug": "not prepared",
    }
    assert calls == [
        (
            str(tmp_path / "snap" / "me/n05-policy"),
            cfg.output_dir / "rehearsal-n05_raw" / "wide",
            2,
            "wide",
            "rehearsal-n05_raw",
        )
    ]
```

Append to `packages/e2l_eval/tests/test_cli.py` (add `from e2l_common.cli import app as root`
and `from typer.testing import CliRunner` at the top if missing):

```python
def test_ablate_commands_have_help():
    runner = CliRunner()
    for args in (
        ["eval", "ablate", "--help"],
        ["eval", "ablate", "prepare", "--help"],
        ["eval", "ablate", "evaluate", "--help"],
    ):
        result = runner.invoke(root, args)
        assert result.exit_code == 0, result.output
    assert "--cells" in runner.invoke(root, ["eval", "ablate", "prepare", "--help"]).output


def test_ablate_evaluate_names_missing_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from e2l_common.paths import repo_root

    cfg = str(repo_root() / "configs" / "eval.yaml")
    sim = str(repo_root() / "configs" / "sim.yaml")
    result = CliRunner().invoke(
        root, ["eval", "ablate", "evaluate", "nope", "-c", cfg, "--sim-config", sim]
    )
    assert result.exit_code != 0
    assert "outputs/ablation/nope/plan.json" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_ablation.py packages/e2l_eval/tests/test_cli.py -q`
Expected: `test_evaluate_skips_done_and_pending_cells` FAILS (`no attribute 'evaluate'`); the two
CLI tests FAIL (`No such command 'prepare'` / exit code from the old `ablate`).

- [ ] **Step 3: Implement `evaluate`**

Add `SimConfig` to the `e2l_common.config` import in `ablation.py`, then append:

```python
def evaluate(
    cfg: EvalConfig, sim_cfg: SimConfig, plan_path: Path, videos: int = 3
) -> dict[str, str]:
    """For each prepared cell whose checkpoint is on the Hub, run every setting that has no
    results.json yet under cfg.output_dir/<plan>-<cell>/<setting>/. Returns label -> status:
    done (all settings present), evaluated, pending (no checkpoint yet), not prepared."""
    from e2l_eval import rollout as rollout_module

    plan = json.loads(Path(plan_path).read_text())
    status = {}
    for row in plan["cells"]:
        label, run = row["label"], f"{plan['name']}-{row['label']}"
        if not row["prepared"]:
            status[label] = "not prepared"
            continue
        todo = [
            s
            for s in rollout_module.SETTINGS
            if not (cfg.output_dir / run / s / "results.json").is_file()
        ]
        if not todo:
            status[label] = "done"
            continue
        if not hub.repo_exists(row["policy_repo"], "model"):
            status[label] = "pending"
            continue
        policy = hub.download_policy(row["policy_repo"])
        for setting in todo:
            rollout_module.rollout(
                str(policy),
                cfg,
                sim_cfg,
                cfg.output_dir / run / setting,
                max_videos=videos,
                setting=setting,
                run=run,
            )
        status[label] = "evaluated"
    return status
```

Run `uv run ruff format packages/e2l_eval` afterwards (the `def evaluate(...)` line is long).

- [ ] **Step 4: Replace the CLI command**

In `packages/e2l_eval/src/e2l_eval/cli.py`, delete the whole `ablate` command (the
`@app.command()` / `def ablate(...)` block) and add in its place:

```python
ablate_app = typer.Typer(
    help="Demo count x augmentation sweep, trained on the DGX Spark via the Hub.",
    no_args_is_help=True,
)
app.add_typer(ablate_app, name="ablate")


@ablate_app.command("prepare")
def ablate_prepare(
    cells: str = typer.Option(None, help="Comma-separated cell labels (default: every cell)."),
    config: Path = config_option("eval"),
    train_config: Path = typer.Option(Path("configs/train/smolvla.yaml"), help="Train config."),
    set_: list[str] = set_option(),
) -> None:
    """Export + push each cell's dataset; write and push plan.json and spark.sh."""
    from e2l_common.config import TrainConfig, load_config
    from e2l_eval.ablation import prepare

    cfg = load_stage_config(config, set_, EvalConfig)
    only = [c.strip() for c in cells.split(",")] if cells else None
    plan = prepare(cfg, load_config(train_config, model=TrainConfig), only)
    typer.echo(f"wrote {plan} and {plan.parent / 'spark.sh'}")
    typer.echo((plan.parent / "spark.sh").read_text().split("set -euo")[0])


@ablate_app.command("evaluate")
def ablate_evaluate(
    plan: str = typer.Argument(..., help="Plan name: outputs/ablation/<plan>/plan.json."),
    videos: int = typer.Option(3, help="Episodes per setting to save as mp4."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    set_: list[str] = set_option(),
) -> None:
    """Evaluate every trained cell of a plan on standard and wide; rerun as cells finish."""
    from e2l_common.config import load_config
    from e2l_eval.ablation import PLAN_ROOT, evaluate

    plan_path = PLAN_ROOT / plan / "plan.json"
    if not plan_path.is_file():
        raise typer.BadParameter(f"no {plan_path}; run `e2l eval ablate prepare`")
    cfg = load_stage_config(config, set_, EvalConfig)
    status = evaluate(cfg, load_config(sim_config, model=SimConfig), plan_path, videos)
    for label, state in status.items():
        typer.echo(f"{label:10s} {state}")
```

In `CLAUDE.md`, under `## Commands`, after the `make rehearsal` bullet (two lines), add:

```markdown
- Ablation / Run A·B via the Spark (Hub login on both): `uv run e2l eval ablate prepare
  [--cells all_raw,all_aug]` → run the printed `spark.sh` steps on the Spark →
  `uv run e2l eval ablate evaluate <plan>` (rehearsal: `--set ablation.name=rehearsal
  ablation.data_root=data/rehearsal ablation.replay_run=rehearsal_replay ablation.generated_run=rehearsal_gen`)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass; ruff clean. If `test_ablate_evaluate_names_missing_plan` shows the path
wrapped across lines by rich, assert on `"nope/plan.json"` instead and ledger the change.

- [ ] **Step 6: Commit**

```bash
git add packages/e2l_eval CLAUDE.md
git commit -m "feat(eval): evaluate Spark-trained ablation cells from the Hub"
```

---

### Task 7: Ablation chart in `e2l eval report`

**Files:**
- Modify: `packages/e2l_eval/src/e2l_eval/report.py` (add `load_plans`, `plot_ablation`;
  `make_report` gains `ablation_dir: Path | None = None`)
- Modify: `packages/e2l_eval/src/e2l_eval/cli.py` (`report` passes `--ablation-dir`)
- Test: `packages/e2l_eval/tests/test_report.py` (append)

**Interfaces:**
- Consumes: plan.json (Task 5), result rows from `load_results` (existing; run label
  `<plan>-<cell>`).
- Produces: `load_plans(ablation_dir: Path) -> list[dict]`;
  `plot_ablation(plan: dict, rows: list[dict], path: Path) -> Path | None`;
  `make_report(results_dir, out_dir, ablation_dir=None)` also writes `ablation_<plan>.png` for
  each plan with at least one evaluated cell. CLI `e2l eval report --ablation-dir
  outputs/ablation` (default).

- [ ] **Step 1: Write the failing tests**

Append to `packages/e2l_eval/tests/test_report.py`:

```python
from e2l_eval.report import load_plans, plot_ablation


def _plan_file(root: Path, name: str = "rehearsal") -> dict:
    plan = {
        "name": name,
        "cells": [
            {"label": "n05_raw", "n_demos": 5, "augmented": False},
            {"label": "n05_aug", "n_demos": 5, "augmented": True},
            {"label": "all_raw", "n_demos": 25, "augmented": False},
            {"label": "all_aug", "n_demos": 25, "augmented": True},
        ],
    }
    (root / name).mkdir(parents=True)
    (root / name / "plan.json").write_text(json.dumps(plan))
    return plan


def test_plot_ablation_uses_only_evaluated_cells(tmp_path):
    plan = _plan_file(tmp_path / "ablation")
    _write(tmp_path / "res", "rehearsal-all_raw", "standard", [True, False])
    _write(tmp_path / "res", "rehearsal-all_aug", "wide", [True, True, False])
    rows = load_results(tmp_path / "res")
    out = plot_ablation(plan, rows, tmp_path / "a.png")
    assert out == tmp_path / "a.png" and out.stat().st_size > 0
    assert plot_ablation(plan, [], tmp_path / "none.png") is None
    assert not (tmp_path / "none.png").exists()


def test_make_report_adds_one_chart_per_plan_with_results(tmp_path):
    _plan_file(tmp_path / "ablation", "rehearsal")
    _plan_file(tmp_path / "ablation", "untrained")
    _write(tmp_path / "res", "rehearsal-all_raw", "standard", [True, False, True])
    assert [p["name"] for p in load_plans(tmp_path / "ablation")] == ["rehearsal", "untrained"]
    written = make_report(tmp_path / "res", tmp_path / "fig", tmp_path / "ablation")
    assert sorted(p.name for p in written) == [
        "ablation_rehearsal.png",
        "results.md",
        "success_rates.png",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval/tests/test_report.py -q`
Expected: collection error `ImportError: cannot import name 'load_plans'`.

- [ ] **Step 3: Implement**

In `report.py`, add after `plot_success_rates`:

```python
def load_plans(ablation_dir: Path) -> list[dict]:
    """Every `<ablation_dir>/<plan>/plan.json` written by `e2l eval ablate prepare`."""
    return [json.loads(p.read_text()) for p in sorted(Path(ablation_dir).glob("*/plan.json"))]


def plot_ablation(plan: dict, rows: list[dict], path: Path) -> Path | None:
    """Success rate vs number of source demos, one line per arm (human only / + generated) with
    95% Wilson error bars, one panel per setting. Only cells with results are drawn; None (and
    no file) if the plan has none yet."""
    by_key = {(r["run"], r["setting"]): r for r in rows}
    points: dict[tuple[str, bool], list[tuple[int, int, int]]] = {}
    for cell in plan["cells"]:
        for setting in SETTINGS:
            row = by_key.get((f"{plan['name']}-{cell['label']}", setting))
            if row is not None:
                k, n = sum(row["successes"]), len(row["successes"])
                points.setdefault((setting, cell["augmented"]), []).append((cell["n_demos"], k, n))
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(3.4 * len(SETTINGS), 3.0), sharey=True)
    for ax, setting in zip(axes, SETTINGS, strict=True):
        for augmented, name in ((False, "human only"), (True, "human + generated")):
            pts = sorted(points.get((setting, augmented), []))
            if not pts:
                continue
            xs = [x for x, _, _ in pts]
            rates = [k / n for _, k, n in pts]
            bounds = [wilson_interval(k, n) for _, k, n in pts]
            err = [
                [r - lo for r, (lo, _) in zip(rates, bounds, strict=True)],
                [hi - r for r, (_, hi) in zip(rates, bounds, strict=True)],
            ]
            ax.errorbar(xs, rates, yerr=err, marker="o", capsize=3, label=name)
        ax.set_title(setting)
        ax.set_xlabel("source demos")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"ablation: {plan['name']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
```

Change `make_report`:

```python
def make_report(
    results_dir: Path, out_dir: Path = Path("docs/figures"), ablation_dir: Path | None = None
) -> list[Path]:
    """`results.md`, `success_rates.png`, (if any videos) `rollouts.gif`, and for each plan in
    `ablation_dir` with evaluated cells an `ablation_<plan>.png`, in `out_dir`."""
    rows = load_results(results_dir)
    if not rows:
        raise FileNotFoundError(f"no <run>/<setting>/results.json under {results_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "results.md"
    table.write_text(results_table(rows))
    written = [table, plot_success_rates(rows, out_dir / "success_rates.png")]
    gif = rollout_grid(rows, out_dir / "rollouts.gif")
    written += [gif] if gif else []
    for plan in load_plans(ablation_dir) if ablation_dir else []:
        chart = plot_ablation(plan, rows, out_dir / f"ablation_{plan['name']}.png")
        written += [chart] if chart else []
    return written
```

In `cli.py` `report`, add the option and pass it:

```python
ablation_dir: Path = (
    typer.Option(Path("outputs/ablation"), help="Folder of <plan>/plan.json (ablation charts)."),
)
```

and call `make_report(results, out_dir, ablation_dir)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `unset PYTHONPATH VIRTUAL_ENV; uv run pytest packages/e2l_eval -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass (the existing `test_make_report_without_videos_writes_table_and_chart_only`
still passes: `ablation_dir` defaults to None).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval
git commit -m "feat(eval): chart success against demo count per ablation plan"
```

---

### Task 8: Smoke chain (laptop → Hub → Spark → Hub → laptop)

**Files:** none (runs). Outputs: `outputs/ablation/smoke/`, `outputs/eval/smoke-all_raw/`,
`data/rehearsal/lerobot/smoke-all_raw/`. This task has **two hand-offs to the user**; stop and
wait at each. Common prefix for every laptop command:
`cd .worktrees/hub-ablation-training && unset PYTHONPATH VIRTUAL_ENV && export HF_HOME=/mnt/data/hf-cache MUJOCO_GL=egl`.

Rehearsal overrides (used here and in Task 9), call them `$REH`:
`--set ablation.data_root=data/rehearsal --set ablation.replay_run=rehearsal_replay --set ablation.generated_run=rehearsal_gen`

- [ ] **Step 1: Check the Hub login**

Run: `uv run python -c "from huggingface_hub import whoami; print(whoami()['name'])"`
Expected: a username. If `LocalTokenNotFoundError`: **hand-off** — ask the user to run
`! HF_HOME=/mnt/data/hf-cache uv run hf auth login` (write token) and to log in on the Spark
too; wait.

- [ ] **Step 2: Prepare the smoke plan**

Run: `uv run e2l eval ablate prepare --cells all_raw --set ablation.name=smoke --set ablation.steps=200 $REH`
Expected: exports `data/rehearsal/lerobot/smoke-all_raw` (25 episodes, 3,291 frames), pushes
`<ns>/e2l-smoke-all_raw` and `<ns>/e2l-smoke-plan`, prints the `spark.sh` header. Check
`plan.json`: 6 cells, only `all_raw` prepared, `frames` 3291.

- [ ] **Step 3: Hand-off: smoke training on the Spark**

Give the user, verbatim (with `<ns>` filled in):

```bash
cd <ego2libero clone> && git fetch && git checkout feat/hub-ablation-training && git pull
uv sync --group train --group dev
unset PYTHONPATH
uv run hf download <ns>/e2l-smoke-plan --repo-type dataset --local-dir outputs/ablation/smoke
bash outputs/ablation/smoke/spark.sh 2>&1 | tee outputs/ablation/smoke/train.log
grep -E "step:|updt_s|it/s" outputs/ablation/smoke/train.log | tail -5
```

Ask them to reply with the last log lines (training speed) once it finishes. Wait.

- [ ] **Step 4: Evaluate the smoke checkpoint (proves loading)**

Run: `uv run e2l eval ablate evaluate smoke --set n_episodes=2 --videos 1`
Expected: `all_raw evaluated`, other cells `not prepared`;
`outputs/eval/smoke-all_raw/{standard,wide}/results.json` each with 2 episodes (success
irrelevant at 200 steps). A load error here (camera keys, rename map, processor files) is a
real finding: debug with superpowers:systematic-debugging before Task 9.

- [ ] **Step 5: Choose `steps`**

From the user's log, compute steps/s (LeRobot logs `updt_s` per step or a step rate).
`steps = min(20000, floor(3 * 3600 * steps_per_s / 100) * 100)`. Record the speed and the
choice in `outputs/ablation/summary.md` and in the ledger as a ruling.

---

### Task 9: Rehearsal Run A / Run B

**Files:** none (runs). Same command prefix and `$REH` as Task 8. One hand-off.

- [ ] **Step 1: Prepare the rehearsal plan**

Run: `uv run e2l eval ablate prepare --cells all_raw,all_aug --set ablation.name=rehearsal --set ablation.steps=<STEPS from Task 8> $REH`
Expected: exports `rehearsal-all_raw` (25 episodes) and `rehearsal-all_aug` (356 episodes,
50,664 frames), pushes both datasets and the plan. Check `df -h /mnt/data` before (needs
≥ 10 GB free).

- [ ] **Step 2: Hand-off: train both on the Spark**

Give the user:

```bash
cd <ego2libero clone> && git pull && uv sync --group train --group dev && unset PYTHONPATH
uv run hf download <ns>/e2l-rehearsal-plan --repo-type dataset --local-dir outputs/ablation/rehearsal
bash outputs/ablation/rehearsal/spark.sh 2>&1 | tee outputs/ablation/rehearsal/train.log
```

Expected duration ≈ 2 × the time budget from Task 8. Ask them to say when it finishes (or
when the first cell has pushed, to start evaluating early). Wait.

- [ ] **Step 3: Evaluate**

Run (background; ~4 × 25 min): `uv run e2l eval ablate evaluate rehearsal`
Expected: `all_raw evaluated`, `all_aug evaluated` (or `pending` if the Spark is still on it;
rerun later). Four `results.json` with 50 episodes each.

- [ ] **Step 4: Report into outputs/**

Run: `uv run e2l eval report --results outputs/eval --out-dir outputs/report`
Expected: `results.md` rows for `reference`, `rehearsal-all_raw`, `rehearsal-all_aug` (and
`smoke-all_raw` with n=2); `ablation_rehearsal.png` with one point per arm per setting;
`git status --short docs/` empty. Copy the table into `outputs/ablation/summary.md`.

- [ ] **Step 5: Free the Track 1 duplicates**

Once both rehearsal datasets are on the Hub: `rm -rf data/rehearsal/lerobot/rehearsal_a data/rehearsal/lerobot/rehearsal_b`
(6.4 GB; identical content to `rehearsal-all_raw` / `rehearsal-all_aug`, spec decision 11).

---

### Task 10: Review and finish

- [ ] **Step 1:** `unset PYTHONPATH VIRTUAL_ENV; make lint && make test` — clean, all pass.
- [ ] **Step 2:** superpowers:requesting-code-review on `main..HEAD` with the spec, this plan
  and its Review Focus; fix findings via superpowers:receiving-code-review.
- [ ] **Step 3:** superpowers:verification-before-completion, then
  superpowers:finishing-a-development-branch. PR body: what changed, the smoke chain result,
  Spark speed and chosen steps, rehearsal Run A / B standard and wide results next to the
  reference. No attribution lines.
