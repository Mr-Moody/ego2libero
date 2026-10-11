# LeRobot Dataset Export and Smoke Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `e2l train export` (successful `SimEpisode`s → a LeRobot v3 dataset in the
exact `HuggingFaceVLA/libero` format), switch fine-tuning to start from `lerobot/smolvla_base`,
and prove the training toolchain end to end with a tiny CPU smoke run on the scripted episodes
already in `data/generated/scripted50/`.

**Architecture:** `e2l_train/export.py` gets a pure per-episode converter (`episode_frames`) and
a writer (`export_dataset`) that streams episodes into `LeRobotDataset.create(...)`. The train
config moves to `lerobot/smolvla_base` plus whatever `lerobot-train` arguments map our LIBERO
feature keys onto the base checkpoint's cameras, found empirically in Task 2 and pinned in
`configs/train/smolvla.yaml`.

**Tech Stack:** Python 3.12, numpy, lerobot 0.6.1 (`LeRobotDataset`, `lerobot-train`), pydantic
config, typer CLI, pytest.

**Spec:** `docs/ego2libero — Project Initialisation Spec.md` (sections `e2l_train`, Conventions,
Risks: "Exported dataset does not match what smolvla_libero was trained on") and
`docs/ego2libero — Submission Checklist.md` (Phase 5 "Before training", Key decision:
fine-tune from `smolvla_base`). No separate design doc: the checklist plus the facts below are
the design.

## Facts established before planning (verified 2026-10-08 against the HF cache)

`HuggingFaceVLA/libero` `meta/info.json` (codebase v3.0, robot_type `panda`, fps **10**):

| key | dtype | shape | names |
|---|---|---|---|
| `observation.images.image` | image | (256, 256, 3) | `["height", "width", "channel"]` |
| `observation.images.image2` | image | (256, 256, 3) | `["height", "width", "channel"]` |
| `observation.state` | float32 | (8,) | `["state"]` |
| `action` | float32 | (7,) | `["actions"]` |

- Task string for our task (LIBERO-Goal 8) in `meta/tasks.parquet`: `put the bowl on the plate`.
  `SimEpisode.task` already holds exactly this string (it is `env.task_description`).
- `SimEpisode.images` keys are `image` and `image2`, each `(T, 256, 256, 3)` uint8, recorded
  from raw LeRobot `LiberoEnv` observations. LeRobot's `LiberoProcessorStep`
  (`lerobot/processor/env_processor.py`) rotates live env images 180° (`torch.flip(img,
  dims=[2, 3])`) before the policy sees them "to match the HuggingFaceVLA/libero camera
  orientation convention". So **exported images must be rotated 180°** (`img[::-1, ::-1]`).
- `SimEpisode.state` is the same 8-D layout as the dataset (eef pos 3, eef axis-angle 3,
  gripper qpos 2); actions are the 7-D LIBERO deltas, one per 20 Hz control step. The reference
  dataset stores one frame per control step and labels it 10 fps; we copy that (fps 10, no
  subsampling).
- `HuggingFaceVLA/smolvla_libero` `config.json`: input features `observation.images.image`,
  `observation.images.image2` (3,256,256), `observation.state` (8); output `action` (7).
- `lerobot/smolvla_base` `config.json`: input features `observation.images.camera1..3`
  (3,256,256), `observation.state` (6); output `action` (6); `empty_cameras: 0`.
  `lerobot-train` accepts `--rename_map` (requires a pretrained `--policy.path`).
- This laptop currently has **no working NVIDIA driver** (`nvidia-smi` fails), so the local smoke
  run is CPU-only and tiny; the 200-step GPU smoke run happens on the DGX Spark, run by the
  human from the command this plan produces.

## Global Constraints

- Run `unset PYTHONPATH` before any `uv run` (ROS Humble pollutes it). `HF_HOME=/mnt/data/hf-cache`
  for anything that downloads checkpoints or datasets (`/` is nearly full).
- `e2l_train` must not import `libero`, `robosuite` or `mujoco`. Import `lerobot` lazily, inside
  functions (as `finetune.py` avoids importing it at all).
- Lint: `uv run ruff check . && uv run ruff format --check .` (line 100). Type hints expected.
- Commits: `<type>(<scope>): <subject>`, scope from `common train configs ...`, header ≤ 72
  chars, footer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch: `feat/lerobot-dataset-export` (exists; worktree `.worktrees/lerobot-dataset-export`,
  venv already set up by `make worktree`). Never commit to `main`.
- Exported feature keys, dtypes, shapes, names, fps (10) and robot_type (`panda`) exactly as in
  the Facts table. Images rotated 180°. Only `success == True` episodes are exported.
- Generated data and outputs live under `data/` and `outputs/`; never `git add` them.

## Review Focus

1. `--max-episodes` larger than the number of successful episodes → export what exists, no error.
2. An existing `dataset_root` → refuse with `FileExistsError` before writing anything (LeRobot
   would otherwise half-overwrite), mirroring `finetune.launch`'s refusal for `output_dir`.
3. No successful episodes in the selection → `ValueError` before a dataset directory is created.
4. An episode with a missing camera or wrong image size → `ValueError` naming the episode, not a
   cryptic LeRobot shape error mid-write.
5. Images rotated the wrong way would silently ruin training: the round-trip test pins the 180°
   rotation, and Task 2 checks a real exported frame against a reference dataset frame by eye.

Tests for 1–5 are in Task 1 (Step 1) and Task 2 (Step 5).

---

### Task 1: `export.py` — SimEpisodes to a LeRobot dataset

**Files:**
- Modify: `packages/e2l_train/src/e2l_train/export.py` (replace the stub)
- Modify: `packages/e2l_train/src/e2l_train/cli.py` (`export` command: add `--max-episodes`)
- Test: `packages/e2l_train/tests/test_export.py` (create)

**Interfaces:**
- Consumes: `e2l_common.schemas.SimEpisode` (fields `task`, `success`, `images`, `state`,
  `actions`), `e2l_common.config.TrainConfig` (`dataset_repo_id`, `dataset_root`).
- Produces: `FEATURES: dict`, `FPS = 10`, `ROBOT_TYPE = "panda"`,
  `episode_frames(ep: SimEpisode) -> list[dict]`,
  `export_dataset(episode_paths: list[Path], cfg: TrainConfig, max_episodes: int | None = None) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `packages/e2l_train/tests/test_export.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from e2l_common.config import TrainConfig
from e2l_common.schemas import SimEpisode
from e2l_train.export import FPS, episode_frames, export_dataset

TASK = "put the bowl on the plate"


def _episode(T: int = 3, success: bool = True, seed: int = 0) -> SimEpisode:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (T, 256, 256, 3), dtype=np.uint8)
    return SimEpisode(
        task=TASK,
        success=success,
        source_demo_ids=["20261003_scripted_000"],
        images={"image": img, "image2": img[:, :, ::-1].copy()},
        state=rng.normal(size=(T, 8)),
        actions=rng.normal(size=(T, 7)),
        object_poses={"bowl": np.eye(4)},
    )


def _save(ep: SimEpisode, path: Path) -> Path:
    npz, _ = ep.save(path)
    return npz


def _cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(dataset_repo_id="local/e2l_test", dataset_root=tmp_path / "ds")


def _load(cfg: TrainConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)


def test_episode_frames_rotates_images_and_casts():
    ep = _episode()
    frames = episode_frames(ep)
    assert len(frames) == 3
    f = frames[1]
    np.testing.assert_array_equal(f["observation.images.image"], np.rot90(ep.images["image"][1], 2))
    np.testing.assert_array_equal(
        f["observation.images.image2"], np.rot90(ep.images["image2"][1], 2)
    )
    assert f["observation.state"].dtype == np.float32 and f["observation.state"].shape == (8,)
    assert f["action"].dtype == np.float32 and f["action"].shape == (7,)
    np.testing.assert_allclose(f["action"], ep.actions[1].astype(np.float32))
    assert f["task"] == TASK


def test_episode_frames_rejects_missing_camera():
    ep = _episode()
    ep.images.pop("image2")
    with pytest.raises(ValueError, match="image2"):
        episode_frames(ep)


def test_episode_frames_rejects_wrong_image_size():
    ep = _episode()
    ep.images["image"] = np.zeros((3, 128, 128, 3), np.uint8)
    with pytest.raises(ValueError, match="256"):
        episode_frames(ep)


def test_export_writes_loadable_dataset_of_successes_only(tmp_path):
    paths = [
        _save(_episode(T=3, seed=0), tmp_path / "ep" / "a"),
        _save(_episode(T=4, success=False, seed=1), tmp_path / "ep" / "b"),
        _save(_episode(T=2, seed=2), tmp_path / "ep" / "c"),
    ]
    cfg = _cfg(tmp_path)
    assert export_dataset(paths, cfg) == cfg.dataset_root

    ds = _load(cfg)
    assert ds.meta.total_episodes == 2
    assert len(ds) == 5
    assert ds.fps == FPS
    assert ds.meta.robot_type == "panda"
    item = ds[0]
    assert tuple(item["observation.images.image"].shape) == (3, 256, 256)
    assert tuple(item["observation.state"].shape) == (8,)
    assert tuple(item["action"].shape) == (7,)
    assert item["task"] == TASK
    # Round trip: stored image (CHW float in [0, 1]) is the 180-degree rotated original.
    stored = item["observation.images.image"].permute(1, 2, 0).numpy() * 255
    original = SimEpisode.load(paths[0]).images["image"][0]
    np.testing.assert_allclose(stored, np.rot90(original, 2), atol=1)


def test_export_max_episodes_caps_and_tolerates_too_large(tmp_path):
    paths = [_save(_episode(seed=s), tmp_path / "ep" / f"e{s}") for s in range(3)]
    cfg = _cfg(tmp_path)
    export_dataset(paths, cfg, max_episodes=2)
    assert _load(cfg).meta.total_episodes == 2

    cfg2 = TrainConfig(dataset_repo_id="local/e2l_test2", dataset_root=tmp_path / "ds2")
    export_dataset(paths, cfg2, max_episodes=10)
    assert _load(cfg2).meta.total_episodes == 3


def test_export_refuses_existing_root(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.dataset_root.mkdir()
    with pytest.raises(FileExistsError):
        export_dataset([_save(_episode(), tmp_path / "ep" / "a")], cfg)


def test_export_without_successes_raises_before_writing(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="no successful"):
        export_dataset([_save(_episode(success=False), tmp_path / "ep" / "a")], cfg)
    assert not cfg.dataset_root.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_train/tests/test_export.py -v`
Expected: FAIL at import (`cannot import name 'FPS'`).

- [ ] **Step 3: Implement `export.py`**

Replace `packages/e2l_train/src/e2l_train/export.py` with:

```python
"""SimEpisodes to a LeRobot dataset in the HuggingFaceVLA/libero format (what smolvla_libero was
trained on): same feature keys, dtypes, shapes, 10 fps label, one frame per control step."""

import json
from pathlib import Path

import numpy as np

from e2l_common.config import TrainConfig
from e2l_common.schemas import SimEpisode

FPS = 10
ROBOT_TYPE = "panda"
CAMERAS = ("image", "image2")
IMAGE_SHAPE = (256, 256, 3)
FEATURES = {
    **{
        f"observation.images.{cam}": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERAS
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def episode_frames(ep: SimEpisode) -> list[dict]:
    """One LeRobot frame per control step. Images are rotated 180 deg: HuggingFaceVLA/libero
    stores them that way, and LeRobot's LiberoProcessorStep rotates live observations to match."""
    for cam in CAMERAS:
        if cam not in ep.images:
            raise ValueError(f"episode has no camera {cam!r} (has {sorted(ep.images)})")
        if ep.images[cam].shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"camera {cam!r} images are {ep.images[cam].shape[1:]}, want 256x256")
    return [
        {
            **{
                f"observation.images.{cam}": np.ascontiguousarray(ep.images[cam][i, ::-1, ::-1])
                for cam in CAMERAS
            },
            "observation.state": ep.state[i].astype(np.float32),
            "action": ep.actions[i].astype(np.float32),
            "task": ep.task,
        }
        for i in range(len(ep.actions))
    ]


def _succeeded(npz: Path) -> bool:
    return bool(json.loads(Path(npz).with_suffix(".json").read_text())["success"])


def export_dataset(
    episode_paths: list[Path], cfg: TrainConfig, max_episodes: int | None = None
) -> Path:
    """Write the successful episodes among `episode_paths` (at most `max_episodes`) to a new
    LeRobot dataset at `cfg.dataset_root`. Refuses to touch an existing directory."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if cfg.dataset_root.exists():
        raise FileExistsError(f"{cfg.dataset_root} exists; choose a new dataset_root")
    keep = [p for p in episode_paths if _succeeded(p)][:max_episodes]
    if not keep:
        raise ValueError(f"no successful episodes among {len(episode_paths)} files")

    dataset = LeRobotDataset.create(
        cfg.dataset_repo_id,
        FPS,
        FEATURES,
        root=cfg.dataset_root,
        robot_type=ROBOT_TYPE,
        use_videos=False,
    )
    for path in keep:
        for frame in episode_frames(SimEpisode.load(path)):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return cfg.dataset_root
```

Notes for the implementer:
- `[:None]` keeps the whole list, so `max_episodes=None` needs no special case.
- If lerobot 0.6.1's `add_frame` / `create` signatures differ from the above (check
  `.venv/lib/python3.12/site-packages/lerobot/datasets/lerobot_dataset.py`, `create` at ~line
  684, `add_frame` ~418, `save_episode` ~434, `finalize` ~474), adapt the calls, not the tests.
- If loading in the test tries to reach the Hub for a local repo id, set
  `monkeypatch.setenv("HF_HUB_OFFLINE", "1")` in the two loading tests rather than changing the
  export.

- [ ] **Step 4: Add `--max-episodes` to the CLI**

In `packages/e2l_train/src/e2l_train/cli.py`, change the `export` command to:

```python
@app.command()
def export(
    run_ids: list[str] = typer.Argument(None, help="Generated runs to include (default: all)."),
    max_episodes: int = typer.Option(None, help="Export at most this many successful episodes."),
    config: Path = config_option("train/smolvla"),
    set_: list[str] = set_option(),
) -> None:
    """Export successful data/generated/<run_id>/ episodes to a LeRobot dataset."""
    from e2l_common.paths import DataPaths
    from e2l_train.export import export_dataset

    cfg = load_stage_config(config, set_, TrainConfig)
    paths = DataPaths.default()
    runs = run_ids or sorted(p.name for p in (paths.root / "generated").glob("*") if p.is_dir())
    episodes = [p for r in runs for p in sorted(paths.generated(r).glob("*.npz"))]
    root = export_dataset(episodes, cfg, max_episodes)
    typer.echo(f"wrote {root}")
```

- [ ] **Step 5: Run the tests and lint**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_train -v && uv run ruff check . && uv run ruff format --check .`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add packages/e2l_train/src/e2l_train/export.py packages/e2l_train/src/e2l_train/cli.py \
  packages/e2l_train/tests/test_export.py
git commit -m "feat(train): export successful sim episodes to a LeRobot dataset" \
  -m "Matches HuggingFaceVLA/libero: same keys, dtypes, 10 fps label, 180-degree images." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Fine-tune from `smolvla_base` and prove it with a smoke run

This task is partly empirical: the exact `lerobot-train` arguments that map our dataset's
features onto `smolvla_base` must be found by running it. The deliverable is a config that
works, a test pinning it, and evidence (a saved checkpoint) in the report.

**Files:**
- Modify: `packages/e2l_common/src/e2l_common/config.py` (`TrainConfig.policy_path` default only)
- Modify: `configs/train/smolvla.yaml`
- Modify: `packages/e2l_train/tests/test_finetune.py`

**Interfaces:**
- Consumes: Task 1's `e2l train export ... --max-episodes N`; `finetune.build_command(cfg)` (unchanged).
- Produces: `configs/train/smolvla.yaml` that trains from `lerobot/smolvla_base` on an exported
  dataset with no further flags except `dataset_root` / `dataset_repo_id` / `output_dir` /
  `steps` / `batch_size` / device.

- [ ] **Step 1: Switch the default policy and pin it in a test**

In `config.py`, `TrainConfig`: `policy_path: str = "lerobot/smolvla_base"` (leave `EvalConfig`
at `HuggingFaceVLA/smolvla_libero`: it is the reference line). In `configs/train/smolvla.yaml`:

```yaml
policy_path: lerobot/smolvla_base            # not smolvla_libero: it already saw LIBERO demos
```

In `test_finetune.py`, change the assertion in `test_build_command_from_config` to
`"--policy.path=lerobot/smolvla_base" in cmd`, and add:

```python
def test_repo_config_trains_from_smolvla_base():
    from e2l_common.config import load_config

    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    cfg = load_config(config, model=TrainConfig)
    assert cfg.policy_path == "lerobot/smolvla_base"
```

Run: `unset PYTHONPATH; uv run pytest packages/e2l_train packages/e2l_common -q` → PASS.

- [ ] **Step 2: Export a 5-episode smoke dataset**

```bash
unset PYTHONPATH
uv run e2l train export scripted50 --max-episodes 5 \
  --set dataset_root=data/lerobot/ego2libero_smoke --set dataset_repo_id=local/ego2libero_smoke
```

Expected: `wrote data/lerobot/ego2libero_smoke`. If the directory exists from an earlier
attempt, delete only `data/lerobot/ego2libero_smoke` and rerun. Then load it:

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/ego2libero_smoke', root='data/lerobot/ego2libero_smoke')
print(ds.meta.total_episodes, len(ds), ds.fps, list(ds.meta.features))
it = ds[0]; print({k: tuple(v.shape) for k, v in it.items() if hasattr(v, 'shape')}, it['task'])"
```

Expected: `5`, about 680 frames, `10`, the feature keys from the Facts table, task
`put the bowl on the plate`.

- [ ] **Step 3: Find the working `lerobot-train` arguments (CPU, 2 steps)**

```bash
unset PYTHONPATH
HF_HOME=/mnt/data/hf-cache uv run e2l train finetune \
  --set dataset_root=data/lerobot/ego2libero_smoke --set dataset_repo_id=local/ego2libero_smoke \
  --set output_dir=outputs/train/smoke_cpu --set steps=2 --set batch_size=1 \
  --set num_workers=0 --set save_freq=2 --set 'extra_args=["--policy.device=cpu"]'
```

(If `--set` cannot take a list, check `e2l_common.cliutils.load_stage_config` for the accepted
syntax; do not change `cliutils`.) `lerobot-train` refuses an existing `output_dir`: delete
`outputs/train/smoke_cpu` between attempts.

The first attempt will most likely fail on feature mismatch (dataset `image`/`image2`/state 8 vs
base `camera1..3`/state 6). Fix it with `lerobot-train` arguments, in this order of preference,
reading `lerobot/configs/train.py` and `lerobot/policies/smolvla/` to confirm each flag exists:
1. `--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}`
   plus `--policy.empty_cameras=1` (base expects three cameras).
2. If state/action dims still clash (8/7 vs 6/6): SmolVLA pads to `max_state_dim` /
   `max_action_dim` = 32, so look for how lerobot reconciles `input_features` from the
   pretrained config with dataset features (search `input_features` in `lerobot/policies/factory.py`
   and `lerobot/scripts/lerobot_train.py`); prefer a flag over code changes.
3. Do not edit anything inside `.venv`. Do not fall back to `smolvla_libero`.

Put the arguments that work into `configs/train/smolvla.yaml` `extra_args` (one list item per
flag, with a comment saying why), so the command above minus `extra_args` overrides works with
only `--policy.device=cpu` added. Re-run the smoke command to confirm.

Expected success: exit 0, and `outputs/train/smoke_cpu/checkpoints/` contains a step-2 checkpoint
with `pretrained_model/` inside (`find outputs/train/smoke_cpu -maxdepth 4`).

If CPU training is impossible for a reason unrelated to config (e.g. an op unsupported on CPU, or
it exceeds ~20 minutes for 2 steps), stop at the first `lerobot-train` log line that shows the
dataset and policy loaded and the first step starting, record that in the report as the evidence,
and say so — do not hide it.

- [ ] **Step 4: Pin the extra args in a test**

Add to `test_finetune.py` a test that loads `configs/train/smolvla.yaml` and asserts every flag
you added in Step 3 appears in `build_command(cfg)`, e.g.:

```python
def test_repo_config_maps_libero_cameras_onto_base():
    from e2l_common.config import load_config

    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    cmd = build_command(load_config(config, model=TrainConfig))
    assert any(a.startswith("--rename_map=") and "camera1" in a for a in cmd)
```

(Adjust the asserted flags to the ones Step 3 actually needed.)

- [ ] **Step 5: Check image orientation against the reference dataset**

Download one data file of the reference dataset and compare a LIBERO-Goal frame with our
exported frame by eye:

```bash
unset PYTHONPATH
HF_HOME=/mnt/data/hf-cache uv run python - <<'EOF'
import io, numpy as np, pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = hf_hub_download("HuggingFaceVLA/libero", "data/chunk-000/file-000.parquet", repo_type="dataset")
df = pd.read_parquet(p)
img = df.iloc[0]["observation.images.image"]
ref = np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) if isinstance(img, dict) else np.asarray(img)
ours = LeRobotDataset("local/ego2libero_smoke", root="data/lerobot/ego2libero_smoke")[0]
ours = (ours["observation.images.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(np.concatenate([ref, ours], axis=1)).save("outputs/export_orientation_check.png")
print("saved outputs/export_orientation_check.png")
EOF
```

Open `outputs/export_orientation_check.png` (Read tool). Left: reference; right: ours. Both must
show the table at the bottom and the robot arm entering from the same side (scenes differ — only
orientation matters). If ours is upside down relative to the reference, the rotation in
`episode_frames` is wrong: fix it, fix the round-trip test, re-export, re-check. Record the
verdict in the report (and attach the path).

- [ ] **Step 6: Lint, full train tests, commit**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_train packages/e2l_common -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add packages/e2l_common/src/e2l_common/config.py configs/train/smolvla.yaml \
  packages/e2l_train/tests/test_finetune.py
git commit -m "feat(train): fine-tune from smolvla_base with LIBERO camera mapping" \
  -m "smolvla_libero already saw LIBERO teleop demos, so results would not be creditable to our data." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Write the Spark command into the report**

In the report file, give the exact command for the human to run the 200-step GPU smoke run on
the DGX Spark (after `uv sync --group train --group dev` and copying
`data/lerobot/ego2libero_smoke` there), e.g.:

```bash
unset PYTHONPATH; HF_HOME=<spark hf cache> uv run e2l train finetune \
  --set dataset_root=data/lerobot/ego2libero_smoke --set dataset_repo_id=local/ego2libero_smoke \
  --set output_dir=outputs/train/smoke_spark --set steps=200 --set save_freq=200 --set batch_size=32
```

plus the CPU seconds-per-step you measured in Step 3.
