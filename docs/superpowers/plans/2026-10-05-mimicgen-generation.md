# MimicGen-style Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `e2l generate run`: multiply each `RobotSegments` demo into many LIBERO
`SimEpisode`s by perturbing object placements around the LIBERO init states, replaying with the
existing `e2l_sim.replay.replay()`, keeping successes and writing a yield manifest.

**Architecture:** `e2l_generate` becomes thin orchestration. `sample.py` perturbs object poses
(pure numpy). `run.py` holds the per-attempt worker (`run_attempt`), the pure manifest builder
(`summarise`) and the driver (`generate`, inline or a spawn `ProcessPoolExecutor`). Two small
helpers are added to `e2l_sim` (init-state count, nominal object poses), because only `e2l_sim`
may touch LIBERO. The `transform`/`stitch` stubs are deleted: replay already transforms and
transits.

**Tech Stack:** Python 3.12, numpy, scipy (via `e2l_common.geometry`), pydantic configs, typer +
rich CLI, `concurrent.futures` with the `spawn` start method, pytest (`sim`/`slow` markers),
LIBERO via LeRobot's `LiberoEnv`.

**Spec:** `docs/superpowers/specs/2026-10-05-mimicgen-generation-design.md`

## Global Constraints

- Poses are 4x4 float64 `T_a_b`; no quaternions in files or configs. Units m, rad, s.
- LIBERO / robosuite / MuJoCo are imported only inside `e2l_sim`. `e2l_generate` calls
  `e2l_sim` functions; it never imports `libero`, `robosuite`, `mujoco` or `lerobot`.
- Run `unset PYTHONPATH` before any `uv run` outside `make` (ROS Humble pollutes it).
- Lint: `uv run ruff check . && uv run ruff format --check .` (line 100). Type hints expected.
- Commits: `<type>(<scope>): <subject>`, scope from `common sim generate configs repo ...`,
  header <= 72 chars, footer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi`.
- Branch: `feat/mimicgen-generation` (already exists, holds the spec and this plan).
- Config defaults (verbatim from spec): `episodes_per_demo: 50`, `xy_range: 0.08`,
  `yaw_range: 0.5`, `min_separation: 0.11`, `max_placement_tries: 20`, `workers: 4`,
  `keep_failures: false`, `seed: 0`; `transit_steps` removed.
- Episode files: `data/generated/<run_id>/<demo_id>_<k:04d>.{npz,json}`; manifest
  `data/generated/<run_id>/manifest.json`; no wall-clock timestamps in it.
- Attempt `k` uses LIBERO init state `k % n_init`; seed
  `SeedSequence([seed, zlib.crc32(demo_id.encode()), k])`.
- Success criterion: `uv run e2l generate run 20261003_scripted_000` with default config
  reports a yield >= 50 %. A lower yield is reported, not hidden by shrinking noise.

## Prerequisites (orchestrator, before `make worktree`)

`/` has ~8 GB free; episodes are tens of MB each. `scripts/worktree_setup.sh` runs
`mkdir -p "$main/data"`, which would create a real directory on `/` if `data/` is missing. So in
the **main checkout** first:

```bash
cd /home/thomas/Competitions/ego2libero
[[ -e data ]] || { mkdir -p /mnt/data/e2l-data && ln -s /mnt/data/e2l-data data; }
ls -ld data   # expect: data -> /mnt/data/e2l-data
```

Then create the worktree on the existing branch (`git worktree add .worktrees/mimicgen-generation
feat/mimicgen-generation` after `git switch main` in the main checkout) and run `make worktree`
inside it.

## Review Focus

1. **Re-running a run id** with stale episodes from an earlier, larger run: without
   `--overwrite` the run must refuse; with it, the old files must be gone (Task 4 test
   `test_overwrite_replaces_stale_files`).
2. **A demo id with no robot segments** (typo on the CLI): fail before any simulator work,
   naming the demo, rather than 50 per-attempt errors (Task 4 test
   `test_missing_demo_fails_before_sim_work`).
3. **A segment referencing an object missing from the scene** (bad `object_map`): each attempt
   is recorded as `error` naming the object, and the run still writes its manifest (Task 4 test
   `test_unknown_object_is_an_attempt_error`).
4. **The bowl spawned on or against the plate**, which would count as a free "success": every
   accepted placement keeps perturbed objects at least `min_separation` apart (Task 1 test
   `test_accepted_placements_respect_min_separation`).
5. **Nothing to do** (no retargeted demos, or `episodes_per_demo=0`): `generate` writes an
   empty manifest without touching LIBERO; the CLI exits 1 with a hint when no demos exist
   (Task 4 `test_no_jobs_writes_empty_manifest`, Task 5 `test_cli_without_demos_exits_with_hint`).

Known limitation (out of scope): a hard worker crash (segfault in MuJoCo) breaks the pool and
the run aborts without a manifest.

---

### Task 1: Placement sampler and config

**Files:**
- Modify: `packages/e2l_common/src/e2l_common/config.py` (`GenerateConfig`, ~line 121)
- Modify: `configs/generate.yaml`
- Modify: `packages/e2l_generate/src/e2l_generate/sample.py`
- Delete: `packages/e2l_generate/src/e2l_generate/transform.py`,
  `packages/e2l_generate/src/e2l_generate/stitch.py`
- Test: `packages/e2l_generate/tests/test_sample.py` (create),
  `packages/e2l_common/tests/test_config.py` (append)

**Interfaces:**
- Consumes: `e2l_common.geometry.make_T`, `rotvec_to_matrix`.
- Produces:
  `sample_object_poses(nominal: dict[str, np.ndarray], names: Iterable[str], cfg: GenerateConfig, rng: np.random.Generator) -> dict[str, np.ndarray] | None`
  — returns only `names` (perturbed), `None` when no placement fits; raises `KeyError` naming
  any name missing from `nominal`.
  `GenerateConfig` fields: `episodes_per_demo, xy_range, yaw_range, min_separation,
  max_placement_tries, workers, keep_failures, seed`.

- [ ] **Step 1: Write the failing tests**

Create `packages/e2l_generate/tests/test_sample.py`:

```python
import numpy as np
import pytest

from e2l_common.config import GenerateConfig
from e2l_common.geometry import make_T, matrix_to_rotvec, rotvec_to_matrix
from e2l_generate.sample import sample_object_poses

NOMINAL = {
    "bowl": make_T(rotvec_to_matrix(np.array([0.0, 0.0, 0.3])), [-0.1, 0.0, 0.9]),
    "plate": make_T(np.eye(3), [0.1, 0.0, 0.9]),
    "cheese": make_T(np.eye(3), [0.0, 0.2, 0.9]),
}


def test_offsets_and_yaw_stay_within_ranges():
    cfg = GenerateConfig(xy_range=0.05, yaw_range=0.4, min_separation=0.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        poses = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, rng)
        assert set(poses) == {"bowl", "plate"}  # unreferenced objects are not returned
        for name, T in poses.items():
            d = T[:3, 3] - NOMINAL[name][:3, 3]
            assert np.all(np.abs(d[:2]) <= 0.05) and d[2] == 0.0
            rv = matrix_to_rotvec(T[:3, :3] @ NOMINAL[name][:3, :3].T)
            assert abs(rv[2]) <= 0.4 + 1e-12
            np.testing.assert_allclose(rv[:2], 0.0, atol=1e-12)  # yaw only: stays upright
            np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_zero_noise_returns_nominal():
    cfg = GenerateConfig(xy_range=0.0, yaw_range=0.0, min_separation=0.0)
    poses = sample_object_poses(NOMINAL, ["bowl"], cfg, np.random.default_rng(1))
    np.testing.assert_allclose(poses["bowl"], NOMINAL["bowl"])


def test_seeded_and_independent_of_name_order():
    cfg = GenerateConfig()
    a = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, np.random.default_rng(7))
    b = sample_object_poses(NOMINAL, ["plate", "bowl"], cfg, np.random.default_rng(7))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_accepted_placements_respect_min_separation():
    # Nominal xy distance 0.2 m, noise +/-0.08 m each: many draws fall below 0.19 m.
    cfg = GenerateConfig(xy_range=0.08, yaw_range=0.0, min_separation=0.19)
    rng = np.random.default_rng(3)
    accepted = 0
    for _ in range(100):
        poses = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, rng)
        if poses is not None:
            accepted += 1
            assert np.linalg.norm(poses["bowl"][:2, 3] - poses["plate"][:2, 3]) >= 0.19
    assert accepted > 0


def test_returns_none_when_no_placement_fits():
    cfg = GenerateConfig(xy_range=0.01, min_separation=1.0, max_placement_tries=5)
    assert sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, np.random.default_rng(0)) is None


def test_unknown_object_is_named():
    with pytest.raises(KeyError, match="bowl_2"):
        sample_object_poses(NOMINAL, ["bowl_2"], GenerateConfig(), np.random.default_rng(0))
```

Append to `packages/e2l_common/tests/test_config.py`:

```python
def test_generate_config_separation_defaults_and_no_transit_steps():
    from e2l_common.config import GenerateConfig

    cfg = GenerateConfig()
    assert cfg.min_separation == 0.11 and cfg.max_placement_tries == 20
    with pytest.raises(ValidationError):
        GenerateConfig(transit_steps=30)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate/tests/test_sample.py packages/e2l_common/tests/test_config.py -q`
Expected: FAIL — `NotImplementedError: e2l_generate.sample.sample_object_poses` and
`AttributeError`/assertion on `min_separation`.

- [ ] **Step 3: Update the config**

In `packages/e2l_common/src/e2l_common/config.py` replace the `GenerateConfig` body with:

```python
class GenerateConfig(_Config):
    episodes_per_demo: int = 50
    xy_range: float = 0.08  # metres, uniform +/- around the init-state object pose
    yaw_range: float = 0.5  # radians, about the sim z axis through the object origin
    min_separation: float = 0.11  # metres, xy distance kept between perturbed objects
    max_placement_tries: int = 20
    workers: int = 4
    keep_failures: bool = False
    seed: int = 0
```

Replace `configs/generate.yaml` with:

```yaml
# MimicGen-style multiplication: RobotSegments -> SimEpisodes.
# Attempt k starts from LIBERO init state k % 50, then perturbs every object the demo's
# segments reference. Transits between segments come from sim.yaml `transit_speed`.

episodes_per_demo: 50                    # attempts per source demo; only successes are kept
xy_range: 0.08                           # object placement noise, +/- m around the init state
yaw_range: 0.5                           # +/- rad about the sim z axis
min_separation: 0.11                     # m, xy; init states put bowl and plate 0.13-0.16 apart
max_placement_tries: 20                  # resamples before an attempt is `placement_failed`
workers: 4                               # parallel replay processes (1 = inline, for debugging)
keep_failures: false                     # store failed rollouts too (debugging only)
seed: 0
```

- [ ] **Step 4: Implement the sampler and delete the dead stubs**

Replace `packages/e2l_generate/src/e2l_generate/sample.py` with:

```python
"""Sample new object placements around nominal (LIBERO init-state) poses."""

from collections.abc import Iterable
from itertools import combinations

import numpy as np

from e2l_common.config import GenerateConfig
from e2l_common.geometry import rotvec_to_matrix


def perturb(T_sim_obj: np.ndarray, d_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Shift an object pose (4,4) by `d_xy` in the sim xy plane and turn it by `yaw` about the
    sim z axis through its own origin; height and tilt are unchanged."""
    T = np.array(T_sim_obj, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, yaw])) @ T[:3, :3]
    T[:2, 3] += d_xy
    return T


def separated(poses: dict[str, np.ndarray], min_distance: float) -> bool:
    """True if every pair of poses is at least `min_distance` apart in xy."""
    return all(
        np.linalg.norm(a[:2, 3] - b[:2, 3]) >= min_distance
        for a, b in combinations(poses.values(), 2)
    )


def sample_object_poses(
    nominal: dict[str, np.ndarray],
    names: Iterable[str],
    cfg: GenerateConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """Perturb the named nominal poses T_sim_obj (4,4) by a uniform xy offset within
    +/- cfg.xy_range and a yaw within +/- cfg.yaw_range. Draws that put two of them closer than
    cfg.min_separation are resampled, up to cfg.max_placement_tries times; then None. Names are
    processed in sorted order so a seeded rng gives the same placement for any input order."""
    names = sorted(set(names))
    missing = [n for n in names if n not in nominal]
    if missing:
        raise KeyError(f"objects {missing} are not in the scene; have {sorted(nominal)}")
    for _ in range(cfg.max_placement_tries):
        poses = {
            n: perturb(
                nominal[n],
                rng.uniform(-cfg.xy_range, cfg.xy_range, size=2),
                rng.uniform(-cfg.yaw_range, cfg.yaw_range),
            )
            for n in names
        }
        if separated(poses, cfg.min_separation):
            return poses
    return None
```

Delete the replaced stubs:

```bash
git rm packages/e2l_generate/src/e2l_generate/transform.py packages/e2l_generate/src/e2l_generate/stitch.py
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate packages/e2l_common -q`
Expected: PASS (including `test_skeleton.py`: `run.generate` is still a stub, so
`check_stubs("e2l_generate") > 0` holds, and `test_repo_configs.py` loads the new yaml).

- [ ] **Step 6: Lint and commit**

```bash
unset PYTHONPATH; uv run ruff check . && uv run ruff format --check .
git add -A packages/e2l_common packages/e2l_generate configs/generate.yaml
git commit -m "feat(generate): sample object placements around init-state poses

Replay already anchors and transits segments, so the transform and stitch
stubs and the transit_steps option go.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi"
```

---

### Task 2: Nominal object poses and init-state count in `e2l_sim`

**Files:**
- Modify: `packages/e2l_sim/src/e2l_sim/libero_setup.py` (add `libero_available`)
- Modify: `packages/e2l_sim/src/e2l_sim/env.py` (add `init_state_count`)
- Modify: `packages/e2l_sim/src/e2l_sim/scene.py` (add `nominal_object_poses`)
- Modify: `packages/e2l_sim/tests/test_replay.py` (use `libero_available`)
- Test: `packages/e2l_sim/tests/test_scene.py` (create)

**Interfaces:**
- Consumes: `e2l_sim.env.make_env(cfg, episode_index)`, `e2l_sim.scene.object_poses(env)`.
- Produces:
  `e2l_sim.libero_setup.libero_available() -> bool`;
  `e2l_sim.env.init_state_count(cfg: SimConfig) -> int`;
  `e2l_sim.scene.nominal_object_poses(cfg: SimConfig, episode_index: int) -> dict[str, np.ndarray]`
  (LIBERO object name -> T_sim_obj, after the same `reset(seed=cfg.seed)` replay does).

- [ ] **Step 1: Move the availability check into `libero_setup`**

Append to `packages/e2l_sim/src/e2l_sim/libero_setup.py`:

```python
def libero_available() -> bool:
    """True if LIBERO imports and its scene assets are installed (gates `sim` tests)."""
    try:
        ensure_libero_config()
        from libero.libero import get_libero_path

        return Path(get_libero_path("assets"), "scenes").is_dir()
    except Exception:
        return False
```

In `packages/e2l_sim/tests/test_replay.py` delete the `_libero_available` function, add
`from e2l_sim.libero_setup import libero_available` to the imports, change the decorator to
`@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")`, and drop
`from pathlib import Path` there if ruff reports it unused. (`libero_setup.py` already imports
`Path`.)

- [ ] **Step 2: Write the failing test**

Create `packages/e2l_sim/tests/test_scene.py`:

```python
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `unset PYTHONPATH; MUJOCO_GL=egl uv run pytest packages/e2l_sim/tests/test_scene.py -q`
Expected: FAIL with `ImportError: cannot import name 'init_state_count'`.

- [ ] **Step 4: Implement**

Append to `packages/e2l_sim/src/e2l_sim/env.py`:

```python
def init_state_count(cfg: SimConfig) -> int:
    """Number of LIBERO init states for `cfg.task_id` (50 for LIBERO-Goal)."""
    ensure_libero_config()
    from lerobot.envs.libero import _get_suite

    return len(_get_suite(cfg.suite).get_task_init_states(cfg.task_id))
```

Append to `packages/e2l_sim/src/e2l_sim/scene.py` (add
`from e2l_common.config import SimConfig` to its imports):

```python
def nominal_object_poses(cfg: SimConfig, episode_index: int) -> dict[str, np.ndarray]:
    """T_sim_obj (4,4) of every movable object in LIBERO init state `episode_index`, read after
    the same seeded reset that `replay` performs."""
    from e2l_sim.env import make_env

    env = make_env(cfg, episode_index=episode_index)
    try:
        env.reset(seed=cfg.seed)
        return object_poses(env)
    finally:
        env.close()
```

- [ ] **Step 5: Run the sim tests to verify they pass**

Run: `unset PYTHONPATH; MUJOCO_GL=egl uv run pytest packages/e2l_sim -q`
Expected: PASS, with `test_scene.py` and `test_scripted_bowl_onto_plate_succeeds` actually
running (not skipped) on this laptop.

- [ ] **Step 6: Lint and commit**

```bash
unset PYTHONPATH; uv run ruff check . && uv run ruff format --check .
git add packages/e2l_sim
git commit -m "feat(sim): read nominal object poses and init-state count

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi"
```

---

### Task 3: Attempt records, seeding and the manifest summary

**Files:**
- Modify: `packages/e2l_generate/src/e2l_generate/run.py` (keep the `generate` stub for now)
- Test: `packages/e2l_generate/tests/test_run.py` (create)

**Interfaces:**
- Consumes: `GenerateConfig`, `SimConfig`, `e2l_common.paths.repo_root`.
- Produces (all in `e2l_generate.run`):
  - `Status = Literal["success", "failure", "placement_failed", "error"]`
  - `@dataclass(frozen=True) AttemptResult(stem: str, demo_id: str, attempt: int, episode_index: int, status: Status, steps: int = 0, saved: bool = False, error: str | None = None, size_mb: float | None = None)`
  - `episode_stem(demo_id: str, k: int) -> str` → `f"{demo_id}_{k:04d}"`
  - `attempt_seed(seed: int, demo_id: str, k: int) -> np.random.Generator`
  - `git_commit() -> str | None`
  - `summarise(results: list[AttemptResult], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `packages/e2l_generate/tests/test_run.py`:

```python
import json

from e2l_common.config import GenerateConfig, SimConfig
from e2l_generate.run import AttemptResult, attempt_seed, episode_stem, summarise


def result(demo, k, status, saved=False, size=None):
    return AttemptResult(
        stem=episode_stem(demo, k),
        demo_id=demo,
        attempt=k,
        episode_index=k % 50,
        status=status,
        steps=10,
        saved=saved,
        size_mb=size,
    )


def test_episode_stem():
    assert episode_stem("20261003_scripted_000", 7) == "20261003_scripted_000_0007"


def test_attempt_seed_is_per_attempt():
    def draw(*a):
        return attempt_seed(*a).random(4).tolist()

    assert draw(0, "d1", 3) == draw(0, "d1", 3)
    assert draw(0, "d1", 3) != draw(0, "d1", 4)
    assert draw(0, "d1", 3) != draw(0, "d2", 3)
    assert draw(0, "d1", 3) != draw(1, "d1", 3)


def test_summarise_counts_yield_and_sizes():
    results = [
        result("d1", 1, "failure"),
        result("d1", 0, "success", saved=True, size=20.0),
        result("d1", 2, "error"),
        result("d1", 3, "placement_failed"),
        result("d2", 0, "success", saved=True, size=30.0),
        result("d2", 1, "failure", saved=True, size=10.0),
    ]
    m = summarise(results, "run0", GenerateConfig(), SimConfig())
    assert m["run_id"] == "run0"
    assert m["demos"]["d1"] == {
        "attempts": 4,
        "successes": 1,
        "failures": 1,
        "placement_failed": 1,
        "errors": 1,
        "yield": 0.25,
    }
    assert m["demos"]["d2"]["yield"] == 0.5
    assert m["mean_episode_mb"] == 20.0  # mean over saved episodes, failures included
    assert [e["stem"] for e in m["episodes"][:2]] == ["d1_0000", "d1_0001"]  # sorted
    assert "size_mb" not in m["episodes"][0]
    assert m["generate_config"]["episodes_per_demo"] == 50
    assert m["sim_config"]["control_freq"] == 20
    json.dumps(m)  # serialisable as-is


def test_summarise_empty():
    m = summarise([], "run0", GenerateConfig(), SimConfig())
    assert m["demos"] == {} and m["episodes"] == [] and m["mean_episode_mb"] is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate/tests/test_run.py -q`
Expected: FAIL with `ImportError: cannot import name 'AttemptResult'`.

- [ ] **Step 3: Implement**

Replace `packages/e2l_generate/src/e2l_generate/run.py` with (the `generate` stub stays until
Task 4):

```python
"""Sample, replay and keep successes; record yield per source demo in a manifest."""

import subprocess
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.paths import repo_root
from e2l_common.stub import not_implemented

Status = Literal["success", "failure", "placement_failed", "error"]
_COUNT_KEY = {
    "success": "successes",
    "failure": "failures",
    "placement_failed": "placement_failed",
    "error": "errors",
}


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one generation attempt; small enough to return from a worker process."""

    stem: str
    demo_id: str
    attempt: int
    episode_index: int
    status: Status
    steps: int = 0
    saved: bool = False
    error: str | None = None
    size_mb: float | None = None


def episode_stem(demo_id: str, k: int) -> str:
    return f"{demo_id}_{k:04d}"


def attempt_seed(seed: int, demo_id: str, k: int) -> np.random.Generator:
    """Generator for attempt `k` of `demo_id`: independent of worker count and scheduling."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(demo_id.encode()), k]))


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def summarise(
    results: list[AttemptResult], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig
) -> dict:
    """Manifest dict: configs, commit, per-demo counts and yield, one entry per attempt."""
    results = sorted(results, key=lambda r: (r.demo_id, r.attempt))
    demos: dict[str, dict] = {}
    for r in results:
        counts = demos.setdefault(
            r.demo_id, {"attempts": 0} | dict.fromkeys(_COUNT_KEY.values(), 0)
        )
        counts["attempts"] += 1
        counts[_COUNT_KEY[r.status]] += 1
    for counts in demos.values():
        counts["yield"] = counts["successes"] / counts["attempts"]
    sizes = [r.size_mb for r in results if r.saved and r.size_mb is not None]
    return {
        "run_id": run_id,
        "git_commit": git_commit(),
        "generate_config": cfg.model_dump(mode="json"),
        "sim_config": sim_cfg.model_dump(mode="json"),
        "demos": demos,
        "mean_episode_mb": round(float(np.mean(sizes)), 2) if sizes else None,
        "episodes": [{k: v for k, v in asdict(r).items() if k != "size_mb"} for r in results],
    }


def generate(
    demo_ids: list[str], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig, data_root: Path
) -> dict:
    """For each source demo, generate `cfg.episodes_per_demo` attempts in parallel workers and
    save successful SimEpisodes to data/generated/<run_id>/. Writes manifest.json with yield
    per source demo and provenance; returns the manifest dict."""
    not_implemented("e2l_generate.run.generate")
```

Note: the `demos.setdefault` line may exceed 100 chars; let `ruff format` wrap it.

- [ ] **Step 4: Run them to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
unset PYTHONPATH; uv run ruff format packages/e2l_generate && uv run ruff check .
git add packages/e2l_generate
git commit -m "feat(generate): record attempts and summarise yield per demo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi"
```

---

### Task 4: Attempt worker and the `generate` driver

**Files:**
- Modify: `packages/e2l_generate/src/e2l_generate/run.py`
- Test: `packages/e2l_generate/tests/test_generate.py` (create, fast, sim faked),
  `packages/e2l_generate/tests/test_generate_sim.py` (create, `sim` + `slow`)

**Interfaces:**
- Consumes: Task 1 `sample_object_poses`; Task 2 `init_state_count`,
  `nominal_object_poses`, `libero_available`; Task 3 `AttemptResult`, `episode_stem`,
  `attempt_seed`, `summarise`; existing `e2l_sim.replay.replay(segments, cfg, T_sim_obj=...,
  episode_index=...) -> SimEpisode`, `e2l_sim.replay.sim_object(name, cfg) -> str`,
  `DataPaths(root).robot_segments/generated/manifest`.
- Produces:
  `run_attempt(demo_id: str, k: int, n_init: int, run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig, data_root: Path) -> AttemptResult`;
  `generate(demo_ids: list[str], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig, data_root: Path, overwrite: bool = False) -> dict`
  (raises `FileExistsError` for a non-empty run dir without `overwrite`, `FileNotFoundError`
  for demos without robot segments).

- [ ] **Step 1: Write the failing fast tests**

Create `packages/e2l_generate/tests/test_generate.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate/tests/test_generate.py -q`
Expected: FAIL — `AttributeError: module 'e2l_generate.run' has no attribute 'init_state_count'`.

- [ ] **Step 3: Implement the worker and driver**

In `packages/e2l_generate/src/e2l_generate/run.py`:

Replace the import block with:

```python
import json
import shutil
import subprocess
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.log import get_logger
from e2l_common.paths import DataPaths, repo_root
from e2l_common.schemas import RobotSegments
from e2l_generate.sample import sample_object_poses
from e2l_sim.env import init_state_count
from e2l_sim.replay import replay, sim_object
from e2l_sim.scene import nominal_object_poses

log = get_logger(__name__)
# Per-process caches: each worker loads a demo and an init state's nominal poses only once.
_segments_cache: dict[Path, RobotSegments] = {}
_nominal_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
```

(`e2l_sim.env`, `.replay` and `.scene` import LIBERO lazily inside functions, so importing them
here is allowed.)

Replace the `generate` stub with:

```python
def _segments(path: Path) -> RobotSegments:
    if path not in _segments_cache:
        _segments_cache[path] = RobotSegments.load(path)
    return _segments_cache[path]


def _nominal(sim_cfg: SimConfig, episode_index: int) -> dict[str, np.ndarray]:
    key = (sim_cfg.model_dump_json(), episode_index)
    if key not in _nominal_cache:
        _nominal_cache[key] = nominal_object_poses(sim_cfg, episode_index)
    return _nominal_cache[key]


def run_attempt(
    demo_id: str,
    k: int,
    n_init: int,
    run_id: str,
    cfg: GenerateConfig,
    sim_cfg: SimConfig,
    data_root: Path,
) -> AttemptResult:
    """One attempt: sample a placement around init state k % n_init, replay, save the episode
    if it succeeded (or if cfg.keep_failures). Exceptions become an `error` result."""
    paths = DataPaths(data_root)
    stem = episode_stem(demo_id, k)
    index = k % n_init
    base = {"stem": stem, "demo_id": demo_id, "attempt": k, "episode_index": index}
    try:
        segments = _segments(paths.robot_segments(demo_id))
        names = {sim_object(seg.ref_object, sim_cfg) for seg in segments.segments}
        rng = attempt_seed(cfg.seed, demo_id, k)
        poses = sample_object_poses(_nominal(sim_cfg, index), names, cfg, rng)
        if poses is None:
            return AttemptResult(**base, status="placement_failed")
        episode = replay(segments, sim_cfg, T_sim_obj=poses, episode_index=index)
        status = "success" if episode.success else "failure"
        steps = len(episode.actions)
        if not (episode.success or cfg.keep_failures):
            return AttemptResult(**base, status=status, steps=steps)
        npz, sidecar = episode.save(paths.generated(run_id, stem))
        size_mb = (npz.stat().st_size + sidecar.stat().st_size) / 1e6
        return AttemptResult(**base, status=status, steps=steps, saved=True, size_mb=size_mb)
    except Exception as e:  # noqa: BLE001 - one bad attempt must not stop the run
        return AttemptResult(**base, status="error", error=f"{type(e).__name__}: {e}")


def _logged(r: AttemptResult) -> AttemptResult:
    detail = f": {r.error}" if r.error else f" ({r.steps} steps)"
    log.info(f"{r.stem} init {r.episode_index}: {r.status}{detail}")
    return r


def generate(
    demo_ids: list[str],
    run_id: str,
    cfg: GenerateConfig,
    sim_cfg: SimConfig,
    data_root: Path,
    overwrite: bool = False,
) -> dict:
    """For each source demo, run `cfg.episodes_per_demo` attempts (inline if cfg.workers <= 1,
    else in spawned worker processes) and save episodes to data/generated/<run_id>/. Writes
    manifest.json with yield per source demo and provenance; returns the manifest dict."""
    paths = DataPaths(Path(data_root))
    run_dir = paths.generated(run_id)
    missing = [d for d in demo_ids if not paths.robot_segments(d).with_suffix(".npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"no robot segments for {missing} in {paths.root / 'robot_segments'}"
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} is not empty; pass overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(d, k) for d in demo_ids for k in range(cfg.episodes_per_demo)]
    results: list[AttemptResult] = []
    if jobs:
        args = (init_state_count(sim_cfg), run_id, cfg, sim_cfg, paths.root)
        if cfg.workers <= 1:
            results = [_logged(run_attempt(d, k, *args)) for d, k in jobs]
        else:
            # spawn, not fork: EGL rendering contexts do not survive a fork.
            with ProcessPoolExecutor(cfg.workers, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(run_attempt, d, k, *args) for d, k in jobs]
                results = [_logged(f.result()) for f in as_completed(futures)]

    manifest = summarise(results, run_id, cfg, sim_cfg)
    paths.manifest(run_id).write_text(json.dumps(manifest, indent=2))
    return manifest
```

Remove the now-unused `from e2l_common.stub import not_implemented` import.

- [ ] **Step 4: Run the fast tests to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate -q`
Expected: `test_generate.py`, `test_run.py`, `test_sample.py` PASS.
`test_skeleton.py::test_stubs_raise_named_not_implemented` now FAILS (`0 > 0`): no stubs
remain. Fix it in this task — replace that test in
`packages/e2l_generate/tests/test_skeleton.py` with:

```python
def test_no_stubs_remain():
    assert check_stubs("e2l_generate") == 0
```

Re-run: `unset PYTHONPATH; uv run pytest packages/e2l_generate -q` → all PASS.

- [ ] **Step 5: Write the sim integration test**

Create `packages/e2l_generate/tests/test_generate_sim.py`:

```python
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
```

- [ ] **Step 6: Run the sim test**

Run: `unset PYTHONPATH; MUJOCO_GL=egl uv run pytest packages/e2l_generate/tests/test_generate_sim.py -v`
Expected: 2 PASSED (not skipped), a few minutes. If the `workers=2` case hangs or errors in
EGL, use superpowers:systematic-debugging; do not switch to `fork`.

- [ ] **Step 7: Lint and commit**

```bash
unset PYTHONPATH; uv run ruff format packages/e2l_generate && uv run ruff check . && uv run ruff format --check .
git add packages/e2l_generate
git commit -m "feat(generate): replay sampled placements in parallel workers

Each attempt starts from init state k % n_init, is seeded per attempt and
saves its episode in the worker; failures and errors land in the manifest.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi"
```

---

### Task 5: CLI, data root and the real run

**Files:**
- Modify: `packages/e2l_generate/src/e2l_generate/cli.py`
- Modify: `Makefile` (`setup` target)
- Modify: `CLAUDE.md` (Environment gotchas)
- Test: `packages/e2l_generate/tests/test_cli.py` (create)

**Interfaces:**
- Consumes: Task 4 `generate(..., overwrite=...) -> dict` (manifest shape from Task 3).
- Produces: `e2l generate run [DEMO_IDS] --run-id --overwrite --config --sim-config --set`;
  `yield_table(manifest: dict) -> rich.table.Table`.

- [ ] **Step 1: Write the failing tests**

Create `packages/e2l_generate/tests/test_cli.py`:

```python
from rich.console import Console
from typer.testing import CliRunner

from e2l_common.paths import repo_root
from e2l_generate.cli import app, yield_table


def test_cli_without_demos_exits_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("E2L_DATA", str(tmp_path))
    configs = repo_root() / "configs"
    result = CliRunner().invoke(
        app,
        ["run", "-c", str(configs / "generate.yaml"), "--sim-config", str(configs / "sim.yaml")],
    )
    assert result.exit_code == 1
    assert "robot_segments" in result.output


def test_yield_table_lists_each_demo():
    manifest = {
        "run_id": "run0",
        "demos": {
            "d1": {
                "attempts": 4,
                "successes": 3,
                "failures": 1,
                "placement_failed": 0,
                "errors": 0,
                "yield": 0.75,
            }
        },
    }
    console = Console(width=120, record=True)
    console.print(yield_table(manifest))
    text = console.export_text()
    assert "d1" in text and "75%" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate/tests/test_cli.py -q`
Expected: FAIL with `ImportError: cannot import name 'yield_table'`.

- [ ] **Step 3: Implement the CLI**

Replace `packages/e2l_generate/src/e2l_generate/cli.py` with:

```python
"""`e2l generate ...`: RobotSegments -> many SimEpisodes."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Multiply demos into simulated episodes.", no_args_is_help=True)

_COLUMNS = ("attempts", "successes", "failures", "placement_failed", "errors")


def yield_table(manifest: dict) -> Table:
    table = Table("demo", *_COLUMNS, "yield", title=f"generated/{manifest['run_id']}")
    for demo_id, counts in manifest["demos"].items():
        table.add_row(demo_id, *(str(counts[c]) for c in _COLUMNS), f"{counts['yield']:.0%}")
    return table


@app.command()
def run(
    demo_ids: list[str] = typer.Argument(None, help="Source demos (default: all retargeted)."),
    run_id: str = typer.Option("run0", help="Output folder under data/generated/."),
    overwrite: bool = typer.Option(False, help="Delete and replace an existing run folder."),
    config: Path = config_option("generate"),
    sim_config: Path = typer.Option(Path("configs/sim.yaml"), help="Simulator config."),
    set_: list[str] = set_option(),
) -> None:
    """Generate episodes from data/robot_segments into data/generated/<run_id>."""
    from e2l_common.config import load_config
    from e2l_generate.run import generate

    cfg = load_stage_config(config, set_, GenerateConfig)
    paths = DataPaths.default()
    demo_ids = demo_ids or paths.demo_ids("robot_segments")
    if not demo_ids:
        typer.echo(
            f"no demos in {paths.root / 'robot_segments'}; run `e2l retarget run` "
            "or `python scripts/scripted_segments.py` first"
        )
        raise typer.Exit(1)
    manifest = generate(
        demo_ids,
        run_id,
        cfg,
        load_config(sim_config, model=SimConfig),
        paths.root,
        overwrite=overwrite,
    )
    Console().print(yield_table(manifest))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_generate -q`
Expected: PASS (the sim test is slow; add `-m "not slow"` to skip it here).

- [ ] **Step 5: Data root in `make setup` and CLAUDE.md**

In `Makefile`, make the `setup` recipe:

```make
setup:            ## install every group (laptop); Spark: uv sync --group train --group dev
	uv sync --all-groups
	uv run pre-commit install
	[[ -e data ]] || { mkdir -p /mnt/data/e2l-data && ln -s /mnt/data/e2l-data data; }
```

In `CLAUDE.md` under "Environment gotchas (this laptop)", after the `/` is nearly full bullet,
add:

```markdown
- `data/` is a symlink to `/mnt/data/e2l-data` (`make setup` creates it): generated episodes
  are tens of MB each. Create it in the main checkout before `make worktree`.
```

- [ ] **Step 6: Lint, full fast suite, commit**

```bash
unset PYTHONPATH; uv run ruff check . && uv run ruff format --check . && uv run pytest -q -m "not slow"
git add packages/e2l_generate Makefile CLAUDE.md
git commit -m "feat(generate): add run options, yield table and data root setup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JSfb9MBAn9FfWqqimJBXwi"
```

- [ ] **Step 7: The real run (success criterion)**

From the worktree (its `data/` links to `/mnt/data/e2l-data`):

```bash
unset PYTHONPATH
uv run python scripts/scripted_segments.py
uv run e2l generate run 20261003_scripted_000 --run-id scripted50
du -sh data/generated/scripted50
python3 -c "import json; m=json.load(open('data/generated/scripted50/manifest.json')); print(m['demos'], m['mean_episode_mb'])"
```

Expected: the yield table prints; yield >= 50 %; no `error` attempts. Record yield,
`mean_episode_mb` and wall time in the PR description. If yield < 50 % or errors appear, stop
and use superpowers:systematic-debugging (look at which init states / placements fail, and
render one failure with `--set keep_failures=true` + `e2l sim replay`-style video); do not
lower `xy_range`/`yaw_range` to pass.
