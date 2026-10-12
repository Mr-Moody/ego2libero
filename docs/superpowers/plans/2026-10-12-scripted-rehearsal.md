# Scripted rehearsal and reference evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run everything after retargeting on 25 jittered scripted demos (replay → generate →
export Run A / Run B rehearsal datasets) and evaluate `smolvla_libero` on both eval settings,
before any real demo exists.

**Architecture:** The scripted bowl-onto-plate demo moves from `scripts/scripted_segments.py`
into a tested `e2l_sim.scripted` module that adds seeded, bounded jitter; the script becomes a
thin `--count/--seed` CLI. `e2l sim replay` gains `--run-id` so rehearsal replays get their own
folder. `scripts/rehearsal.sh` (`make rehearsal`) chains write → replay → generate → export. The
reference eval uses the existing `e2l eval rollout` / `report` unchanged.

**Tech Stack:** Python 3.12, numpy, typer (+ `typer.testing.CliRunner`), pytest, bash, LeRobot
0.6.1 (`LeRobotDataset`), LIBERO via `e2l_sim`.

**Spec:** `docs/superpowers/specs/2026-10-12-scripted-rehearsal-design.md`

## Global Constraints

- Work in `.worktrees/scripted-rehearsal` on branch `feat/scripted-rehearsal`. Run
  `make worktree` there once before the first test run. `unset PYTHONPATH` before any `uv run`.
- Commits: `<type>(<scope>)?: <subject>`; scopes from `common sim generate train eval scripts
  configs repo`. **No `Co-Authored-By:`, `Claude-Session:` or "Generated with Claude Code"
  lines, ever** (commits and PR body).
- Poses are 4x4 float64 `T_a_b`; no quaternions; metres, radians, seconds.
- `e2l_sim` modules must import without LIBERO (the stub checker imports them all);
  `e2l_sim.scripted` imports only numpy and `e2l_common`.
- Demo ids match `^\d{8}_[a-z0-9]+_\d{3}$`: scripted ids are `20261003_scripted_000`..`999`.
- Demo `000` is the nominal demo, identical (`np.array_equal`) to today's `scripts/scripted_segments.py`
  output. Demos `i ≥ 1` use `np.random.default_rng([seed, i])`.
- Jitter bounds (uniform): grasp angle ±0.4 rad on rim radius 0.05 m; approach/retreat height
  0.10 ± 0.02 m; place height 0.03 ± 0.01 m; place xy ± 0.01 m; segment duration × [0.7, 1.3].
- Rehearsal names: `data/generated/rehearsal_replay`, `data/generated/rehearsal_gen`,
  `data/lerobot/rehearsal_a` (replay only), `data/lerobot/rehearsal_b` (replay + gen).
  `scripted50` is never touched or exported here.
- Nothing in this plan writes to `docs/figures/`, README or LICENSE (Track 3).
- ruff, line length 100; type hints; comment density like the surrounding files.

## Review Focus

1. Rerunning `make rehearsal` over an existing rehearsal → rebuilds cleanly (export refuses
   existing dataset dirs, so the script must clear only the `rehearsal_*` dirs it owns). Pinned
   in Task 4 Step 4 (tiny run, twice).
2. `--count 0` or `--count 1001` → a clear `ValueError`, never a malformed 4-digit demo id.
   Pinned by `test_count_outside_range_is_rejected` (Task 1).
3. `e2l sim replay --run-id ../x` → rejected before anything is written. Pinned by
   `test_replay_rejects_run_id_with_separator` (Task 2).
4. Jittered demos keep the gripper flips exactly at segment ends (grasp at the last approach
   sample, release at the last carry sample). Pinned by `test_structure_is_preserved` (Task 1).
5. Plain `e2l sim replay <id>` (no `--run-id`) still writes to `data/generated/replay/`, so real
   demo replays and docs keep working. Pinned by `test_replay_defaults_to_replay_folder` (Task 2).

---

### Task 1: `e2l_sim.scripted`: nominal and jittered scripted demos

**Files:**
- Create: `packages/e2l_sim/src/e2l_sim/scripted.py`
- Create: `packages/e2l_sim/tests/test_scripted.py`
- Modify: `scripts/scripted_segments.py` (whole file becomes a thin CLI)
- Modify: `packages/e2l_sim/tests/test_replay.py:78-90` (import from `e2l_sim.scripted`)

**Interfaces:**
- Produces:
  - `FPS: float = 20.0`, `DEMO_PREFIX = "20261003_scripted"`
  - `scripted_demo_id(index: int) -> str` (raises `ValueError` outside 0..999)
  - `scripted_segments(demo_id: str, rng: np.random.Generator | None = None) -> RobotSegments`
  - `write_scripted(paths: DataPaths, count: int, seed: int = 0) -> list[Path]` (npz paths,
    in index order; raises `ValueError` unless `1 <= count <= 1000`)
  - CLI: `uv run python scripts/scripted_segments.py [--count N] [--seed S] [--root DIR]`

- [ ] **Step 1: Write the failing tests**

`packages/e2l_sim/tests/test_scripted.py`:

```python
import numpy as np
import pytest

from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegments
from e2l_sim.scripted import FPS, scripted_demo_id, scripted_segments, write_scripted

R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


def _draws(n: int = 200) -> list[RobotSegments]:
    return [scripted_segments(scripted_demo_id(1), np.random.default_rng([7, i])) for i in range(n)]


def test_nominal_demo_matches_the_original_script():
    segs = scripted_segments(scripted_demo_id(0)).segments
    assert [s.ref_object for s in segs] == ["bowl", "plate", "plate"]
    assert [len(s.T_obj_ee) for s in segs] == [30, 30, 20]
    for s in segs:
        np.testing.assert_array_equal(
            s.T_obj_ee[:, :3, :3], np.broadcast_to(R_TOP_DOWN, (len(s.T_obj_ee), 3, 3))
        )
    np.testing.assert_allclose(segs[0].T_obj_ee[0, :3, 3], [0.05, 0.0, 0.145])
    np.testing.assert_allclose(segs[0].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.045])
    np.testing.assert_allclose(segs[1].T_obj_ee[0, :3, 3], [0.05, 0.0, 0.225])
    np.testing.assert_allclose(segs[1].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.075])
    np.testing.assert_allclose(segs[2].T_obj_ee[-1, :3, 3], [0.05, 0.0, 0.225])


def test_structure_is_preserved():
    for demo in [scripted_segments(scripted_demo_id(0)), *_draws(50)]:
        a, c, r = demo.segments
        assert [s.ref_object for s in demo.segments] == ["bowl", "plate", "plate"]
        assert (a.gripper[:-1] == -1).all() and a.gripper[-1] == 1  # grasp on the last sample
        assert (c.gripper[:-1] == 1).all() and c.gripper[-1] == -1  # release on the last sample
        assert (r.gripper == -1).all()
        assert all(s.feasible.all() and len(s.feasible) == len(s.T_obj_ee) for s in demo.segments)
        assert demo.fps == FPS


def test_jitter_is_seeded():
    a = scripted_segments("20261003_scripted_005", np.random.default_rng([0, 5]))
    b = scripted_segments("20261003_scripted_005", np.random.default_rng([0, 5]))
    c = scripted_segments("20261003_scripted_005", np.random.default_rng([1, 5]))
    for sa, sb in zip(a.segments, b.segments, strict=True):
        np.testing.assert_array_equal(sa.T_obj_ee, sb.T_obj_ee)
    assert not np.array_equal(a.segments[0].T_obj_ee[-1], c.segments[0].T_obj_ee[-1])


def test_jitter_stays_in_bounds():
    for demo in _draws():
        a, c, r = demo.segments
        grasp, place = a.T_obj_ee[-1, :3, 3], c.T_obj_ee[-1, :3, 3]
        angle = np.arctan2(grasp[1], grasp[0])
        assert abs(angle) <= 0.4 + 1e-9
        np.testing.assert_allclose(np.hypot(grasp[0], grasp[1]), 0.05)
        assert grasp[2] == pytest.approx(0.045)
        assert 0.08 - 1e-9 <= a.T_obj_ee[0, 2, 3] - grasp[2] <= 0.12 + 1e-9
        d = place - grasp
        assert np.abs(d[:2]).max() <= 0.01 + 1e-9
        assert 0.02 - 1e-9 <= d[2] <= 0.04 + 1e-9
        for seg, nominal in zip(demo.segments, (30, 30, 20), strict=True):
            assert round(0.7 * nominal) - 1 <= len(seg.T_obj_ee) <= round(1.3 * nominal) + 1
            # grip-site z axis points straight down; fingers close radially at the grasp angle
            np.testing.assert_allclose(
                seg.T_obj_ee[:, :3, 2], np.tile([0, 0, -1.0], (len(seg.T_obj_ee), 1)), atol=1e-12
            )
            np.testing.assert_allclose(seg.T_obj_ee[0, :2, 0], [np.cos(angle), np.sin(angle)])


def test_write_scripted_writes_nominal_first_and_jittered_rest(tmp_path):
    paths = DataPaths(tmp_path)
    npzs = write_scripted(paths, count=3, seed=0)
    assert [p.stem for p in npzs] == [f"20261003_scripted_{i:03d}" for i in range(3)]
    assert paths.demo_ids("robot_segments") == [p.stem for p in npzs]
    first = RobotSegments.load(npzs[0])
    np.testing.assert_array_equal(
        first.segments[0].T_obj_ee, scripted_segments(npzs[0].stem).segments[0].T_obj_ee
    )
    second = RobotSegments.load(npzs[1])
    assert not np.array_equal(second.segments[0].T_obj_ee[-1], first.segments[0].T_obj_ee[-1])
    again = write_scripted(DataPaths(tmp_path / "again"), count=3, seed=0)
    np.testing.assert_array_equal(
        RobotSegments.load(again[1]).segments[1].T_obj_ee, second.segments[1].T_obj_ee
    )


@pytest.mark.parametrize("count", [0, 1001])
def test_count_outside_range_is_rejected(tmp_path, count):
    with pytest.raises(ValueError, match="count"):
        write_scripted(DataPaths(tmp_path), count=count)
    assert not (tmp_path / "robot_segments").exists()


def test_demo_id_outside_range_is_rejected():
    with pytest.raises(ValueError):
        scripted_demo_id(1000)
```

Run `uv run ruff format packages/e2l_sim/tests/test_scripted.py` after pasting: two
`assert_allclose` lines exceed 100 characters and ruff wraps them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_scripted.py -v`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'e2l_sim.scripted'`.

- [ ] **Step 3: Write the module**

`packages/e2l_sim/src/e2l_sim/scripted.py`:

```python
"""Hand-designed bowl-onto-plate `RobotSegments`: the nominal demo, or a jittered copy that
varies the way human demos do, to rehearse everything after retargeting without perception.

Segments follow the same cut as real demos (approach -> grasp, carry -> release, retreat) and
are expressed in the real object names of configs/sim.yaml `object_map`. The object frames are
LIBERO's body frames (origin at the base centre, z up), so the grasp is a top-down pinch on the
bowl rim and the release is just above the plate centre.
"""

from pathlib import Path

import numpy as np

from e2l_common.geometry import make_T
from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegment, RobotSegments

FPS = 20.0
DEMO_PREFIX = "20261003_scripted"
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])  # grip site z down, fingers closing along object x
RIM_RADIUS = 0.05  # m; bowl rim r 0.056 m, h 0.053 m; angle 0 is away from the cabinet
GRASP_Z = 0.045
UP = 0.10  # m, approach height above the grasp; the place approach and retreat use 1.5x
PLACE_Z = 0.03  # m above the grasp height: bowl base ~1.5 cm above the plate's base
DURATIONS = (1.5, 1.5, 1.0)  # s: approach, carry, retreat
# Jitter, uniform: +/- bounds, and the per-segment duration scale range.
GRASP_ANGLE = 0.4  # rad around the rim
UP_JITTER = 0.02
PLACE_Z_JITTER = 0.01
PLACE_XY_JITTER = 0.01
DURATION_SCALE = (0.7, 1.3)


def scripted_demo_id(index: int) -> str:
    if not 0 <= index < 1000:
        raise ValueError(f"scripted demo index {index} is outside 0..999")
    return f"{DEMO_PREFIX}_{index:03d}"


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _line(R: np.ndarray, p0: np.ndarray, p1: np.ndarray, seconds: float) -> np.ndarray:
    """Poses with rotation R from p0 to p1 (object frame), FPS samples per second."""
    s = np.linspace(0.0, 1.0, max(2, round(seconds * FPS)))[:, None]
    return make_T(R, (1 - s) * p0 + s * p1)


def _segment(ref: str, T: np.ndarray, closed: bool, flip_at_end: bool = False) -> RobotSegment:
    gripper = np.full(len(T), 1.0 if closed else -1.0)
    if flip_at_end:
        gripper[-1] = -gripper[-1]
    return RobotSegment(
        ref_object=ref, T_obj_ee=T, gripper=gripper, feasible=np.ones(len(T), dtype=bool)
    )


def scripted_segments(demo_id: str, rng: np.random.Generator | None = None) -> RobotSegments:
    """The nominal demo when `rng` is None, else one draw of the jitter bounds above. The grip
    yaw turns with the grasp angle, so fingers always close across the rim, and the carry keeps
    the same grip-to-object pose relative to the plate, so the bowl lands centred on it."""
    angle, up, place_z, place_xy, scale = 0.0, UP, PLACE_Z, np.zeros(2), np.ones(3)
    if rng is not None:
        angle = rng.uniform(-GRASP_ANGLE, GRASP_ANGLE)
        up = UP + rng.uniform(-UP_JITTER, UP_JITTER)
        place_z = PLACE_Z + rng.uniform(-PLACE_Z_JITTER, PLACE_Z_JITTER)
        place_xy = rng.uniform(-PLACE_XY_JITTER, PLACE_XY_JITTER, size=2)
        scale = rng.uniform(*DURATION_SCALE, size=3)
    R = _rot_z(angle) @ R_TOP_DOWN
    grasp = np.array([RIM_RADIUS * np.cos(angle), RIM_RADIUS * np.sin(angle), GRASP_Z])
    place = grasp + np.array([place_xy[0], place_xy[1], place_z])
    lift = np.array([0.0, 0.0, up])
    t = np.asarray(DURATIONS) * scale
    return RobotSegments(
        demo_id=demo_id,
        fps=FPS,
        segments=[
            _segment("bowl", _line(R, grasp + lift, grasp, t[0]), closed=False, flip_at_end=True),
            _segment(
                "plate", _line(R, place + 1.5 * lift, place, t[1]), closed=True, flip_at_end=True
            ),
            _segment("plate", _line(R, place, place + 1.5 * lift, t[2]), closed=False),
        ],
    )


def write_scripted(paths: DataPaths, count: int, seed: int = 0) -> list[Path]:
    """Write demos 000..count-1 to robot_segments/: 000 nominal, i >= 1 jittered with
    rng([seed, i]). Returns the npz paths in index order."""
    if not 1 <= count <= 1000:
        raise ValueError(f"count must be in 1..1000, got {count}")
    out = []
    for i in range(count):
        demo_id = scripted_demo_id(i)
        rng = None if i == 0 else np.random.default_rng([seed, i])
        npz, _ = scripted_segments(demo_id, rng).save(paths.robot_segments(demo_id))
        out.append(npz)
    return out
```

Note: `_rot_z(0.0)` is exactly the identity, so demo 000 equals the old script's output exactly
(the old `line()` used `R_TOP_DOWN` directly; `I @ R_TOP_DOWN == R_TOP_DOWN` exactly).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_scripted.py -v`
Expected: 8 passed. If `test_jitter_stays_in_bounds` fails on the duration bound, check
`_line` rounds `seconds * FPS` (not truncates).

- [ ] **Step 5: Replace the script with a thin CLI**

`scripts/scripted_segments.py` (whole file):

```python
"""Write hand-designed bowl-onto-plate `RobotSegments` files, to test replay and everything after
it without perception. Demo 000 is the nominal demo; 001.. are jittered the way human demos vary
(see e2l_sim.scripted).

    uv run python scripts/scripted_segments.py                # 20261003_scripted_000 only
    uv run python scripts/scripted_segments.py --count 25     # rehearsal set (make rehearsal)
    uv run e2l sim replay 20261003_scripted_000 --video
"""

from pathlib import Path

import typer

from e2l_common.paths import DataPaths
from e2l_sim.scripted import write_scripted


def main(
    count: int = typer.Option(1, help="Demos to write: 000 nominal, the rest jittered."),
    seed: int = typer.Option(0, help="Jitter seed; demo i uses rng([seed, i])."),
    root: Path | None = typer.Option(None, help="Data root (default: data/ or $E2L_DATA)."),
) -> None:
    paths = DataPaths(root) if root else DataPaths.default()
    for npz in write_scripted(paths, count, seed):
        print(f"wrote {npz}")


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 6: Point the LIBERO replay test at the module**

In `packages/e2l_sim/tests/test_replay.py`, `test_scripted_bowl_onto_plate_succeeds`: delete the
`importlib` block (the `import importlib.util`, `path = ...`, `spec = ...`, `module = ...`,
`spec.loader.exec_module(module)` lines) and replace
`segments: RobotSegments = module.scripted("20261003_scripted_000")` with:

```python
    from e2l_sim.scripted import scripted_segments

    segments: RobotSegments = scripted_segments("20261003_scripted_000")
```

Update its docstring to `"""The nominal scripted demo (e2l_sim.scripted) solves the chosen task."""`.
Remove imports that became unused (`ruff check` will name them).

- [ ] **Step 7: Run the sim test, the script, and lint**

Run:
```bash
unset PYTHONPATH
uv run pytest packages/e2l_sim -v -m "sim or not sim"
uv run python scripts/scripted_segments.py --root "$(mktemp -d)" --count 3
uv run ruff check . && uv run ruff format --check .
```
Expected: all `e2l_sim` tests pass, including `test_scripted_bowl_onto_plate_succeeds`; the
script prints three `wrote .../robot_segments/20261003_scripted_00{0,1,2}.npz` lines; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add packages/e2l_sim/src/e2l_sim/scripted.py packages/e2l_sim/tests/test_scripted.py \
  packages/e2l_sim/tests/test_replay.py scripts/scripted_segments.py
git commit -m "feat(sim): jitter scripted demos to stand in for human ones"
```

---

### Task 2: `e2l sim replay --run-id`

**Files:**
- Modify: `packages/e2l_sim/src/e2l_sim/cli.py` (`replay` command)
- Create: `packages/e2l_sim/tests/test_cli.py`

**Interfaces:**
- Consumes: `write_scripted(paths, count, seed)` from Task 1 (test fixture data).
- Produces: `e2l sim replay <demo_id> [--run-id RUN] [--episode-index K] [--video]`; saves to
  `data/generated/<RUN>/<demo_id>.{npz,json}`; `RUN` defaults to `replay` and must match
  `^[A-Za-z0-9][A-Za-z0-9._-]*$`.

- [ ] **Step 1: Write the failing tests**

`packages/e2l_sim/tests/test_cli.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import e2l_sim.replay
from e2l_common.cli import app
from e2l_common.paths import DataPaths, repo_root
from e2l_sim.scripted import write_scripted

DEMO = "20261003_scripted_000"


@pytest.fixture
def fake_replay(tmp_path, monkeypatch):
    """Scripted demo on disk under a temp data root; replay records its call and saves stubs."""
    monkeypatch.setenv("E2L_DATA", str(tmp_path))
    write_scripted(DataPaths(tmp_path), count=1)
    calls = {}

    def save(path):
        calls["saved"] = path
        return path.with_suffix(".npz"), path.with_suffix(".json")

    def replay(segments, cfg, episode_index=0, **_):
        calls["episode_index"] = episode_index
        return SimpleNamespace(success=True, actions=np.zeros((5, 7)), save=save)

    monkeypatch.setattr(e2l_sim.replay, "replay", replay)
    return tmp_path, calls


def _invoke(*args: str):
    cfg = str(repo_root() / "configs" / "sim.yaml")
    return CliRunner().invoke(app, ["sim", "replay", DEMO, "-c", cfg, *args])


def test_replay_defaults_to_replay_folder(fake_replay):
    root, calls = fake_replay
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert calls["saved"] == DataPaths(root).generated("replay", DEMO)
    assert calls["episode_index"] == 0


def test_replay_saves_under_run_id(fake_replay):
    root, calls = fake_replay
    result = _invoke("--run-id", "rehearsal_replay", "--episode-index", "3")
    assert result.exit_code == 0, result.output
    assert calls["saved"] == DataPaths(root).generated("rehearsal_replay", DEMO)
    assert calls["episode_index"] == 3


@pytest.mark.parametrize("run_id", ["../x", "a/b", ".hidden", ""])
def test_replay_rejects_run_id_with_separator(fake_replay, run_id):
    _, calls = fake_replay
    result = _invoke("--run-id", run_id)
    assert result.exit_code != 0
    assert "saved" not in calls and "episode_index" not in calls
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_cli.py -v`
Expected: `test_replay_defaults_to_replay_folder` passes (current behaviour);
`test_replay_saves_under_run_id` and the four `rejects` cases FAIL with "No such option:
--run-id" (exit code 2 makes the rejects cases pass for the wrong reason — that's fine, they
must still pass after Step 3).

If `test_replay_defaults_to_replay_folder` errors because `e2l_common.cli` cannot find the `sim`
sub-app, check `uv run e2l --help` lists `sim` in the worktree venv (`make worktree`).

- [ ] **Step 3: Add the option**

In `packages/e2l_sim/src/e2l_sim/cli.py`, add at module level (after `app = ...`):

```python
import re

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # one folder under data/generated/
```

(put `import re` with the other stdlib imports at the top). Change `replay`'s signature and
body:

```python
@app.command()
def replay(
    demo_id: str,
    config: Path = config_option("sim"),
    episode_index: int = typer.Option(0, help="LIBERO init state giving the object poses."),
    run_id: str = typer.Option("replay", help="Output folder under data/generated/."),
    video: bool = typer.Option(False, help="Also write outputs/replay/<demo_id>.mp4."),
    set_: list[str] = set_option(),
) -> None:
    """Replay data/robot_segments/<demo_id> once at the nominal object poses into
    data/generated/<run_id>/."""
    if not _RUN_ID.match(run_id):
        raise typer.BadParameter(f"{run_id!r} is not a single folder name", param_hint="--run-id")
    import numpy as np

    ...
    npz, _ = episode.save(paths.generated(run_id, demo_id))
```

(only the `episode.save(...)` argument changes in the body; keep everything else).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_cli.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint and commit**

```bash
unset PYTHONPATH; uv run ruff check . && uv run ruff format --check .
git add packages/e2l_sim/src/e2l_sim/cli.py packages/e2l_sim/tests/test_cli.py
git commit -m "feat(sim): choose the replay output folder with --run-id"
```

---

### Task 3: Pin the eval/generation separation in the repo configs

**Files:**
- Modify: `packages/e2l_common/tests/test_repo_configs.py` (append one test)

**Interfaces:**
- Consumes: `load_config` (existing), `configs/generate.yaml`, `configs/eval.yaml`.
- Produces: nothing; a guard that fails if someone widens generation into the wide-eval range.

This is an invariant test over today's (correct) configs, so it passes on first run. Prove it
can fail by breaking a config temporarily.

- [ ] **Step 1: Write the test**

Append to `packages/e2l_common/tests/test_repo_configs.py`:

```python
def test_wide_eval_lies_outside_generation_range():
    """Held-out evaluation: wide placements start where generation sampling stops, and the
    standard and wide seed ranges never overlap."""
    configs = Path(__file__).parents[3] / "configs"
    gen = load_config(configs / "generate.yaml")
    ev = load_config(configs / "eval.yaml")
    assert ev.wide.min_offset >= gen.xy_range
    assert ev.wide.min_yaw >= gen.yaw_range
    assert ev.wide.seed_offset >= ev.n_episodes
```

- [ ] **Step 2: Check it can fail**

Run:
```bash
unset PYTHONPATH
sed -i 's/^xy_range: 0.08 /xy_range: 0.10 /' configs/generate.yaml && \
uv run pytest packages/e2l_common/tests/test_repo_configs.py::test_wide_eval_lies_outside_generation_range -q; \
git checkout configs/generate.yaml
```
Expected: the run with `xy_range: 0.10` FAILS on `ev.wide.min_offset >= gen.xy_range`; the
config is restored afterwards (`git diff configs/` is empty).

- [ ] **Step 3: Run it on the real configs**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_common/tests/test_repo_configs.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add packages/e2l_common/tests/test_repo_configs.py
git commit -m "test(configs): keep wide eval outside the generation range"
```

---

### Task 4: `make rehearsal`: replay, generate and export the rehearsal datasets

**Files:**
- Create: `scripts/rehearsal.sh` (executable)
- Modify: `Makefile` (`.PHONY` list and a `rehearsal` target)
- Modify: `CLAUDE.md` (Commands section, one line)

**Interfaces:**
- Consumes: `scripts/scripted_segments.py --count` (Task 1), `e2l sim replay --run-id
  --episode-index` (Task 2), existing `e2l generate run <ids> --run-id --overwrite --set`,
  existing `e2l train export <run_ids> --set dataset_root=... --set dataset_repo_id=...`.
- Produces: `data/generated/rehearsal_replay/`, `data/generated/rehearsal_gen/` (with
  `manifest.json`), `data/lerobot/rehearsal_a/`, `data/lerobot/rehearsal_b/`. Track 2 rsyncs the
  two datasets to the Spark.

- [ ] **Step 1: Write the driver**

`scripts/rehearsal.sh`:

```bash
#!/usr/bin/env bash
# Rehearse everything after retargeting on jittered scripted demos: replay each (Run A source),
# multiply them (Run B source), export both LeRobot datasets. Rebuilds from scratch on rerun.
# Spec: docs/superpowers/specs/2026-10-12-scripted-rehearsal-design.md
#   COUNT=25 PER_DEMO=16 scripts/rehearsal.sh      (make rehearsal)
set -euo pipefail
unset PYTHONPATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
COUNT="${COUNT:-25}"
PER_DEMO="${PER_DEMO:-16}"
e2l() { uv run e2l "$@"; }

uv run python scripts/scripted_segments.py --count "$COUNT"
mapfile -t ids < <(for ((i = 0; i < COUNT; i++)); do printf '20261003_scripted_%03d\n' "$i"; done)

rm -rf data/generated/rehearsal_replay data/lerobot/rehearsal_a data/lerobot/rehearsal_b
for i in "${!ids[@]}"; do
  e2l sim replay "${ids[$i]}" --run-id rehearsal_replay --episode-index $((i % 50))
done
ok=$(grep -l '"success": true' data/generated/rehearsal_replay/*.json | wc -l)
echo "raw replay: $ok / $COUNT succeeded"

e2l generate run "${ids[@]}" --run-id rehearsal_gen --overwrite --set "episodes_per_demo=$PER_DEMO"

e2l train export rehearsal_replay \
  --set dataset_root=data/lerobot/rehearsal_a --set dataset_repo_id=local/rehearsal_a
e2l train export rehearsal_replay rehearsal_gen \
  --set dataset_root=data/lerobot/rehearsal_b --set dataset_repo_id=local/rehearsal_b
```

Then `chmod +x scripts/rehearsal.sh`.

Before relying on the `grep` count, check the sidecar's success key formatting:
`python3 -c "import json;print(json.load(open('data/generated/scripted50/20261003_scripted_000_0000.json'))['success'])"`
prints `True`, and `grep -c '"success": true' data/generated/scripted50/20261003_scripted_000_0000.json`
prints `1`. If the JSON is written without the space after the colon, change the pattern to
`'"success":\s*true'` with `grep -lE`.

- [ ] **Step 2: Add the Makefile target**

In `Makefile`, append ` rehearsal` to the `.PHONY` list (the continuation line ending in
`eval report`), and add after the `report:` target:

```make
rehearsal:        ## scripted stand-in demos -> replay, generate, export (COUNT=25 PER_DEMO=16)
	scripts/rehearsal.sh
```

(The recipe line starts with a tab. `COUNT=... make rehearsal` passes through the environment.)

- [ ] **Step 3: Document it**

In `CLAUDE.md`, under `## Commands`, after the "Replay check without perception" bullet (two
lines), add:

```markdown
- Rehearsal without demos: `make rehearsal` (25 jittered scripted demos → `rehearsal_replay`,
  `rehearsal_gen`, datasets `data/lerobot/rehearsal_{a,b}`)
```

- [ ] **Step 4: Tiny run, twice (Review Focus 1)**

Run:
```bash
COUNT=2 PER_DEMO=2 make rehearsal && COUNT=2 PER_DEMO=2 make rehearsal
```
Expected: both runs complete with exit 0; the second does not fail with
`FileExistsError ... choose a new dataset_root`; each prints `raw replay: N / 2 succeeded`
and the generate yield table for two demos. If `rehearsal_replay` has 0 successes the Run A
export fails with `no successful episodes` — that is correct behaviour; rerun with `COUNT=3`
(demo 000 is the nominal demo and succeeds).

- [ ] **Step 5: Lint, test, commit**

```bash
unset PYTHONPATH; uv run ruff check . && uv run ruff format --check . && uv run pytest -q
git add scripts/rehearsal.sh Makefile CLAUDE.md
git commit -m "feat(scripts): rehearse replay, generation and export on scripted demos"
```

- [ ] **Step 6: Full run**

Check disk first: `df -h /mnt/data` needs ≥ 15 GB free (~400 attempts, ~330 kept × ~22 MB,
plus two image datasets). Then, in the background (expect hours; generation is the long part):

```bash
mkdir -p outputs/rehearsal && make rehearsal 2>&1 | tee outputs/rehearsal/rehearsal.log
```

Expected at the end of the log: `raw replay: N / 25 succeeded`, a yield table for
`generated/rehearsal_gen` with 25 rows, and two `wrote data/lerobot/rehearsal_{a,b}` lines.

If total successes in `rehearsal_gen` are below 300, rerun only generation and export with a
larger `PER_DEMO` (attempt seeds are per (demo, attempt), so the first attempts reproduce):

```bash
PER = <ceil(300 / (25 * observed_yield))>
uv run e2l generate run $(printf '20261003_scripted_%03d ' $(seq 0 24)) --run-id rehearsal_gen --overwrite --set episodes_per_demo=$PER
rm -rf data/lerobot/rehearsal_b
uv run e2l train export rehearsal_replay rehearsal_gen --set dataset_root=data/lerobot/rehearsal_b --set dataset_repo_id=local/rehearsal_b
```

- [ ] **Step 7: Verify both datasets load and record the numbers**

Run:
```bash
unset PYTHONPATH; uv run python - <<'EOF'
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset
for name in ("rehearsal_a", "rehearsal_b"):
    ds = LeRobotDataset(f"local/{name}", root=f"data/lerobot/{name}")
    print(name, "episodes", ds.num_episodes, "frames", ds.num_frames, "keys", sorted(ds[0]))
m = json.load(open("data/generated/rehearsal_gen/manifest.json"))
att = sum(d["attempts"] for d in m["demos"].values())
ok = sum(d["successes"] for d in m["demos"].values())
never = [k for k, d in m["demos"].items() if d["successes"] == 0]
print(f"generation: {ok}/{att} = {ok/att:.0%}; demos never succeeding: {never}")
EOF
du -sh data/generated/rehearsal_* data/lerobot/rehearsal_*
```
Expected: `rehearsal_a` episodes = the raw-replay success count; `rehearsal_b` episodes = that
plus the generation successes (≥ 300); keys include `observation.images.image`,
`observation.images.image2`, `observation.state`, `action`, `task`. Write these numbers
(replay successes, generation successes/attempts and yield, never-succeeding demos, episode
and frame counts, sizes) into `outputs/rehearsal/summary.md` for the PR description and Track 2.

---

### Task 5: Reference evaluation of `smolvla_libero`

**Files:** none (runs only; outputs under `outputs/`, which is gitignored).

**Interfaces:**
- Consumes: existing `e2l eval rollout --run reference --setting {standard,wide}` and
  `e2l eval report --results outputs/eval --out-dir DIR`.
- Produces: `outputs/eval/reference/standard/results.json`,
  `outputs/eval/reference/wide/results.json` (+ 3 videos each) — kept for the write-up — and a
  throwaway check report in `outputs/report/`.

Can run in parallel with Task 4 Step 6 only if GPU memory allows: generation is CPU-only
(MuJoCo + EGL rendering), the policy uses ~2–3 GB on the 8 GB RTX 4060. Check
`nvidia-smi` while generation runs; if free memory is under 4 GB, run this after it.

- [ ] **Step 1: Standard setting**

Run:
```bash
unset PYTHONPATH; HF_HOME=/mnt/data/hf-cache uv run e2l eval rollout --run reference --setting standard 2>&1 | tee outputs/eval_reference_standard.log
```
Expected: 50 episodes; `outputs/eval/reference/standard/results.json` with 50 `successes` and
seeds 1000..1049; 3 mp4s under `videos/`.

- [ ] **Step 2: Wide setting**

Run:
```bash
unset PYTHONPATH; HF_HOME=/mnt/data/hf-cache uv run e2l eval rollout --run reference --setting wide 2>&1 | tee outputs/eval_reference_wide.log
```
Expected: 50 episodes, seeds 11000..11049; `placements` recorded for each, with
`placement_failed` false for (nearly) all; 3 mp4s.

- [ ] **Step 3: Check placements really are wide**

Run:
```bash
python3 - <<'EOF'
import json
r = json.load(open("outputs/eval/reference/wide/results.json"))
p = r["placements"]
print(len(p), "placements;", sum(x["placement_failed"] for x in p), "fell back to the init state")
EOF
```
Expected: 50 placements, at most a few fallbacks (the spec of the wide setting measured 0/40).
More than 5 fallbacks → stop and report; the wide row would be mislabelled.

- [ ] **Step 4: Report into outputs/, not docs/**

Run:
```bash
unset PYTHONPATH; uv run e2l eval report --results outputs/eval --out-dir outputs/report
```
Expected: prints paths to `outputs/report/results.md`, `success_rates.png`, `rollouts.gif`;
`results.md` has a `reference` row with standard and wide success rates and Wilson intervals.
`git status --short docs/` shows nothing new. Copy the reference row into
`outputs/rehearsal/summary.md`.

(`outputs/eval/baseline/` from `make smoke` has no `run`/`setting` keys; if the report trips on
it, pass `--results` a folder containing only `reference/`, e.g. by moving `baseline` to
`outputs/eval_baseline/`, and note it in the summary.)

---

### Task 6: Review and finish the branch

**Files:** none new.

- [ ] **Step 1: Full checks**

Run: `unset PYTHONPATH; make lint && make test`
Expected: ruff clean; all tests pass (sim-marked ones included on this laptop).

- [ ] **Step 2: Code review**

Use superpowers:requesting-code-review on the branch diff against `main`, with the spec and this
plan as context. Address findings with superpowers:receiving-code-review.

- [ ] **Step 3: Verify, then finish**

Use superpowers:verification-before-completion (re-run Step 1, re-read
`outputs/rehearsal/summary.md`), then superpowers:finishing-a-development-branch. The spec and
this plan are committed on the branch (`docs(repo): add scripted rehearsal spec and plan`) if
not already. PR body: what changed, plus the numbers from `outputs/rehearsal/summary.md` (raw
replay rate, generation yield, dataset sizes, reference standard/wide success with intervals).
No attribution lines.
