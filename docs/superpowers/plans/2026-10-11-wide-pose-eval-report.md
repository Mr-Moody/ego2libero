# Wide-pose evaluation and results report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate policies on a `wide` setting (task objects moved beyond the generation range)
next to the `standard` LIBERO init states, store one `results.json` per (run, setting), and build
the README table, bar chart and rollout GIF from them.

**Architecture:** Pure sampling maths in `e2l_common.geometry`; placement validity and a lazily
built `LiberoEnv` subclass in `e2l_sim.wide`; `e2l_eval.rollout` swaps in that env for
`--setting wide` but keeps LeRobot's `eval_policy` loop; `e2l_eval.report` reads a results tree
and writes `results.md`, `success_rates.png`, `rollouts.gif`.

**Tech Stack:** Python 3.12, numpy, pydantic, typer, gymnasium, LeRobot 0.6.1 / LIBERO,
matplotlib (Agg), imageio + imageio-ffmpeg (read mp4), Pillow (labels, GIF).

**Spec:** `docs/superpowers/specs/2026-10-11-wide-pose-eval-report-design.md`

## Global Constraints

- Work in `.worktrees/wide-pose-eval-report` on branch `feat/wide-pose-eval-report`. Run
  `make worktree` there once before the first test run. `unset PYTHONPATH` before any `uv run`.
- Commits: `<type>(<scope>)?: <subject>`, types feat/fix/refactor/test/docs/…; scopes are
  package names (`common`, `sim`, `eval`, `generate`). **No `Co-Authored-By:`, `Claude-Session:`
  or "Generated with Claude Code" lines, ever.**
- Poses are 4x4 float64 `T_a_b`; no quaternions in files/configs (results store `xyz` + `rotvec`).
- LIBERO / robosuite / MuJoCo / LeRobot env imports only inside `e2l_sim`, and lazily inside
  functions in `e2l_eval`. Every `e2l_sim` module must import without LIBERO (the stub checker
  imports them all), so `e2l_sim.wide` builds its `LiberoEnv` subclass inside a function.
- ruff, line length 100. Type hints. Lean code, comment density like the surrounding files.
- Wide defaults (verbatim from spec): `min_offset 0.08`, `max_offset 0.15`, `min_yaw 0.5`,
  `max_yaw 1.0`, `min_separation 0.11`, `max_tries 20`, `settle_tol 0.02`, `seed_offset 10_000`.
- Results layout: `outputs/eval/<run>/<setting>/results.json` (+ `videos/eval_episode_<i>.mp4`).
- Run order in table/chart: `A`, `B`, `reference`, then others alphabetically.

## Spec refinements made while planning

- **Separation check**: the spec says `separated(all movable object poses)`. LIBERO init states
  may already place two *unmoved* objects closer than 0.11 m, which would reject every draw. The
  check is applied only to pairs that include a moved object (`placement_valid`, Task 3).
- **Settle check** rejects `|dz| > settle_tol` in both directions (an object landing on a
  fixture rises; one falling off sinks), not only drops.
- Placements are recorded per reset on the env (`env.placements`) and copied into
  `results.json` before the env is closed.

## Review Focus

1. **Results copied from another machine** (Spark → laptop): `video_paths` in `results.json` are
   relative to the other machine's cwd. The report must find `videos/<name>` next to the
   `results.json`. Test in Task 6.
2. **Missing (run, setting) cells**: before Run A exists, the report must still write a table
   (`—` cells), a chart without those bars, and not crash. Test in Task 6.
3. **No videos / no successes / no failures**: GIF tiles that cannot be filled are skipped; no
   GIF file at all when no run has videos. Test in Task 6.
4. **Init states with unmoved objects already close together** must not make every wide draw
   fail. Test in Task 3 (`placement_valid` ignores unmoved–unmoved pairs).
5. **Typo in `--run` or `--setting`**: a clear CLI error listing valid values, never a rollout of
   the wrong policy. Test in Task 5.

---

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `packages/e2l_common/src/e2l_common/geometry.py` | modify | gains `perturb`, `separated`, `sample_wide_offset` |
| `packages/e2l_common/tests/test_geometry.py` | modify | tests for the three |
| `packages/e2l_generate/src/e2l_generate/sample.py` | modify | import `perturb`, `separated` from geometry |
| `packages/e2l_common/src/e2l_common/config.py` | modify | `WidePoseConfig`; `EvalConfig.runs`, `.wide` |
| `packages/e2l_common/tests/test_config.py` | modify | eval config defaults / overrides |
| `configs/eval.yaml` | modify | `runs:` and `wide:` blocks |
| `packages/e2l_sim/src/e2l_sim/wide.py` | create | `sample_wide_placement`, `placement_valid`, `settled`, `wide_pose_env_class()` |
| `packages/e2l_sim/src/e2l_sim/env.py` | modify | `make_env(..., wide=None)` |
| `packages/e2l_sim/tests/test_wide.py` | create | pure tests + one slow LIBERO test |
| `packages/e2l_eval/src/e2l_eval/rollout.py` | modify | `setting`, `run`, wide env, `write_results` |
| `packages/e2l_eval/tests/test_rollout.py` | modify | `write_results` test |
| `packages/e2l_eval/src/e2l_eval/cli.py` | modify | `--run`, `--setting`, `--videos`; `resolve_policy`; report defaults |
| `packages/e2l_eval/tests/test_cli.py` | create | CLI resolution tests |
| `packages/e2l_eval/src/e2l_eval/report.py` | rewrite | loader, table, chart, GIF, `make_report` |
| `packages/e2l_eval/tests/test_report.py` | create | synthetic results-tree tests |
| `README.md`, `CLAUDE.md` | modify | commands |

---

### Task 1: Placement maths in `e2l_common.geometry`

**Files:**
- Modify: `packages/e2l_common/src/e2l_common/geometry.py` (append)
- Modify: `packages/e2l_generate/src/e2l_generate/sample.py:1-27`
- Modify: `packages/e2l_common/tests/test_geometry.py` (append)

**Interfaces:**
- Consumes: `rotvec_to_matrix` (already in geometry).
- Produces:
  - `perturb(T_sim_obj: np.ndarray, d_xy: np.ndarray, yaw: float) -> np.ndarray`
  - `separated(poses: dict[str, np.ndarray], min_distance: float) -> bool`
  - `sample_wide_offset(rng: np.random.Generator, min_offset: float, max_offset: float, min_yaw: float, max_yaw: float) -> tuple[np.ndarray, float]`
    (plain floats, not the config, so `e2l_common.geometry` does not import `config`, which
    imports geometry.)

- [ ] **Step 1: Write the failing tests** — append to `packages/e2l_common/tests/test_geometry.py`
  and add `perturb, sample_wide_offset, separated` to its geometry import list:

```python
def test_perturb_shifts_xy_and_turns_about_z():
    T = make_T(rotvec_to_matrix(np.array([0.0, 0.0, 0.3])), [0.1, -0.2, 0.9])
    P = perturb(T, np.array([0.05, -0.01]), 0.4)
    np.testing.assert_allclose(P[:3, 3], [0.15, -0.21, 0.9])
    np.testing.assert_allclose(matrix_to_rotvec(P[:3, :3]), [0.0, 0.0, 0.7], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.1, -0.2, 0.9])  # input untouched


def test_separated_uses_xy_only():
    a = make_T(np.eye(3), [0.0, 0.0, 0.0])
    b = make_T(np.eye(3), [0.1, 0.0, 5.0])
    assert separated({"a": a, "b": b}, 0.1)
    assert not separated({"a": a, "b": b}, 0.11)
    assert separated({"a": a}, 1.0)


def test_sample_wide_offset_stays_outside_square_inside_disc():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d, yaw = sample_wide_offset(rng, 0.08, 0.15, 0.5, 1.0)
        assert d.shape == (2,)
        assert np.linalg.norm(d) <= 0.15 + 1e-12
        assert np.max(np.abs(d)) > 0.08
        assert 0.5 <= abs(yaw) <= 1.0


def test_sample_wide_offset_is_seeded_and_covers_both_yaw_signs():
    a = [sample_wide_offset(np.random.default_rng(5), 0.08, 0.15, 0.5, 1.0) for _ in range(3)]
    b = [sample_wide_offset(np.random.default_rng(5), 0.08, 0.15, 0.5, 1.0) for _ in range(3)]
    for (da, ya), (db, yb) in zip(a, b):
        np.testing.assert_array_equal(da, db)
        assert ya == yb
    rng = np.random.default_rng(1)
    signs = {np.sign(sample_wide_offset(rng, 0.08, 0.15, 0.5, 1.0)[1]) for _ in range(50)}
    assert signs == {-1.0, 1.0}
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_common/tests/test_geometry.py -q`
Expected: collection error `ImportError: cannot import name 'perturb'`.

- [ ] **Step 3: Implement** — append to `packages/e2l_common/src/e2l_common/geometry.py`
  (add `from itertools import combinations` to the imports at the top):

```python
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


def sample_wide_offset(
    rng: np.random.Generator, min_offset: float, max_offset: float, min_yaw: float, max_yaw: float
) -> tuple[np.ndarray, float]:
    """One (d_xy, yaw) draw: d_xy uniform over the disc of radius `max_offset`, rejecting draws
    inside the square +/- `min_offset` on both axes; |yaw| uniform in [min_yaw, max_yaw] with a
    random sign."""
    while True:
        d_xy = rng.uniform(-max_offset, max_offset, size=2)
        if np.linalg.norm(d_xy) <= max_offset and np.max(np.abs(d_xy)) > min_offset:
            break
    yaw = rng.uniform(min_yaw, max_yaw) * rng.choice([-1.0, 1.0])
    return d_xy, float(yaw)
```

  In `packages/e2l_generate/src/e2l_generate/sample.py` delete the local `perturb` and
  `separated` definitions and the `combinations` / `rotvec_to_matrix` imports; import instead:

```python
from e2l_common.geometry import perturb, separated
```

- [ ] **Step 4: Run tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_common/tests/test_geometry.py packages/e2l_generate/tests/test_sample.py -q && uv run ruff check packages/e2l_common packages/e2l_generate`
Expected: all pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_common packages/e2l_generate/src/e2l_generate/sample.py
git commit -m "refactor(common): move placement perturbation into geometry and add wide offsets"
```

---

### Task 2: Eval config — `WidePoseConfig`, `runs`

**Files:**
- Modify: `packages/e2l_common/src/e2l_common/config.py:147-154` and `__all__`
- Modify: `configs/eval.yaml`
- Modify: `packages/e2l_common/tests/test_config.py` (append)

**Interfaces:**
- Produces: `WidePoseConfig` (fields below), `EvalConfig.runs: dict[str, str]`,
  `EvalConfig.wide: WidePoseConfig`.

- [ ] **Step 1: Write the failing test** — append to `packages/e2l_common/tests/test_config.py`:

```python
def test_eval_config_wide_defaults_and_overrides(tmp_path):
    from e2l_common.config import EvalConfig, WidePoseConfig

    cfg = EvalConfig()
    assert cfg.runs == {"reference": "HuggingFaceVLA/smolvla_libero"}
    assert cfg.wide == WidePoseConfig()
    assert (cfg.wide.min_offset, cfg.wide.max_offset) == (0.08, 0.15)
    assert (cfg.wide.min_yaw, cfg.wide.max_yaw) == (0.5, 1.0)
    assert cfg.wide.seed_offset == 10_000
    path = tmp_path / "eval.yaml"
    path.write_text("runs: {B: outputs/train/b}\n")
    cfg = load_config(path, ["wide.max_offset=0.2"], EvalConfig)
    assert cfg.runs == {"B": "outputs/train/b"} and cfg.wide.max_offset == 0.2
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_common/tests/test_config.py -q`
Expected: FAIL, `ImportError: cannot import name 'WidePoseConfig'`.

- [ ] **Step 3: Implement** — in `config.py` add `"WidePoseConfig"` to `__all__` and replace
  the `EvalConfig` class with:

```python
class WidePoseConfig(_Config):
    """Wide-pose evaluation: task objects moved beyond the generation range after reset."""

    min_offset: float = 0.08  # m, offsets inside this square (both axes) are rejected
    max_offset: float = 0.15  # m, max radial xy offset from the init-state pose
    min_yaw: float = 0.5  # rad, |yaw| about the sim z axis
    max_yaw: float = 1.0
    min_separation: float = 0.11  # m, xy, from each moved object to every other object
    table_xy: tuple[tuple[float, float], tuple[float, float]] = ((-0.35, 0.25), (-0.40, 0.40))
    max_tries: int = 20  # draws per reset before falling back to the init state
    settle_tol: float = 0.02  # m, max xy drift / |dz| of a moved object while settling
    seed_offset: int = 10_000  # wide episode i uses seed + seed_offset + i


class EvalConfig(_Config):
    policy_path: str = "HuggingFaceVLA/smolvla_libero"
    runs: dict[str, str] = Field(
        default_factory=lambda: {"reference": "HuggingFaceVLA/smolvla_libero"}
    )  # label -> checkpoint dir or hub id
    n_episodes: int = 50
    seed: int = 1000
    output_dir: Path = Path("outputs/eval")
    device: str = "cuda"
    wide: WidePoseConfig = Field(default_factory=WidePoseConfig)
    demo_counts: list[int] = Field(default_factory=lambda: [5, 10, 20, 40])
    augmentation: list[bool] = Field(default_factory=lambda: [False, True])
```

  In `configs/eval.yaml`, after `policy_path`, add (the `table_xy` comment is finalised in
  Task 3 after measuring):

```yaml
runs:                                        # e2l eval rollout --run <label>
  reference: HuggingFaceVLA/smolvla_libero   # LIBERO teleop data; context only
  # A: outputs/train/run_a/checkpoints/last/pretrained_model   # human demos only
  # B: outputs/train/run_b/checkpoints/last/pretrained_model   # + generated episodes

wide:                                        # --setting wide: task objects moved after reset
  min_offset: 0.08                           # m; offsets inside +/- this square are rejected (= generate xy_range)
  max_offset: 0.15                           # m; radial
  min_yaw: 0.5                               # rad; |yaw| (= generate yaw_range)
  max_yaw: 1.0
  min_separation: 0.11                       # m, xy, moved object to any other object
  table_xy: [[-0.35, 0.25], [-0.40, 0.40]]   # sim-frame xy box objects must stay in
  max_tries: 20
  settle_tol: 0.02                           # m
  seed_offset: 10000                         # wide seeds = seed + 10000 + i
```

- [ ] **Step 4: Run tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_common -q && uv run ruff check packages/e2l_common`
Expected: pass (including `test_repo_configs.py::test_config_loads[eval]`).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_common configs/eval.yaml
git commit -m "feat(common): add wide-pose and run-label settings to the eval config"
```

---

### Task 3: `e2l_sim.wide` — placements and the wide-pose env

**Files:**
- Create: `packages/e2l_sim/src/e2l_sim/wide.py`
- Modify: `packages/e2l_sim/src/e2l_sim/env.py` (`make_env`)
- Create: `packages/e2l_sim/tests/test_wide.py`
- Modify: `configs/eval.yaml` (`table_xy` after measuring)

**Interfaces:**
- Consumes: `perturb`, `sample_wide_offset` (Task 1); `WidePoseConfig` (Task 2);
  `e2l_sim.scene.object_poses(env)`, `set_object_poses(env, poses)`, `_problem_env(env)`.
- Produces:
  - `sample_wide_placement(nominal: dict[str, np.ndarray], names: Iterable[str], cfg: WidePoseConfig, rng: np.random.Generator) -> dict[str, np.ndarray] | None` (one draw; `None` if invalid)
  - `placement_valid(moved: dict[str, np.ndarray], others: dict[str, np.ndarray], cfg: WidePoseConfig) -> bool`
  - `settled(actual: dict[str, np.ndarray], targets: dict[str, np.ndarray], tol: float) -> bool`
  - `wide_pose_env_class() -> type` (cached `LiberoEnv` subclass; instances have
    `placements: list[dict]`, each `{"seed": int | None, "placement_failed": bool, "T_sim_obj": {name: {"xyz": [3], "rotvec": [3]}}}`)
  - `e2l_sim.env.make_env(cfg, episode_index=0, obs_type="pixels_agent_pos", wide: WidePoseConfig | None = None)`

- [ ] **Step 1: Write the failing pure tests** — `packages/e2l_sim/tests/test_wide.py`:

```python
import os

import numpy as np
import pytest

from e2l_common.config import EvalConfig, SimConfig, WidePoseConfig, load_config
from e2l_common.geometry import make_T, matrix_to_rotvec
from e2l_common.paths import repo_root
from e2l_sim.libero_setup import libero_available
from e2l_sim.wide import placement_valid, sample_wide_placement, settled

NOMINAL = {
    "bowl": make_T(np.eye(3), [-0.05, 0.0, 0.9]),
    "plate": make_T(np.eye(3), [0.05, 0.15, 0.9]),
    "cheese": make_T(np.eye(3), [-0.2, -0.3, 0.9]),
    "bottle": make_T(np.eye(3), [-0.2, -0.25, 0.9]),  # 5 cm from cheese, neither moves
}
CFG = WidePoseConfig(table_xy=((-0.5, 0.5), (-0.6, 0.6)))


def test_placements_move_named_objects_beyond_the_generation_square():
    rng = np.random.default_rng(0)
    accepted = 0
    for _ in range(300):
        poses = sample_wide_placement(NOMINAL, ["plate", "bowl"], CFG, rng)
        if poses is None:
            continue
        accepted += 1
        assert set(poses) == {"bowl", "plate"}
        for name, T in poses.items():
            d = T[:3, 3] - NOMINAL[name][:3, 3]
            assert np.max(np.abs(d[:2])) > CFG.min_offset and np.linalg.norm(d[:2]) <= 0.15
            assert d[2] == 0.0
            assert CFG.min_yaw - 1e-9 <= abs(matrix_to_rotvec(T[:3, :3])[2]) <= CFG.max_yaw
    assert accepted > 100  # close unmoved cheese/bottle must not block every draw


def test_placement_is_seeded_and_order_independent():
    a = sample_wide_placement(NOMINAL, ["bowl", "plate"], CFG, np.random.default_rng(3))
    b = sample_wide_placement(NOMINAL, ["plate", "bowl"], CFG, np.random.default_rng(3))
    assert a is not None
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_placement_valid_checks_table_and_moved_pairs_only():
    others = {"cheese": NOMINAL["cheese"], "bottle": NOMINAL["bottle"]}
    ok = {"bowl": make_T(np.eye(3), [0.1, 0.1, 0.9])}
    assert placement_valid(ok, others, CFG)
    off_table = {"bowl": make_T(np.eye(3), [0.6, 0.1, 0.9])}
    assert not placement_valid(off_table, others, CFG)
    near_cheese = {"bowl": make_T(np.eye(3), [-0.2, -0.35, 0.9])}
    assert not placement_valid(near_cheese, others, CFG)
    two_close = {"bowl": make_T(np.eye(3), [0.1, 0.1, 0.9]),
                 "plate": make_T(np.eye(3), [0.15, 0.1, 0.9])}
    assert not placement_valid(two_close, others, CFG)


def test_unknown_object_is_named():
    with pytest.raises(KeyError, match="bowl_2"):
        sample_wide_placement(NOMINAL, ["bowl_2"], CFG, np.random.default_rng(0))


def test_settled_checks_xy_drift_and_height_change():
    target = {"bowl": make_T(np.eye(3), [0.1, 0.1, 0.9])}
    assert settled({"bowl": make_T(np.eye(3), [0.11, 0.1, 0.895])}, target, 0.02)
    assert not settled({"bowl": make_T(np.eye(3), [0.13, 0.1, 0.9])}, target, 0.02)
    assert not settled({"bowl": make_T(np.eye(3), [0.1, 0.1, 0.95])}, target, 0.02)  # on a fixture
    assert not settled({"bowl": make_T(np.eye(3), [0.1, 0.1, 0.5])}, target, 0.02)  # fell off
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_wide.py -q`
Expected: `ModuleNotFoundError: No module named 'e2l_sim.wide'`.

- [ ] **Step 3: Implement** `packages/e2l_sim/src/e2l_sim/wide.py`:

```python
"""Wide-pose evaluation: after LIBERO's reset, move the task objects to placements beyond the
generation range. The `LiberoEnv` subclass is built lazily so this module imports without
LIBERO (the stub checker imports every module)."""

from collections.abc import Iterable
from functools import cache

import numpy as np

from e2l_common.config import WidePoseConfig
from e2l_common.geometry import matrix_to_rotvec, perturb, sample_wide_offset


def placement_valid(
    moved: dict[str, np.ndarray], others: dict[str, np.ndarray], cfg: WidePoseConfig
) -> bool:
    """Moved objects stay inside cfg.table_xy and at least cfg.min_separation (xy) from every
    other object, moved or not. Pairs of unmoved objects are not checked."""
    (x0, x1), (y0, y1) = cfg.table_xy
    for name, T in moved.items():
        x, y = T[:2, 3]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        rest = [P for n, P in {**others, **moved}.items() if n != name]
        if any(np.linalg.norm(T[:2, 3] - P[:2, 3]) < cfg.min_separation for P in rest):
            return False
    return True


def sample_wide_placement(
    nominal: dict[str, np.ndarray],
    names: Iterable[str],
    cfg: WidePoseConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """One draw: each named nominal pose T_sim_obj (4,4) moved by `sample_wide_offset`, in
    sorted name order. None if the draw fails `placement_valid` against the other objects."""
    names = sorted(set(names))
    missing = [n for n in names if n not in nominal]
    if missing:
        raise KeyError(f"objects {missing} are not in the scene; have {sorted(nominal)}")
    moved = {}
    for n in names:
        d_xy, yaw = sample_wide_offset(rng, cfg.min_offset, cfg.max_offset, cfg.min_yaw, cfg.max_yaw)
        moved[n] = perturb(nominal[n], d_xy, yaw)
    others = {n: T for n, T in nominal.items() if n not in moved}
    return moved if placement_valid(moved, others, cfg) else None


def settled(actual: dict[str, np.ndarray], targets: dict[str, np.ndarray], tol: float) -> bool:
    """Every target object ended within `tol` of its target in xy and in height."""
    return all(
        np.linalg.norm(actual[n][:2, 3] - T[:2, 3]) <= tol and abs(actual[n][2, 3] - T[2, 3]) <= tol
        for n, T in targets.items()
    )


def _pose_record(T: np.ndarray) -> dict[str, list[float]]:
    return {"xyz": T[:3, 3].tolist(), "rotvec": matrix_to_rotvec(T[:3, :3]).tolist()}


@cache
def wide_pose_env_class() -> type:
    """`LiberoEnv` subclass whose reset moves `names` to a wide placement drawn from the reset
    seed. Construct with the usual LiberoEnv kwargs plus `wide: WidePoseConfig`, `names`."""
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    from e2l_sim.scene import _problem_env, object_poses, set_object_poses

    class WidePoseLiberoEnv(LiberoEnv):
        def __init__(self, *args, wide: WidePoseConfig, names: Iterable[str], **kwargs):
            super().__init__(*args, **kwargs)
            self.wide = wide
            self.names = sorted(names)
            self.placements: list[dict] = []

        def reset(self, seed=None, **kwargs):
            super().reset(seed=seed, **kwargs)  # init state, settle steps, control mode
            rng = np.random.default_rng(seed)
            sim = _problem_env(self).sim
            saved = sim.get_state().flatten()
            nominal = object_poses(self)
            raw_obs, targets = None, None
            for _ in range(self.wide.max_tries):
                draw = sample_wide_placement(nominal, self.names, self.wide, rng)
                if draw is None:
                    continue
                self._env.regenerate_obs_from_state(saved)
                set_object_poses(self, draw)
                for _ in range(self.num_steps_wait):
                    raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
                if settled(object_poses(self), draw, self.wide.settle_tol):
                    targets = draw
                    break
            if targets is None:  # run the episode from the unchanged init state
                raw_obs = self._env.regenerate_obs_from_state(saved)
            final = object_poses(self)
            self.placements.append(
                {
                    "seed": seed,
                    "placement_failed": targets is None,
                    "T_sim_obj": {n: _pose_record(final[n]) for n in self.names},
                }
            )
            return self._format_raw_obs(raw_obs), {"is_success": False}

    return WidePoseLiberoEnv
```

  In `packages/e2l_sim/src/e2l_sim/env.py` change `make_env` to accept `wide` and pick the class
  (add `from e2l_common.config import SimConfig, WidePoseConfig`):

```python
def make_env(
    cfg: SimConfig,
    episode_index: int = 0,
    obs_type: str = "pixels_agent_pos",
    wide: WidePoseConfig | None = None,
):
    """LeRobot LiberoEnv for `cfg.suite` / `cfg.task_id`, rendering `cfg.camera_names` at
    `cfg.image_size` with `cfg.control_freq` Hz relative (delta) control. `episode_index` picks
    the LIBERO init state. With `wide`, each reset also moves the `cfg.object_map` objects to a
    wide placement (`e2l_sim.wide`). Rendering is headless (MUJOCO_GL=egl unless already set)."""
    ...  # existing body up to the task-name check, unchanged
    kwargs = dict(
        task_suite=suite,
        task_id=cfg.task_id,
        task_suite_name=cfg.suite,
        episode_length=cfg.max_steps,
        camera_name=[libero_camera(c) for c in cfg.camera_names],
        obs_type=obs_type,
        observation_width=cfg.image_size,
        observation_height=cfg.image_size,
        init_states=True,
        episode_index=episode_index,
        control_freq=cfg.control_freq,
        control_mode="relative",
    )
    if wide is None:
        return LiberoEnv(**kwargs)
    from e2l_sim.wide import wide_pose_env_class

    return wide_pose_env_class()(wide=wide, names=cfg.object_map.values(), **kwargs)
```

- [ ] **Step 4: Run pure tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_wide.py packages/e2l_sim/tests/test_skeleton.py -q`
Expected: 5 pure tests + skeleton pass.

- [ ] **Step 5: Add the slow LIBERO test** — append to `test_wide.py`:

```python
@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")
def test_wide_env_reset_moves_task_objects_reproducibly():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.env import make_env
    from e2l_sim.scene import object_poses

    sim_cfg = load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig)
    wide = load_config(repo_root() / "configs" / "eval.yaml", model=EvalConfig).wide
    names = sorted(sim_cfg.object_map.values())
    base = make_env(sim_cfg)
    try:
        base.reset(seed=11_000)
        nominal = object_poses(base)
    finally:
        base.close()
    poses = []
    for _ in range(2):
        env = make_env(sim_cfg, wide=wide)
        try:
            obs, _ = env.reset(seed=11_000)
            assert "pixels" in obs
            rec = env.placements[-1]
            assert not rec["placement_failed"]
            poses.append(object_poses(env))
        finally:
            env.close()
    (x0, x1), (y0, y1) = wide.table_xy
    for n in names:
        d = poses[0][n][:2, 3] - nominal[n][:2, 3]
        assert np.max(np.abs(d)) > wide.min_offset - wide.settle_tol
        assert x0 <= poses[0][n][0, 3] <= x1 and y0 <= poses[0][n][1, 3] <= y1
        np.testing.assert_allclose(poses[0][n], poses[1][n], atol=1e-6)
```

- [ ] **Step 6: Measure the scene and set `table_xy`** — run this throwaway snippet (not
  committed) to print where the task objects and other bodies sit across init states:

```bash
unset PYTHONPATH; MUJOCO_GL=egl uv run python - <<'EOF'
import numpy as np
from e2l_common.config import SimConfig, load_config
from e2l_sim.env import make_env
from e2l_sim.scene import _problem_env, object_poses
cfg = load_config("configs/sim.yaml", model=SimConfig)
xy = {}
for k in range(0, 50, 5):
    env = make_env(cfg, episode_index=k); env.reset(seed=0)
    for n, T in object_poses(env).items():
        xy.setdefault(n, []).append(T[:2, 3])
    if k == 0:
        sim = _problem_env(env).sim
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name and name.endswith("_main"):
                print("body", name, np.round(sim.data.body_xpos[i], 3))
    env.close()
for n, v in xy.items():
    v = np.array(v); print(n, "min", v.min(0).round(3), "max", v.max(0).round(3))
EOF
```

  Pick `table_xy` so that every nominal task-object position ± 0.15 m stays on the table and
  clear of the fixture bodies printed (cabinet, stove, rack), and the robot can reach it (stay
  within x ≤ 0.25). Put the values in `configs/eval.yaml` and `WidePoseConfig.table_xy` with a
  comment `# measured 2026-10-11 from init states 0..45 of put_the_bowl_on_the_plate`. If the
  bounds you need differ from the default, update both.

- [ ] **Step 7: Run the slow test**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_sim/tests/test_wide.py -q -m "sim"`
Expected: PASS. If `placement_failed` is True, print `env.placements` and the draws: the usual
cause is `table_xy` too tight or `min_separation` against a nearby object; fix the config, not
the test.

- [ ] **Step 8: Commit**

```bash
uv run ruff check packages/e2l_sim && uv run ruff format --check packages/e2l_sim
git add packages/e2l_sim configs/eval.yaml packages/e2l_common/src/e2l_common/config.py
git commit -m "feat(sim): move task objects beyond the generation range for wide-pose evaluation"
```

---

### Task 4: Rollout with `setting` and `run`, and `write_results`

**Files:**
- Modify: `packages/e2l_eval/src/e2l_eval/rollout.py`
- Modify: `packages/e2l_eval/tests/test_rollout.py` (append)

**Interfaces:**
- Consumes: `e2l_sim.env.make_env(cfg, episode_index, wide=...)` and env `.placements` (Task 3);
  `EvalConfig.wide.seed_offset` (Task 2).
- Produces:
  - `RolloutResult` fields: `policy, task, successes, seeds, video_paths, run="custom", setting="standard", placements=[]`
  - `SETTINGS = ("standard", "wide")`
  - `write_results(result: RolloutResult, out_dir: Path) -> Path` — writes `results.json` with keys
    `policy, task, run, setting, successes, seeds, video_paths, video_successes, placements, success_rate, wilson_95`.
  - `rollout(policy_path, cfg, sim_cfg, out_dir, n_episodes=None, max_videos=1, setting="standard", run="custom") -> RolloutResult`

- [ ] **Step 1: Write the failing test** — append to `packages/e2l_eval/tests/test_rollout.py`
  (extend the import to `from e2l_eval.rollout import RolloutResult, checkpoint_rename_map, write_results`):

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval/tests/test_rollout.py -q`
Expected: `ImportError: cannot import name 'write_results'`.

- [ ] **Step 3: Implement** in `rollout.py`:

```python
SETTINGS = ("standard", "wide")


@dataclass
class RolloutResult:
    policy: str
    task: str
    successes: list[bool]
    seeds: list[int]
    video_paths: list[str] = field(default_factory=list)
    run: str = "custom"
    setting: str = "standard"
    placements: list[dict] = field(default_factory=list)  # wide only: per-episode object poses

    @property
    def success_rate(self) -> float:
        return sum(self.successes) / len(self.successes) if self.successes else float("nan")


def write_results(result: RolloutResult, out_dir: Path) -> Path:
    """`out_dir/results.json`: the result plus success rate, 95% Wilson interval and the outcome
    of each rendered episode (video i is episode i)."""
    lo, hi = wilson_interval(sum(result.successes), len(result.successes))
    summary = {
        **result.__dict__,
        "video_successes": result.successes[: len(result.video_paths)],
        "success_rate": result.success_rate,
        "wilson_95": [lo, hi],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(summary, indent=2))
    return path
```

  Change `rollout`'s signature and env construction; keep the policy/processor code as is:

```python
def rollout(
    policy_path: str,
    cfg: EvalConfig,
    sim_cfg: SimConfig,
    out_dir: Path,
    n_episodes: int | None = None,
    max_videos: int = 1,
    setting: str = "standard",
    run: str = "custom",
) -> RolloutResult:
    """Run `n_episodes` (default cfg.n_episodes) seeded episodes of a policy on the chosen task,
    saving up to `max_videos` mp4s and `results.json` in `out_dir`. `standard` uses seeds
    cfg.seed + i on LIBERO's init states; `wide` uses seeds cfg.seed + cfg.wide.seed_offset + i
    and moves the task objects after each reset (`e2l_sim.wide`)."""
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    ...  # env vars, ensure_libero_config, lerobot imports exactly as before; add:
    import gymnasium as gym

    n_episodes = n_episodes or cfg.n_episodes
    set_seed(cfg.seed)
    env_cfg = _libero_env_config(sim_cfg)
    ...
    if setting == "wide":
        from e2l_sim.env import make_env as make_sim_env

        env = gym.vector.SyncVectorEnv([partial(make_sim_env, sim_cfg, 0, wide=cfg.wide)])
        start_seed = cfg.seed + cfg.wide.seed_offset
    else:
        env = make_env(env_cfg, n_envs=1)[sim_cfg.suite][sim_cfg.task_id]
        start_seed = cfg.seed
    try:
        ...  # unchanged, but eval_policy(..., start_seed=start_seed)
        placements = list(env.envs[0].placements) if setting == "wide" else []
    finally:
        env.close()

    episodes = info["per_episode"]
    result = RolloutResult(
        policy=policy_path,
        task=f"{sim_cfg.suite}/{sim_cfg.task_id}",
        successes=[bool(e["success"]) for e in episodes],
        seeds=[int(e["seed"]) for e in episodes],
        video_paths=[str(p) for p in info.get("video_paths", [])],
        run=run,
        setting=setting,
        placements=placements,
    )
    write_results(result, out_dir)
    return result
```

  Put `from functools import partial` at module top; `gymnasium` stays a lazy import inside
  `rollout`. Remove the old inline `summary`/`results.json` lines. `baseline.py` is unchanged
  (it uses the defaults).

- [ ] **Step 4: Run tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval -q && uv run ruff check packages/e2l_eval`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval
git commit -m "feat(eval): run rollouts on standard or wide-pose settings and label results"
```

---

### Task 5: `e2l eval rollout --run --setting --videos`

**Files:**
- Modify: `packages/e2l_eval/src/e2l_eval/cli.py`
- Create: `packages/e2l_eval/tests/test_cli.py`

**Interfaces:**
- Consumes: `EvalConfig.runs` (Task 2); `rollout(..., setting=, run=)`, `SETTINGS` (Task 4).
- Produces: `resolve_policy(cfg: EvalConfig, run: str | None, policy: str | None) -> tuple[str, str]`
  (label, checkpoint); output dir `cfg.output_dir / label / setting`.

- [ ] **Step 1: Write the failing tests** — `packages/e2l_eval/tests/test_cli.py`:

```python
import pytest
import typer
from typer.testing import CliRunner

from e2l_common.config import EvalConfig
from e2l_common.paths import repo_root
from e2l_eval import rollout as rollout_module
from e2l_eval.cli import app, resolve_policy

CFG = EvalConfig(runs={"reference": "hub/ref", "B": "ckpt/b"})


def test_resolve_policy():
    assert resolve_policy(CFG, "B", None) == ("B", "ckpt/b")
    assert resolve_policy(CFG, "B", "other/ckpt") == ("B", "other/ckpt")
    assert resolve_policy(CFG, None, "x/y") == ("custom", "x/y")
    assert resolve_policy(CFG, None, None) == ("custom", CFG.policy_path)
    with pytest.raises(typer.BadParameter, match="B, reference"):
        resolve_policy(CFG, "C", None)


def test_rollout_cli_writes_under_run_and_setting(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rollout_module, "rollout", lambda *a, **k: calls.append((a, k)))
    configs = repo_root() / "configs"
    args = ["rollout", "--run", "reference", "--setting", "wide", "--videos", "2",
            "-c", str(configs / "eval.yaml"), "--sim-config", str(configs / "sim.yaml"),
            "--set", f"output_dir={tmp_path}"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    (policy, _, _, out_dir), kwargs = calls[0]
    assert policy == "HuggingFaceVLA/smolvla_libero"
    assert out_dir == tmp_path / "reference" / "wide"
    assert kwargs == {"max_videos": 2, "setting": "wide", "run": "reference"}


def test_rollout_cli_rejects_unknown_setting_and_run(monkeypatch):
    monkeypatch.setattr(rollout_module, "rollout", lambda *a, **k: None)
    configs = repo_root() / "configs"
    base = ["rollout", "-c", str(configs / "eval.yaml"), "--sim-config", str(configs / "sim.yaml")]
    bad_setting = CliRunner().invoke(app, [*base, "--setting", "wid"])
    assert bad_setting.exit_code != 0 and "standard" in bad_setting.output
    bad_run = CliRunner().invoke(app, [*base, "--run", "Z"])
    assert bad_run.exit_code != 0 and "reference" in bad_run.output
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval/tests/test_cli.py -q`
Expected: `ImportError: cannot import name 'resolve_policy'`.

- [ ] **Step 3: Implement** in `cli.py` (replace the `rollout` and `report` commands):

```python
def resolve_policy(cfg: EvalConfig, run: str | None, policy: str | None) -> tuple[str, str]:
    """(run label, checkpoint): an explicit `policy` wins; else `run` is looked up in cfg.runs;
    no run means label `custom` with `policy` or cfg.policy_path."""
    if run is None:
        return "custom", policy or cfg.policy_path
    if policy:
        return run, policy
    if run not in cfg.runs:
        raise typer.BadParameter(
            f"unknown run {run!r}; eval.yaml runs: {', '.join(sorted(cfg.runs))}", param_hint="--run"
        )
    return run, cfg.runs[run]


@app.command()
def rollout(
    policy: str = typer.Option(None, help="Checkpoint path or hub id (overrides --run)."),
    run: str = typer.Option(None, help="Run label from eval.yaml `runs` (A, B, reference)."),
    setting: str = typer.Option("standard", help="standard | wide"),
    videos: int = typer.Option(3, help="Episodes to save as mp4."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    set_: list[str] = set_option(),
) -> None:
    """Evaluate a policy for n_episodes seeded episodes into output_dir/<run>/<setting>/."""
    from e2l_common.config import load_config
    from e2l_eval import rollout as rollout_module

    if setting not in rollout_module.SETTINGS:
        raise typer.BadParameter(
            f"{setting!r} is not one of {', '.join(rollout_module.SETTINGS)}", param_hint="--setting"
        )
    cfg = load_stage_config(config, set_, EvalConfig)
    label, policy_path = resolve_policy(cfg, run, policy)
    sim_cfg = load_config(sim_config, model=SimConfig)
    out_dir = cfg.output_dir / label / setting
    rollout_module.rollout(
        policy_path, cfg, sim_cfg, out_dir, max_videos=videos, setting=setting, run=label
    )


@app.command()
def report(
    results: Path = typer.Option(Path("outputs/eval"), help="Folder of <run>/<setting>/results.json."),
    out_dir: Path = typer.Option(Path("docs/figures")),
) -> None:
    """Results table, success-rate chart and rollout GIF into docs/figures/."""
    from e2l_eval.report import make_report

    for path in make_report(results, out_dir):
        typer.echo(path)
```

  (Calling through `rollout_module.rollout` is what lets the test monkeypatch it.)

- [ ] **Step 4: Run tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval -q && uv run ruff check packages/e2l_eval`
Expected: pass (including `test_skeleton.py::test_cli_help`).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval
git commit -m "feat(eval): select runs and settings from the rollout command"
```

---

### Task 6: `e2l_eval.report`

**Files:**
- Rewrite: `packages/e2l_eval/src/e2l_eval/report.py`
- Create: `packages/e2l_eval/tests/test_report.py`

**Interfaces:**
- Consumes: `results.json` as written by `write_results` (Task 4); `wilson_interval`.
- Produces:
  - `load_results(results_dir: Path) -> list[dict]` — each dict is the file's JSON plus `"dir": Path`; `run`/`setting` fall back to folder names; files without `successes` or with an unknown setting are skipped with a warning.
  - `results_table(rows: list[dict]) -> str` (markdown)
  - `plot_success_rates(rows: list[dict], path: Path) -> Path`
  - `rollout_grid(rows: list[dict], path: Path, fps: int = 10, tile: int = 256, max_seconds: float = 15.0) -> Path | None`
  - `make_report(results_dir: Path, out_dir: Path = Path("docs/figures")) -> list[Path]`

- [ ] **Step 1: Write the failing tests** — `packages/e2l_eval/tests/test_report.py`:

```python
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from e2l_eval.report import load_results, make_report, results_table, rollout_grid


def _write(root: Path, run: str, setting: str, successes, videos=0, placements=(), legacy=False):
    d = root / run / setting
    (d / "videos").mkdir(parents=True)
    paths = []
    for i in range(videos):
        p = d / "videos" / f"eval_episode_{i}.mp4"
        frames = np.full((12, 64, 64, 3), 40 * (i + 1), dtype=np.uint8)
        imageio.mimwrite(p, frames, fps=20)
        paths.append(f"/other/machine/outputs/eval/{run}/{setting}/videos/{p.name}")
    data = {"successes": successes, "video_paths": paths,
            "video_successes": successes[:videos], "placements": list(placements)}
    if not legacy:
        data |= {"run": run, "setting": setting}
    (d / "results.json").write_text(json.dumps(data))


def test_load_results_falls_back_to_folders_and_skips_bad_files(tmp_path):
    _write(tmp_path, "A", "standard", [True, False], legacy=True)
    (tmp_path / "B" / "junk").mkdir(parents=True)
    (tmp_path / "B" / "junk" / "results.json").write_text(json.dumps({"successes": [True]}))
    rows = load_results(tmp_path)
    assert [(r["run"], r["setting"]) for r in rows] == [("A", "standard")]


def test_table_orders_runs_and_marks_missing_cells(tmp_path):
    _write(tmp_path, "reference", "standard", [True] * 8 + [False] * 2)
    _write(tmp_path, "B", "standard", [True] * 5 + [False] * 5)
    _write(tmp_path, "B", "wide", [True, False, False, False],
           placements=[{"placement_failed": True}, {"placement_failed": False}])
    _write(tmp_path, "zeta", "wide", [False])
    table = results_table(load_results(tmp_path))
    lines = table.splitlines()
    assert lines[0].startswith("| Run | standard | wide |")
    body = [ln.split("|")[1].strip() for ln in lines[2:] if ln.startswith("| ")]
    assert body == ["B", "reference", "zeta"]
    assert "8/10 = 80% [49%–94%]" in table  # Wilson 95% for 8/10
    assert "1/4 = 25%" in table
    assert "| — |" in table  # reference has no wide result
    assert "1 of 2 wide episodes" in table  # placements recorded for 2 episodes


def test_grid_finds_videos_next_to_results_and_skips_missing_tiles(tmp_path):
    _write(tmp_path, "B", "standard", [True, True], videos=2)  # no failure video
    _write(tmp_path, "B", "wide", [False], videos=1)
    out = rollout_grid(load_results(tmp_path), tmp_path / "grid.gif", tile=64)
    assert out == tmp_path / "grid.gif"
    gif = Image.open(out)
    assert gif.size == (2 * 64, 64 + 16)  # two tiles (success, wide) plus label strip
    assert gif.n_frames >= 2


def test_make_report_without_videos_writes_table_and_chart_only(tmp_path):
    _write(tmp_path / "res", "reference", "standard", [True, False, True])
    written = make_report(tmp_path / "res", tmp_path / "fig")
    assert sorted(p.name for p in written) == ["results.md", "success_rates.png"]
    assert (tmp_path / "fig" / "success_rates.png").stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval/tests/test_report.py -q`
Expected: `ImportError: cannot import name 'load_results'`.

- [ ] **Step 3: Implement** `packages/e2l_eval/src/e2l_eval/report.py`:

```python
"""Results table, success-rate chart and rollout GIF for the README, from
`<results_dir>/<run>/<setting>/results.json` files written by `e2l eval rollout`."""

import json
import logging
from pathlib import Path

import numpy as np

from e2l_eval.metrics import wilson_interval
from e2l_eval.rollout import SETTINGS

log = logging.getLogger(__name__)
RUN_ORDER = ("A", "B", "reference")
HEADLINE = "B"


def load_results(results_dir: Path) -> list[dict]:
    """Every `<run>/<setting>/results.json` under `results_dir`, with `dir` added; run and
    setting fall back to the folder names for files written before they were recorded."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/*/results.json")):
        data = json.loads(path.read_text())
        data.setdefault("run", path.parent.parent.name)
        data.setdefault("setting", path.parent.name)
        if not data.get("successes") or data["setting"] not in SETTINGS:
            log.warning("skipping %s: no successes or unknown setting", path)
            continue
        rows.append({**data, "dir": path.parent})
    return rows


def _runs(rows: list[dict]) -> list[str]:
    present = {r["run"] for r in rows}
    return [r for r in RUN_ORDER if r in present] + sorted(present - set(RUN_ORDER))


def _cell(row: dict | None) -> str:
    if row is None:
        return "—"
    k, n = sum(row["successes"]), len(row["successes"])
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def results_table(rows: list[dict]) -> str:
    """Markdown table: one row per run, one column per setting, `k/n = rate [95% Wilson]`."""
    by_key = {(r["run"], r["setting"]): r for r in rows}
    lines = ["| Run | " + " | ".join(SETTINGS) + " |", "|---" * (len(SETTINGS) + 1) + "|"]
    for run in _runs(rows):
        cells = [_cell(by_key.get((run, s))) for s in SETTINGS]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    wide = [p for r in rows if r["setting"] == "wide" for p in r.get("placements", [])]
    failed = sum(bool(p.get("placement_failed")) for p in wide)
    if failed:
        lines += ["", f"{failed} of {len(wide)} wide episodes kept the init-state placement "
                  "(no valid wide placement settled)."]
    return "\n".join(lines) + "\n"


def plot_success_rates(rows: list[dict], path: Path) -> Path:
    """Grouped bars (x = run, one bar per setting) with 95% Wilson error bars and n on top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _runs(rows)
    by_key = {(r["run"], r["setting"]): r for r in rows}
    width = 0.8 / len(SETTINGS)
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(runs), 3.2))
    for j, setting in enumerate(SETTINGS):
        xs, rates, err = [], [], [[], []]
        for i, run in enumerate(runs):
            row = by_key.get((run, setting))
            if row is None:
                continue
            k, n = sum(row["successes"]), len(row["successes"])
            lo, hi = wilson_interval(k, n)
            xs.append(i + (j - (len(SETTINGS) - 1) / 2) * width)
            rates.append(k / n)
            err[0].append(k / n - lo)
            err[1].append(hi - k / n)
            ax.text(xs[-1], hi + 0.02, f"n={n}", ha="center", fontsize=7)
        ax.bar(xs, rates, width, yerr=err, capsize=3, label=setting)
    ax.set_xticks(range(len(runs)), runs)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _video(row: dict, index: int) -> Path | None:
    """Video `index` of a result, looked up next to its results.json when the recorded path is
    from another machine or working directory."""
    recorded = Path(row["video_paths"][index])
    for p in (recorded, row["dir"] / "videos" / recorded.name):
        if p.is_file():
            return p
    return None


def _tiles(rows: list[dict]) -> list[tuple[str, Path]]:
    """(label, video) for the headline run: first standard success, first standard failure,
    first wide video."""
    runs = _runs(rows)
    run = HEADLINE if HEADLINE in runs else (runs[0] if runs else None)
    by_key = {(r["run"], r["setting"]): r for r in rows}
    tiles = []
    for setting, want, label in (
        ("standard", True, "success"),
        ("standard", False, "failure"),
        ("wide", None, "wide pose"),
    ):
        row = by_key.get((run, setting))
        if row is None:
            continue
        for i, ok in enumerate(row.get("video_successes", [])):
            if want is None or ok == want:
                video = _video(row, i)
                if video is not None:
                    outcome = "" if want is not None else (" ✓" if ok else " ✗")
                    tiles.append((f"{run}: {label}{outcome}", video))
                    break
    return tiles


def _read_frames(path: Path, fps: int, tile: int, max_frames: int) -> list[np.ndarray]:
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(path)
    step = max(1, round(reader.get_meta_data().get("fps", fps) / fps))
    frames = []
    for i, frame in enumerate(reader):
        if i % step == 0:
            frames.append(np.asarray(Image.fromarray(frame).resize((tile, tile))))
        if len(frames) >= max_frames:
            break
    reader.close()
    return frames


def rollout_grid(
    rows: list[dict], path: Path, fps: int = 10, tile: int = 256, max_seconds: float = 15.0
) -> Path | None:
    """Labelled side-by-side GIF of up to three rollouts (see `_tiles`); None if no videos.
    Shorter clips hold their last frame."""
    from PIL import Image, ImageDraw

    tiles = _tiles(rows)
    if not tiles:
        return None
    clips = [_read_frames(v, fps, tile, int(max_seconds * fps)) for _, v in tiles]
    n = max(len(c) for c in clips)
    strip = 16
    frames = []
    for t in range(n):
        canvas = Image.new("RGB", (tile * len(clips), tile + strip), "white")
        draw = ImageDraw.Draw(canvas)
        for k, ((label, _), clip) in enumerate(zip(tiles, clips)):
            canvas.paste(Image.fromarray(clip[min(t, len(clip) - 1)]), (k * tile, strip))
            draw.text((k * tile + 4, 2), label, fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
    return path


def make_report(results_dir: Path, out_dir: Path = Path("docs/figures")) -> list[Path]:
    """`results.md`, `success_rates.png` and (if any videos) `rollouts.gif` in `out_dir`."""
    rows = load_results(results_dir)
    if not rows:
        raise FileNotFoundError(f"no <run>/<setting>/results.json under {results_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "results.md"
    table.write_text(results_table(rows))
    written = [table, plot_success_rates(rows, out_dir / "success_rates.png")]
    gif = rollout_grid(rows, out_dir / "rollouts.gif")
    return written + ([gif] if gif else [])
```

  Note: `✓`/`✗` may not exist in Pillow's default bitmap font; if the label renders as boxes,
  use `(ok)` / `(fail)` instead.

  This removes the last stub in `report.py`; `e2l_eval.ablation.run_ablation` remains a stub, so
  `test_skeleton.py::check_stubs("e2l_eval") > 0` still holds.

- [ ] **Step 4: Run tests**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_eval -q && uv run ruff check packages/e2l_eval && uv run ruff format --check packages/e2l_eval`
Expected: pass. If `test_table_orders_runs_and_marks_missing_cells` fails on the Wilson string,
compute `wilson_interval(8, 10)` and check the formatting, not the test arithmetic: it is
(0.490, 0.943).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_eval
git commit -m "feat(eval): build the results table, success chart and rollout grid"
```

---

### Task 7: End-to-end check with the reference policy, docs

**Files:**
- Modify: `README.md` (Eval row / commands), `CLAUDE.md` (Commands)

- [ ] **Step 1: Full test suite and lint**

Run: `unset PYTHONPATH; make lint && make test`
Expected: all pass (sim-marked tests run on the laptop).

- [ ] **Step 2: Reference rollouts, 10 episodes each** (needs GPU; `HF_HOME=/mnt/data/hf-cache`):

```bash
unset PYTHONPATH; export HF_HOME=/mnt/data/hf-cache
uv run e2l eval rollout --run reference --setting standard --set n_episodes=10
uv run e2l eval rollout --run reference --setting wide --set n_episodes=10
uv run e2l eval report --out-dir outputs/eval/figures-check
```

Expected: two `results.json` under `outputs/eval/reference/{standard,wide}/`; the report prints
three paths. Open `rollouts.gif` and check the wide tile shows the bowl/plate visibly moved from
the standard layout and nothing floating or inside a fixture. Look at
`outputs/eval/reference/wide/results.json` → `placements`: `placement_failed` should be rare
(< 10 %). Record standard vs wide success rates in the PR description.

- [ ] **Step 3: Docs** — in `CLAUDE.md` Commands add one line:

```markdown
- `uv run e2l eval rollout --run reference --setting wide` · `uv run e2l eval report` (→ docs/figures)
```

  In `README.md` keep the Eval table row; under the evaluation section (or its TODO) add:

```markdown
Each policy is evaluated on LIBERO's standard initial states and on a **wide-pose** setting
where the bowl and plate are moved 8–15 cm and 0.5–1.0 rad beyond anything generation sampled
(`configs/eval.yaml: wide`). `e2l eval rollout --run <A|B|reference> --setting <standard|wide>`
writes `outputs/eval/<run>/<setting>/results.json`; `e2l eval report` turns them into
`docs/figures/results.md`, `success_rates.png` and `rollouts.gif`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs(eval): document wide-pose rollouts and the results report"
```
