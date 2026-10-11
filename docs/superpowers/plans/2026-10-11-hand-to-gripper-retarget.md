# Hand-to-Gripper Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `e2l retarget run` (object-centric `HandSegments` → `RobotSegments` that
`e2l sim replay` executes) plus `e2l retarget calibrate`, which measures the wrist-to-grip offset
and finger closing axis from a recorded demo, so real demos can reach the simulator.

**Architecture:** `hand_to_ee.py` puts the grip point at a fixed offset in the hand frame and, by
default, replaces the hand's rotation with a top-down grasp whose fingers close along the hand's
closing axis projected onto the horizontal plane. `feasibility.py` flags frames that are below
the table, far from the object, or too fast. `calibrate.py` estimates the offset and closing axis
from the thumb/index keypoints of a `DemoTrajectory` while the grip is closed. A round-trip test
rebuilds hand segments from `scripts/scripted_segments.py`, retargets them and replays them in
LIBERO.

**Tech Stack:** Python 3.12, numpy, scipy (via `e2l_common.geometry`), pydantic config, typer,
pytest (`sim`/`slow` markers), LIBERO through `e2l_sim.replay`.

**Spec:** `docs/ego2libero — Project Initialisation Spec.md` (Conventions; `e2l_retarget`) and the
submission checklist, Phase 3: "Implement `hand_to_ee.py`: fixed wrist-to-gripper offset; start
with top-down orientation canonicalisation" and "Implement `feasibility.py` only as far as needed:
workspace bounds and velocity limits; skip IK reachability unless replays fail on reach".

## Decisions

- **Frames.** Segments are in the reference object's frame (`T_obj_hand`); object frames sit on
  the table with z up (LIBERO body frames: origin at the base centre). "Down" is −z of the object
  frame. HandSegments carry no table pose, so workspace checks are made in the object frame.
- **Grip point.** `T_obj_ee = T_obj_hand @ cfg.T_hand_ee.T`: the offset is in the hand (palm)
  frame, whose origin is the wrist and whose x points from the wrist to the middle knuckle.
- **Top-down canonicalisation** (default on). Rotation columns: z = (0,0,−1); x = horizontal
  projection of `R_obj_hand @ cfg.closing_axis` (new field, hand frame, default (0,1,0): across
  the knuckles), normalised; y = z × x. If the projection is shorter than 0.2 (axis near
  vertical), reuse the previous frame's x (object x for the first frame). The Panda gripper is
  symmetric under a half turn about z, so x is flipped to stay within 90° of the previous x (the
  first frame: of object x). For the scripted demo's rotation `diag(1,−1,−1)` this gives exactly
  x = object x.
- **Feasibility.** Per frame: `z ≥ cfg.min_height` (−0.02 m), `‖p‖ ≤ cfg.max_object_distance`
  (0.6 m), and from the previous frame linear speed ≤ `max_linear_speed`, angular speed ≤
  `max_angular_speed`. No IK. Replay does not read `feasible`; it is reported by the CLI.
  `RetargetConfig.workspace_min/max` (table frame, unusable here) are replaced by
  `min_height` and `max_object_distance`.
- **Gripper.** `grip` 1 → +1.0 closed, 0 → −1.0 open (robosuite convention).
- **Calibration.** Over hand-valid frames where the grip is closed: offset = median pinch point
  (thumb-tip/index-tip midpoint) in the hand frame; closing axis = normalised mean of the unit
  thumb→index direction in the hand frame. `e2l retarget calibrate <demo>` prints the two
  `retarget.yaml` lines. Grip comes from `e2l_segment.events.aperture_to_grip`, so e2l-retarget
  gains an e2l-segment dependency.

## Global Constraints

- Poses are 4x4 float64 `T_a_b` (`p_a = T_a_b @ p_b`); no quaternions in files or configs;
  metres, radians, seconds. SE(3) helpers from `e2l_common.geometry` (`make_T`, `inv_T`,
  `rotvec_to_matrix`, `matrix_to_rotvec` all accept batches).
- Gripper: segment `grip` 0/1 → robot `gripper` −1 open / +1 closed.
- LIBERO / robosuite / MuJoCo are imported only inside `e2l_sim`; tests reach them through
  `e2l_sim.replay`.
- Run `unset PYTHONPATH` before any `uv run`; sim tests need `MUJOCO_GL=egl` (set by `make`).
- Lint: `uv run ruff check . && uv run ruff format --check .` (line 100). Type hints expected.
- Commits: `<type>(<scope>): <subject>`, scope `retarget`, `common` or `configs`, header ≤ 72
  chars. **No attribution:** no `Co-Authored-By:` or `Claude-Session:` trailers, ever.
- Branch `feat/hand-to-gripper-retarget` (worktree `.worktrees/hand-to-gripper-retarget`, venv
  set up by `make worktree`). Never commit to `main`; never `git add` `data/` or `outputs/`.

## Review Focus

1. A closing axis near vertical (hand rolled on its side) → no NaN rotation; previous x reused.
   (Task 1 `test_top_down_keeps_x_continuous_and_handles_vertical_axis`.)
2. Half-turn flips of the closing axis between frames → no 180° wrist spin.
   (Same test.)
3. A single-frame segment → `check_segment` must not crash on empty diffs.
   (Task 2 `test_check_segment_single_frame`.)
4. A demo with no closed-grip valid frame → `calibrate` raises a `ValueError` naming the demo.
   (Task 3 test.)
5. NaN keypoints on some closed frames → excluded, not propagated into the estimate.
   (Task 3 test.)

---

### Task 1: `hand_to_ee` with top-down canonicalisation

**Files:**
- Modify: `packages/e2l_retarget/src/e2l_retarget/hand_to_ee.py` (implement `hand_to_ee`, add
  `top_down_rotations`; leave `retarget_segments` stubbed for Task 2)
- Modify: `packages/e2l_common/src/e2l_common/config.py` (`RetargetConfig.closing_axis`)
- Modify: `configs/retarget.yaml`
- Test: `packages/e2l_retarget/tests/test_hand_to_ee.py` (create)

**Interfaces:**
- Produces: `top_down_rotations(closing: np.ndarray) -> np.ndarray` ((M,3) → (M,3,3)),
  `hand_to_ee(T_obj_hand: np.ndarray, cfg: RetargetConfig) -> np.ndarray` ((M,4,4) → (M,4,4)),
  `RetargetConfig.closing_axis: Vec3 = (0.0, 1.0, 0.0)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from e2l_common.config import Pose, RetargetConfig
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_retarget.hand_to_ee import hand_to_ee, top_down_rotations

R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])  # scripted demo's grip-site rotation
# Hand axes in the object frame: fingers (x) along +y, palm normal (z) down, so the hand's y
# (across the knuckles) lies along the object's x.
R_HAND = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def test_top_down_from_horizontal_closing_axis():
    R = top_down_rotations(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(R[0], R_TOP_DOWN, atol=1e-12)
    np.testing.assert_allclose(R[1][:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    for r in R:
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(r), 1.0)
        np.testing.assert_allclose(r[:, 2], [0.0, 0.0, -1.0])


def test_top_down_keeps_x_continuous_and_handles_vertical_axis():
    closing = np.array([[-1.0, 0.1, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    R = top_down_rotations(closing)
    assert np.isfinite(R).all()
    x = R[:, :, 0]
    assert x[0] @ [1.0, 0.0, 0.0] > 0  # first frame flipped towards object x
    assert all(x[i] @ x[i - 1] >= 0 for i in range(1, len(x)))  # no half-turn jumps
    np.testing.assert_allclose(x[2], x[1])  # near-vertical axis: previous x reused


def test_hand_to_ee_applies_offset_in_hand_frame_and_snaps_rotation():
    tilt = rotvec_to_matrix(np.array([0.3, 0.0, 0.0]))  # roll about object x keeps hand y on x
    R = tilt @ R_HAND
    T_obj_hand = make_T(R, np.array([0.05, 0.0, 0.15]))[None]
    cfg = RetargetConfig(T_hand_ee=Pose(xyz=(0.09, 0.0, 0.02)), closing_axis=(0.0, 1.0, 0.0))
    T = hand_to_ee(T_obj_hand, cfg)
    np.testing.assert_allclose(T[0, :3, 3], T_obj_hand[0, :3, 3] + R @ [0.09, 0.0, 0.02])
    np.testing.assert_allclose(T[0, :3, :3], R_TOP_DOWN, atol=1e-12)


def test_hand_to_ee_without_canonicalisation_composes_offset():
    T_obj_hand = make_T(R_HAND, np.array([0.0, 0.1, 0.2]))[None]
    off = Pose(xyz=(0.0, 0.0, 0.1), rotvec=(0.0, 0.0, 0.5))
    T = hand_to_ee(T_obj_hand, RetargetConfig(T_hand_ee=off, canonicalise_top_down=False))
    np.testing.assert_allclose(T[0], T_obj_hand[0] @ off.T)
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_retarget/tests/test_hand_to_ee.py -v`
Expected: FAIL (`cannot import name 'top_down_rotations'`, or a pydantic error for `closing_axis`).

- [ ] **Step 3: Config field**

In `RetargetConfig` (config.py), after `canonicalise_top_down`:

```python
    closing_axis: Vec3 = (0.0, 1.0, 0.0)  # hand-frame axis the fingers close along
```

In `configs/retarget.yaml` replace the `T_hand_ee` and `canonicalise_top_down` lines with:

```yaml
# Wrist (palm frame origin; x towards the middle knuckle) to the grip point, in the hand frame.
# Measure it from a recorded demo with `e2l retarget calibrate <demo_id>`.
T_hand_ee: {xyz: [0.08, 0.0, 0.0], rotvec: [0.0, 0.0, 0.0]}
canonicalise_top_down: true              # grip straight down; fingers close along closing_axis
closing_axis: [0.0, 1.0, 0.0]            # hand frame; `e2l retarget calibrate` measures it too
```

- [ ] **Step 4: Implement** — in `hand_to_ee.py` keep the imports `retarget_segments` needs,
add `_UP`, `_MIN_HORIZONTAL`, `top_down_rotations`, and replace the `hand_to_ee` stub:

```python
_UP = np.array([0.0, 0.0, 1.0])
_MIN_HORIZONTAL = 0.2  # closing axis with a shorter horizontal projection is ambiguous


def top_down_rotations(closing: np.ndarray) -> np.ndarray:
    """Rotations (M,3,3) with z straight down and x along the horizontal projection of the
    closing direction (M,3) in the object frame (z up). A near-vertical closing direction reuses
    the previous x (object x before any). The gripper is symmetric under a half turn about z, so
    x is flipped to stay within 90 deg of the previous x (first frame: object x)."""
    x_prev = np.array([1.0, 0.0, 0.0])
    z = -_UP
    R = np.empty((len(closing), 3, 3))
    for i, a in enumerate(np.asarray(closing, dtype=np.float64)):
        h = a - (a @ _UP) * _UP
        n = np.linalg.norm(h)
        x = h / n if n >= _MIN_HORIZONTAL else x_prev
        if x @ x_prev < 0:
            x = -x
        R[i] = np.column_stack([x, np.cross(z, x), z])
        x_prev = x
    return R


def hand_to_ee(T_obj_hand: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """T_obj_ee (M,4,4) = T_obj_hand (M,4,4) @ cfg.T_hand_ee.T: the grip point at a fixed offset
    in the hand frame. With `cfg.canonicalise_top_down` the rotation becomes a top-down grasp
    whose fingers close along the hand's `cfg.closing_axis` (see `top_down_rotations`)."""
    T_obj_ee = np.asarray(T_obj_hand, dtype=np.float64) @ cfg.T_hand_ee.T
    if cfg.canonicalise_top_down:
        closing = T_obj_hand[:, :3, :3] @ np.asarray(cfg.closing_axis, dtype=np.float64)
        T_obj_ee[:, :3, :3] = top_down_rotations(closing)
    return T_obj_ee
```

- [ ] **Step 5: Run tests → PASS; also** `uv run pytest packages/e2l_common -q` (repo configs
must still validate) **and lint.**

- [ ] **Step 6: Commit** — `feat(retarget): map hand poses to top-down gripper poses`
(no attribution trailers).

---

### Task 2: feasibility checks and `retarget_segments`

**Files:**
- Modify: `packages/e2l_retarget/src/e2l_retarget/feasibility.py`
- Modify: `packages/e2l_retarget/src/e2l_retarget/hand_to_ee.py` (`retarget_segments`)
- Modify: `packages/e2l_retarget/src/e2l_retarget/cli.py` (`run` prints a summary)
- Modify: `packages/e2l_common/src/e2l_common/config.py`, `configs/retarget.yaml`
- Modify: `packages/e2l_retarget/tests/test_skeleton.py`
- Test: `packages/e2l_retarget/tests/test_feasibility.py`, extend `test_hand_to_ee.py`

**Interfaces:**
- Consumes: Task 1 `hand_to_ee`.
- Produces: `check_segment(T_obj_ee: np.ndarray, t: np.ndarray, cfg: RetargetConfig) -> np.ndarray`
  (bool (M,)); `retarget_segments(segments: HandSegments, cfg: RetargetConfig) -> RobotSegments`;
  `GRIPPER_OPEN = -1.0`, `GRIPPER_CLOSED = 1.0` in `hand_to_ee.py`;
  `RetargetConfig.min_height = -0.02`, `.max_object_distance = 0.6`.

- [ ] **Step 1: Write the failing tests**

`test_feasibility.py`:

```python
import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_retarget.feasibility import check_segment

CFG = RetargetConfig()
FPS = 30.0


def _line(n: int = 10) -> np.ndarray:
    T = np.tile(make_T(np.diag([1.0, -1.0, -1.0]), np.array([0.05, 0.0, 0.1])), (n, 1, 1))
    T[:, 2, 3] = np.linspace(0.15, 0.05, n)  # 0.1 m over 9 frames = 0.33 m/s
    return T


def test_smooth_approach_is_feasible():
    assert check_segment(_line(), np.arange(10) / FPS, CFG).all()


def test_fast_jump_and_spin_are_flagged_at_the_frame_they_arrive():
    T = _line()
    T[5:, 0, 3] += 0.1  # 3 m/s into frame 5
    T[8:, :3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, 1.0])) @ T[8, :3, :3]  # 30 rad/s
    ok = check_segment(T, np.arange(10) / FPS, CFG)
    assert not ok[5] and not ok[8]
    assert ok[[0, 1, 2, 3, 4, 6, 7, 9]].all()


def test_below_table_and_far_from_object_are_flagged():
    T = _line(3)
    T[1, 2, 3] = -0.05
    T[2, :3, 3] = [0.7, 0.0, 0.1]
    ok = check_segment(T, np.arange(3) / 0.5, CFG)  # slow clock: only the position limits bite
    np.testing.assert_array_equal(ok, [True, False, False])


def test_check_segment_single_frame():
    assert check_segment(_line(1), np.zeros(1), CFG).tolist() == [True]
```

Append to `test_hand_to_ee.py`:

```python
from e2l_common.schemas import HandSegment, HandSegments  # noqa: E402  (move to the top)
from e2l_retarget.hand_to_ee import retarget_segments  # noqa: E402  (move to the top)


def test_retarget_segments_maps_poses_grip_and_feasibility():
    cfg = RetargetConfig(T_hand_ee=Pose(xyz=(0.09, 0.0, 0.02)))
    T_hand = make_T(R_HAND, np.array([0.0, 0.0, 0.15]))
    segs = HandSegments(
        demo_id="20261011_test_000",
        fps=30.0,
        segments=[
            HandSegment(
                ref_object="bowl",
                t_start=0.0,
                t_end=0.1,
                T_obj_hand=np.tile(T_hand, (4, 1, 1)),
                grip=np.array([0, 0, 0, 1], np.int8),
            ),
            HandSegment(
                ref_object="plate",
                t_start=0.1,
                t_end=0.2,
                T_obj_hand=np.tile(T_hand, (3, 1, 1)),
                grip=np.array([1, 1, 0], np.int8),
            ),
        ],
    )
    robot = retarget_segments(segs, cfg)
    assert robot.demo_id == segs.demo_id and robot.fps == 30.0
    assert [s.ref_object for s in robot.segments] == ["bowl", "plate"]
    np.testing.assert_array_equal(robot.segments[0].gripper, [-1.0, -1.0, -1.0, 1.0])
    np.testing.assert_array_equal(robot.segments[1].gripper, [1.0, 1.0, -1.0])
    np.testing.assert_allclose(
        robot.segments[0].T_obj_ee, hand_to_ee(segs.segments[0].T_obj_hand, cfg)
    )
    assert robot.segments[0].feasible.dtype == bool and robot.segments[1].feasible.all()
```

(Put the two new imports at the top of the file with the others; the `noqa` comments above are
only there so the snippet reads standalone.)

In `test_skeleton.py` replace the stub test with:

```python
def test_no_stubs_remain():
    assert check_stubs("e2l_retarget") == 0
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/e2l_retarget -v` → FAIL.

- [ ] **Step 3: Config** — in `RetargetConfig` replace `workspace_min` / `workspace_max` with:

```python
    min_height: float = -0.02  # m, grip point above the object's base plane (the table)
    max_object_distance: float = 0.6  # m, grip point to the reference object's origin
```

and in `configs/retarget.yaml` replace the two `workspace_*` lines with:

```yaml
min_height: -0.02                        # m, object frame: grip point not below the table
max_object_distance: 0.6                 # m, grip point within this of the reference object
```

(`grep -rn "workspace_min\|workspace_max" packages configs scripts` must then return nothing.)

- [ ] **Step 4: Implement `feasibility.py`**

```python
"""Workspace and speed checks for retargeted segments."""

import numpy as np

from e2l_common.config import RetargetConfig
from e2l_common.geometry import matrix_to_rotvec


def check_segment(T_obj_ee: np.ndarray, t: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """Per-frame feasibility (M,) bool for gripper poses in the reference object's frame
    (M,4,4) at times t (M,) seconds: not below `cfg.min_height` (object frames sit on the table,
    z up), within `cfg.max_object_distance` of the object, and reached from the previous frame
    within `cfg.max_linear_speed` (m/s) and `cfg.max_angular_speed` (rad/s). IK reachability
    is left to replay."""
    p = T_obj_ee[:, :3, 3]
    ok = (p[:, 2] >= cfg.min_height) & (np.linalg.norm(p, axis=1) <= cfg.max_object_distance)
    if len(p) > 1:
        dt = np.diff(t)
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
        R = T_obj_ee[:, :3, :3]
        dR = np.swapaxes(R[:-1], 1, 2) @ R[1:]
        w = np.linalg.norm(matrix_to_rotvec(dR), axis=1) / dt
        ok[1:] &= (v <= cfg.max_linear_speed) & (w <= cfg.max_angular_speed)
    return ok
```

- [ ] **Step 5: Implement `retarget_segments`** (drop the `not_implemented` import; import
`RobotSegment` and `check_segment`):

```python
GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0


def retarget_segments(segments: HandSegments, cfg: RetargetConfig) -> RobotSegments:
    """Map every HandSegment to a RobotSegment: poses via `hand_to_ee`, grip {0,1} to robosuite
    gripper {-1,+1}, and `feasible` from `feasibility.check_segment` at the demo fps."""
    robot = []
    for seg in segments.segments:
        T_obj_ee = hand_to_ee(seg.T_obj_hand, cfg)
        t = np.arange(len(T_obj_ee)) / segments.fps
        robot.append(
            RobotSegment(
                ref_object=seg.ref_object,
                T_obj_ee=T_obj_ee,
                gripper=np.where(seg.grip > 0, GRIPPER_CLOSED, GRIPPER_OPEN),
                feasible=check_segment(T_obj_ee, t, cfg),
            )
        )
    return RobotSegments(demo_id=segments.demo_id, fps=segments.fps, segments=robot)
```

In `cli.py` `run`, after saving:

```python
    typer.echo(
        f"{demo_id}: "
        + " | ".join(
            f"{s.ref_object} {len(s.T_obj_ee)} frames, {s.feasible.mean():.0%} feasible"
            for s in robot.segments
        )
    )
```

- [ ] **Step 6: Run** `uv run pytest packages/e2l_retarget packages/e2l_common -q` → PASS; lint.

- [ ] **Step 7: Commit** — `feat(retarget): retarget hand segments with feasibility flags`.

---

### Task 3: `e2l retarget calibrate` — measure the grip offset and closing axis

**Files:**
- Create: `packages/e2l_retarget/src/e2l_retarget/calibrate.py`
- Modify: `packages/e2l_retarget/src/e2l_retarget/cli.py` (new `calibrate` command)
- Modify: `packages/e2l_retarget/pyproject.toml` (add `e2l-segment`), then `uv lock`
- Test: `packages/e2l_retarget/tests/test_calibrate.py`

**Interfaces:**
- Consumes: `DemoTrajectory`; `e2l_segment.events.aperture_to_grip(t, aperture, cfg)`.
- Produces: `estimate_grip(demo: DemoTrajectory, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_common.schemas import DemoTrajectory
from e2l_retarget.calibrate import estimate_grip

THUMB = np.array([0.08, -0.01, -0.02])  # hand frame, metres
INDEX = np.array([0.10, 0.01, -0.02])


def _demo(n: int = 20) -> DemoTrajectory:
    rng = np.random.default_rng(0)
    T = make_T(
        rotvec_to_matrix(rng.normal(scale=0.5, size=(n, 3))), rng.normal(scale=0.1, size=(n, 3))
    )
    kp = np.zeros((n, 21, 3))
    kp[:, 4] = np.einsum("nij,j->ni", T[:, :3, :3], THUMB) + T[:, :3, 3]
    kp[:, 8] = np.einsum("nij,j->ni", T[:, :3, :3], INDEX) + T[:, :3, 3]
    return DemoTrajectory(
        demo_id="20261011_test_000",
        fps=30.0,
        object_names=["bowl"],
        t=np.arange(n) / 30.0,
        T_table_hand=T,
        hand_valid=np.ones(n, bool),
        keypoints_table=kp,
        aperture=np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1),
        T_table_obj=np.tile(np.eye(4), (n, 1, 1, 1)),
        obj_valid=np.ones((n, 1), bool),
    )


def test_estimate_grip_recovers_pinch_point_and_axis_in_hand_frame():
    demo = _demo()
    closed = np.zeros(20, bool)
    closed[5:15] = True
    xyz, axis, n = estimate_grip(demo, closed)
    np.testing.assert_allclose(xyz, (THUMB + INDEX) / 2, atol=1e-9)
    want = (INDEX - THUMB) / np.linalg.norm(INDEX - THUMB)
    np.testing.assert_allclose(axis, want, atol=1e-9)
    assert n == 10


def test_estimate_grip_skips_invalid_and_nan_frames():
    demo = _demo()
    demo.hand_valid[5] = False
    demo.keypoints_table[6, 8] = np.nan
    demo.keypoints_table[7, 4] = 99.0  # garbage, but frame 7 is open so it is not used
    closed = np.zeros(20, bool)
    closed[[5, 6, 8, 9]] = True
    xyz, _, n = estimate_grip(demo, closed)
    assert n == 2
    np.testing.assert_allclose(xyz, (THUMB + INDEX) / 2, atol=1e-9)


def test_estimate_grip_without_closed_frames_raises():
    with pytest.raises(ValueError, match="20261011_test_000"):
        estimate_grip(_demo(), np.zeros(20, bool))
```

- [ ] **Step 2: Run to verify failure** → FAIL (`No module named 'e2l_retarget.calibrate'`).

- [ ] **Step 3: Implement `calibrate.py`**

```python
"""Measure the wrist-to-grip offset and the finger closing axis from a recorded demo."""

import numpy as np

from e2l_common.schemas import DemoTrajectory

THUMB_TIP, INDEX_TIP = 4, 8


def estimate_grip(demo: DemoTrajectory, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Over hand-valid frames where the grip is `closed` (N,) bool and both fingertips are
    finite: the median pinch point (thumb/index tip midpoint) and the normalised mean
    thumb-to-index direction, both in the hand frame, plus the number of frames used. These are
    `T_hand_ee.xyz` and `closing_axis` for retarget.yaml. ValueError if no frame qualifies."""
    tips = demo.keypoints_table[:, [THUMB_TIP, INDEX_TIP]]
    use = demo.hand_valid & closed & np.isfinite(tips).all(axis=(1, 2))
    if not use.any():
        raise ValueError(f"{demo.demo_id}: no hand-valid frame with the grip closed")
    R = demo.T_table_hand[use, :3, :3]
    origin = demo.T_table_hand[use, :3, 3]
    thumb, index = tips[use, 0], tips[use, 1]
    pinch = np.einsum("nji,nj->ni", R, (thumb + index) / 2 - origin)  # R^T (p - o)
    d = np.einsum("nji,nj->ni", R, index - thumb)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    axis = d.mean(axis=0)
    return np.median(pinch, axis=0), axis / np.linalg.norm(axis), int(use.sum())
```

- [ ] **Step 4: Dependency** — in `packages/e2l_retarget/pyproject.toml` add `"e2l-segment"` to
`dependencies` and `e2l-segment = { workspace = true }` under `[tool.uv.sources]`; run
`unset PYTHONPATH; uv lock && uv sync --all-groups`; `git diff --stat uv.lock` must be small (the
e2l-retarget entry only).

- [ ] **Step 5: CLI command** (in `cli.py`):

```python
@app.command()
def calibrate(
    demo_id: str,
    segment_config: Path = typer.Option(
        Path("configs/segment.yaml"), help="Segment config (grip thresholds)."
    ),
) -> None:
    """Measure T_hand_ee and closing_axis from data/perception/<demo_id>; prints the two
    retarget.yaml lines to paste."""
    from e2l_common.config import SegmentConfig, load_config
    from e2l_common.schemas import DemoTrajectory
    from e2l_retarget.calibrate import estimate_grip
    from e2l_segment.events import aperture_to_grip

    demo = DemoTrajectory.load(DataPaths.default().perception(demo_id))
    seg_cfg = load_config(segment_config, model=SegmentConfig)
    closed = aperture_to_grip(demo.t, demo.aperture, seg_cfg) == 1
    xyz, axis, n = estimate_grip(demo, closed)
    typer.echo(f"# {demo_id}: {n} closed-grip frames")
    typer.echo(
        f"T_hand_ee: {{xyz: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}], rotvec: [0.0, 0.0, 0.0]}}"
    )
    typer.echo(f"closing_axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
```

(`ruff format` will wrap the long echo line; run it.)

- [ ] **Step 6: Run** `uv run pytest packages/e2l_retarget -q`, `uv run e2l retarget calibrate --help`
(exit 0), lint → PASS.

- [ ] **Step 7: Commit** — `feat(retarget): measure grip offset and closing axis from a demo`.

---

### Task 4: round trip through the scripted demo, replayed in LIBERO

**Files:**
- Test: `packages/e2l_retarget/tests/test_roundtrip.py` (create)

**Interfaces:**
- Consumes: Tasks 1–2 `retarget_segments`; `scripts/scripted_segments.py` `scripted(demo_id) -> RobotSegments`
  (rotation `diag(1,-1,-1)`, i.e. x = object x); `e2l_sim.replay.replay(segments, cfg) -> SimEpisode`;
  `e2l_sim.libero_setup.libero_available()`; `e2l_common.paths.repo_root()`.

- [ ] **Step 1: Write the tests**

```python
import importlib.util
import os

import numpy as np
import pytest

from e2l_common.config import Pose, RetargetConfig, SimConfig, load_config
from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_common.paths import repo_root
from e2l_common.schemas import HandSegment, HandSegments, RobotSegments
from e2l_retarget.hand_to_ee import retarget_segments
from e2l_sim.libero_setup import libero_available

# Hand axes in the object frame: fingers along +y, palm down, knuckle axis (hand y) along x.
R_HAND = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
CFG = RetargetConfig(T_hand_ee=Pose(xyz=(0.09, 0.0, 0.02)), closing_axis=(0.0, 1.0, 0.0))


def _scripted() -> RobotSegments:
    path = repo_root() / "scripts" / "scripted_segments.py"
    spec = importlib.util.spec_from_file_location("scripted_segments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scripted("20261011_retarget_000")


def _as_hand(robot: RobotSegments) -> HandSegments:
    """Hand segments a person could have shown: the hand rolls up to 0.3 rad about the object x
    while its knuckle axis stays along x, and its grip point follows the scripted gripper."""
    off = np.asarray(CFG.T_hand_ee.xyz)
    hand = []
    for s in robot.segments:
        m = len(s.T_obj_ee)
        roll = np.zeros((m, 3))
        roll[:, 0] = np.linspace(-0.3, 0.3, m)
        R = rotvec_to_matrix(roll) @ R_HAND
        p = s.T_obj_ee[:, :3, 3] - np.einsum("nij,j->ni", R, off)
        hand.append(
            HandSegment(
                ref_object=s.ref_object,
                t_start=0.0,
                t_end=(m - 1) / robot.fps,
                T_obj_hand=make_T(R, p),
                grip=(s.gripper > 0).astype(np.int8),
            )
        )
    return HandSegments(demo_id=robot.demo_id, fps=robot.fps, segments=hand)


def test_retarget_recovers_scripted_gripper_poses():
    scripted = _scripted()
    robot = retarget_segments(_as_hand(scripted), CFG)
    for got, want in zip(robot.segments, scripted.segments, strict=True):
        assert got.ref_object == want.ref_object
        np.testing.assert_allclose(got.T_obj_ee, want.T_obj_ee, atol=1e-9)
        np.testing.assert_array_equal(got.gripper, want.gripper)
        assert got.feasible.all()


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.skipif(not libero_available(), reason="needs LIBERO and its assets")
def test_retargeted_hand_demo_replays_successfully():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from e2l_sim.replay import replay

    robot = retarget_segments(_as_hand(_scripted()), CFG)
    episode = replay(robot, load_config(repo_root() / "configs" / "sim.yaml", model=SimConfig))
    assert episode.success
```

- [ ] **Step 2: Run** `uv run pytest packages/e2l_retarget/tests/test_roundtrip.py -v` (with
`MUJOCO_GL=egl`): both PASS. If the replay test fails while the pose test passes, the defect is
in the retarget output format, not in replay (the scripted demo itself replays successfully:
`packages/e2l_sim/tests/test_replay.py::test_scripted_bowl_onto_plate_succeeds`). Investigate
before changing anything.

- [ ] **Step 3: Full suite and lint** — `unset PYTHONPATH; uv run pytest -q` and ruff → PASS.

- [ ] **Step 4: Commit** — `test(retarget): replay a retargeted hand demo in LIBERO`.
