# Hand Trajectory and Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement every perception and segmentation function still stubbed (`hand/pose.py`,
`smoothing.py`, `pipeline.py`, `viz.py`, `events.py`, `assign.py`, `segment.py`, `segment viz`)
with synthetic-data tests, so tonight's first real phone demos go straight through
`e2l perception run → viz → e2l segment run → viz` and only threshold tuning remains.

**Architecture:** Perception splits into a thin, untested-by-unit video loop
(`observe_frames`: markers + MediaPipe + PnP per frame) and a pure, fully tested assembler
(`assemble_trajectory`: camera → table frame, short-gap filling, optional Kalman + RTS smoothing,
aperture). `T_table_hand` is derived from the (smoothed) table-frame keypoints with the same palm
frame used for PnP, so the pose and keypoints can never disagree. Segmentation is pure numpy:
hysteresis + dwell → grip events → cut → re-express relative to the reference object.

**Tech Stack:** Python 3.12, numpy, OpenCV (`solvePnP` SQPNP), MediaPipe (existing detector),
matplotlib (Agg), imageio, pydantic configs, typer, pytest.

**Spec:** `docs/ego2libero — Project Initialisation Spec.md` (Conventions; `e2l_perception`,
`e2l_segment` sections) and `docs/ego2libero — Submission Checklist.md` Phase 2. No separate
design doc; design decisions are under "Decisions" below.

## Decisions

- **Hand pose.** PnP of MediaPipe's 21 metric world landmarks against their 2D detections gives
  `T_cam_world` (MediaPipe's world-landmark frame). The hand frame is the palm frame
  (`palm.compute_palm_frame` axes) with origin at the wrist landmark (index 0):
  `T_world_hand = make_T(compute_palm_frame(world), world[0])`.
  `keypoints_cam = T_cam_world · world`. Aperture = ‖kp[4] − kp[8]‖ (thumb tip – index tip).
- **Table frame.** `T_table_x = inv(T_cam_table) @ T_cam_x`, per frame. A hand frame is valid
  only when the hand pose *and* the board are both detected that frame.
- **Gaps.** Interior runs of invalid hand frames lasting at most `max_gap_s` (new config field,
  default 0.2 s) become valid and are filled; longer gaps stay invalid. Leading/trailing invalid
  runs stay invalid.
- **Smoothing.** Constant-velocity Kalman filter per keypoint coordinate (white-acceleration
  noise), forward pass with missing measurements, then an RTS backward pass. All 63 scalar
  tracks share one model and one validity mask, so they share one covariance; states are
  stacked `(n, M)`. The ported 126-state `JointHandEKF` is not used (its noise values are in
  unknown units and its P history costs ~127 kB/frame). Without smoothing, short gaps are filled
  by linear interpolation. Fallback per the checklist if RTS misbehaves on real data: a
  zero-phase low-pass filter (not in this plan).
- **Config.** In `PerceptionConfig` replace `ekf_process_noise` / `ekf_measurement_noise` with
  `smooth_accel_noise: float = 1.0` (m²/s³, white-acceleration spectral density) and
  `smooth_meas_std: float = 0.01` (m), and add `max_gap_s: float = 0.2`. Nothing outside config,
  the YAML and this plan uses the old fields (Task 2 verifies with grep).
- **Invalid frames** hold NaN in `T_table_hand`, `keypoints_table`, `aperture`, and in
  `T_table_obj` where `obj_valid` is False. Consumers must use the validity masks.
- **Segments.** Bounds are `[0, e1, e2, …, N−1]` over grip-event frame indices; segment *s* covers
  frames `bounds[s]..bounds[s+1]` inclusive (neighbours share the event frame), keeping only
  hand-valid frames; segments with fewer than 2 valid frames are dropped. The segment closing at
  event *e* uses the object nearest the hand at *e*, held at its pose at *e*; the trailing
  segment (after the last event) reuses the last event's object and pose; a demo with no events
  is one segment closed at its last frame. At a release the carried object is nearest the hand,
  so release events exclude the object referenced by the preceding grasp (the held one); the
  reference is then the place target. This matches `scripts/scripted_segments.py`
  (approach→grasp ends with grip 1; carry→release ends with grip 0; then a retreat).
- **Dwell.** Runs of constant grip shorter than `min_dwell_s` (run length × median frame period)
  are flipped, shortest first, until none remain or one run is left.
- **Not in this plan** (needs real data): the per-demo depth-scale correction at grasps, and
  threshold tuning. The per-demo valid-frame fraction is printed by `e2l perception run`.

## Global Constraints

- Poses are 4x4 float64 `T_a_b` (`p_a = T_a_b @ p_b`); no quaternions in files/configs; metres,
  radians, seconds; timestamps = frame index / fps.
- SE(3) helpers come from `e2l_common.geometry` (`make_T`, `inv_T`, `compose`, `rotvec_to_matrix`,
  …); do not re-implement them.
- Gripper: `aperture` (m) → `grip` int8 0 open / 1 closed.
- No LIBERO / robosuite / MuJoCo imports in these packages.
- Run `unset PYTHONPATH` before any `uv run`. Use `MPLBACKEND=Agg` behaviour: call
  `matplotlib.use("Agg")` inside plotting functions before importing pyplot.
- Lint: `uv run ruff check . && uv run ruff format --check .` (line 100). Type hints expected.
- Commits: `<type>(<scope>): <subject>`, scope `perception`, `segment`, `common` or `configs`,
  header ≤ 72 chars, footer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `feat/hand-trajectory-segments` (exists; worktree `.worktrees/hand-trajectory-segments`,
  venv set up by `make worktree`). Never commit to `main`. Never `git add` `data/` or `outputs/`.
- Remove the `not_implemented` import from every file whose last stub you implement.

## Review Focus

1. A demo whose aperture is NaN for long stretches (hand out of view) → grip holds its last state
   through NaNs; no spurious events. (Task 5 test `test_nan_holds_state`.)
2. A frame where the hand is detected but the board is not → hand invalid that frame (the table
   frame is unknown). (Task 3 test `test_board_missing_invalidates_hand`.)
3. An object never seen at all → `obj_valid[:, k]` all False, poses NaN, and `reference_object`
   skips it instead of crashing; if no object is ever seen, `ValueError` with the demo id.
   (Task 6 tests.)
4. Calibration made at a different resolution than the video → `ValueError` before any MediaPipe
   work. (Task 3 test `test_observe_frames_rejects_mismatched_intrinsics`.)
5. A grip event on the last frame, or no events at all → no zero/one-frame segments, still a
   valid `HandSegments`. (Task 6 tests.)

---

### Task 1: `hand/pose.py` — metric hand pose by PnP

**Files:**
- Modify: `packages/e2l_perception/src/e2l_perception/hand/pose.py`
- Test: `packages/e2l_perception/tests/test_pose.py` (create)

**Interfaces:**
- Consumes: `e2l_perception.hand.palm.compute_palm_frame(pts) -> R (3,3)`,
  `e2l_perception.markers.pnp_to_T(rvec, tvec) -> (4,4)`.
- Produces: `palm_pose(world_landmarks) -> T_world_hand (4,4)`,
  `estimate_T_cam_world(world_landmarks, image_landmarks_px, K, dist) -> (4,4) | None`,
  `estimate_T_cam_hand(world_landmarks, image_landmarks_px, K, dist, T_world_hand) -> (4,4) | None`.

- [ ] **Step 1: Write the failing tests**

`packages/e2l_perception/tests/test_pose.py` (the `hand_pts` fixture is in this package's
`conftest.py`: a plausible right hand, wrist at the origin, metres):

```python
import cv2
import numpy as np

from e2l_common.geometry import make_T, rotvec_to_matrix
from e2l_perception.hand.pose import estimate_T_cam_hand, estimate_T_cam_world, palm_pose

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
T_CAM_WORLD = make_T(rotvec_to_matrix(np.array([0.3, -0.2, 0.1])), np.array([0.05, -0.02, 0.45]))


def _project(T_cam_world: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    px, _ = cv2.projectPoints(pts, rvec, T_cam_world[:3, 3], K, DIST)
    return px.reshape(-1, 2)


def test_palm_pose_origin_is_wrist_and_rotation_is_proper(hand_pts):
    shifted = hand_pts + np.array([0.01, 0.02, 0.03])
    T = palm_pose(shifted)
    np.testing.assert_allclose(T[:3, 3], shifted[0])
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_recovers_known_pose_exactly(hand_pts):
    px = _project(T_CAM_WORLD, hand_pts)
    T_world_hand = palm_pose(hand_pts)
    T = estimate_T_cam_hand(hand_pts, px, K, DIST, T_world_hand)
    np.testing.assert_allclose(T, T_CAM_WORLD @ T_world_hand, atol=1e-5)


def test_pose_is_robust_to_pixel_noise(hand_pts):
    rng = np.random.default_rng(0)
    px = _project(T_CAM_WORLD, hand_pts) + rng.normal(scale=0.5, size=(21, 2))
    T = estimate_T_cam_world(hand_pts, px, K, DIST)
    assert np.linalg.norm(T[:3, 3] - T_CAM_WORLD[:3, 3]) < 0.01


def test_non_finite_input_returns_none(hand_pts):
    px = _project(T_CAM_WORLD, hand_pts)
    px[3] = np.nan
    assert estimate_T_cam_world(hand_pts, px, K, DIST) is None
    assert estimate_T_cam_hand(hand_pts, px, K, DIST, palm_pose(hand_pts)) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_perception/tests/test_pose.py -v`
Expected: FAIL (`cannot import name 'estimate_T_cam_world'`).

- [ ] **Step 3: Implement**

Replace `hand/pose.py` with:

```python
"""Recover the metric hand pose in the camera frame."""

import cv2
import numpy as np

from e2l_common.geometry import compose, make_T
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.markers import pnp_to_T


def palm_pose(world_landmarks: np.ndarray) -> np.ndarray:
    """T_world_hand (4,4): palm frame (`compute_palm_frame` axes) with its origin at the wrist
    landmark, expressed in MediaPipe's world-landmark frame. Input (21,3) metres."""
    pts = np.asarray(world_landmarks, dtype=np.float64)
    return make_T(compute_palm_frame(pts), pts[0])


def estimate_T_cam_world(
    world_landmarks: np.ndarray, image_landmarks_px: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray | None:
    """T_cam_world (4,4) from solvePnP (SQPNP) of the 21 world landmarks (21,3), metres, against
    their 2D detections (21,2), pixels. None if the input is non-finite or PnP fails or puts the
    hand behind the camera."""
    obj = np.asarray(world_landmarks, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_landmarks_px, dtype=np.float64).reshape(-1, 2)
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    T = pnp_to_T(rvec, tvec)
    return T if np.isfinite(T).all() and T[2, 3] > 0 else None


def estimate_T_cam_hand(
    world_landmarks: np.ndarray,
    image_landmarks_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_world_hand: np.ndarray,
) -> np.ndarray | None:
    """T_cam_hand (4,4) = T_cam_world @ T_world_hand, with `T_world_hand` from `palm_pose`.
    None if PnP fails."""
    T_cam_world = estimate_T_cam_world(world_landmarks, image_landmarks_px, K, dist)
    return None if T_cam_world is None else compose(T_cam_world, T_world_hand)
```

- [ ] **Step 4: Run tests → PASS; lint**

Run: `unset PYTHONPATH; uv run pytest packages/e2l_perception -q && uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_perception/src/e2l_perception/hand/pose.py packages/e2l_perception/tests/test_pose.py
git commit -m "feat(perception): recover metric hand pose by PnP of world landmarks" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `smoothing.py` — Kalman + RTS, and the config fields

**Files:**
- Modify: `packages/e2l_perception/src/e2l_perception/smoothing.py`
- Modify: `packages/e2l_common/src/e2l_common/config.py` (`PerceptionConfig` only)
- Modify: `configs/perception.yaml`
- Test: `packages/e2l_perception/tests/test_smoothing.py` (create)

**Interfaces:**
- Produces: `rts_smooth(x_filt, P_filt, x_pred, P_pred, F) -> (x_smooth, P_smooth)` and
  `smooth_tracks(z, valid, dt, accel_noise, meas_std) -> np.ndarray`;
  `PerceptionConfig.smooth_accel_noise`, `.smooth_meas_std`, `.max_gap_s`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from e2l_perception.smoothing import rts_smooth, smooth_tracks


def test_rts_matches_batch_least_squares():
    """For a linear-Gaussian random walk the RTS smoother equals the batch MAP solution."""
    rng = np.random.default_rng(0)
    N, q, r, m0, P0 = 30, 0.1, 0.5, 0.0, 2.0
    z = np.cumsum(rng.normal(scale=q**0.5, size=N)) + rng.normal(scale=r**0.5, size=N)
    F = np.ones((N, 1, 1))
    xf, Pf, xp, Pp = [], [], [], []
    x, P = np.array([m0]), np.array([[P0]])
    for k in range(N):
        if k > 0:
            x, P = F[k] @ x, F[k] @ P @ F[k].T + q
        xp.append(x.copy())
        Pp.append(P.copy())
        gain = P / (P + r)
        x, P = x + gain[:, 0] * (z[k] - x), (1 - gain) * P
        xf.append(x.copy())
        Pf.append(P.copy())
    xs, Ps = rts_smooth(np.array(xf), np.array(Pf), np.array(xp), np.array(Pp), F)

    A, b = np.zeros((N, N)), np.zeros(N)
    A[0, 0], b[0] = 1 / P0, m0 / P0
    A[np.arange(N), np.arange(N)] += 1 / r
    b += z / r
    for k in range(1, N):
        A[k, k] += 1 / q
        A[k - 1, k - 1] += 1 / q
        A[k, k - 1] -= 1 / q
        A[k - 1, k] -= 1 / q
    np.testing.assert_allclose(xs[:, 0], np.linalg.solve(A, b), atol=1e-9)
    np.testing.assert_allclose(Ps[:, 0, 0], np.diag(np.linalg.inv(A)), atol=1e-9)


def test_rts_accepts_constant_F_and_stacked_tracks():
    N, n, M = 5, 2, 3
    rng = np.random.default_rng(1)
    P = np.tile(np.eye(n), (N, 1, 1))
    x = rng.normal(size=(N, n, M))
    xs, Ps = rts_smooth(x, P, x, P + np.eye(n), np.eye(n))
    assert xs.shape == (N, n, M) and Ps.shape == (N, n, n)
    np.testing.assert_allclose(xs[-1], x[-1])


def _line(N: int = 90, dt: float = 1 / 30) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N) * dt
    truth = np.stack([0.1 + 0.2 * t, -0.05 * t, 0.3 + 0 * t], axis=1)  # (N, 3) metres
    return t, truth


def test_smooth_tracks_reduces_noise_and_fills_gaps():
    t, truth = _line()
    rng = np.random.default_rng(2)
    z = truth + rng.normal(scale=0.01, size=truth.shape)
    valid = np.ones(len(t), bool)
    valid[40:45] = False
    z[~valid] = np.nan
    out = smooth_tracks(z, valid, dt=1 / 30, accel_noise=1.0, meas_std=0.01)
    assert out.shape == z.shape and np.isfinite(out).all()
    raw_rmse = np.sqrt(np.nanmean((z - truth) ** 2))
    rmse = np.sqrt(np.mean((out - truth) ** 2))
    assert rmse < 0.6 * raw_rmse
    assert np.abs(out[40:45] - truth[40:45]).max() < 0.01


def test_smooth_tracks_leaves_leading_invalid_rows_and_handles_all_invalid():
    t, truth = _line(20)
    valid = np.ones(20, bool)
    valid[:3] = False
    z = truth.copy()
    z[:3] = np.nan
    out = smooth_tracks(z, valid, dt=1 / 30, accel_noise=1.0, meas_std=0.01)
    assert np.isnan(out[:3]).all() and np.isfinite(out[3:]).all()
    none = smooth_tracks(np.full((5, 2), np.nan), np.zeros(5, bool), 1 / 30, 1.0, 0.01)
    assert np.isnan(none).all()
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/e2l_perception/tests/test_smoothing.py -v` → FAIL (`cannot import name 'smooth_tracks'`).

- [ ] **Step 3: Implement**

```python
"""Offline smoothing: constant-velocity Kalman filter forward, Rauch-Tung-Striebel backward."""

import numpy as np


def rts_smooth(
    x_filt: np.ndarray, P_filt: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray, F: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Backward RTS pass. Per step k: filtered x (N,n) and P (N,n,n), predicted x_pred[k] =
    F[k] @ x_filt[k-1] (N,n) and P_pred (N,n,n), transition F (N,n,n) or one constant (n,n).
    x may carry trailing track axes, (N,n,M), for M tracks sharing one covariance.
    Index 0 of the predictions is unused. Returns smoothed (x, P)."""
    N = len(x_filt)
    F = np.broadcast_to(F, (N, *P_filt.shape[1:]))
    xs, Ps = x_filt.astype(float).copy(), P_filt.astype(float).copy()
    for k in range(N - 2, -1, -1):
        C = np.linalg.solve(P_pred[k + 1], F[k + 1] @ P_filt[k]).T
        xs[k] = x_filt[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P_filt[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T
    return xs, Ps


def smooth_tracks(
    z: np.ndarray, valid: np.ndarray, dt: float, accel_noise: float, meas_std: float
) -> np.ndarray:
    """Smooth M scalar tracks z (N,M) sampled every dt seconds; rows where `valid` (N,) is False
    are missing and get filled. Model: constant velocity with white acceleration of spectral
    density `accel_noise` (m^2/s^3), measurement std `meas_std` (m). Rows before the first
    valid one stay NaN."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    first = idx[0]
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = accel_noise * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0]])
    R = meas_std**2

    x = np.stack([z[first], np.zeros(z.shape[1])])  # (2, M)
    P = np.diag([R, 1.0])
    xf, Pf, xp, Pp = [], [], [], []
    for k in range(first, len(z)):
        if k > first:
            x, P = F @ x, F @ P @ F.T + Q
        xp.append(x)
        Pp.append(P)
        if valid[k]:
            gain = P @ H.T / (H @ P @ H.T + R)  # (2, 1)
            x = x + gain @ (z[k][None, :] - H @ x)
            P = (np.eye(2) - gain @ H) @ P
        xf.append(x)
        Pf.append(P)
    xs, _ = rts_smooth(np.array(xf), np.array(Pf), np.array(xp), np.array(Pp), F)
    out[first:] = xs[:, 0, :]
    return out
```

- [ ] **Step 4: Config fields**

In `packages/e2l_common/src/e2l_common/config.py`, `PerceptionConfig`: delete
`ekf_process_noise` and `ekf_measurement_noise`; add (keep `smooth: bool = True`):

```python
    max_gap_s: float = 0.2  # interior hand-tracking gaps up to this long are filled
    smooth_accel_noise: float = 1.0  # m^2/s^3, white-acceleration density of the CV model
    smooth_meas_std: float = 0.01  # m, keypoint measurement noise
```

In `configs/perception.yaml` replace the two `ekf_*` lines with:

```yaml
max_gap_s: 0.2                           # fill interior tracking gaps up to this long (s)
smooth_accel_noise: 1.0                  # m^2/s^3, constant-velocity Kalman model
smooth_meas_std: 0.01                    # m, keypoint measurement noise
smooth: true                             # Kalman forward + RTS backward over keypoints
```

(remove the old `smooth:` line so it appears once). Then verify nothing else used the old names:
`grep -rn "ekf_process_noise\|ekf_measurement_noise" packages configs scripts` → no output.

- [ ] **Step 5: Run tests → PASS** — `uv run pytest packages/e2l_perception packages/e2l_common -q`, then lint.

- [ ] **Step 6: Commit**

```bash
git add packages/e2l_perception/src/e2l_perception/smoothing.py \
  packages/e2l_perception/tests/test_smoothing.py \
  packages/e2l_common/src/e2l_common/config.py configs/perception.yaml
git commit -m "feat(perception): smooth keypoint tracks with Kalman and RTS passes" \
  -m "Replaces the unused ekf_* config fields with documented-unit smoothing fields." \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `pipeline.py` — per-frame observations to a `DemoTrajectory`

**Files:**
- Modify: `packages/e2l_perception/src/e2l_perception/pipeline.py`
- Modify: `packages/e2l_perception/src/e2l_perception/cli.py` (`run`: print valid fraction)
- Test: `packages/e2l_perception/tests/test_pipeline.py` (create)

**Interfaces:**
- Consumes: Task 1 `palm_pose`, `estimate_T_cam_hand`; Task 2 `smooth_tracks`,
  `cfg.max_gap_s`, `cfg.smooth_accel_noise`, `cfg.smooth_meas_std`, `cfg.smooth`;
  existing `MarkerTracker(capture, K, dist).observe(frame) -> MarkerObservation`,
  `HandDetector(model_path, num_hands, mirrored, min_detection_confidence, min_tracking_confidence=...)`
  as a context manager with `.detect_all(frame, t)`, `HandFilter(hand, max_jump_px).select(dets)`,
  `VideoReader(path)` (iterates `(frame, t)`, has `.fps`, `.size`), `load_intrinsics(path)`
  (`.K`, `.dist`, `.size` = (width, height)).
- Produces: `FrameObservation` dataclass
  (`T_cam_table`, `T_cam_obj: dict[str, (4,4)]`, `T_cam_hand`, `keypoints_cam (21,3) | None`,
  `detection: HandDetection | None`); `fill_short_gaps(valid, max_gap) -> valid`;
  `observe_frames(video, cfg, capture) -> Iterator[tuple[float, np.ndarray, FrameObservation]]`
  (also yields the BGR frame, for overlays); `assemble_trajectory(demo_id, fps, observations,
  object_names, cfg) -> DemoTrajectory`; `run_demo(video, demo_id, cfg, capture) -> DemoTrajectory`.

- [ ] **Step 1: Write the failing tests**

`packages/e2l_perception/tests/test_pipeline.py`:

```python
import cv2
import numpy as np
import pytest

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.geometry import compose, inv_T, make_T, rotvec_to_matrix
from e2l_perception.calibration import Intrinsics
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.pipeline import (
    FrameObservation,
    assemble_trajectory,
    fill_short_gaps,
    observe_frames,
)

FPS = 30.0
T_CAM_TABLE = make_T(rotvec_to_matrix(np.array([2.2, 0.0, 0.0])), np.array([0.0, 0.1, 0.6]))
T_TABLE_BOWL = make_T(t=np.array([0.1, 0.0, 0.0]))


def _hand_table(i: int, hand_pts: np.ndarray) -> np.ndarray:
    """Keypoints in the table frame at frame i: the canonical hand moving 1 cm/frame along x."""
    return hand_pts + np.array([-0.2 + 0.01 * i, 0.05, 0.15])


def _obs(i: int, hand_pts: np.ndarray, hand=True, board=True, bowl=True) -> FrameObservation:
    kp_table = _hand_table(i, hand_pts)
    kp_cam = (T_CAM_TABLE[:3, :3] @ kp_table.T).T + T_CAM_TABLE[:3, 3]
    T_cam_hand = compose(T_CAM_TABLE, make_T(compute_palm_frame(kp_table), kp_table[0]))
    return FrameObservation(
        T_cam_table=T_CAM_TABLE if board else None,
        T_cam_obj={"bowl": compose(T_CAM_TABLE, T_TABLE_BOWL)} if bowl else {},
        T_cam_hand=T_cam_hand if hand else None,
        keypoints_cam=kp_cam if hand else None,
        detection=None,
    )


def _cfg(**kw) -> PerceptionConfig:
    return PerceptionConfig(smooth=False, **kw)


def test_fill_short_gaps_only_fills_interior_gaps_up_to_max():
    v = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0], bool)
    out = fill_short_gaps(v, max_gap=2)
    np.testing.assert_array_equal(out, [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0])


def test_assemble_expresses_hand_and_objects_in_table_frame(hand_pts):
    obs = [_obs(i, hand_pts) for i in range(10)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl", "plate"], _cfg())
    assert demo.hand_valid.all()
    np.testing.assert_allclose(demo.t, np.arange(10) / FPS)
    np.testing.assert_allclose(demo.keypoints_table[4], _hand_table(4, hand_pts), atol=1e-9)
    np.testing.assert_allclose(demo.T_table_hand[4, :3, 3], _hand_table(4, hand_pts)[0], atol=1e-9)
    np.testing.assert_allclose(demo.aperture, np.linalg.norm(hand_pts[4] - hand_pts[8]), atol=1e-9)
    np.testing.assert_allclose(demo.T_table_obj[3, 0], T_TABLE_BOWL, atol=1e-9)
    assert demo.obj_valid[:, 0].all() and not demo.obj_valid[:, 1].any()
    assert np.isnan(demo.T_table_obj[:, 1]).all()


def test_short_gap_is_interpolated_long_gap_stays_invalid(hand_pts):
    hand = np.ones(40, bool)
    hand[10:14] = False  # 4 frames = 0.13 s <= 0.2 s
    hand[20:30] = False  # 10 frames = 0.33 s > 0.2 s
    obs = [_obs(i, hand_pts, hand=bool(hand[i])) for i in range(40)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], _cfg(max_gap_s=0.2))
    assert demo.hand_valid[10:14].all() and not demo.hand_valid[20:30].any()
    np.testing.assert_allclose(demo.keypoints_table[12], _hand_table(12, hand_pts), atol=1e-9)
    assert np.isnan(demo.keypoints_table[25]).all() and np.isnan(demo.aperture[25])
    assert np.isnan(demo.T_table_hand[25]).all()


def test_board_missing_invalidates_hand(hand_pts):
    obs = [_obs(i, hand_pts, board=i != 0) for i in range(5)]
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], _cfg())
    assert not demo.hand_valid[0] and demo.hand_valid[1:].all()
    assert not demo.obj_valid[0, 0]


def test_smoothing_keeps_linear_motion(hand_pts):
    obs = [_obs(i, hand_pts) for i in range(30)]
    cfg = PerceptionConfig(smooth=True)
    demo = assemble_trajectory("20261008_test_000", FPS, obs, ["bowl"], cfg)
    truth = np.stack([_hand_table(i, hand_pts) for i in range(30)])
    assert np.abs(demo.keypoints_table - truth).max() < 0.005


def test_observe_frames_rejects_mismatched_intrinsics(tmp_path):
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    calib = Intrinsics(K=np.eye(3), dist=np.zeros(5), size=(1920, 1080)).save(tmp_path / "c.json")
    capture = CaptureConfig(intrinsics=calib)
    with pytest.raises(ValueError, match="1920"):
        next(observe_frames(video, PerceptionConfig(), capture))
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/e2l_perception/tests/test_pipeline.py -v` → FAIL at import.

- [ ] **Step 3: Implement**

Replace `pipeline.py` with:

```python
"""Per-demo perception: normalised video -> DemoTrajectory in the table frame."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.geometry import compose, inv_T, make_T
from e2l_common.schemas import DemoTrajectory
from e2l_perception.calibration import load_intrinsics
from e2l_perception.hand.detector import HandDetection
from e2l_perception.hand.palm import compute_palm_frame
from e2l_perception.hand.pose import estimate_T_cam_hand, palm_pose
from e2l_perception.smoothing import smooth_tracks
from e2l_perception.video import VideoReader

THUMB_TIP, INDEX_TIP = 4, 8


@dataclass
class FrameObservation:
    """Everything seen in one frame, in the camera frame. None / missing keys = not detected."""

    T_cam_table: np.ndarray | None
    T_cam_obj: dict[str, np.ndarray] = field(default_factory=dict)
    T_cam_hand: np.ndarray | None = None
    keypoints_cam: np.ndarray | None = None  # (21,3) metres
    detection: HandDetection | None = None


def _transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def observe_frames(
    video: Path, cfg: PerceptionConfig, capture: CaptureConfig
) -> Iterator[tuple[float, np.ndarray, FrameObservation]]:
    """Yield (t, BGR frame, observation) for every frame: board and object poses from ArUco,
    the demonstrating hand from MediaPipe + HandFilter, and its metric pose by PnP."""
    from e2l_perception.hand.detector import HandDetector
    from e2l_perception.hand.filter import HandFilter
    from e2l_perception.markers import MarkerTracker

    reader = VideoReader(video)
    intr = load_intrinsics(capture.intrinsics)
    if tuple(intr.size) != tuple(reader.size):
        raise ValueError(f"intrinsics are for {intr.size}, video {video} is {reader.size}")
    tracker = MarkerTracker(capture, intr.K, intr.dist)
    hand_filter = HandFilter(cfg.hand, cfg.max_wrist_jump_px)
    with HandDetector(
        cfg.hand_model,
        cfg.num_hands,
        cfg.mirrored,
        cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    ) as detector:
        for frame, t in reader:
            markers = tracker.observe(frame)
            detection = hand_filter.select(detector.detect_all(frame, t))
            obs = FrameObservation(markers.T_cam_table, markers.T_cam_obj, detection=detection)
            if detection is not None:
                world = detection.landmarks_3d.astype(np.float64)
                T_world_hand = palm_pose(world)
                obs.T_cam_hand = estimate_T_cam_hand(
                    world, detection.landmarks_2d, intr.K, intr.dist, T_world_hand
                )
                if obs.T_cam_hand is not None:
                    obs.keypoints_cam = _transform(compose(obs.T_cam_hand, inv_T(T_world_hand)), world)
            yield t, frame, obs


def fill_short_gaps(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Mark interior runs of False no longer than `max_gap` frames as True."""
    out = valid.copy()
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            out[a + 1 : b] = True
    return out


def assemble_trajectory(
    demo_id: str,
    fps: float,
    observations: list[FrameObservation],
    object_names: list[str],
    cfg: PerceptionConfig,
) -> DemoTrajectory:
    """Camera-frame observations -> table-frame DemoTrajectory. Hand frames need both the hand
    and the board; interior gaps up to `cfg.max_gap_s` are filled (Kalman+RTS if `cfg.smooth`,
    else linear interpolation). T_table_hand is the palm frame of the table-frame keypoints.
    Invalid entries are NaN."""
    N, K = len(observations), len(object_names)
    kp = np.full((N, 21, 3), np.nan)
    T_table_obj = np.full((N, K, 4, 4), np.nan)
    for i, obs in enumerate(observations):
        if obs.T_cam_table is None:
            continue
        T_table_cam = inv_T(obs.T_cam_table)
        if obs.keypoints_cam is not None:
            kp[i] = _transform(T_table_cam, obs.keypoints_cam)
        for k, name in enumerate(object_names):
            if name in obs.T_cam_obj:
                T_table_obj[i, k] = compose(T_table_cam, obs.T_cam_obj[name])
    seen = np.isfinite(kp).all(axis=(1, 2))
    hand_valid = fill_short_gaps(seen, int(np.floor(cfg.max_gap_s * fps + 1e-9)))

    flat = kp.reshape(N, -1)
    if cfg.smooth:
        flat = smooth_tracks(flat, seen, 1 / fps, cfg.smooth_accel_noise, cfg.smooth_meas_std)
    elif seen.any():
        frames = np.arange(N)
        flat = np.stack(
            [np.interp(frames, frames[seen], col[seen]) for col in flat.T], axis=1
        )
    kp = flat.reshape(N, 21, 3)
    kp[~hand_valid] = np.nan

    T_table_hand = np.full((N, 4, 4), np.nan)
    for i in np.flatnonzero(hand_valid):
        T_table_hand[i] = make_T(compute_palm_frame(kp[i]), kp[i, 0])
    return DemoTrajectory(
        demo_id=demo_id,
        fps=fps,
        object_names=object_names,
        t=np.arange(N) / fps,
        T_table_hand=T_table_hand,
        hand_valid=hand_valid,
        keypoints_table=kp,
        aperture=np.linalg.norm(kp[:, THUMB_TIP] - kp[:, INDEX_TIP], axis=1),
        T_table_obj=T_table_obj,
        obj_valid=np.isfinite(T_table_obj).all(axis=(2, 3)),
    )


def run_demo(
    video: Path, demo_id: str, cfg: PerceptionConfig, capture: CaptureConfig
) -> DemoTrajectory:
    """Track a normalised video and assemble its table-frame DemoTrajectory."""
    observations = [obs for _, _, obs in observe_frames(video, cfg, capture)]
    fps = VideoReader(video).fps
    return assemble_trajectory(demo_id, fps, observations, list(capture.objects), cfg)
```

If `HandDetector`'s positional order differs from the call above, use keywords; check
`hand/detector.py` (`model_path, num_hands, mirrored, min_detection_confidence, ...`). The
detector is constructed only after the intrinsics check, so the mismatch test never needs the
MediaPipe model file.

In `cli.py` `run`, after `demo.save(...)`, print the valid fraction:

```python
    typer.echo(
        f"{demo_id}: {len(demo.t)} frames, hand valid {demo.hand_valid.mean():.1%}, "
        + ", ".join(f"{n} {demo.obj_valid[:, k].mean():.1%}" for k, n in enumerate(demo.object_names))
    )
```

- [ ] **Step 4: Run tests → PASS; lint** (`uv run pytest packages/e2l_perception -q`, ruff).

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_perception/src/e2l_perception/pipeline.py \
  packages/e2l_perception/src/e2l_perception/cli.py packages/e2l_perception/tests/test_pipeline.py
git commit -m "feat(perception): assemble table-frame demo trajectories from video" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: perception `viz.py` — overlay video and trajectory plot

**Files:**
- Modify: `packages/e2l_perception/src/e2l_perception/viz.py` (replace the two stubs; add `draw_observation`)
- Modify: `packages/e2l_perception/src/e2l_perception/cli.py` (`viz` command)
- Test: `packages/e2l_perception/tests/test_viz.py` (create)

**Interfaces:**
- Consumes: Task 3 `FrameObservation`, `observe_frames`; existing `draw_detection`, `draw_axes`, `draw_hud`.
- Produces: `draw_observation(frame, obs, K, dist) -> None`,
  `render_overlay(video, out, cfg, capture) -> Path`, `plot_trajectory(demo, out) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from e2l_common.geometry import make_T
from e2l_common.schemas import DemoTrajectory
from e2l_perception.pipeline import FrameObservation
from e2l_perception.viz import draw_observation, plot_trajectory

K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def test_draw_observation_draws_hand_table_and_objects(make_detection):
    frame = np.zeros((480, 640, 3), np.uint8)
    T = make_T(t=np.array([0.0, 0.0, 0.5]))
    obs = FrameObservation(
        T_cam_table=T,
        T_cam_obj={"bowl": make_T(t=np.array([0.05, 0.0, 0.5]))},
        T_cam_hand=make_T(t=np.array([-0.05, 0.0, 0.5])),
        detection=make_detection("Right", (320.0, 240.0)),
    )
    draw_observation(frame, obs, K, np.zeros(5))
    assert frame.any()


def test_draw_observation_tolerates_empty_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_observation(frame, FrameObservation(T_cam_table=None), K, np.zeros(5))
    assert frame.any()  # HUD text still says nothing was detected


def test_plot_trajectory_writes_png_with_invalid_frames(tmp_path):
    N = 20
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = np.linspace(0, 0.2, N)
    T[5:8] = np.nan
    valid = np.ones(N, bool)
    valid[5:8] = False
    aperture = np.linspace(0.08, 0.02, N)
    aperture[5:8] = np.nan
    demo = DemoTrajectory(
        demo_id="20261008_test_000",
        fps=30.0,
        object_names=["bowl"],
        t=np.arange(N) / 30,
        T_table_hand=T,
        hand_valid=valid,
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=np.tile(np.eye(4), (N, 1, 1, 1)),
        obj_valid=np.ones((N, 1), bool),
    )
    out = plot_trajectory(demo, tmp_path / "plots" / "traj.png")
    assert out.is_file() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure** → FAIL (`cannot import name 'draw_observation'`).

- [ ] **Step 3: Implement**

In `viz.py`: remove the `not_implemented` import; add `from e2l_common.config import
CaptureConfig, PerceptionConfig`, `from e2l_common.schemas import DemoTrajectory`,
`from e2l_perception.calibration import load_intrinsics`, `from e2l_perception.pipeline import
FrameObservation, observe_frames` (check for an import cycle: `pipeline` must not import `viz`),
and replace the two stubs with:

```python
def draw_observation(
    frame: np.ndarray, obs: FrameObservation, K: np.ndarray, dist: np.ndarray
) -> None:
    """Landmarks, palm axes, table axes (10 cm) and object axes (5 cm), plus a status HUD."""
    if obs.detection is not None:
        draw_detection(frame, obs.detection)
    if obs.T_cam_hand is not None:
        draw_axes(frame, obs.T_cam_hand, K, dist, 0.05)
    if obs.T_cam_table is not None:
        draw_axes(frame, obs.T_cam_table, K, dist, 0.10)
    for T_cam_obj in obs.T_cam_obj.values():
        draw_axes(frame, T_cam_obj, K, dist, 0.05)
    draw_hud(
        frame,
        [
            f"board {'ok' if obs.T_cam_table is not None else '--'}  "
            f"hand {'ok' if obs.T_cam_hand is not None else '--'}",
            "objects: " + (", ".join(sorted(obs.T_cam_obj)) or "--"),
        ],
    )


def render_overlay(
    video: Path, out: Path, cfg: PerceptionConfig, capture: CaptureConfig
) -> Path:
    """Re-track `video` and write an mp4 with `draw_observation` overlays to `out`."""
    import imageio.v2 as imageio

    from e2l_perception.video import VideoReader

    intr = load_intrinsics(capture.intrinsics)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=VideoReader(video).fps, macro_block_size=1) as writer:
        for t, frame, obs in observe_frames(video, cfg, capture):
            draw_observation(frame, obs, intr.K, intr.dist)
            draw_hud(frame[-40:], [f"t={t:.2f}s"])
            writer.append_data(frame[:, :, ::-1])
    return out


def plot_trajectory(demo: DemoTrajectory, out: Path) -> Path:
    """3D wrist path with object start positions, wrist xyz over time, and aperture; PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrist = demo.T_table_hand[:, :3, 3]
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(2, 2, (1, 3), projection="3d")
    ax3d.plot(*wrist.T, lw=1.5, label="wrist")
    for k, name in enumerate(demo.object_names):
        seen = np.flatnonzero(demo.obj_valid[:, k])
        if seen.size:
            ax3d.scatter(*demo.T_table_obj[seen[0], k, :3, 3], s=40, label=name)
    ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title=f"{demo.demo_id} (table frame)")
    ax3d.legend()
    ax_xyz = fig.add_subplot(2, 2, 2)
    for i, axis in enumerate("xyz"):
        ax_xyz.plot(demo.t, wrist[:, i], label=axis)
    ax_xyz.set(ylabel="wrist position (m)")
    ax_xyz.legend()
    ax_ap = fig.add_subplot(2, 2, 4, sharex=ax_xyz)
    ax_ap.plot(demo.t, demo.aperture)
    ax_ap.set(xlabel="t (s)", ylabel="aperture (m)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
```

(`draw_hud(frame[-40:], ...)` draws into a view of the bottom strip; if that renders off-strip,
draw it at the top with `draw_hud(frame, [..., f"t={t:.2f}s"])` instead — keep it simple.)

Replace the CLI `viz` command:

```python
@app.command()
def viz(
    demo_id: str,
    config: Path = config_option("perception"),
    capture_config: Path = typer.Option(Path("configs/capture.yaml"), help="Capture config."),
    out_dir: Path = typer.Option(Path("outputs/perception")),
    set_: list[str] = set_option(),
) -> None:
    """Overlay video and trajectory plots for one demo."""
    from e2l_common.config import load_config
    from e2l_common.schemas import DemoTrajectory
    from e2l_perception.viz import plot_trajectory, render_overlay

    cfg = load_stage_config(config, set_, PerceptionConfig)
    paths = DataPaths.default()
    capture = load_config(capture_config, model=CaptureConfig)
    render_overlay(paths.normalised(demo_id), out_dir / f"{demo_id}_overlay.mp4", cfg, capture)
    plot_trajectory(
        DemoTrajectory.load(paths.perception(demo_id)), out_dir / f"{demo_id}_trajectory.png"
    )
```

- [ ] **Step 4: Run tests → PASS; lint.** Also `uv run e2l perception viz --help` exits 0.

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_perception/src/e2l_perception/viz.py \
  packages/e2l_perception/src/e2l_perception/cli.py packages/e2l_perception/tests/test_viz.py
git commit -m "feat(perception): render tracking overlays and trajectory plots" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `events.py` — aperture to grip, grip events

**Files:**
- Modify: `packages/e2l_segment/src/e2l_segment/events.py`
- Test: `packages/e2l_segment/tests/test_events.py` (create)

**Interfaces:**
- Consumes: `SegmentConfig(close_threshold=0.035, open_threshold=0.055, min_dwell_s=0.25)`.
- Produces: `aperture_to_grip(t, aperture, cfg) -> grip (N,) int8`,
  `grip_events(t, grip) -> list[GripEvent]` (existing `GripEvent(kind, index, t)` dataclass).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from e2l_common.config import SegmentConfig
from e2l_segment.events import aperture_to_grip, grip_events

CFG = SegmentConfig(close_threshold=0.035, open_threshold=0.055, min_dwell_s=0.25)
FPS = 30.0


def _t(n: int) -> np.ndarray:
    return np.arange(n) / FPS


def test_pinch_closes_and_opens_at_thresholds():
    ap = np.r_[np.full(30, 0.08), np.full(30, 0.02), np.full(30, 0.08)]
    grip = aperture_to_grip(_t(90), ap, CFG)
    assert grip.dtype == np.int8
    np.testing.assert_array_equal(grip, np.r_[np.zeros(30), np.ones(30), np.zeros(30)])


def test_hysteresis_keeps_grip_closed_between_thresholds():
    ap = np.r_[np.full(20, 0.08), np.full(20, 0.02), np.tile([0.04, 0.05], 20), np.full(20, 0.08)]
    grip = aperture_to_grip(_t(len(ap)), ap, CFG)
    assert grip[20:80].all() and not grip[80:].any()


def test_short_blips_are_removed():
    ap = np.full(90, 0.08)
    ap[40:43] = 0.02  # 0.1 s < 0.25 s
    assert not aperture_to_grip(_t(90), ap, CFG).any()
    ap = np.full(90, 0.02)
    ap[40:43] = 0.08
    assert aperture_to_grip(_t(90), ap, CFG).all()


def test_nan_holds_state():
    ap = np.r_[np.full(20, 0.08), np.full(20, 0.02), np.full(30, np.nan), np.full(20, 0.08)]
    grip = aperture_to_grip(_t(len(ap)), ap, CFG)
    assert grip[20:70].all() and not grip[70:].any()


def test_all_nan_is_open():
    assert not aperture_to_grip(_t(10), np.full(10, np.nan), CFG).any()


def test_grip_events_kinds_indices_and_times():
    grip = np.r_[np.zeros(30), np.ones(30), np.zeros(30)].astype(np.int8)
    events = grip_events(_t(90), grip)
    assert [(e.kind, e.index) for e in events] == [("grasp", 30), ("release", 60)]
    assert np.isclose(events[1].t, 2.0)
    assert grip_events(_t(5), np.zeros(5, np.int8)) == []
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest packages/e2l_segment/tests/test_events.py -v` → FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement**

Replace the two stub bodies (and drop the `not_implemented` import):

```python
def aperture_to_grip(t: np.ndarray, aperture: np.ndarray, cfg: SegmentConfig) -> np.ndarray:
    """(docstring unchanged)"""
    grip = np.zeros(len(aperture), np.int8)
    state = 0
    for i, a in enumerate(aperture):
        if np.isfinite(a):
            if state == 0 and a < cfg.close_threshold:
                state = 1
            elif state == 1 and a > cfg.open_threshold:
                state = 0
        grip[i] = state
    return _drop_short_runs(t, grip, cfg.min_dwell_s)


def _runs(grip: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of constant-value runs."""
    edges = np.r_[0, np.flatnonzero(np.diff(grip)) + 1, len(grip)]
    return list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))


def _drop_short_runs(t: np.ndarray, grip: np.ndarray, min_dwell_s: float) -> np.ndarray:
    """Flip runs shorter than `min_dwell_s` (run length x median frame period), shortest first,
    until none remain or only one run is left."""
    if len(grip) < 2:
        return grip
    dt = float(np.median(np.diff(t)))
    grip = grip.copy()
    while len(runs := _runs(grip)) > 1:
        start, stop = min(runs, key=lambda r: r[1] - r[0])
        if (stop - start) * dt >= min_dwell_s:
            break
        grip[start:stop] = 1 - grip[start]
    return grip


def grip_events(t: np.ndarray, grip: np.ndarray) -> list[GripEvent]:
    """(docstring unchanged)"""
    return [
        GripEvent("grasp" if grip[i] == 1 else "release", int(i), float(t[i]))
        for i in np.flatnonzero(np.diff(grip)) + 1
    ]
```

- [ ] **Step 4: Run tests → PASS; lint.**

- [ ] **Step 5: Commit**

```bash
git add packages/e2l_segment/src/e2l_segment/events.py packages/e2l_segment/tests/test_events.py
git commit -m "feat(segment): binarise aperture with hysteresis and minimum dwell" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `assign.py` + `segment.py` — object-centric hand segments

**Files:**
- Modify: `packages/e2l_segment/src/e2l_segment/assign.py`
- Modify: `packages/e2l_segment/src/e2l_segment/segment.py`
- Modify: `packages/e2l_segment/src/e2l_segment/cli.py` (`run`: print a summary line)
- Test: `packages/e2l_segment/tests/test_segment.py` (create)

**Interfaces:**
- Consumes: Task 5 `aperture_to_grip`, `grip_events`; `DemoTrajectory`, `HandSegment(s)`.
- Produces: `nearest_valid_index(valid, index) -> int`,
  `reference_object(demo, event_index, exclude: frozenset[str] = frozenset()) -> str`, `segment_demo(demo, cfg) -> HandSegments`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.assign import nearest_valid_index, reference_object
from e2l_segment.segment import segment_demo

FPS = 30.0
BOWL = np.array([0.1, 0.0, 0.0])
PLATE = np.array([-0.1, 0.1, 0.0])


def _demo(hand_valid: np.ndarray | None = None, plate_seen: bool = True) -> DemoTrajectory:
    """Approach the bowl (grasp at 30), carry it to the plate (release at 60), retreat to 89."""
    N = 90
    wrist = np.zeros((N, 3))
    wrist[:31] = np.linspace(BOWL + [0, 0, 0.2], BOWL + [0, 0, 0.05], 31)
    wrist[30:61] = np.linspace(BOWL + [0, 0, 0.05], PLATE + [0, 0, 0.08], 31)
    wrist[60:] = np.linspace(PLATE + [0, 0, 0.08], PLATE + [0, 0, 0.25], 30)
    T_hand = np.tile(np.eye(4), (N, 1, 1))
    T_hand[:, :3, 3] = wrist
    T_obj = np.tile(np.eye(4), (N, 2, 1, 1))
    T_obj[:, 0, :3, 3] = BOWL
    T_obj[30:61, 0, :3, 3] = wrist[30:61] - [0, 0, 0.05]  # bowl travels with the hand
    T_obj[61:, 0, :3, 3] = PLATE + [0, 0, 0.03]
    T_obj[:, 1, :3, 3] = PLATE
    obj_valid = np.ones((N, 2), bool)
    if not plate_seen:
        obj_valid[:, 1] = False
        T_obj[:, 1] = np.nan
    aperture = np.full(N, 0.08)
    aperture[30:60] = 0.02
    valid = np.ones(N, bool) if hand_valid is None else hand_valid
    T_hand[~valid] = np.nan
    aperture[~valid] = np.nan
    return DemoTrajectory(
        demo_id="20261008_test_000",
        fps=FPS,
        object_names=["bowl", "plate"],
        t=np.arange(N) / FPS,
        T_table_hand=T_hand,
        hand_valid=valid,
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=T_obj,
        obj_valid=obj_valid,
    )


def test_nearest_valid_index():
    v = np.array([0, 1, 0, 0, 1, 0], bool)
    assert nearest_valid_index(v, 0) == 1
    assert nearest_valid_index(v, 3) == 4
    assert nearest_valid_index(v, 2) == 1  # tie -> earlier
    with pytest.raises(ValueError):
        nearest_valid_index(np.zeros(3, bool), 1)


def test_reference_object_is_nearest_at_event_and_skips_unseen():
    demo = _demo()
    assert reference_object(demo, 30) == "bowl"
    assert reference_object(demo, 60) == "bowl"  # the carried bowl is nearest at release
    assert reference_object(demo, 60, exclude=frozenset({"bowl"})) == "plate"
    assert reference_object(_demo(plate_seen=False), 60) == "bowl"


def test_reference_object_without_any_object_raises():
    demo = _demo()
    demo.obj_valid[:] = False
    with pytest.raises(ValueError, match="20261008_test_000"):
        reference_object(demo, 30)


def test_segments_cut_at_events_relative_to_reference_object():
    segs = segment_demo(_demo(), SegmentConfig())
    assert segs.demo_id == "20261008_test_000" and segs.fps == FPS
    assert [s.ref_object for s in segs.segments] == ["bowl", "plate", "plate"]
    s0, s1, s2 = segs.segments
    assert len(s0.T_obj_hand) == 31 and np.isclose(s0.t_start, 0) and np.isclose(s0.t_end, 1.0)
    np.testing.assert_allclose(s0.T_obj_hand[-1, :3, 3], [0, 0, 0.05], atol=1e-9)
    assert s0.grip[-1] == 1 and not s0.grip[:-1].any()  # ends on the grasp frame
    assert s1.grip[:-1].all() and s1.grip[-1] == 0  # ends on the release frame
    np.testing.assert_allclose(s1.T_obj_hand[-1, :3, 3], [0, 0, 0.08], atol=1e-9)
    assert np.isclose(s2.t_start, 2.0) and np.isclose(s2.t_end, 89 / FPS)


def test_invalid_hand_frames_are_dropped_from_segments():
    valid = np.ones(90, bool)
    valid[10:20] = False
    segs = segment_demo(_demo(hand_valid=valid), SegmentConfig())
    assert len(segs.segments[0].T_obj_hand) == 21
    assert all(np.isfinite(s.T_obj_hand).all() for s in segs.segments)


def test_no_events_gives_one_segment_and_event_on_last_frame_adds_none():
    demo = _demo()
    demo.aperture[:] = 0.08
    segs = segment_demo(demo, SegmentConfig())
    assert len(segs.segments) == 1 and len(segs.segments[0].T_obj_hand) == 90

    demo = _demo()
    demo.aperture[:] = 0.08
    demo.aperture[60:] = 0.02  # grasp at 60, held to the end: no trailing release
    demo.aperture[89] = 0.02
    segs = segment_demo(demo, SegmentConfig())
    assert all(len(s.T_obj_hand) >= 2 for s in segs.segments)
```

- [ ] **Step 2: Run to verify failure** → FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement `assign.py`** (drop the `not_implemented` import):

```python
def reference_object(
    demo: DemoTrajectory, event_index: int, exclude: frozenset[str] = frozenset()
) -> str:
    """(docstring unchanged; add:) Objects in `exclude` (e.g. the one being carried at a
    release) and objects never seen are skipped; ValueError if none is left."""
    hand = demo.T_table_hand[nearest_valid_index(demo.hand_valid, event_index), :3, 3]
    best, best_d = None, np.inf
    for k, name in enumerate(demo.object_names):
        if name in exclude or not demo.obj_valid[:, k].any():
            continue
        j = nearest_valid_index(demo.obj_valid[:, k], event_index)
        d = float(np.linalg.norm(demo.T_table_obj[j, k, :3, 3] - hand))
        if d < best_d:
            best, best_d = name, d
    if best is None:
        raise ValueError(f"{demo.demo_id}: no object is ever visible")
    return best


def nearest_valid_index(valid: np.ndarray, index: int) -> int:
    """(docstring unchanged; add:) Ties go to the earlier frame; ValueError if none is valid."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("no valid frames")
    return int(idx[np.argmin(np.abs(idx - index))])
```

- [ ] **Step 4: Implement `segment.py`** (drop the `not_implemented` import):

```python
import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.geometry import inv_T
from e2l_common.schemas import DemoTrajectory, HandSegment, HandSegments
from e2l_segment.assign import nearest_valid_index, reference_object
from e2l_segment.events import aperture_to_grip, grip_events


def segment_demo(demo: DemoTrajectory, cfg: SegmentConfig) -> HandSegments:
    """(existing docstring, plus:) Segments share their boundary event frame; only hand-valid
    frames are kept and segments with fewer than 2 are dropped. The trailing segment after the
    last event reuses that event's object and pose."""
    grip = aperture_to_grip(demo.t, demo.aperture, cfg)
    found = grip_events(demo.t, grip)
    events = [e.index for e in found]
    last = len(demo.t) - 1
    bounds = [0, *events, last]
    # Reference object per closing event; a release skips the object grasped before it.
    refs: list[str] = []
    held: frozenset[str] = frozenset()
    for e in found:
        refs.append(reference_object(demo, e.index, held if e.kind == "release" else frozenset()))
        held = frozenset({refs[-1]}) if e.kind == "grasp" else frozenset()
    segments = []
    for s, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        frames = np.arange(i0, i1 + 1)
        frames = frames[demo.hand_valid[frames]]
        if len(frames) < 2:
            continue
        closing = events[s] if s < len(events) else (events[-1] if events else last)
        ref = refs[s] if s < len(refs) else (refs[-1] if refs else reference_object(demo, last))
        k = demo.object_names.index(ref)
        T_obj_table = inv_T(demo.T_table_obj[nearest_valid_index(demo.obj_valid[:, k], closing), k])
        segments.append(
            HandSegment(
                ref_object=ref,
                t_start=float(demo.t[frames[0]]),
                t_end=float(demo.t[frames[-1]]),
                T_obj_hand=T_obj_table @ demo.T_table_hand[frames],
                grip=grip[frames],
            )
        )
    return HandSegments(demo_id=demo.demo_id, fps=demo.fps, segments=segments)
```

In `cli.py` `run`, after saving, add:

```python
    typer.echo(
        f"{demo_id}: {len(segments.segments)} segments "
        + " | ".join(f"{s.ref_object} {s.t_start:.2f}-{s.t_end:.2f}s" for s in segments.segments)
    )
```

- [ ] **Step 5: Run tests → PASS; lint.** `uv run pytest packages/e2l_segment -q`.

- [ ] **Step 6: Commit**

```bash
git add packages/e2l_segment/src/e2l_segment/assign.py packages/e2l_segment/src/e2l_segment/segment.py \
  packages/e2l_segment/src/e2l_segment/cli.py packages/e2l_segment/tests/test_segment.py
git commit -m "feat(segment): cut demos at grip events into object-centric segments" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `e2l segment viz` — the segmentation plot for the README

**Files:**
- Create: `packages/e2l_segment/src/e2l_segment/viz.py`
- Modify: `packages/e2l_segment/src/e2l_segment/cli.py` (`viz` command)
- Modify: `packages/e2l_segment/pyproject.toml` (add `"matplotlib>=3.8"`), then `uv lock`
- Test: `packages/e2l_segment/tests/test_viz.py` (create)

**Interfaces:**
- Consumes: Task 5/6 functions; `DemoTrajectory`.
- Produces: `plot_segments(demo, cfg, out) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.viz import plot_segments


def test_plot_segments_writes_png(tmp_path):
    N = 90
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = np.linspace(0.1, -0.1, N)
    aperture = np.full(N, 0.08)
    aperture[30:60] = 0.02
    aperture[70:75] = np.nan
    demo = DemoTrajectory(
        demo_id="20261008_test_000",
        fps=30.0,
        object_names=["bowl"],
        t=np.arange(N) / 30,
        T_table_hand=T,
        hand_valid=np.ones(N, bool),
        keypoints_table=np.zeros((N, 21, 3)),
        aperture=aperture,
        T_table_obj=np.tile(np.eye(4), (N, 1, 1, 1)),
        obj_valid=np.ones((N, 1), bool),
    )
    out = plot_segments(demo, SegmentConfig(), tmp_path / "seg.png")
    assert out.is_file() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure** → FAIL (`No module named 'e2l_segment.viz'`).

- [ ] **Step 3: Add the dependency** — in `packages/e2l_segment/pyproject.toml`
`dependencies = ["e2l-common", "matplotlib>=3.8"]`; run `unset PYTHONPATH; uv lock && uv sync --all-groups`
(matplotlib is already in the lock via e2l-perception, so only the e2l-segment entry changes;
check `git diff --stat uv.lock` is small).

- [ ] **Step 4: Implement `viz.py`**

```python
"""Segmentation plot: aperture with thresholds, binary grip, and segment boundaries."""

from pathlib import Path

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.events import aperture_to_grip, grip_events
from e2l_segment.segment import segment_demo


def plot_segments(demo: DemoTrajectory, cfg: SegmentConfig, out: Path) -> Path:
    """Aperture (m) with close/open thresholds, grip shaded where closed, grasp/release lines
    labelled with each segment's reference object. Writes a PNG to `out`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grip = aperture_to_grip(demo.t, demo.aperture, cfg)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(demo.t, demo.aperture, color="black", lw=1.2, label="aperture")
    ax.axhline(cfg.close_threshold, color="tab:red", ls="--", lw=0.8, label="close")
    ax.axhline(cfg.open_threshold, color="tab:green", ls="--", lw=0.8, label="open")
    ax.fill_between(demo.t, 0, 1, where=grip == 1, transform=ax.get_xaxis_transform(),
                    color="tab:orange", alpha=0.15, label="grip closed")
    for e in grip_events(demo.t, grip):
        ax.axvline(e.t, color="tab:red" if e.kind == "grasp" else "tab:green", lw=1)
        ax.annotate(e.kind, (e.t, 1.0), xycoords=("data", "axes fraction"), rotation=90,
                    va="top", ha="right", fontsize=8)
    for s in segment_demo(demo, cfg).segments:
        ax.annotate(s.ref_object, ((s.t_start + s.t_end) / 2, 0.02),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=8)
    ax.set(xlabel="t (s)", ylabel="aperture (m)", title=f"{demo.demo_id} segmentation")
    ax.set_ylim(0, np.nanmax(np.r_[demo.aperture, cfg.open_threshold]) * 1.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
```

(`ruff format` will reflow the hand-wrapped calls; run it.)

- [ ] **Step 5: Replace the CLI `viz` stub**

```python
@app.command()
def viz(
    demo_id: str,
    config: Path = config_option("segment"),
    out_dir: Path = typer.Option(Path("outputs/segment")),
    set_: list[str] = set_option(),
) -> None:
    """Plot aperture, grip and segment boundaries for one demo."""
    from e2l_common.schemas import DemoTrajectory
    from e2l_segment.viz import plot_segments

    cfg = load_stage_config(config, set_, SegmentConfig)
    demo = DemoTrajectory.load(DataPaths.default().perception(demo_id))
    typer.echo(f"wrote {plot_segments(demo, cfg, out_dir / f'{demo_id}_segments.png')}")
```

- [ ] **Step 6: Run the whole suite and lint** —
`unset PYTHONPATH; uv run pytest -q && uv run ruff check . && uv run ruff format --check .`;
also `grep -rn "not_implemented" packages/e2l_perception/src packages/e2l_segment/src` must list
nothing from this plan's modules.

- [ ] **Step 7: Commit**

```bash
git add packages/e2l_segment/src/e2l_segment/viz.py packages/e2l_segment/src/e2l_segment/cli.py \
  packages/e2l_segment/pyproject.toml uv.lock packages/e2l_segment/tests/test_viz.py
git commit -m "feat(segment): plot aperture, grip and segment boundaries" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
