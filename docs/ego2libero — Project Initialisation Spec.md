# ego2libero — Project Initialisation Spec

Oct 3, 2026 · @Thomas

## Purpose and context

This spec tells Claude Code how to initialise **ego2libero**: a uv workspace with one package per processing stage, pinned dependencies and runnable stub CLIs. The algorithms are filled in later; this pass only builds the skeleton, ports existing hand-tracking code and proves the toolchain installs and runs.

**The challenge.** Humanoid's research-intern application asks for real manipulation data, recorded personally by the applicant (e.g. phone video), to drive a robot manipulator in a simple simulator such as LIBERO, showcasing VLA and/or world-model skills. Submission is a public GitHub repo with a README covering run instructions, example outputs, design choices, and what worked and what didn't. Deadline: Friday 9 October 2026, 23:59 BST.

**What the judges reward.** Creativity, with the personally collected data playing a real role. Policy performance in sim. Implementation simplicity and clear, honest presentation without filler. Code and README should stay lean.

**Chosen approach: demo multiplication.** Record roughly 20 to 50 phone videos of a hand doing a LIBERO-like pick-and-place task with real objects tagged with ArUco markers. Recover the wrist trajectory and gripper aperture in the table frame, then split each demo into object-centric segments at grasp and release events. Transform those segments to new object poses in LIBERO (MimicGen-style), replay them, and keep only successful rollouts. Fine-tune SmolVLA on the result and report success rate against the number of human demos, with and without augmentation.

**Starting assets.** The existing repo [Mr-Moody/Hand-Tracking-Fusion](https://github.com/Mr-Moody/Hand-Tracking-Fusion) (MediaPipe detection, hand-locking filter, palm-frame EKF, bone-length enforcement, stereo calibration) is ported into the perception package (see Porting plan). Prior experience covers SmolVLA on a Franka Panda end to end.

## Pipeline overview

Seven stages run in sequence, each a package with its own CLI command, each reading the previous stage's files from `data/` and writing its own. Every stage can be rerun alone, which keeps debugging local.

&#91;embedded content: ego2libero pipeline · 7 stages, one package each\]

The generate stage is the multiplication step: one human demo yields many simulated episodes, and only successful replays reach training.

## Repository layout

One repo, one uv workspace, one installable package per stage under `packages/`. Each package uses the `src/` layout, has its own `pyproject.toml` and `tests/`, and exposes a Typer CLI registered under the single `e2l` command.

```text
ego2libero/
├── pyproject.toml            # workspace root: members, dependency groups, ruff/pytest config
├── uv.lock
├── .python-version           # 3.12
├── CLAUDE.md                 # short: links this spec, conventions, common commands
├── README.md                 # submission README (written last; stub now)
├── Makefile                  # setup, lint, test, smoke, per-stage targets
├── .pre-commit-config.yaml
├── configs/                  # one YAML per stage; experiments override these
│   ├── capture.yaml          # camera intrinsics path, marker dictionary + sizes, object→marker map
│   ├── perception.yaml
│   ├── segment.yaml
│   ├── retarget.yaml
│   ├── sim.yaml              # LIBERO suite, task id, camera names, image size, control freq
│   ├── generate.yaml
│   ├── train/smolvla.yaml
│   └── eval.yaml
├── packages/
│   ├── e2l_common/           # schemas, SE(3) maths, IO, config loading, logging
│   ├── e2l_perception/       # video → table-frame hand + object trajectories
│   ├── e2l_segment/          # trajectories → object-centric segments
│   ├── e2l_retarget/         # hand segments → Panda end-effector segments
│   ├── e2l_sim/              # LIBERO wrapper: env, object placement, replay, rendering
│   ├── e2l_generate/         # MimicGen-style transform + stitch + replay + filter
│   ├── e2l_train/            # LeRobot dataset export + SmolVLA fine-tuning
│   └── e2l_eval/             # rollouts, success rate, videos, ablation tables/plots
├── scripts/                  # one-offs: calibrate_phone.py, make_markers.py
├── docs/figures/             # images used by the README (committed)
├── data/                     # gitignored; see Conventions for subfolders
└── outputs/                  # gitignored; checkpoints, eval videos, logs
```

Package internals follow one pattern:

```text
packages/e2l_segment/
├── pyproject.toml            # name = "e2l-segment", depends on e2l-common
├── src/e2l_segment/
│   ├── __init__.py
│   ├── cli.py                # Typer app, mounted as `e2l segment ...`
│   └── ...                   # modules listed in Package specifications
└── tests/
```

The root CLI lives in `e2l_common.cli` and mounts each package's Typer app if that package is installed. Machines that only install some groups (the DGX Spark) still get a working `e2l` command.

## Technology stack

Python 3.12 everywhere, managed by uv. LeRobot's LIBERO integration and MediaPipe both support 3.12; pin exact versions in `uv.lock` once the first sync resolves.

| Layer | Choice | Notes |
| --- | --- | --- |
| Env / packaging | uv workspace, hatchling build backend | One lockfile; per-machine installs via dependency groups |
| Lint / test | ruff (lint + format), pytest, pre-commit | No mypy for now; type hints still expected |
| Config | YAML + pydantic v2 models | One model per stage in `e2l_common.config`; CLI flags override YAML |
| CLI / logging | Typer, rich | Single `e2l` entry point |
| Maths | numpy, scipy (`Rotation`, `Slerp`) | All SE(3) helpers live in `e2l_common.geometry` |
| Hand tracking | mediapipe (Tasks `HandLandmarker`, `hand_landmarker.task`) | Model downloaded by `make models`, not committed |
| Vision / markers | opencv-contrib-python (`cv2.aruco`, `solvePnP`, calibration) | Use only the contrib wheel: mediapipe depends on it, and mixing it with `opencv-python` breaks `cv2` |
| Video IO | ffmpeg (system) + OpenCV `VideoCapture` | Phone video is normalised to constant-frame-rate H.264 before processing |
| Simulator | LIBERO via `lerobot[libero]` (hf-libero, robosuite 1.4.1, MuJoCo) | Headless rendering with `MUJOCO_GL=egl` |
| Policy | LeRobot `lerobot[smolvla]`, PyTorch | Start from `HuggingFaceVLA/smolvla_libero`; `lerobot/smolvla_base` as the alternative |
| Dataset format | Intermediate: `.npz` + JSON sidecar. Training: LeRobot dataset (parquet + mp4) | Schemas in Conventions |
| Plots / video | matplotlib, imageio + imageio-ffmpeg | Figures land in `docs/figures/` |
| Experiment tracking | Weights & Biases, optional | Off by default |

**Dependency groups.** The root `pyproject.toml` defines groups that pull workspace members, so each machine installs only what it needs:

| Group | Members | Laptop (RTX 4060) | Desktop (RTX 3080) | DGX Spark (aarch64) |
| --- | --- | --- | --- | --- |
| `dev` | ruff, pytest, pre-commit | Yes | Yes | Yes |
| `perception` | e2l-perception, e2l-segment | Yes | Optional | No |
| `sim` | e2l-sim, e2l-retarget, e2l-generate, e2l-eval | Yes | Yes | No |
| `train` | e2l-train | Inference + small runs | Fallback training | Main fine-tuning |

`e2l-common` is a dependency of every member. Typical commands: laptop `uv sync --all-groups`; Spark `uv sync --group train --group dev`.

**PyTorch wheels.** Configure uv indexes per platform with environment markers: CUDA 12.x wheels on x86\_64 (laptop, desktop), CUDA 13 aarch64 wheels on the Spark. Verify `torch.cuda.is_available()` on each machine as part of the smoke test.

## Conventions

Every pose is a 4×4 float64 homogeneous transform named `T_a_b`: the pose of frame `b` expressed in frame `a`, so `p_a = T_a_b @ p_b`. Units are metres, radians and seconds. These rules live in `e2l_common.geometry` and are enforced by tests, because frame bugs are the most likely failure in this project.

**Frames**

| Frame | Definition | Source |
| --- | --- | --- |
| `cam` | OpenCV camera frame: x right, y down, z forward | Phone intrinsics from `scripts/calibrate_phone.py` |
| `table` | Origin at the centre of the table ArUco board, z up out of the table, x/y along the board | Board pose per frame via `solvePnP` |
| `obj:<name>` | Canonical frame of each real object | Object marker pose × fixed marker-to-object offset from `capture.yaml` |
| `hand` | Palm frame with origin at the wrist (from `compute_palm_frame`) | MediaPipe landmarks + `solvePnP` for translation |
| `ee` | Panda end-effector site in robosuite | Simulator |
| `sim` | LIBERO / robosuite world frame | Fixed `T_sim_table` in `sim.yaml` |

**Rotations.** Store rotation matrices or full transforms in files, never quaternions. Where a library needs quaternions, name the variable `quat_xyzw` (scipy, robosuite) or `quat_wxyz` (MuJoCo) and convert only at that call site.

**Gripper.** Perception stores raw `aperture` (thumb-tip to index-tip distance, metres). Segmentation produces binary `grip` (0 open, 1 closed) with hysteresis. Robot actions use robosuite's convention: −1 open, +1 closed.

**Actions.** LIBERO actions are 7-D: end-effector delta position (3), delta axis-angle rotation (3), gripper (1). Use the same scaling as the LIBERO datasets the `smolvla_libero` checkpoint was trained on. `e2l_sim` owns the conversion from absolute `T_sim_ee` targets to these deltas.

**Data directories** (all under gitignored `data/`; demo ids look like `20261004_bowlplate_007`):

| Path | Contents | Written by |
| --- | --- | --- |
| `raw/<session>/<demo_id>.mp4` | Original phone video, untouched | You |
| `normalised/<demo_id>.mp4` | Constant-frame-rate H.264 copy | `e2l perception normalise` |
| `calib/<device>.json` | Camera matrix, distortion, resolution | `scripts/calibrate_phone.py` |
| `perception/<demo_id>.npz` + `.json` | Table-frame hand and object trajectories | `e2l perception run` |
| `segments/<demo_id>.npz` + `.json` | Object-centric hand segments | `e2l segment run` |
| `robot_segments/<demo_id>.npz` + `.json` | Object-centric end-effector segments | `e2l retarget run` |
| `generated/<run_id>/<episode>.npz` + `manifest.json` | Replayed sim episodes, success flag, provenance | `e2l generate run` |
| `lerobot/<dataset_name>/` | LeRobot dataset for training | `e2l train export` |

**File schemas.** Arrays go in `.npz`; metadata (ids, object order, fps, config hash, provenance) goes in a JSON sidecar. Each schema is a pydantic model in `e2l_common.schemas` with `save()` / `load()` that validate shapes.

| Schema | Key arrays (N = frames, K = objects, M = segment frames) |
| --- | --- |
| `DemoTrajectory` | `t (N)`, `T_table_hand (N,4,4)`, `hand_valid (N)`, `keypoints_table (N,21,3)`, `aperture (N)`, `T_table_obj (N,K,4,4)`, `obj_valid (N,K)` |
| `HandSegments` | Per segment: `ref_object`, `t_start`, `t_end`, `T_obj_hand (M,4,4)`, `grip (M)` |
| `RobotSegments` | Per segment: `ref_object`, `T_obj_ee (M,4,4)`, `gripper (M)`, `feasible (M)` |
| `SimEpisode` | `images` per camera `(T,H,W,3)` uint8, `state (T,8)`, `actions (T,7)`, `success`, `object_poses`, `source_demo_ids` |

## Package specifications

Each package lists its modules, CLI commands, dependencies beyond `e2l-common`, and its **init scope**: what to implement in this pass and what to leave as a typed stub. A stub has a full signature, a docstring stating inputs, outputs and frames, and raises `NotImplementedError`.

### e2l\_common

Shared foundations every stage imports.

- `geometry.py`: `make_T`, `inv_T`, `compose`, rotation-vector and matrix conversions, pose interpolation (linear translation + slerp), averaging of rotations, `delta_action(T_prev, T_next)`.
- `schemas.py`: the pydantic schemas from Conventions with validated `save()` / `load()`.
- `config.py`: one pydantic model per stage, `load_config(path, overrides)`.
- `paths.py`: data directory layout and demo-id helpers. `cli.py`: root Typer app mounting installed packages. `log.py`: rich logging.
- Dependencies: numpy, scipy, pydantic, pyyaml, typer, rich.
- **Init scope: implement fully, with tests** (round-trip transforms, schema save/load, config overrides).

### e2l\_perception

Phone video to `DemoTrajectory` in the table frame.

- `video.py`: ffmpeg normalisation to constant frame rate; `VideoReader` yielding `(frame, t)` with t from frame index ÷ fps.
- `calibration.py`: load intrinsics; checkerboard calibration ported from the existing repo.
- `markers.py`: ArUco detection, `T_cam_table` from the table board, `T_cam_obj` per tagged object.
- `hand/`: `detector.py`, `filter.py`, `palm.py`, `ekf.py`, `fusion.py` (ported), plus `pose.py` to recover `T_cam_hand` via `solvePnP` of the 21 metric world landmarks against the 2D landmarks.
- `smoothing.py`: offline Rauch–Tung–Striebel smoother over the forward EKF.
- `pipeline.py`: per-demo run producing `DemoTrajectory`. `viz.py`: overlay video plus trajectory and aperture plots.
- CLI: `e2l perception normalise | run | viz`.
- Dependencies: mediapipe, opencv-contrib-python, matplotlib, imageio.
- **Init scope: port the existing code (see Porting plan), implement `video.py` and `markers.py`; stub `pose.py`, `smoothing.py` and `pipeline.py`.**

### e2l\_segment

`DemoTrajectory` to object-centric `HandSegments`.

- `events.py`: aperture to binary grip with hysteresis and minimum dwell time; grasp and release events.
- `assign.py`: reference object per segment (the object nearest the hand at the event that ends it).
- `segment.py`: cut at events, re-express each segment as `T_obj_hand`.
- CLI: `e2l segment run | viz`. No extra dependencies.
- **Init scope: stubs.**

### e2l\_retarget

`HandSegments` to `RobotSegments`.

- `hand_to_ee.py`: fixed `T_hand_ee` offset from config (wrist to gripper centre), optional orientation canonicalisation for top-down grasps.
- `feasibility.py`: workspace bounds, velocity limits, and an IK reachability check through `e2l_sim`.
- CLI: `e2l retarget run`. Depends on e2l-sim.
- **Init scope: stubs.**

### e2l\_sim

The only package that imports LIBERO, robosuite or MuJoCo.

- `env.py`: `make_env(suite, task)` with seeds, camera names and image size from `sim.yaml`.
- `scene.py`: set object poses in the sim from table-frame poses via `T_sim_table`.
- `control.py`: absolute `T_sim_ee` targets to 7-D LIBERO delta actions; closed-loop target tracking.
- `replay.py`: execute `RobotSegments` and record a `SimEpisode`. `render.py`: frames and mp4 output.
- CLI: `e2l sim check | replay`.
- Dependencies: lerobot\[libero\].
- **Init scope: implement `env.py` and `e2l sim check`** (create the chosen task, step 50 random actions, save one frame per camera and print observation and action specs). Stub the rest.

### e2l\_generate

MimicGen-style multiplication: many `SimEpisode`s from few `RobotSegments`.

- `sample.py`: new object poses within configured bounds.
- `transform.py`: `T_sim_ee_new = T_sim_obj_new @ T_obj_ee` for each segment.
- `stitch.py`: interpolated transit motions between segments.
- `run.py`: sample, transform, stitch, replay and keep successes; parallel workers; writes `manifest.json` with yield statistics.
- CLI: `e2l generate run`. Depends on e2l-sim.
- **Init scope: stubs.**

### e2l\_train

- `export.py`: `SimEpisode`s to a LeRobot dataset with the same feature keys, image size and task-string format the `smolvla_libero` checkpoint expects.
- `finetune.py`: thin wrapper that builds and launches the `lerobot-train` command from `configs/train/smolvla.yaml`.
- CLI: `e2l train export | finetune`.
- Dependencies: lerobot\[smolvla\], torch.
- **Init scope: implement `finetune.py` as a dry-run that prints the resolved command; stub `export.py`.**

### e2l\_eval

- `rollout.py`: run a policy in the chosen LIBERO task for N seeded episodes, saving videos.
- `metrics.py`: success rate with Wilson confidence interval.
- `ablation.py`: sweep number of human demos × augmentation on/off. `report.py`: plots, tables and video grids into `docs/figures/`.
- CLI: `e2l eval baseline | rollout | ablate | report`. Depends on e2l-sim and e2l-train.
- **Init scope: implement `e2l eval baseline`** (run `HuggingFaceVLA/smolvla_libero` on the chosen task for 5 episodes, print success rate, save one video). Stub the rest.

## Porting plan from Hand-Tracking-Fusion

Copy from commit `07947d1` of [Hand-Tracking-Fusion](https://github.com/Mr-Moody/Hand-Tracking-Fusion) into `e2l_perception`, with a header comment naming the source file and commit. Replace `sys.path` imports with package-relative imports. Apply the fixes below in the same pass; each gets a regression test.

| Source file | Destination | Required changes |
| --- | --- | --- |
| `src/hand_detector.py` | `hand/detector.py` | The x-negation of world landmarks assumes a mirrored webcam feed. Make it a `mirrored: bool = False` parameter. Also return MediaPipe's handedness label and score. |
| `src/hand_filter.py` | `hand/filter.py` | Hands are matched by `landmarks_3d[0]`, but world landmarks are wrist-centred, so that value is near zero for every hand and the distance gate never separates them. Match on handedness plus 2D wrist pixel distance instead. Merge the two near-duplicate filter methods into one. |
| `src/palm_utils.py` | `hand/palm.py` | None. |
| `src/joint_hand_ekf.py` | `hand/ekf.py` | Optionally record per-step `x`, `P` and predicted values so `smoothing.py` can run an RTS backward pass. |
| `src/fusion.py` | `hand/fusion.py` | Keep the single-camera path as the default. The stereo branch never runs from `main.py`, which passes `[filtered_detection, None]`; keep it but mark it untested. |
| `src/calibration/` | `calibration.py` | Port checkerboard intrinsics calibration for one camera; drop the live stereo capture loop. Do not port `stereo_cal.json`. |
| `src/visualiser.py` | `viz.py` | Drawing helpers only; operate on video frames rather than live windows. |
| `src/camera.py`, `main.py` | Not ported | Replaced by `video.py` and the CLI. |
| `src/hand_12dof_utils.py`, `src/plot3d.py` | Not ported | Out of scope for the Panda pipeline. |

Timestamps must come from frame index ÷ fps of the normalised video, not `time.monotonic()`. MediaPipe's `detect_for_video` requires strictly increasing integer milliseconds, so convert with `round(t * 1000)`.

## Initialisation tasks for Claude Code

Work through these in order on the laptop (Ubuntu 22.04, RTX 4060), one commit per task. Each task is done only when its acceptance check passes.

- [ ] **1. Scaffold the workspace.** Root `pyproject.toml` with workspace members and dependency groups, `.python-version` (3.12), ruff and pytest config, `.gitignore` (`data/`, `outputs/`, `models/`, video files outside `docs/`), pre-commit, and a Makefile with `setup`, `lint`, `test`, `models`, `smoke`. *Check:* `uv sync --all-groups` succeeds and `uv run e2l --help` runs.
- [ ] **2. Configure PyTorch indexes** per platform as described in Technology stack. *Check:* `torch.cuda.is_available()` is true on the laptop.
- [ ] **3. Implement `e2l_common`** (geometry, schemas, config, paths, root CLI). *Check:* tests cover transform round-trips, delta actions, schema save/load and config overrides, and all pass.
- [ ] **4. Create all package skeletons** with modules, Typer sub-apps and typed stubs from Package specifications. *Check:* `e2l <package> <command> --help` works for every command; calling a stub gives a clear `NotImplementedError` naming the function.
- [ ] **5. Write `configs/*.yaml`** with commented defaults. For `sim.yaml`, list the available LIBERO-Goal tasks and pick the bowl-onto-plate task if one exists; otherwise pick the simplest single pick-and-place task and note why. *Check:* every config loads through its pydantic model.
- [ ] **6. Port the hand-tracking code** per the Porting plan, including both bug fixes. *Check:* tests for the `mirrored` flag and for handedness-based hand selection pass.
- [ ] **7. Implement `video.py`, `markers.py` and the two scripts.** `scripts/make_markers.py` writes a printable PDF of the table board and object markers at a stated physical size; `scripts/calibrate_phone.py` runs checkerboard calibration from a video. *Check:* a synthetic test renders a marker with known pose and `markers.py` recovers it within 2 mm and 1°.
- [ ] **8. Implement `e2l sim check`.** *Check:* runs headless with `MUJOCO_GL=egl`, writes one PNG per camera to `outputs/sim_check/` and prints observation and action specs.
- [ ] **9. Implement `e2l eval baseline`.** *Check:* 5 seeded episodes of `HuggingFaceVLA/smolvla_libero` on the chosen task, prints success rate, saves one mp4.
- [ ] **10. Implement `e2l train finetune --dry-run`.** *Check:* prints the full resolved `lerobot-train` command.
- [ ] **11. Write `CLAUDE.md`** (under 60 lines): three-line project summary, link to this spec, common commands, the `T_a_b` and units rules, no quaternions in files, and LIBERO imports only inside `e2l_sim`.
- [ ] **12. Stub `README.md`** with the sections the submission needs: Overview, Setup, Pipeline, Results, Design choices, What worked and what didn't.
- [ ] **13. Wire `make smoke`** to run lint, tests, `e2l sim check` and `e2l eval baseline`. *Check:* passes from a fresh clone.

**Out of scope for this pass**

- Implementing any function marked as a stub in Package specifications.
- Isaac Sim, Isaac Lab or any simulator other than LIBERO.
- Extra frameworks such as Hydra or PyTorch Lightning; vendoring or forking LeRobot.
- Committing data, videos, model weights or the MediaPipe `.task` file.
- Notebooks as the home of pipeline code.

## Risks and open questions

The biggest risk is replay: if retargeted human trajectories rarely succeed in LIBERO, nothing downstream works. The first end-to-end milestone is one recorded demo replaying successfully.

| Risk | Mitigation |
| --- | --- |
| LIBERO install fails (native build steps, robosuite version pins) | Resolve in task 1 before writing other code; record the working versions in `CLAUDE.md` |
| Monocular hand depth is biased (MediaPipe's metric scale is a learned estimate) | At each grasp event, compare hand position to the grasped object's marker; fit a per-demo scale correction |
| Phone video is variable frame rate, blurred or rolling-shutter distorted | Normalise to constant frame rate; record at 60 fps in good light with slow, deliberate motions |
| Real objects differ in size and shape from LIBERO assets | Per-object grasp offsets in config; segments are object-relative, so only the offset changes |
| Exported dataset does not match what `smolvla_libero` was trained on (keys, image size, normalisation stats) | Inspect the checkpoint config and a sample from the `HuggingFaceVLA/libero` dataset before writing `export.py` |
| Low generation yield makes augmentation slow | Log yield per source demo from the first run; drop demos whose segments never succeed |
| MediaPipe or LIBERO unavailable on the DGX Spark | Not needed there: the `train` group only installs LeRobot and PyTorch |

**Open questions to settle before recording**

- Which exact LIBERO task (task 5 lists the candidates).
- Head-mounted phone or tripod: head-mounted fits "egocentric" in the brief but makes marker visibility and motion blur harder.
- Phone model, resolution and frame rate, which set the calibration file and marker size.
