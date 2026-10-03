# ego2libero

Phone videos of a hand doing pick-and-place (ArUco-tagged objects) become wrist + gripper
trajectories in the table frame, are split into object-centric segments, multiplied
MimicGen-style into many LIBERO episodes, and used to fine-tune SmolVLA.
Spec: [docs/ego2libero — Project Initialisation Spec.md](docs/ego2libero%20—%20Project%20Initialisation%20Spec.md)

## Layout
uv workspace, one package per stage in `packages/e2l_*` (src layout), one `e2l` CLI.
Stages: perception → segment → retarget → sim → generate → train → eval. Each reads the
previous stage's files under `data/` (gitignored) and writes its own. Configs: `configs/*.yaml`,
one pydantic model each in `e2l_common.config`; override with `--set key.sub=value`.

## Commands
- `make setup` (laptop: all groups) · Spark: `uv sync --group train --group dev`
- `make lint` · `make test` · `make models` (MediaPipe hand_landmarker.task) · `make smoke`
- `uv run e2l --help`, `uv run e2l info` (installed stages + CUDA)
- `uv run e2l sim check` · `uv run e2l eval baseline` · `uv run e2l train finetune --dry-run`
- Replay check without perception: `uv run python scripts/scripted_segments.py`, then
  `uv run e2l sim replay 20261003_scripted_000 --video` (should print `success=True`)
- `uv run python scripts/make_markers.py` · `uv run python scripts/calibrate_phone.py <video>`

## Conventions (enforced by tests)
- Poses are 4x4 float64 `T_a_b` = pose of frame b in frame a, so `p_a = T_a_b @ p_b`.
  Frames: cam, table (board centre, z up), obj:<name>, hand, ee, sim.
- Units: metres, radians, seconds. Timestamps = frame index / fps, never wall clock.
- No quaternions in files or configs (matrices / rotvecs). At call sites name them
  `quat_xyzw` (scipy, robosuite) or `quat_wxyz` (MuJoCo).
- Gripper: perception `aperture` (m) → segment `grip` 0/1 → robot actions -1 open / +1 closed.
- LIBERO, robosuite and MuJoCo are imported only inside `e2l_sim` (and lazily, inside
  functions, in `e2l_eval`). Call `e2l_sim.libero_setup.ensure_libero_config()` first.
- SE(3) maths lives in `e2l_common.geometry`; schemas (npz + JSON sidecar) in
  `e2l_common.schemas`. Unimplemented functions call `e2l_common.stub.not_implemented`.
- Lean code and README; ruff (line 100) is the only linter; type hints expected.

## Git and multi-agent work — read [CONTRIBUTING.md](CONTRIBUTING.md) before branching
- Branches `<type>/<kebab-slug>`; commits `<type>(<scope>)?: <subject>`; types feat, fix,
  refactor, perf, test, docs, build, ci, chore. Hooks reject anything else.
- Never put agent, subagent, wave, phase, task, worker or model names, or bare numbers, in a
  branch name or scope. Name branches from the change, before dispatching subagents.
- Follow the superpowers chain: brainstorming → writing-plans → using-git-worktrees →
  subagent-driven-development / dispatching-parallel-agents (+ test-driven-development) →
  requesting-code-review → verification-before-completion → finishing-a-development-branch.
- Worktrees: native tool or `.worktrees/<slug>`; rename `worktree-*` branches with
  `git branch -m`; run `make worktree` inside it (own venv on /mnt/data, shared data/models).

## Environment gotchas (this laptop)
- `~/.zshrc` sources ROS Humble, whose py3.10 `PYTHONPATH` breaks the venv: run
  `unset PYTHONPATH` before `uv run` (the Makefile already unexports it).
- `/` is nearly full: `.venv` is a symlink to `/mnt/data/venvs/ego2libero`; uv cache is on
  `/mnt/data`. Use `HF_HOME=/mnt/data/hf-cache` for checkpoints.
- Headless rendering: `MUJOCO_GL=egl` (set by the Makefile and `e2l_sim`).
- Only `opencv-contrib-python` may provide `cv2`; root `pyproject.toml` overrides drop
  `opencv-python` / `opencv-python-headless` pulled by lerobot, hf-libero and robosuite.

## Working versions (uv.lock, Oct 2026)
Python 3.12.13 · torch 2.11.0+cu128 (cu130 on aarch64) · lerobot 0.6.1 · hf-libero 0.1.4 ·
robosuite 1.4.0 (pinned by hf-libero) · mujoco 3.8.1 · mediapipe 1.0.1 ·
opencv-contrib-python 4.14.0.94 · numpy 2.2.6
