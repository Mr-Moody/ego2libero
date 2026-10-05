# MimicGen-style generation — design

Status: approved in brainstorming, 2026-10-05. Implements `e2l generate run`.

## Goal

Multiply each source demo (`data/robot_segments/<demo_id>`) into many LIBERO episodes by
randomising the object placement, replaying the object-centric segments, and keeping the
successes. Output: `data/generated/<run_id>/<demo_id>_<k:04d>.{npz,json}` (`SimEpisode`) plus
`manifest.json`. This is the input to `e2l_train.export.export_dataset` (separate spec).

The work does not need human video: `scripts/scripted_segments.py` writes a valid
`RobotSegments` file (`20261003_scripted_000`) whose nominal replay already succeeds.

## Decisions

1. **Generate is thin orchestration over `e2l_sim.replay.replay()`.** Replay already anchors
   each segment to its reference object's pose at the segment start (MimicGen's rule), adds a
   straight transit into it and tracks it closed-loop. Generate only samples placements, runs
   replays in parallel, filters and records. `e2l_generate/transform.py`, `stitch.py` and the
   `transit_steps` config field are deleted; transits are governed by `sim.yaml`
   `transit_speed`.
2. **Placements cycle through the LIBERO init states and are perturbed.** Attempt `k` uses
   init state `episode_index = k % n_init` (`n_init` = the task's init-state count, 50 for
   LIBERO-Goal), so training layouts cover the evaluation layouts, then
   adds uniform noise to every object the demo's segments reference.
3. **Full `SimEpisode`s with images are stored**, under a data root on `/mnt/data`.

## Components

### `e2l_sim.scene.nominal_object_poses(cfg: SimConfig, episode_index: int) -> dict`

Create the env for `episode_index`, reset with `cfg.seed`, return `object_poses(env)` (LIBERO
object name -> `T_sim_obj` (4,4)), close the env. Lives in `e2l_sim` because it touches LIBERO.

### `e2l_generate.sample`

`sample_object_poses(nominal, names, cfg, rng) -> dict[str, np.ndarray] | None`

- `nominal`: LIBERO name -> `T_sim_obj`; `names`: the LIBERO names to perturb (the segments'
  `ref_object`s mapped through `sim_cfg.object_map`).
- For each name: xy offset uniform in `±cfg.xy_range` per axis, yaw uniform in
  `±cfg.yaw_range`, applied as `T_new = Trans(p + d) @ Rz(yaw) @ Trans(-p) @ T` with `p` the
  object origin, i.e. a rotation about the sim z axis through the object origin. z height and
  tilt are unchanged.
- Reject the draw if any pair of the perturbed objects is closer than `cfg.min_separation` in
  xy; retry up to `cfg.max_placement_tries` times, then return `None`.
- Returns only the perturbed objects (other scene objects keep their init-state poses).

### `e2l_generate.run`

`attempt_seed(seed, demo_id, k) -> np.random.Generator`: `default_rng(SeedSequence([seed,
zlib.crc32(demo_id.encode()), k]))`. Deterministic per attempt, independent of worker count
and scheduling.

`run_attempt(demo_id, k, n_init, run_id, cfg, sim_cfg, data_root) -> AttemptResult` (one worker job):

1. Load `RobotSegments` for `demo_id` (cached per process).
2. `episode_index = k % n_init` (`n_init` from `e2l_sim.env.init_state_count`, read once by
   the parent); nominal poses via `nominal_object_poses` (cached per process
   and init state).
3. `sample_object_poses`; `None` -> status `placement_failed`.
4. `replay(segments, sim_cfg, T_sim_obj=sampled, episode_index=episode_index)`.
5. Save the episode in the worker (episodes are ~100 MB; never pickled back) to
   `generated/<run_id>/<demo_id>_<k:04d>` if it succeeded, or if `cfg.keep_failures`.
6. Any exception in steps 2-5 -> status `error` with `f"{type(e).__name__}: {e}"`; the run
   continues.

`AttemptResult` (small dataclass): `stem, demo_id, attempt, episode_index, status, steps,
saved, error, size_mb`. `status` is one of `success`, `failure`, `placement_failed`, `error`.

`generate(demo_ids, run_id, cfg, sim_cfg, data_root, overwrite=False) -> dict`:

- Refuse (`FileExistsError`) if `generated/<run_id>/` exists and is not empty, unless
  `overwrite`, which deletes it first (no stale episodes from an earlier run survive).
- Refuse (`FileNotFoundError`) before any sim work if a demo has no robot segments.
- Jobs = every `(demo_id, k)` for `k < cfg.episodes_per_demo`.
- `cfg.workers == 1`: run inline. Otherwise `ProcessPoolExecutor(cfg.workers,
  mp_context=get_context("spawn"))` — EGL contexts do not survive `fork`.
- Log one line per finished attempt; write `manifest.json`; return it.

`summarise(results, run_id, cfg, sim_cfg) -> dict` (pure, builds the manifest):

```json
{
  "run_id": "run0",
  "git_commit": "ddb6201…",
  "generate_config": {...}, "sim_config": {...},
  "demos": {"<demo_id>": {"attempts": 50, "successes": 31, "failures": 17,
            "placement_failed": 0, "errors": 2, "yield": 0.62}},
  "mean_episode_mb": 23.4,
  "episodes": [{"stem": "<demo_id>_0000", "demo_id": "...", "attempt": 0,
                "episode_index": 0, "status": "success", "steps": 212,
                "saved": true, "error": null}]
}
```

`generate` adds `complete`: false if the run was interrupted before every attempt was
collected (workers may then have saved episodes the manifest does not list). `yield =
successes / attempts`. `mean_episode_mb` is over saved episodes (null if none). No
wall-clock timestamps. `git_commit` comes from `git rev-parse HEAD` (null outside a repo).

### CLI

`e2l generate run [DEMO_IDS...] --run-id run0 [--overwrite] [--set key=value]` — defaults to
every demo in `data/robot_segments`; ends by printing a per-demo yield table.

### Config (`GenerateConfig`, `configs/generate.yaml`)

Remove `transit_steps`. Add `min_separation: float = 0.11` (m, xy, between perturbed
objects; init states put the bowl and plate 0.128-0.157 m apart, and closer placements risk
the bowl starting on the plate) and `max_placement_tries: int = 20`. Keep `episodes_per_demo`, `xy_range`, `yaw_range`,
`workers`, `keep_failures`, `seed`.

### Data root

`make setup` creates `data -> /mnt/data/e2l-data` when `data/` does not exist (`/` has ~8 GB
free). `scripts/worktree_setup.sh` already links worktrees to the main checkout's `data/`.
One line in CLAUDE.md "Environment gotchas".

## Testing

Fast (no sim):

- Sampler: offsets within `xy_range`, yaw within `yaw_range`, rotation z axis and height
  unchanged; seeded determinism; pairs closer than `min_separation` are rejected; returns
  `None` when tries run out (e.g. `min_separation` larger than achievable).
- `attempt_seed`: same inputs -> same draws; different `k` or `demo_id` -> different draws.
- `summarise`: counts, yield, `mean_episode_mb`, episode entries.
- `generate` refuses a non-empty run directory without `overwrite`; `overwrite` removes stale
  files; a demo without robot segments fails before any sim work.
- `test_skeleton.py`: drop the `check_stubs` assertion (no stubs remain); keep the CLI help
  check.

`sim` + `slow`: generate on the scripted demo into a temp data root with
`episodes_per_demo=2`, once with `workers=1` and once with `workers=2` (spawn path); saved
episodes load as `SimEpisode`, manifest counts match the files.

## Success criteria

`uv run e2l generate run 20261003_scripted_000` with default config (50 attempts, 4 workers)
completes, writes the manifest, and reports a yield of at least 50 %. A lower yield is a
finding to investigate (tracking, grasp robustness to yaw), not something to hide by
shrinking the noise.

## Out of scope

Segment `feasible` flags (retarget), LeRobot export, resuming partial runs, perturbing
objects the demo does not reference.
