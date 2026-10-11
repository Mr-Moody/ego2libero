# Wide-pose evaluation and results report — design

Status: approved in brainstorming, 2026-10-11. Implements `e2l eval rollout --run --setting`
and `e2l eval report`.

## Goal

Score each policy (Run A: human only, Run B: multiplied, reference: `smolvla_libero`) on two
settings and turn the results into README figures:

- **standard**: LIBERO's own init states, unchanged (episode `i` uses init state `i % 50`).
- **wide**: the same init states, then the task objects moved to placements *outside* the range
  generation trained on. Demo multiplication should help most here.

Output per (run, setting): `outputs/eval/<run>/<setting>/results.json` plus videos. `e2l eval
report` gathers them into `docs/figures/results.md`, `success_rates.png` and `rollouts.gif`.

Nothing here needs recorded demos: `smolvla_libero` fills the reference row, and the report is
tested on synthetic result files.

## Decisions

1. **Wide placements lie strictly outside the generation range.** Generation perturbs objects by
   up to ±`generate.xy_range` (0.08 m) per axis and ±`generate.yaw_range` (0.5 rad). A wide
   xy offset is drawn uniformly over the disc of radius `wide.max_offset` (0.15 m), rejecting
   draws inside the square ±`wide.min_offset` (0.08 m, = the generation range) on both axes, so
   every wide offset is outside anything generation produced. Yaw magnitude is uniform in
   [`wide.min_yaw`, `wide.max_yaw`] = [0.5, 1.0] rad with a random sign.
2. **Only the task objects move**: the LIBERO names in `sim.object_map` (bowl, plate). Other
   movable objects and fixtures stay where the init state puts them.
3. **Placements depend only on the episode seed.** `rng = np.random.default_rng(seed)`, so every
   policy sees identical wide placements for the same seed. Wide seeds start at
   `eval.seed + eval.wide.seed_offset` (default 1000 + 10000), disjoint from standard seeds and
   from generation (`generate.seed` = 0, which only seeds a sampler, never an eval episode).
4. **LeRobot's eval loop is kept.** Wide rollouts swap the env, not the loop: a
   `SyncVectorEnv` of one `WidePoseLiberoEnv` replaces `lerobot.envs.factory.make_env`, and
   `eval_policy` runs as before, so the standard setting stays comparable with `lerobot-eval`.
5. **One results file per (run, setting), gathered by directory scan.** Runs happen on
   different machines (A on the RTX 3080, B on the DGX Spark); result folders are copied
   together and `e2l eval report` scans `outputs/eval/*/*/results.json`.
6. **`perturb` and `separated` move to `e2l_common.geometry`** (SE(3) maths lives there).
   `e2l_generate.sample` and `e2l_sim` both import them from there.

## Components

### `e2l_common.config`

```python
class WidePoseConfig(_Config):
    min_offset: float = 0.08  # m, offsets inside this square (both axes) are rejected
    max_offset: float = 0.15  # m, max radial xy offset from the init-state pose
    min_yaw: float = 0.5  # rad, |yaw| about the sim z axis
    max_yaw: float = 1.0
    min_separation: float = 0.11  # m, xy, between any two movable objects after the move
    table_xy: list[list[float]] = [
        [-0.35, 0.25],
        [-0.40, 0.40],
    ]  # [[xmin,xmax],[ymin,ymax]], m, sim frame
    max_tries: int = 20  # draws per reset before falling back to the init state
    settle_tol: float = 0.02  # m, max xy drift / z drop of a moved object while settling
    seed_offset: int = 10_000


class EvalConfig:  # additions
    runs: dict[str, str] = {"reference": "HuggingFaceVLA/smolvla_libero"}  # label -> checkpoint
    wide: WidePoseConfig = WidePoseConfig()
```

`table_xy` defaults are a placeholder within LIBERO's 1.0 × 1.2 m table (centred at the world
origin). The implementation checks them against the init-state object positions and the
fixtures of `put_the_bowl_on_the_plate` and narrows them if needed; the measured values go in
`configs/eval.yaml` with a comment.

### `e2l_common.geometry`

`perturb(T, d_xy, yaw)` and `separated(poses, min_distance)` move here unchanged from
`e2l_generate.sample`, plus:

`sample_wide_offset(rng, cfg: WidePoseConfig) -> tuple[np.ndarray, float]`: one `(d_xy, yaw)`
draw as in decision 1. Pure numpy, unit-tested.

### `e2l_sim.wide.WidePoseLiberoEnv(LiberoEnv)`

New module `e2l_sim/wide.py`; `e2l_sim.env.make_env` gains a `wide: WidePoseConfig | None`
argument that returns this class instead of `LiberoEnv`.

`reset(seed)`:

1. `super().reset(seed=seed)` (init state, 10 settle steps, control mode set).
2. Record `nominal = object_poses(self)` and the MuJoCo state (`sim.get_state()`).
3. Up to `max_tries` times, with `rng = default_rng(seed)` created once per reset:
   restore the state; for each name in `object_map` values (sorted) draw an offset and
   `perturb` its nominal pose; reject if any moved object leaves `table_xy` or
   `separated(all movable object poses, min_separation)` fails; else teleport with
   `set_object_poses`, step `num_steps_wait` no-op actions, and accept if every moved object
   is within `settle_tol` of its target in xy and has not dropped by more than `settle_tol`.
4. If no draw is accepted, restore the init state and set `self.placement_failed = True`
   (the episode still runs; results record it).
5. Store `self.T_sim_obj = {name: T}` for the accepted placement and return the formatted
   observation of the last step.

### `e2l_eval.rollout`

`rollout(policy_path, cfg, sim_cfg, out_dir, n_episodes=None, max_videos=1,
setting="standard", run=None)`.

- `setting == "wide"`: build `gym.vector.SyncVectorEnv([lambda: make_env(sim_cfg, wide=cfg.wide)])`
  and pass `start_seed=cfg.seed + cfg.wide.seed_offset`. LeRobot's `make_env` is used for
  `standard`, as now.
- `results.json` gains `run`, `setting`, `video_successes` (success of each rendered episode;
  `eval_episode_<i>.mp4` is episode `i`) and, for wide, `placements`: per episode, the moved
  objects' `T_sim_obj` as `{xyz, rotvec}` and `placement_failed`.
- `RolloutResult` gains `run` and `setting` fields.

### `e2l eval rollout` CLI

`--run <label>` resolves the policy from `cfg.runs` (an explicit `--policy` wins) and
`--setting standard|wide`; output goes to `cfg.output_dir / run / setting`. Without `--run`
the label is `custom`. `--videos N` (default 3) sets `max_videos`.

### `e2l_eval.report.make_report(results_dir, out_dir) -> list[Path]`

Reads every `results_dir/*/*/results.json`; the run and setting come from the file (folder names
are only the fallback for old files without them). Writes:

- `results.md`: one row per run (order: `A`, `B`, `reference`, then any others alphabetically),
  columns `standard` and `wide`, each cell `k/n = xx% [lo–hi]` with the 95 % Wilson interval
  recomputed from `successes`; `—` where a combination is missing.
- `success_rates.png`: grouped bars (x = run, one bar per setting) with Wilson error bars,
  episode count annotated on each bar; matplotlib, Agg backend.
- `rollouts.gif`: up to three labelled tiles side by side, chosen from the headline run (`B`,
  else the first run present): first successful standard video, first failed standard video,
  first wide video. Missing tiles are skipped; no GIF if there are no videos. Frames
  downsampled to 10 fps and 256 px per tile with imageio.

`e2l eval report` defaults change to `--results outputs/eval --out-dir docs/figures`.

## Error handling

- A `results.json` with no `successes` or an unknown setting is skipped with a printed warning.
- `--run` naming a label not in `cfg.runs` (and no `--policy`) is a CLI error listing the labels.
- `placement_failed` episodes count in the success rate (they are honest evaluations of the
  init state) but are reported in the table footnote with their count.

## Testing

- `e2l_common`: `sample_wide_offset` over many draws: radius ≤ `max_offset`, never inside the
  ±`min_offset` square, |yaw| in range; same seed gives the same draws;
  `perturb` / `separated` tests move with them.
- `e2l_generate`: existing sample tests still pass after the import move.
- `e2l_eval.report`: synthetic results tree (A and B, both settings, one missing cell) → table
  text, Wilson numbers, run order, PNG written; GIF built from tiny synthetic mp4s.
- `e2l_eval` CLI: `--run` resolution and output path, without launching LIBERO (monkeypatched
  `rollout`).
- Slow (marked like the existing sim tests): one `WidePoseLiberoEnv` reset — moved objects
  are offset beyond the ±`min_offset` square, stay inside `table_xy`, and two resets with the
  same seed give identical poses.
- Manual: `e2l eval rollout --run reference --setting standard` and `--setting wide` with
  `--set n_episodes=10`, then `e2l eval report`; look at the GIF and the wide placements.

## Changes during implementation (2026-10-11)

- The separation check covers only pairs that include a moved object (unmoved objects may
  already sit closer than `min_separation` in an init state).
- The settle check covers every movable object, moved or not, in both height directions.
- `table_xy` stays at the loose default. Measured fixtures (cabinet and rack at y < -0.125,
  stove at x < -0.16, y > 0.11) are not boxed out: draws that hit them fail the settle check and
  are redrawn. Over 40 seeds no episode fell back to the init state; a tight box did on 11/40.
