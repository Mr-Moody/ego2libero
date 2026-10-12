# Scripted rehearsal and reference evaluation — design

Status: approved in brainstorming, 2026-10-12. Track 1 of the pre-demo work; Track 2 (Spark
smoke training, rehearsal fine-tunes, ablation) builds on its outputs; Track 3 (README, LICENSE,
tidy) waits for real results.

## Goal

Everything after retargeting runs end to end before a single demo is recorded, on stand-in
demos that vary like human ones, and the reference row of the results table exists:

- 25 jittered scripted demos replace the 25 human demos the real run will have.
- Their raw replays give a rehearsal **Run A** dataset; replays plus ~300 generated episodes give
  a rehearsal **Run B** dataset. Both are exported and load in LeRobot, ready to rsync to the
  Spark in Track 2.
- `smolvla_libero` is evaluated on `standard` and `wide` (50 episodes each). This is a real
  result, not a rehearsal, and is kept for the write-up.
- The eval/generation separation is pinned by a test on the repo configs.

When real demos arrive, the same commands run with real demo ids in place of the scripted ones.

## Decisions

1. **Scripted demo construction moves into `e2l_sim.scripted`.** `scripts/scripted_segments.py`
   becomes a thin CLI over it, so the jitter is unit-tested by `make test` (scripts are not in
   `testpaths`). The module imports only `e2l_common`, so it needs no LIBERO.
2. **Demo 000 stays the nominal demo, unchanged.** `scripted_segments(demo_id)` with no rng
   returns exactly today's segments, so the documented replay check
   (`e2l sim replay 20261003_scripted_000` → `success=True`) and the `scripted50` pilot stay valid.
   Demos `001`–`024` draw jitter from `np.random.default_rng([seed, index])`.
3. **Jitter mimics human variation, inside what the task tolerates:**
   - grasp point: angle θ ~ U(−0.4, 0.4) rad around the bowl rim (radius 0.05 m, away from the
     cabinet at θ = 0); gripper yaw turns with θ so fingers still close across the rim;
   - approach and retreat height: 0.10 ± 0.02 m; place height: 0.03 ± 0.01 m above the plate
     anchor; place xy: ± 0.01 m;
   - each segment's duration × U(0.7, 1.3).
   Jittered demos that fail raw replay are kept on disk and simply drop out of Run A, as failed
   human demos will. The raw-replay success rate is a rehearsal number worth reporting.
4. **Each demo replays at its own init state.** Demo `i` replays at LIBERO init state `i % 50`
   (`--episode-index`), so Run A covers varied scenes the way varied real start poses will.
5. **Replay output folder is selectable.** `e2l sim replay` gains `--run-id` (default `replay`,
   today's behaviour) so rehearsal replays (`rehearsal_replay`) never mix with real ones.
6. **Run names.** `data/generated/rehearsal_replay` (Run A source), `data/generated/rehearsal_gen`
   (generated), `data/lerobot/rehearsal_a` = export of `rehearsal_replay`,
   `data/lerobot/rehearsal_b` = export of `rehearsal_replay` + `rehearsal_gen`. The existing
   `scripted50` pilot is left alone and excluded.
7. **Generation size.** 25 demos × `episodes_per_demo=16` = 400 attempts; at the pilot's 84%
   yield that is ~330 successes (≥ 300 target). If yield is lower, raise `episodes_per_demo` and
   rerun with `--overwrite`; seeds are per (demo, attempt) so the first 16 attempts reproduce.
   Disk: ~22 MB per episode → ~7 GB on `/mnt/data`; checked before launching.
8. **One reproducible driver.** `scripts/rehearsal.sh` runs write → replay → generate → export
   with the names above; `make rehearsal` calls it. Each step is idempotent or uses
   `--overwrite`, so a rerun rebuilds from scratch.
9. **Reference eval writes outside `docs/`.** `e2l eval rollout --run reference --setting
   {standard,wide}` writes `outputs/eval/reference/<setting>/`; `e2l eval report --out-dir
   outputs/report` checks the report path without touching `docs/figures` (Track 3).
10. **Held-out evaluation is a config invariant.** Wide placements already lie outside the
    generation square (`test_wide.py`). A repo-config test adds: `eval.wide.min_offset ≥
    generate.xy_range`, `eval.wide.min_yaw ≥ generate.yaw_range`, and `eval.wide.seed_offset ≥
    eval.n_episodes` (standard and wide seed ranges disjoint). The standard setting reuses
    LIBERO's init states, around which generation samples; it is in-distribution by design and
    the write-up says so.

## Components

| Unit | Change |
| --- | --- |
| `e2l_sim/scripted.py` (new) | `scripted_segments(demo_id, rng=None) -> RobotSegments`; nominal + jitter |
| `scripts/scripted_segments.py` | `--count N --seed S`; writes `20261003_scripted_000..N-1` |
| `e2l_sim/cli.py` `replay` | `--run-id` option |
| `scripts/rehearsal.sh`, `Makefile` | driver + `rehearsal` target |
| `e2l_common/tests/test_repo_configs.py` | eval/generate separation test |
| `CLAUDE.md` | rehearsal command line |

## Testing

- Unit (no LIBERO): nominal demo equals today's (array_equal); jitter is seeded and reproducible;
  jitter stays within the stated bounds; segment structure (3 segments, refs bowl/plate/plate,
  gripper flips) is preserved; grip-site z axis still points down.
- CLI: `e2l sim replay --run-id X` saves under `data/generated/X/` (replay monkeypatched).
- Config: the separation test above.
- Runs (verification, not tests): replay success count, generation manifest yield, both
  datasets load with `LeRobotDataset` and report episode/frame counts, reference results.json
  for both settings, `e2l eval report --out-dir outputs/report` succeeds.

## Out of scope

Training of any kind, the ablation, README/LICENSE/figures in `docs/`, anything needing real
video.
