# Hub-mediated training and ablation — design

Status: approved in brainstorming, 2026-10-12. Track 2 of the pre-demo work; builds on Track 1
(`2026-10-12-scripted-rehearsal-design.md`, PR #9). Track 3 (README, LICENSE, figures in
`docs/`) still waits for real results.

## Goal

One code path takes any set of replayed + generated episodes to trained SmolVLA checkpoints and
evaluated results, with training on the DGX Spark and everything else on the laptop:

- **Ablation**: demo count × augmentation. Run A (`all_raw`) and Run B (`all_aug`) are two
  cells of the same grid, so the headline runs and the ablation share one implementation.
- **Smoke run**: 200 steps on the Spark proves Hub push → train → Hub push → laptop pull →
  `e2l eval rollout` loads the checkpoint, and measures training speed.
- **Rehearsal Run A / Run B**: the two headline cells trained on Track 1's scripted rehearsal
  data and evaluated on `standard` and `wide`, 50 episodes each.

When real demos exist the same commands run with the real `replay` / generated runs and the full
grid.

## Decisions

1. **The Hugging Face Hub carries all transfers.** The laptop pushes each cell's LeRobot dataset
   as a private dataset repo; the Spark trains from the Hub id (no local copy needed) and
   `lerobot-train` pushes the final checkpoint (which includes `train_config.json`, so the
   camera rename map survives) to a private model repo; the laptop downloads it with
   `snapshot_download`. Prerequisite on both machines: `hf auth login` with a write token, using
   `HF_HOME=/mnt/data/hf-cache` on the laptop.
2. **Repo names.** Dataset `<ns>/e2l-<plan>-<cell>`, policy `<ns>/e2l-<plan>-<cell>-policy`,
   where `<ns>` is `ablation.hub_namespace` or, if unset, the logged-in user (`whoami`).
   `<plan>` is `ablation.name` (`rehearsal`, later e.g. `real`); `<cell>` is the cell label.
3. **Cells.** For each `n` in `demo_counts` and each arm in `augmentation`: label
   `n05_raw` / `n05_aug` (zero-padded to 2 digits), or `all_raw` / `all_aug` for `"all"`.
   `raw` = the successful raw replays of the cell's demos; `aug` = those replays plus every
   successful generated episode whose `source_demo_ids` are all inside the cell's demos.
   Counts larger than the number of eligible demos are an error at prepare time, naming both.
4. **Eligible demos and nested subsets.** Eligible = demos whose raw replay in
   `ablation.replay_run` succeeded (a demo the robot cannot replay contributes nothing to `raw`,
   so it is excluded from both arms to keep them comparable). Eligible ids are sorted, shuffled
   once with `np.random.default_rng(ablation.subset_seed)`, and cell `n` takes the first `n`, so
   the 5-demo set is inside the 10-demo set. `"all"` takes every eligible demo.
5. **Equal training budget.** Every cell trains for the same `ablation.steps`, with the batch
   size and seed from `configs/train/smolvla.yaml`. `steps` is set from the smoke run's measured
   speed (Decision 9).
6. **Two commands around the Spark, driven by a plan file.**
   - `e2l eval ablate prepare [--cells a,b]` exports each selected cell's dataset under
     `<data_root>/lerobot/<plan>-<cell>`, pushes it, and writes
     `outputs/ablation/<plan>/plan.json` (cells, demo ids, episode and frame counts, repo ids,
     steps, configs) and `outputs/ablation/<plan>/spark.sh`. `--cells` limits which cells are
     exported, pushed and scripted; `plan.json` still lists every cell of the grid, each with a
     `prepared` flag. A cell whose dataset repo already exists on the Hub is not re-pushed.
   - `spark.sh` is run by the user in a clone of this repo on the Spark (`git pull`,
     `uv sync --group train --group dev` first): one `e2l train finetune` per prepared cell, in
     order, each pushing its checkpoint. It stops at the first failure (`set -euo pipefail`).
   - `e2l eval ablate evaluate <plan>` runs, for every prepared cell whose policy repo exists on
     the Hub and whose `outputs/eval/<plan>-<cell>/<setting>/results.json` is missing,
     `rollout(..., setting)` for `standard` and `wide`. Cells not yet trained are listed as
     pending, not errors; rerunning after each Spark job picks up new checkpoints.
7. **Train config changes.** `TrainConfig.dataset_root` becomes optional (`None` = load
   `dataset_repo_id` from the Hub; the local `meta/info.json` check is skipped and no
   `--dataset.root` is passed). New `push_repo_id: str | None` and `private: bool = True` add
   `--policy.push_to_hub=true --policy.repo_id=<id> --policy.private=<bool>`; without
   `push_repo_id` the command is unchanged (`--policy.push_to_hub=false`).
8. **Config.** `EvalConfig.demo_counts` / `augmentation` move into a new `ablation:` section:
   `name`, `data_root`, `replay_run`, `generated_run`, `demo_counts: [5, 10, "all"]`,
   `augmentation: [false, true]`, `subset_seed: 0`, `steps: 20000`, `hub_namespace: null`,
   `private: true`. Defaults describe the real run (`data_root: data`, `replay_run: replay`,
   `generated_run: run0`). The rehearsal and smoke plans are `--set ablation.*` overrides, not
   extra config files; `plan.json` records the resolved values.
9. **Smoke run = a one-cell plan.** `prepare --cells all_raw` with `ablation.name=smoke`,
   `ablation.steps=200` on the rehearsal data; after the Spark run, `evaluate smoke` with
   `--set n_episodes=2` proves loading. Steps/s is read from the `lerobot-train` log on the
   Spark; `steps` for the rehearsal A/B is then chosen so each fine-tune takes ≤ 3 h, capped at
   the 20 000 default.
10. **Reporting.** `e2l eval report` keeps its table/chart/GIF and, when
    `outputs/ablation/*/plan.json` exists, adds `ablation_<plan>.png`: success rate vs demo count,
    `raw` and `aug` lines with Wilson error bars, one panel per setting, plotting only cells with
    results. Track 2 always passes `--out-dir outputs/report` (`docs/` untouched).
11. **Disk.** `/mnt/data` is the constraint. `prepare` writes only the selected cells' datasets;
    after a cell is pushed its local export is kept (evaluate does not need it, but a re-push
    would). Track 1's `data/rehearsal/lerobot/rehearsal_{a,b}` (6.4 GB) duplicate the rehearsal
    `all_raw` / `all_aug` exports and may be deleted once those are pushed.

## Components

| Unit | Change |
| --- | --- |
| `e2l_common/config.py` | `AblationConfig`; `EvalConfig.ablation`; `TrainConfig.dataset_root` optional, `push_repo_id`, `private` |
| `configs/eval.yaml` | `ablation:` section replaces `demo_counts` / `augmentation` |
| `e2l_train/finetune.py` | Hub dataset (no root) and push flags in `build_command` / `launch` |
| `e2l_train/export.py` | unchanged API; `prepare` calls `export_dataset` with an episode list |
| `e2l_eval/ablation.py` | `Cell`, `eligible_demos`, `select_cells`, `prepare`, `spark_script`, `evaluate` |
| `e2l_eval/hub.py` (new) | thin wrappers: `namespace`, `repo_exists`, `push_dataset`, `download_policy` (lazy `huggingface_hub` / LeRobot imports) |
| `e2l_eval/cli.py` | `ablate prepare`, `ablate evaluate` (replaces `ablate`) |
| `e2l_eval/report.py` | `plot_ablation`; called from `make_report` per plan |
| `CLAUDE.md` | ablation commands |

## Testing

- Unit, no network, no LIBERO: cell labels; eligibility from replay sidecars; nested subsets
  stable under the seed; `aug` picks exactly the generated episodes whose sources are inside the
  cell; too-large counts raise naming both numbers; `plan.json` round-trip and `prepared`
  flags; `spark.sh` contains one finetune per prepared cell with the right repo ids, steps and
  `set -euo pipefail`; `build_command` with/without `dataset_root` and `push_repo_id`;
  `evaluate` skip logic with `hub` and `rollout` monkeypatched (done, pending, new);
  `plot_ablation` on synthetic results; config loads with the new section.
- Hub wrappers are tested with `huggingface_hub` calls monkeypatched (argument shapes only).
- Runs (verification): smoke chain end to end; rehearsal `all_raw` / `all_aug` trained and
  evaluated, report built into `outputs/report`.

## Out of scope

Training anywhere but the Spark; the full rehearsal grid (only `all_raw` / `all_aug` are
trained; other cells are unit-tested); real-demo runs; README / LICENSE / `docs/figures`;
automating the Spark side beyond a generated shell script.
