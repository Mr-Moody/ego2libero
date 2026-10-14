"""Demo count x augmentation sweep, trained on the DGX Spark through the Hugging Face Hub.

`prepare` exports and pushes one LeRobot dataset per cell and writes plan.json + spark.sh; the
user runs spark.sh on the Spark (one `e2l train finetune` per cell, each pushing its checkpoint);
`evaluate` pulls every trained checkpoint and runs both eval settings. Run A and Run B are the
`all_raw` and `all_aug` cells."""

import json
import shlex
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from e2l_common.config import AblationConfig, EvalConfig, SimConfig, TrainConfig
from e2l_common.paths import DataPaths
from e2l_eval import hub
from e2l_train.export import export_dataset

PLAN_ROOT = Path("outputs/ablation")


@dataclass(frozen=True)
class Cell:
    label: str
    n_demos: int
    augmented: bool
    demo_ids: tuple[str, ...]
    episodes: tuple[Path, ...]  # SimEpisode npz files: the demos' replays, then generated


def cell_label(count: int | str, augmented: bool) -> str:
    size = "all" if count == "all" else f"n{int(count):02d}"
    return f"{size}_{'aug' if augmented else 'raw'}"


def _sidecar(npz: Path) -> dict:
    return json.loads(npz.with_suffix(".json").read_text())


def eligible_demos(replay_dir: Path) -> list[str]:
    """Sorted demo ids whose raw replay in `replay_dir` succeeded (file stem = demo id)."""
    return sorted(p.stem for p in replay_dir.glob("*.npz") if _sidecar(p)["success"])


def select_cells(cfg: AblationConfig) -> list[Cell]:
    """Every (demo count, augmentation) cell in config order. Eligible demos are shuffled once
    with rng(subset_seed) and cell n takes the first n, so smaller subsets nest in larger ones.
    `aug` cells add every successful generated episode whose sources are all in the subset."""
    paths = DataPaths(cfg.data_root)
    replay_dir = paths.generated(cfg.replay_run)
    eligible = eligible_demos(replay_dir)
    if not eligible:
        raise ValueError(f"no successful replays in {replay_dir}")
    order = [eligible[i] for i in np.random.default_rng(cfg.subset_seed).permutation(len(eligible))]
    gen_dir = paths.generated(cfg.generated_run)
    generated = [
        (p, set(meta["source_demo_ids"]))
        for p in sorted(gen_dir.glob("*.npz"))
        if (meta := _sidecar(p))["success"]
    ]
    cells = []
    for count in cfg.demo_counts:
        n = len(order) if count == "all" else count
        if n > len(order):
            raise ValueError(
                f"demo count {count} exceeds the {len(order)} demos whose replay succeeded "
                f"in {replay_dir}"
            )
        demos = tuple(sorted(order[:n]))
        raw = tuple(replay_dir / f"{d}.npz" for d in demos)
        for augmented in cfg.augmentation:
            extra = tuple(p for p, src in generated if src <= set(demos)) if augmented else ()
            if augmented and not extra:
                raise ValueError(
                    f"no successful generated episodes for {cell_label(count, True)} in {gen_dir}"
                )
            cells.append(Cell(cell_label(count, augmented), n, augmented, demos, raw + extra))
    return cells


def repo_ids(namespace: str, plan: str, label: str) -> tuple[str, str]:
    """(dataset repo, policy repo) for one cell."""
    base = f"{namespace}/e2l-{plan}-{label}"
    return base, f"{base}-policy"


def prepare(
    cfg: EvalConfig,
    train_cfg: TrainConfig,
    only: list[str] | None = None,
    plan_root: Path = PLAN_ROOT,
) -> Path:
    """Export and push the datasets of the selected cells (`only`, default all), then write and
    push plan.json + spark.sh. An existing local export is reused if it holds every episode of
    the cell (delete it to rebuild) and a dataset already on the Hub is not pushed again. Cells
    prepared by an earlier run with the same `ablation` settings stay prepared. Returns the
    plan.json path."""
    ab = cfg.ablation
    cells = select_cells(ab)
    labels = [c.label for c in cells]
    chosen = labels if only is None else only
    unknown = [c for c in chosen if c not in labels]
    if unknown:
        raise ValueError(f"unknown cells {unknown}; this plan has {labels}")
    out = plan_root / ab.name
    earlier = _earlier_prepared(out / "plan.json", ab)
    ns = hub.namespace(ab.hub_namespace)
    data = DataPaths(ab.data_root)
    rows = []
    for cell in cells:
        dataset_repo, policy_repo = repo_ids(ns, ab.name, cell.label)
        row = {
            "label": cell.label,
            "n_demos": cell.n_demos,
            "augmented": cell.augmented,
            "demo_ids": list(cell.demo_ids),
            "episodes": len(cell.episodes),
            "frames": None,
            "dataset_repo": dataset_repo,
            "policy_repo": policy_repo,
            "prepared": cell.label in chosen or cell.label in earlier,
        }
        if row["prepared"]:
            root = data.root / "lerobot" / f"{ab.name}-{cell.label}"
            if not root.exists():
                update = {"dataset_root": root, "dataset_repo_id": dataset_repo}
                export_dataset(list(cell.episodes), train_cfg.model_copy(update=update))
            info = json.loads((root / "meta" / "info.json").read_text())
            if info["total_episodes"] != len(cell.episodes):
                raise ValueError(
                    f"{root} has {info['total_episodes']} episodes but cell {cell.label} has "
                    f"{len(cell.episodes)} (interrupted export or changed settings); delete it "
                    "to rebuild"
                )
            row["frames"] = info["total_frames"]
            if not hub.has_file(dataset_repo, "meta/info.json", "dataset"):
                hub.push_dataset(root, dataset_repo, ab.private)
        rows.append(row)
    plan = {
        "name": ab.name,
        "created": date.today().isoformat(),
        "namespace": ns,
        "plan_repo": f"{ns}/e2l-{ab.name}-plan",
        "steps": ab.steps,
        "batch_size": train_cfg.batch_size,
        "ablation": ab.model_dump(mode="json"),
        "cells": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "spark.sh").write_text(spark_script(plan))
    hub.push_folder(out, plan["plan_repo"], ab.private)
    return out / "plan.json"


def _earlier_prepared(plan_path: Path, ab: AblationConfig) -> set[str]:
    """Labels prepared by an earlier `prepare` of this plan with identical ablation settings."""
    if not plan_path.is_file():
        return set()
    plan = json.loads(plan_path.read_text())
    if plan["ablation"] != ab.model_dump(mode="json"):
        return set()
    return {row["label"] for row in plan["cells"] if row["prepared"]}


def spark_script(plan: dict) -> str:
    """Bash for the DGX Spark: one `e2l train finetune` per prepared cell, training from the
    cell's Hub dataset and pushing the checkpoint. A finished cell (done marker) is skipped; an
    output dir without the marker is a crashed run and stops the script until it is deleted."""
    name = plan["name"]
    private = str(plan["ablation"]["private"]).lower()
    lines = [
        "#!/usr/bin/env bash",
        f"# Ablation plan {name!r}, written by `e2l eval ablate prepare` on {plan['created']}.",
        "# On the DGX Spark, in a clone of ego2libero:",
        "#   git pull && uv sync --group train --group dev",
        f"#   uv run hf download {plan['plan_repo']} --repo-type dataset "
        f"--local-dir outputs/ablation/{name}",
        f"#   bash outputs/ablation/{name}/spark.sh",
        "set -euo pipefail",
        "unset PYTHONPATH",
    ]
    for row in plan["cells"]:
        if not row["prepared"]:
            continue
        run = f"{name}-{row['label']}"
        sets = {
            "dataset_root": "null",
            "dataset_repo_id": row["dataset_repo"],
            "output_dir": f"outputs/train/{run}",
            "steps": plan["steps"],
            "push_repo_id": row["policy_repo"],
            "private": private,
        }
        args = " ".join(f"--set {shlex.quote(f'{k}={v}')}" for k, v in sets.items())
        out = f"outputs/train/{run}"
        lines += [
            f"if [ -f {out}/.e2l-done ]; then",
            f"  echo 'skip {run}: already trained and pushed (delete {out} to retrain)'",
            f"elif [ -d {out} ]; then",
            f"  echo 'partial run in {out}: delete it to retrain' >&2",
            "  exit 1",
            "else",
            f"  uv run e2l train finetune {args}",
            f"  touch {out}/.e2l-done",
            "fi",
        ]
    return "\n".join(lines) + "\n"


def evaluate(
    cfg: EvalConfig, sim_cfg: SimConfig, plan_path: Path, videos: int = 3
) -> dict[str, str]:
    """For each prepared cell whose checkpoint is on the Hub, run every setting that has no
    results.json yet under cfg.output_dir/<plan>-<cell>/<setting>/. Returns label -> status:
    done (all settings present), evaluated, pending (no checkpoint yet), not prepared."""
    from e2l_eval import rollout as rollout_module

    plan = json.loads(Path(plan_path).read_text())
    status = {}
    for row in plan["cells"]:
        label, run = row["label"], f"{plan['name']}-{row['label']}"
        if not row["prepared"]:
            status[label] = "not prepared"
            continue
        todo = [
            s
            for s in rollout_module.SETTINGS
            if not (cfg.output_dir / run / s / "results.json").is_file()
        ]
        if not todo:
            status[label] = "done"
            continue
        if not hub.has_file(row["policy_repo"], "config.json", "model"):
            status[label] = "pending"
            continue
        policy = hub.download_policy(row["policy_repo"])
        for setting in todo:
            rollout_module.rollout(
                str(policy),
                cfg,
                sim_cfg,
                cfg.output_dir / run / setting,
                max_videos=videos,
                setting=setting,
                run=run,
            )
        status[label] = "evaluated"
    return status
