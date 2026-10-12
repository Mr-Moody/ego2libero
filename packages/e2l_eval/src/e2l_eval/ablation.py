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

from e2l_common.config import AblationConfig, EvalConfig, TrainConfig
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
    push plan.json + spark.sh. An existing local export is reused (delete it to rebuild) and a
    dataset repo already on the Hub is not pushed again. Returns the plan.json path."""
    ab = cfg.ablation
    cells = select_cells(ab)
    labels = [c.label for c in cells]
    chosen = labels if only is None else only
    unknown = [c for c in chosen if c not in labels]
    if unknown:
        raise ValueError(f"unknown cells {unknown}; this plan has {labels}")
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
            "prepared": cell.label in chosen,
        }
        if row["prepared"]:
            root = data.root / "lerobot" / f"{ab.name}-{cell.label}"
            if not root.exists():
                update = {"dataset_root": root, "dataset_repo_id": dataset_repo}
                export_dataset(list(cell.episodes), train_cfg.model_copy(update=update))
            row["frames"] = json.loads((root / "meta" / "info.json").read_text())["total_frames"]
            if not hub.repo_exists(dataset_repo, "dataset"):
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
    out = plan_root / ab.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    (out / "spark.sh").write_text(spark_script(plan))
    hub.push_folder(out, plan["plan_repo"], ab.private)
    return out / "plan.json"


def spark_script(plan: dict) -> str:
    """Bash for the DGX Spark: one `e2l train finetune` per prepared cell, training from the
    cell's Hub dataset and pushing the checkpoint; cells with an output dir are skipped."""
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
        lines += [
            f"if [ -d outputs/train/{run} ]; then",
            f"  echo 'skip {run}: outputs/train/{run} exists (delete it to retrain)'",
            "else",
            f"  uv run e2l train finetune {args}",
            "fi",
        ]
    return "\n".join(lines) + "\n"
