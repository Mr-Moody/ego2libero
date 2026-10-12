"""Demo count x augmentation sweep, trained on the DGX Spark through the Hugging Face Hub.

`prepare` exports and pushes one LeRobot dataset per cell and writes plan.json + spark.sh; the
user runs spark.sh on the Spark (one `e2l train finetune` per cell, each pushing its checkpoint);
`evaluate` pulls every trained checkpoint and runs both eval settings. Run A and Run B are the
`all_raw` and `all_aug` cells."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e2l_common.config import AblationConfig
from e2l_common.paths import DataPaths


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
