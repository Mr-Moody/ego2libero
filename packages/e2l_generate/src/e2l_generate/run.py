"""Sample, replay and keep successes; record yield per source demo in a manifest."""

import subprocess
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.paths import repo_root
from e2l_common.stub import not_implemented

Status = Literal["success", "failure", "placement_failed", "error"]
_COUNT_KEY = {
    "success": "successes",
    "failure": "failures",
    "placement_failed": "placement_failed",
    "error": "errors",
}


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one generation attempt; small enough to return from a worker process."""

    stem: str
    demo_id: str
    attempt: int
    episode_index: int
    status: Status
    steps: int = 0
    saved: bool = False
    error: str | None = None
    size_mb: float | None = None


def episode_stem(demo_id: str, k: int) -> str:
    return f"{demo_id}_{k:04d}"


def attempt_seed(seed: int, demo_id: str, k: int) -> np.random.Generator:
    """Generator for attempt `k` of `demo_id`: independent of worker count and scheduling."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(demo_id.encode()), k]))


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def summarise(
    results: list[AttemptResult], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig
) -> dict:
    """Manifest dict: configs, commit, per-demo counts and yield, one entry per attempt."""
    results = sorted(results, key=lambda r: (r.demo_id, r.attempt))
    demos: dict[str, dict] = {}
    for r in results:
        counts = demos.setdefault(
            r.demo_id, {"attempts": 0} | dict.fromkeys(_COUNT_KEY.values(), 0)
        )
        counts["attempts"] += 1
        counts[_COUNT_KEY[r.status]] += 1
    for counts in demos.values():
        counts["yield"] = counts["successes"] / counts["attempts"]
    sizes = [r.size_mb for r in results if r.saved and r.size_mb is not None]
    return {
        "run_id": run_id,
        "git_commit": git_commit(),
        "generate_config": cfg.model_dump(mode="json"),
        "sim_config": sim_cfg.model_dump(mode="json"),
        "demos": demos,
        "mean_episode_mb": round(float(np.mean(sizes)), 2) if sizes else None,
        "episodes": [{k: v for k, v in asdict(r).items() if k != "size_mb"} for r in results],
    }


def generate(
    demo_ids: list[str], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig, data_root: Path
) -> dict:
    """For each source demo, generate `cfg.episodes_per_demo` attempts in parallel workers and
    save successful SimEpisodes to data/generated/<run_id>/. Writes manifest.json with yield
    per source demo and provenance; returns the manifest dict."""
    not_implemented("e2l_generate.run.generate")
