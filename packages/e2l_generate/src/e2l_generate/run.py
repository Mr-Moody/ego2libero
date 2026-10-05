"""Sample, replay and keep successes; record yield per source demo in a manifest."""

import json
import shutil
import subprocess
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import numpy as np

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.log import get_logger
from e2l_common.paths import DataPaths, repo_root
from e2l_common.schemas import RobotSegments
from e2l_generate.sample import sample_object_poses
from e2l_sim.env import init_state_count
from e2l_sim.replay import replay, sim_object
from e2l_sim.scene import nominal_object_poses

log = get_logger(__name__)
# Per-process caches: each worker loads a demo and an init state's nominal poses only once.
_segments_cache: dict[Path, RobotSegments] = {}
_nominal_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

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


def _segments(path: Path) -> RobotSegments:
    if path not in _segments_cache:
        _segments_cache[path] = RobotSegments.load(path)
    return _segments_cache[path]


def _nominal(sim_cfg: SimConfig, episode_index: int) -> dict[str, np.ndarray]:
    key = (sim_cfg.model_dump_json(), episode_index)
    if key not in _nominal_cache:
        _nominal_cache[key] = nominal_object_poses(sim_cfg, episode_index)
    return _nominal_cache[key]


def run_attempt(
    demo_id: str,
    k: int,
    n_init: int,
    run_id: str,
    cfg: GenerateConfig,
    sim_cfg: SimConfig,
    data_root: Path,
) -> AttemptResult:
    """One attempt: sample a placement around init state k % n_init, replay, save the episode
    if it succeeded (or if cfg.keep_failures). Exceptions become an `error` result."""
    paths = DataPaths(data_root)
    stem = episode_stem(demo_id, k)
    index = k % n_init
    base = {"stem": stem, "demo_id": demo_id, "attempt": k, "episode_index": index}
    try:
        segments = _segments(paths.robot_segments(demo_id))
        names = {sim_object(seg.ref_object, sim_cfg) for seg in segments.segments}
        rng = attempt_seed(cfg.seed, demo_id, k)
        poses = sample_object_poses(_nominal(sim_cfg, index), names, cfg, rng)
        if poses is None:
            return AttemptResult(**base, status="placement_failed")
        episode = replay(segments, sim_cfg, T_sim_obj=poses, episode_index=index)
        status = "success" if episode.success else "failure"
        steps = len(episode.actions)
        if not (episode.success or cfg.keep_failures):
            return AttemptResult(**base, status=status, steps=steps)
        npz, sidecar = episode.save(paths.generated(run_id, stem))
        size_mb = (npz.stat().st_size + sidecar.stat().st_size) / 1e6
        return AttemptResult(**base, status=status, steps=steps, saved=True, size_mb=size_mb)
    except Exception as e:  # noqa: BLE001 - one bad attempt must not stop the run
        return AttemptResult(**base, status="error", error=f"{type(e).__name__}: {e}")


def _logged(r: AttemptResult) -> AttemptResult:
    detail = f": {r.error}" if r.error else f" ({r.steps} steps)"
    log.info(f"{r.stem} init {r.episode_index}: {r.status}{detail}")
    return r


def generate(
    demo_ids: list[str],
    run_id: str,
    cfg: GenerateConfig,
    sim_cfg: SimConfig,
    data_root: Path,
    overwrite: bool = False,
) -> dict:
    """For each source demo, run `cfg.episodes_per_demo` attempts (inline if cfg.workers <= 1,
    else in spawned worker processes) and save episodes to data/generated/<run_id>/. Writes
    manifest.json with yield per source demo and provenance; returns the manifest dict."""
    paths = DataPaths(Path(data_root))
    run_dir = paths.generated(run_id)
    missing = [d for d in demo_ids if not paths.robot_segments(d).with_suffix(".npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"no robot segments for {missing} in {paths.root / 'robot_segments'}"
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} is not empty; pass overwrite to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(d, k) for d in demo_ids for k in range(cfg.episodes_per_demo)]
    results: list[AttemptResult] = []
    if jobs:
        args = (init_state_count(sim_cfg), run_id, cfg, sim_cfg, paths.root)
        if cfg.workers <= 1:
            results = [_logged(run_attempt(d, k, *args)) for d, k in jobs]
        else:
            # spawn, not fork: EGL rendering contexts do not survive a fork.
            with ProcessPoolExecutor(cfg.workers, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(run_attempt, d, k, *args) for d, k in jobs]
                results = [_logged(f.result()) for f in as_completed(futures)]

    manifest = summarise(results, run_id, cfg, sim_cfg)
    paths.manifest(run_id).write_text(json.dumps(manifest, indent=2))
    return manifest
