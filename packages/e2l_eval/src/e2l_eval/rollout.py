"""Policy rollouts in LIBERO."""

from pathlib import Path

from e2l_common.config import EvalConfig, SimConfig
from e2l_common.stub import not_implemented


def rollout(policy_path: str, cfg: EvalConfig, sim_cfg: SimConfig, out_dir: Path) -> list[bool]:
    """Run `cfg.n_episodes` seeded episodes (seeds cfg.seed + i) of a policy on the chosen
    task, saving one mp4 per episode to `out_dir`. Returns per-episode success."""
    not_implemented("e2l_eval.rollout.rollout")
