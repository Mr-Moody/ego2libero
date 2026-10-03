"""`e2l eval baseline`: run the pretrained smolvla_libero checkpoint on the chosen task."""

from pathlib import Path

from rich.console import Console

from e2l_common.config import EvalConfig, SimConfig
from e2l_eval.metrics import wilson_interval
from e2l_eval.rollout import RolloutResult, rollout


def run_baseline(cfg: EvalConfig, sim_cfg: SimConfig, episodes: int, out_dir: Path) -> float:
    """Evaluate `cfg.policy_path` for `episodes` seeded episodes; print the success rate with a
    95% Wilson interval and save one rollout video. Returns the success rate."""
    result: RolloutResult = rollout(
        cfg.policy_path, cfg, sim_cfg, out_dir, n_episodes=episodes, max_videos=1
    )
    k, n = sum(result.successes), len(result.successes)
    lo, hi = wilson_interval(k, n)
    console = Console()
    console.print(f"[bold]{result.policy}[/bold] on {result.task} ({sim_cfg.task_name})")
    console.print(f"seeds {result.seeds}: ", end="")
    console.print(" ".join("✓" if s else "✗" for s in result.successes))
    console.print(f"success rate {k}/{n} = {result.success_rate:.0%} (95% CI {lo:.0%}-{hi:.0%})")
    for path in result.video_paths:
        console.print(f"video: {path}")
    return result.success_rate
