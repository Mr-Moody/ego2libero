"""Policy rollouts in LIBERO through LeRobot's evaluation loop.

Uses lerobot's env factory, policy factory, pre/post-processors and `eval_policy`, so results
are directly comparable with `lerobot-eval` and published smolvla_libero numbers."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from e2l_common.config import EvalConfig, SimConfig
from e2l_eval.metrics import wilson_interval


@dataclass
class RolloutResult:
    policy: str
    task: str
    successes: list[bool]
    seeds: list[int]
    video_paths: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(self.successes) / len(self.successes) if self.successes else float("nan")


def _libero_env_config(sim_cfg: SimConfig):
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    from e2l_sim.env import libero_camera

    return LiberoEnvConfig(
        task=sim_cfg.suite,
        task_ids=[sim_cfg.task_id],
        fps=sim_cfg.control_freq,
        episode_length=sim_cfg.max_steps,
        camera_name=",".join(libero_camera(c) for c in sim_cfg.camera_names),
        observation_height=sim_cfg.image_size,
        observation_width=sim_cfg.image_size,
    )


def rollout(
    policy_path: str,
    cfg: EvalConfig,
    sim_cfg: SimConfig,
    out_dir: Path,
    n_episodes: int | None = None,
    max_videos: int = 1,
) -> RolloutResult:
    """Run `n_episodes` (default cfg.n_episodes) seeded episodes (seeds cfg.seed + i) of a policy
    on the chosen task, saving up to `max_videos` mp4s and `results.json` in `out_dir`."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    from e2l_sim.libero_setup import ensure_libero_config

    ensure_libero_config()
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    n_episodes = n_episodes or cfg.n_episodes
    set_seed(cfg.seed)
    env_cfg = _libero_env_config(sim_cfg)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = cfg.device

    env = make_env(env_cfg, n_envs=1)[sim_cfg.suite][sim_cfg.task_id]
    try:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            info = eval_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=n_episodes,
                max_episodes_rendered=max_videos,
                videos_dir=out_dir / "videos",
                start_seed=cfg.seed,
            )
    finally:
        env.close()

    episodes = info["per_episode"]
    result = RolloutResult(
        policy=policy_path,
        task=f"{sim_cfg.suite}/{sim_cfg.task_id}",
        successes=[bool(e["success"]) for e in episodes],
        seeds=[int(e["seed"]) for e in episodes],
        video_paths=[str(p) for p in info.get("video_paths", [])],
    )
    lo, hi = wilson_interval(sum(result.successes), len(result.successes))
    summary = {**result.__dict__, "success_rate": result.success_rate, "wilson_95": [lo, hi]}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return result
