"""Policy rollouts in LIBERO through LeRobot's evaluation loop.

Uses lerobot's env factory, policy factory, pre/post-processors and `eval_policy`, so results
are directly comparable with `lerobot-eval` and published smolvla_libero numbers."""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from e2l_common.config import EvalConfig, SimConfig
from e2l_eval.metrics import wilson_interval

SETTINGS = ("standard", "wide")


@dataclass
class RolloutResult:
    policy: str
    task: str
    successes: list[bool]
    seeds: list[int]
    video_paths: list[str] = field(default_factory=list)
    run: str = "custom"
    setting: str = "standard"
    placements: list[dict] = field(default_factory=list)  # wide only: per-episode object poses

    @property
    def success_rate(self) -> float:
        return sum(self.successes) / len(self.successes) if self.successes else float("nan")


def write_results(result: RolloutResult, out_dir: Path) -> Path:
    """`out_dir/results.json`: the result plus success rate, 95% Wilson interval and the outcome
    of each rendered episode (video i is episode i)."""
    lo, hi = wilson_interval(sum(result.successes), len(result.successes))
    summary = {
        **result.__dict__,
        "video_successes": result.successes[: len(result.video_paths)],
        "success_rate": result.success_rate,
        "wilson_95": [lo, hi],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


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


def checkpoint_rename_map(policy_path: str) -> dict[str, str]:
    """Camera rename_map a fine-tuned checkpoint was trained with ({} for hub ids / no file)."""
    train_config = Path(policy_path) / "train_config.json"
    if not train_config.is_file():
        return {}
    return dict(json.loads(train_config.read_text()).get("rename_map") or {})


def rollout(
    policy_path: str,
    cfg: EvalConfig,
    sim_cfg: SimConfig,
    out_dir: Path,
    n_episodes: int | None = None,
    max_videos: int = 1,
    setting: str = "standard",
    run: str = "custom",
) -> RolloutResult:
    """Run `n_episodes` (default cfg.n_episodes) seeded episodes of a policy on the chosen task,
    saving up to `max_videos` mp4s and `results.json` in `out_dir`. `standard` uses seeds
    cfg.seed + i on LIBERO's init states; `wide` uses seeds cfg.seed + cfg.wide.seed_offset + i
    and moves the task objects after each reset (`e2l_sim.wide`)."""
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    from e2l_sim.libero_setup import ensure_libero_config

    ensure_libero_config()
    import gymnasium as gym
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
    if cfg.n_action_steps is not None:
        policy_cfg.n_action_steps = cfg.n_action_steps

    if setting == "wide":
        from e2l_sim.env import make_env as make_sim_env

        env = gym.vector.SyncVectorEnv([partial(make_sim_env, sim_cfg, 0, wide=cfg.wide)])
        start_seed = cfg.seed + cfg.wide.seed_offset
    else:
        env = make_env(env_cfg, n_envs=1)[sim_cfg.suite][sim_cfg.task_id]
        start_seed = cfg.seed
    try:
        policy = make_policy(
            cfg=policy_cfg, env_cfg=env_cfg, rename_map=checkpoint_rename_map(policy_path)
        )
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
                start_seed=start_seed,
            )
        placements = list(env.envs[0].placements) if setting == "wide" else []
    finally:
        env.close()

    episodes = info["per_episode"]
    result = RolloutResult(
        policy=policy_path,
        task=f"{sim_cfg.suite}/{sim_cfg.task_id}",
        successes=[bool(e["success"]) for e in episodes],
        seeds=[int(e["seed"]) for e in episodes],
        video_paths=[str(p) for p in info.get("video_paths", [])],
        run=run,
        setting=setting,
        placements=placements,
    )
    write_results(result, out_dir)
    return result
