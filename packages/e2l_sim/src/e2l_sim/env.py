"""Create LIBERO environments from `configs/sim.yaml`.

Wraps LeRobot's `LiberoEnv` so data generation, `e2l sim check` and policy evaluation all see
the observation format the smolvla_libero checkpoint was trained on (cameras `image` and
`image2`, 256x256, LIBERO init states, 10 settle steps after reset)."""

import os

from e2l_common.config import SimConfig
from e2l_sim.libero_setup import ensure_libero_config


def libero_camera(name: str) -> str:
    """robosuite camera name -> LIBERO observation key (agentview -> agentview_image)."""
    return name if name.endswith("_image") else f"{name}_image"


def make_env(cfg: SimConfig, episode_index: int = 0, obs_type: str = "pixels_agent_pos"):
    """LeRobot LiberoEnv for `cfg.suite` / `cfg.task_id`, rendering `cfg.camera_names` at
    `cfg.image_size` with `cfg.control_freq` Hz relative (delta) control. `episode_index` picks
    the LIBERO init state. Rendering is headless (MUJOCO_GL=egl unless already set)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    ensure_libero_config()
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(cfg.suite)
    task = suite.get_task(cfg.task_id)
    if cfg.task_name and task.name != cfg.task_name:
        raise ValueError(
            f"{cfg.suite} task {cfg.task_id} is {task.name!r}, but sim.yaml says {cfg.task_name!r}"
        )
    return LiberoEnv(
        task_suite=suite,
        task_id=cfg.task_id,
        task_suite_name=cfg.suite,
        episode_length=cfg.max_steps,
        camera_name=[libero_camera(c) for c in cfg.camera_names],
        obs_type=obs_type,
        observation_width=cfg.image_size,
        observation_height=cfg.image_size,
        init_states=True,
        episode_index=episode_index,
        control_freq=cfg.control_freq,
        control_mode="relative",
    )
