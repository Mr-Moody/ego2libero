"""`e2l sim check`: prove LIBERO creates, steps and renders headless."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from rich.console import Console
from rich.tree import Tree

from e2l_common.config import SimConfig
from e2l_sim.env import make_env


def _spec_tree(tree: Tree, value) -> None:
    for key, sub in value.items():
        if isinstance(sub, dict):
            _spec_tree(tree.add(f"[bold]{key}[/bold]"), sub)
        else:
            arr = np.asarray(sub)
            tree.add(f"{key}: {arr.dtype} {arr.shape}")


def object_positions(env) -> dict[str, np.ndarray]:
    """World positions of the task's movable objects (LIBERO sim frame, metres)."""
    inner = env._env.env  # LiberoEnv -> OffScreenRenderEnv -> LIBERO problem env
    return {
        name: inner.sim.data.body_xpos[inner.obj_body_id[name]].copy()
        for name in getattr(inner, "objects_dict", {})
    }


def run_check(cfg: SimConfig, steps: int, out_dir: Path) -> None:
    console = Console()
    env = make_env(cfg)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        console.print(f"[bold]{cfg.suite}[/bold] task {cfg.task_id}: {env.task}")
        console.print(f'instruction: "{env.task_description}"')

        rng = np.random.default_rng(cfg.seed)
        low, high = env.action_space.low, env.action_space.high
        success = False
        for _ in range(steps):
            obs, _, terminated, _, info = env.step(rng.uniform(low, high).astype(np.float32))
            success |= bool(info["is_success"])
            if terminated:
                break

        out_dir.mkdir(parents=True, exist_ok=True)
        for cam, img in obs["pixels"].items():
            # LIBERO renders upside down; flip for viewing only (policies see the raw image).
            path = out_dir / f"{cam}.png"
            iio.imwrite(path, np.ascontiguousarray(img[::-1, ::-1]))
            console.print(f"wrote {path}")

        tree = Tree("[bold]observation spec[/bold]")
        _spec_tree(tree, obs)
        console.print(tree)
        console.print(
            f"[bold]action spec[/bold]: {env.action_space.dtype} {env.action_space.shape}, "
            f"low {low.tolist()}, high {high.tolist()}  "
            "(dpos xyz, drot axis-angle xyz, gripper -1 open / +1 closed)"
        )
        for name, pos in object_positions(env).items():
            console.print(f"object {name}: T_sim_obj translation {np.round(pos, 3).tolist()}")
        console.print(f"stepped {steps} random actions, success={success}")
    finally:
        env.close()
