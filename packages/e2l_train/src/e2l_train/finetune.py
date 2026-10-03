"""Build and launch `lerobot-train` for SmolVLA from `configs/train/smolvla.yaml`."""

import shlex
import shutil
import subprocess

from rich.console import Console

from e2l_common.config import TrainConfig


def build_command(cfg: TrainConfig) -> list[str]:
    """The full `lerobot-train` argv for a config (fine-tune from `cfg.policy_path`)."""
    return [
        "lerobot-train",
        f"--policy.path={cfg.policy_path}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={cfg.dataset_repo_id}",
        f"--dataset.root={cfg.dataset_root}",
        f"--output_dir={cfg.output_dir}",
        f"--job_name={cfg.output_dir.name}",
        f"--steps={cfg.steps}",
        f"--batch_size={cfg.batch_size}",
        f"--num_workers={cfg.num_workers}",
        f"--save_freq={cfg.save_freq}",
        f"--wandb.enable={str(cfg.wandb).lower()}",
        *cfg.extra_args,
    ]


def launch(cfg: TrainConfig, dry_run: bool = False) -> list[str]:
    """Print the resolved command; unless `dry_run`, check inputs and run it."""
    cmd = build_command(cfg)
    Console().print(shlex.join(cmd), soft_wrap=True, highlight=False)
    if dry_run:
        return cmd
    if not (cfg.dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"no LeRobot dataset at {cfg.dataset_root}; run `e2l train export`")
    if cfg.output_dir.exists():
        raise FileExistsError(f"{cfg.output_dir} exists; lerobot-train will not overwrite it")
    executable = shutil.which("lerobot-train")
    if executable is None:
        raise RuntimeError(
            "lerobot-train not found; install the train group (uv sync --group train)"
        )
    subprocess.run([executable, *cmd[1:]], check=True)
    return cmd
