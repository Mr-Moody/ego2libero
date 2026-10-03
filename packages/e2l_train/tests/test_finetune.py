from pathlib import Path

from typer.testing import CliRunner

from e2l_common.config import TrainConfig
from e2l_train.cli import app
from e2l_train.finetune import build_command


def test_build_command_from_config():
    cfg = TrainConfig(steps=10, batch_size=2, wandb=True, extra_args=["--policy.device=cpu"])
    cmd = build_command(cfg)
    assert cmd[0] == "lerobot-train"
    assert "--policy.path=HuggingFaceVLA/smolvla_libero" in cmd
    assert "--steps=10" in cmd and "--batch_size=2" in cmd and "--wandb.enable=true" in cmd
    assert cmd[-1] == "--policy.device=cpu"


def test_cli_dry_run_prints_resolved_command():
    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    result = CliRunner().invoke(
        app, ["finetune", "-c", str(config), "--dry-run", "--set", "steps=7"]
    )
    assert result.exit_code == 0, result.output
    assert "lerobot-train" in result.output and "--steps=7" in result.output
