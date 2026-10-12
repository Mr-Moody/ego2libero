from pathlib import Path

from typer.testing import CliRunner

from e2l_common.config import TrainConfig
from e2l_train.cli import app
from e2l_train.finetune import build_command


def test_build_command_from_config():
    cfg = TrainConfig(steps=10, batch_size=2, wandb=True, extra_args=["--policy.device=cpu"])
    cmd = build_command(cfg)
    assert cmd[0] == "lerobot-train"
    assert "--policy.path=lerobot/smolvla_base" in cmd
    assert "--steps=10" in cmd and "--batch_size=2" in cmd and "--wandb.enable=true" in cmd
    assert cmd[-1] == "--policy.device=cpu"


def test_cli_dry_run_prints_resolved_command():
    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    result = CliRunner().invoke(
        app, ["finetune", "-c", str(config), "--dry-run", "--set", "steps=7"]
    )
    assert result.exit_code == 0, result.output
    assert "lerobot-train" in result.output and "--steps=7" in result.output


def test_repo_config_trains_from_smolvla_base():
    from e2l_common.config import load_config

    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    cfg = load_config(config, model=TrainConfig)
    assert cfg.policy_path == "lerobot/smolvla_base"


def test_repo_config_maps_libero_cameras_onto_base():
    from e2l_common.config import load_config

    config = Path(__file__).parents[3] / "configs" / "train" / "smolvla.yaml"
    cmd = build_command(load_config(config, model=TrainConfig))
    rename = next(a for a in cmd if a.startswith("--rename_map="))
    assert '"observation.images.image": "observation.images.camera1"' in rename
    assert '"observation.images.image2": "observation.images.camera2"' in rename
    assert "--policy.empty_cameras=1" in cmd


def test_build_command_trains_from_hub_and_pushes_checkpoint():
    cfg = TrainConfig(
        dataset_root=None,
        dataset_repo_id="me/e2l-x-all_raw",
        push_repo_id="me/e2l-x-all_raw-policy",
    )
    cmd = build_command(cfg)
    assert "--dataset.repo_id=me/e2l-x-all_raw" in cmd
    assert not any(a.startswith("--dataset.root") for a in cmd)
    assert "--policy.push_to_hub=true" in cmd
    assert "--policy.repo_id=me/e2l-x-all_raw-policy" in cmd
    assert "--policy.private=true" in cmd
    assert "--policy.push_to_hub=false" not in cmd


def test_build_command_without_push_is_unchanged():
    cmd = build_command(TrainConfig())
    assert "--policy.push_to_hub=false" in cmd
    assert not any(a.startswith(("--policy.repo_id", "--policy.private")) for a in cmd)
    assert "--dataset.root=data/lerobot/ego2libero" in cmd


def test_launch_from_hub_skips_local_dataset_check(tmp_path, monkeypatch):
    import subprocess

    from e2l_train import finetune

    calls = []
    monkeypatch.setattr(finetune.shutil, "which", lambda _: "/bin/lerobot-train")
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
    cfg = TrainConfig(dataset_root=None, output_dir=tmp_path / "out")
    finetune.launch(cfg)
    assert calls and calls[0][0] == "/bin/lerobot-train"
