import pytest
import typer
from typer.testing import CliRunner

from e2l_common.cli import app as root
from e2l_common.config import EvalConfig
from e2l_common.paths import repo_root
from e2l_eval import rollout as rollout_module
from e2l_eval.cli import app, resolve_policy

CFG = EvalConfig(runs={"reference": "hub/ref", "B": "ckpt/b"})
CONFIGS = repo_root() / "configs"
BASE = ["rollout", "-c", str(CONFIGS / "eval.yaml"), "--sim-config", str(CONFIGS / "sim.yaml")]


def test_resolve_policy():
    assert resolve_policy(CFG, "B", None) == ("B", "ckpt/b")
    assert resolve_policy(CFG, "B", "other/ckpt") == ("B", "other/ckpt")
    assert resolve_policy(CFG, None, "x/y") == ("custom", "x/y")
    assert resolve_policy(CFG, None, None) == ("custom", CFG.policy_path)
    with pytest.raises(typer.BadParameter, match="B, reference"):
        resolve_policy(CFG, "C", None)


def test_rollout_cli_writes_under_run_and_setting(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rollout_module, "rollout", lambda *a, **k: calls.append((a, k)))
    args = [*BASE, "--run", "reference", "--setting", "wide", "--videos", "2"]
    result = CliRunner().invoke(app, [*args, "--set", f"output_dir={tmp_path}"])
    assert result.exit_code == 0, result.output
    (policy, _, _, out_dir), kwargs = calls[0]
    assert policy == "HuggingFaceVLA/smolvla_libero"
    assert out_dir == tmp_path / "reference" / "wide"
    assert kwargs == {"max_videos": 2, "setting": "wide", "run": "reference"}


def test_rollout_cli_rejects_unknown_setting_and_run(monkeypatch):
    calls = []
    monkeypatch.setattr(rollout_module, "rollout", lambda *a, **k: calls.append(a))
    bad_setting = CliRunner().invoke(app, [*BASE, "--setting", "wid"])
    assert bad_setting.exit_code != 0 and "standard" in bad_setting.output
    bad_run = CliRunner().invoke(app, [*BASE, "--run", "Z"])
    assert bad_run.exit_code != 0 and "reference" in bad_run.output
    assert calls == []


def test_ablate_commands_have_help():
    runner = CliRunner()
    for args in (
        ["eval", "ablate", "--help"],
        ["eval", "ablate", "prepare", "--help"],
        ["eval", "ablate", "evaluate", "--help"],
    ):
        result = runner.invoke(root, args)
        assert result.exit_code == 0, result.output
    assert "--cells" in runner.invoke(root, ["eval", "ablate", "prepare", "--help"]).output


def test_ablate_evaluate_names_missing_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from e2l_common.paths import repo_root

    cfg = str(repo_root() / "configs" / "eval.yaml")
    sim = str(repo_root() / "configs" / "sim.yaml")
    result = CliRunner().invoke(
        root, ["eval", "ablate", "evaluate", "nope", "-c", cfg, "--sim-config", sim]
    )
    assert result.exit_code != 0
    assert "outputs/ablation/nope/plan.json" in result.output
