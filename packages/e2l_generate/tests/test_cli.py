from rich.console import Console
from typer.testing import CliRunner

from e2l_common.cli import app
from e2l_common.paths import repo_root
from e2l_generate.cli import yield_table


def test_cli_without_demos_exits_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("E2L_DATA", str(tmp_path))
    configs = repo_root() / "configs"
    # Through the root CLI: on its own, the one-command sub-app would read "run" as a demo id.
    args = ["-c", str(configs / "generate.yaml"), "--sim-config", str(configs / "sim.yaml")]
    result = CliRunner().invoke(app, ["generate", "run", *args])
    assert result.exit_code == 1
    assert "robot_segments" in result.output


def test_yield_table_lists_each_demo():
    manifest = {
        "run_id": "run0",
        "demos": {
            "d1": {
                "attempts": 4,
                "successes": 3,
                "failures": 1,
                "placement_failed": 0,
                "errors": 0,
                "yield": 0.75,
            }
        },
    }
    console = Console(width=120, record=True)
    console.print(yield_table(manifest))
    text = console.export_text()
    assert "d1" in text and "75%" in text
