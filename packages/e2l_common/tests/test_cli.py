from typer.testing import CliRunner

from e2l_common.cli import app


def test_root_help_and_info():
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0 and "e2l-common" in result.output
