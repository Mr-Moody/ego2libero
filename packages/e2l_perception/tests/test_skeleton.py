from e2l_common.testing import check_cli_help
from e2l_perception.cli import app


def test_cli_help():
    assert check_cli_help(app)
