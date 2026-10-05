from e2l_common.testing import check_cli_help, check_stubs
from e2l_generate.cli import app


def test_cli_help():
    assert check_cli_help(app)


def test_no_stubs_remain():
    assert check_stubs("e2l_generate") == 0
