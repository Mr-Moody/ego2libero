from e2l_common.testing import check_cli_help, check_stubs
from e2l_segment.cli import app


def test_cli_help():
    assert check_cli_help(app)


def test_stubs_raise_named_not_implemented():
    assert check_stubs("e2l_segment") > 0
