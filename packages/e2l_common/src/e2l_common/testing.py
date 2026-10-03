"""Test helpers shared by the stage packages."""

import importlib
import inspect
import pkgutil

import pytest
import typer
from typer.testing import CliRunner


def iter_stubs(package: str):
    """(qualname, callable) for every function or method in `package` that calls not_implemented."""
    pkg = importlib.import_module(package)
    for info in pkgutil.walk_packages(pkg.__path__, f"{package}."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if getattr(obj, "__module__", None) != info.name:
                continue
            members = [(name, obj)]
            if inspect.isclass(obj):
                members = [(f"{name}.{m}", f) for m, f in vars(obj).items()]
            for qual, fn in members:
                if inspect.isfunction(fn) and "not_implemented(" in _source(fn):
                    yield f"{info.name}.{qual}", fn


def _source(fn) -> str:
    try:
        return inspect.getsource(fn)
    except OSError:  # generated methods, e.g. dataclass __init__
        return ""


def check_stubs(package: str) -> int:
    """Call every stub with None arguments; each must raise NotImplementedError naming itself."""
    count = 0
    for qualname, fn in iter_stubs(package):
        args = [None] * len(inspect.signature(fn).parameters)
        with pytest.raises(NotImplementedError, match=qualname.replace(".", r"\.")):
            fn(*args)
        count += 1
    return count


def check_cli_help(app: typer.Typer) -> list[str]:
    """`--help` must succeed for the sub-app and each of its commands."""
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    names = [c.name or c.callback.__name__ for c in app.registered_commands]
    for name in names:
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, f"{name} --help failed:\n{result.output}"
    return names
