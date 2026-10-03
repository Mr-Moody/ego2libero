"""Root `e2l` command. Mounts each stage's Typer app if that package is installed."""

import importlib

import typer

STAGES = ("perception", "segment", "retarget", "sim", "generate", "train", "eval")

app = typer.Typer(help="ego2libero: phone demos to LIBERO training data.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """ego2libero: phone demos to LIBERO training data."""


def _mount_stages() -> None:
    for stage in STAGES:
        module = f"e2l_{stage}.cli"
        try:
            stage_cli = importlib.import_module(module)
        except ModuleNotFoundError as exc:
            # Package not installed on this machine (e.g. Spark has no perception group).
            if exc.name in (f"e2l_{stage}", module):
                continue
            raise
        app.add_typer(stage_cli.app, name=stage)


@app.command()
def info() -> None:
    """Show which stage packages are installed on this machine."""
    from importlib.metadata import PackageNotFoundError, version

    from rich.console import Console
    from rich.table import Table

    table = Table("package", "version")
    for name in ("common", *STAGES):
        try:
            table.add_row(f"e2l-{name}", version(f"e2l-{name}"))
        except PackageNotFoundError:
            table.add_row(f"e2l-{name}", "[dim]not installed[/dim]")
    Console().print(table)


_mount_stages()


def main() -> None:
    app()
