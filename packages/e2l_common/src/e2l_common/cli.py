"""Root `e2l` command. Mounts each stage's Typer app if that package is installed."""

import importlib

import typer
from rich.console import Console

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

    from rich.table import Table

    table = Table("package", "version")
    for name in ("common", *STAGES):
        try:
            table.add_row(f"e2l-{name}", version(f"e2l-{name}"))
        except PackageNotFoundError:
            table.add_row(f"e2l-{name}", "[dim]not installed[/dim]")
    Console().print(table)
    Console().print(_torch_status())


def _torch_status() -> str:
    """Torch build and CUDA visibility; part of the per-machine smoke test."""
    try:
        import torch
    except ModuleNotFoundError:
        return "torch: not installed"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}: [red]CUDA not available[/red]"
    return (
        f"torch {torch.__version__}: CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}"
    )


_mount_stages()


def main() -> None:
    try:
        app()
    except NotImplementedError as exc:
        Console(stderr=True).print(f"[red]NotImplementedError:[/red] {exc}")
        raise SystemExit(2) from None
