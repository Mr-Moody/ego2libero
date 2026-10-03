"""Options and helpers shared by the stage CLIs."""

from pathlib import Path

import typer
from pydantic import BaseModel

from e2l_common.config import load_config
from e2l_common.log import setup_logging


def config_option(stage: str):
    """`--config/-c` option defaulting to `configs/<stage>.yaml`."""
    return typer.Option(Path(f"configs/{stage}.yaml"), "--config", "-c", help="Stage YAML config.")


def set_option():
    return typer.Option(None, "--set", help="Config override key.sub=value (repeatable).")


def load_stage_config[M: BaseModel](path: Path, overrides: list[str] | None, model: type[M]) -> M:
    """Set up logging and load a stage config with CLI overrides."""
    setup_logging()
    return load_config(path, overrides or [], model)
