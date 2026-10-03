"""Rich logging shared by every stage."""

import logging

from rich.logging import RichHandler

__all__ = ["get_logger", "setup_logging"]


def setup_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
