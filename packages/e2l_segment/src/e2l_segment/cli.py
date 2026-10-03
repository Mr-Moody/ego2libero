"""`e2l segment ...`: DemoTrajectory -> HandSegments."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import SegmentConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Split demos into object-centric hand segments.", no_args_is_help=True)


@app.command()
def run(
    demo_id: str,
    config: Path = config_option("segment"),
    set_: list[str] = set_option(),
) -> None:
    """Segment data/perception/<demo_id> into data/segments/<demo_id>."""
    from e2l_common.schemas import DemoTrajectory
    from e2l_segment.segment import segment_demo

    cfg = load_stage_config(config, set_, SegmentConfig)
    paths = DataPaths.default()
    segments = segment_demo(DemoTrajectory.load(paths.perception(demo_id)), cfg)
    segments.save(paths.segments(demo_id))


@app.command()
def viz(demo_id: str) -> None:
    """Plot aperture, grip and segment boundaries for one demo."""
    from e2l_common.stub import not_implemented

    not_implemented("e2l_segment.cli.viz")
