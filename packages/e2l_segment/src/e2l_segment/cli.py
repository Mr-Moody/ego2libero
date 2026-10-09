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
    typer.echo(
        f"{demo_id}: {len(segments.segments)} segments "
        + " | ".join(f"{s.ref_object} {s.t_start:.2f}-{s.t_end:.2f}s" for s in segments.segments)
    )


@app.command()
def viz(
    demo_id: str,
    config: Path = config_option("segment"),
    out_dir: Path = typer.Option(Path("outputs/segment")),
    set_: list[str] = set_option(),
) -> None:
    """Plot aperture, grip and segment boundaries for one demo."""
    from e2l_common.schemas import DemoTrajectory
    from e2l_segment.viz import plot_segments

    cfg = load_stage_config(config, set_, SegmentConfig)
    demo = DemoTrajectory.load(DataPaths.default().perception(demo_id))
    typer.echo(f"wrote {plot_segments(demo, cfg, out_dir / f'{demo_id}_segments.png')}")
