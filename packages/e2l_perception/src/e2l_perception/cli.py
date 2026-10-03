"""`e2l perception ...`: phone video -> DemoTrajectory."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import CaptureConfig, PerceptionConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Hand and object tracking from phone video.", no_args_is_help=True)


@app.command()
def normalise(
    src: Path = typer.Argument(Path("data/raw"), help="Video file or folder of raw videos."),
    config: Path = config_option("perception"),
    set_: list[str] = set_option(),
) -> None:
    """Re-encode raw videos to constant-frame-rate H.264 in data/normalised/."""
    from e2l_perception.video import normalise as normalise_video

    cfg = load_stage_config(config, set_, PerceptionConfig)
    paths = DataPaths.default()
    videos = sorted(src.rglob("*.mp4")) + sorted(src.rglob("*.MOV")) if src.is_dir() else [src]
    for video in videos:
        normalise_video(video, paths.normalised(video.stem), cfg.fps)


@app.command()
def run(
    demo_id: str,
    config: Path = config_option("perception"),
    capture_config: Path = typer.Option(Path("configs/capture.yaml"), help="Capture config."),
    set_: list[str] = set_option(),
) -> None:
    """Track hand and objects in data/normalised/<demo_id>.mp4 -> data/perception/<demo_id>."""
    from e2l_common.config import load_config
    from e2l_perception.pipeline import run_demo

    cfg = load_stage_config(config, set_, PerceptionConfig)
    paths = DataPaths.default()
    capture = load_config(capture_config, model=CaptureConfig)
    demo = run_demo(paths.normalised(demo_id), demo_id, cfg, capture)
    demo.save(paths.perception(demo_id))


@app.command()
def viz(demo_id: str, out_dir: Path = typer.Option(Path("outputs/perception"))) -> None:
    """Overlay video and trajectory plots for one demo."""
    from e2l_perception.viz import plot_trajectory, render_overlay

    paths = DataPaths.default()
    render_overlay(paths.normalised(demo_id), demo_id, out_dir / f"{demo_id}_overlay.mp4")
    plot_trajectory(demo_id, out_dir / f"{demo_id}_trajectory.png")
