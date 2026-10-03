"""`e2l sim ...`: LIBERO checks and replay."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import SimConfig

app = typer.Typer(help="LIBERO environment, replay and rendering.", no_args_is_help=True)


@app.command()
def check(
    config: Path = config_option("sim"),
    steps: int = typer.Option(50, help="Random actions to step."),
    out_dir: Path = typer.Option(Path("outputs/sim_check")),
    set_: list[str] = set_option(),
) -> None:
    """Create the task, step random actions, save one PNG per camera, print specs."""
    from e2l_sim.check import run_check

    run_check(load_stage_config(config, set_, SimConfig), steps, out_dir)


@app.command()
def replay(
    demo_id: str,
    config: Path = config_option("sim"),
    episode_index: int = typer.Option(0, help="LIBERO init state giving the object poses."),
    video: bool = typer.Option(False, help="Also write outputs/replay/<demo_id>.mp4."),
    set_: list[str] = set_option(),
) -> None:
    """Replay data/robot_segments/<demo_id> once at the nominal object poses."""
    import numpy as np

    from e2l_common.paths import DataPaths
    from e2l_common.schemas import RobotSegments
    from e2l_sim.render import save_video, upright
    from e2l_sim.replay import replay as replay_segments

    cfg = load_stage_config(config, set_, SimConfig)
    paths = DataPaths.default()
    segments = RobotSegments.load(paths.robot_segments(demo_id))
    episode = replay_segments(segments, cfg, episode_index=episode_index)
    npz, _ = episode.save(paths.generated("replay", demo_id))
    typer.echo(f"{demo_id}: success={episode.success}, {len(episode.actions)} steps -> {npz}")
    if video:
        frames = np.concatenate([upright(v) for v in episode.images.values()], axis=2)
        out = save_video(frames, Path("outputs/replay") / f"{demo_id}.mp4", cfg.control_freq)
        typer.echo(f"wrote {out}")
