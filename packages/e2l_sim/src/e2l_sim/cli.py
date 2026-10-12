"""`e2l sim ...`: LIBERO checks and replay."""

import re
from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import SimConfig

app = typer.Typer(help="LIBERO environment, replay and rendering.", no_args_is_help=True)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # one folder under data/generated/


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
    run_id: str = typer.Option("replay", help="Output folder under data/generated/."),
    video: bool = typer.Option(False, help="Also write outputs/replay/<demo_id>.mp4."),
    set_: list[str] = set_option(),
) -> None:
    """Replay data/robot_segments/<demo_id> once at the nominal object poses into
    data/generated/<run_id>/."""
    if not _RUN_ID.match(run_id):
        raise typer.BadParameter(f"{run_id!r} is not a single folder name", param_hint="--run-id")
    import numpy as np

    from e2l_common.paths import DataPaths
    from e2l_common.schemas import RobotSegments
    from e2l_sim.render import save_video, upright
    from e2l_sim.replay import replay as replay_segments

    cfg = load_stage_config(config, set_, SimConfig)
    paths = DataPaths.default()
    segments = RobotSegments.load(paths.robot_segments(demo_id))
    episode = replay_segments(segments, cfg, episode_index=episode_index)
    npz, _ = episode.save(paths.generated(run_id, demo_id))
    typer.echo(f"{demo_id}: success={episode.success}, {len(episode.actions)} steps -> {npz}")
    if video:
        frames = np.concatenate([upright(v) for v in episode.images.values()], axis=2)
        out = save_video(frames, Path("outputs/replay") / f"{demo_id}.mp4", cfg.control_freq)
        typer.echo(f"wrote {out}")
