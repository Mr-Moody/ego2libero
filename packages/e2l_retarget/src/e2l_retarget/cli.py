"""`e2l retarget ...`: HandSegments -> RobotSegments."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import RetargetConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Retarget hand segments to the Panda end effector.", no_args_is_help=True)


@app.command()
def run(
    demo_id: str,
    config: Path = config_option("retarget"),
    set_: list[str] = set_option(),
) -> None:
    """Retarget data/segments/<demo_id> into data/robot_segments/<demo_id>."""
    from e2l_common.schemas import HandSegments
    from e2l_retarget.hand_to_ee import retarget_segments

    cfg = load_stage_config(config, set_, RetargetConfig)
    paths = DataPaths.default()
    robot = retarget_segments(HandSegments.load(paths.segments(demo_id)), cfg)
    robot.save(paths.robot_segments(demo_id))
