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
    typer.echo(
        f"{demo_id}: "
        + " | ".join(
            f"{s.ref_object} {len(s.T_obj_ee)} frames, {s.feasible.mean():.0%} feasible"
            for s in robot.segments
        )
    )


@app.command()
def calibrate(
    demo_id: str,
    segment_config: Path = config_option("segment"),
    set_: list[str] = set_option(),
) -> None:
    """Measure T_hand_ee and closing_axis from data/perception/<demo_id>; prints the two
    retarget.yaml lines to paste."""
    from e2l_common.config import SegmentConfig
    from e2l_common.schemas import DemoTrajectory
    from e2l_retarget.calibrate import estimate_grip
    from e2l_segment.anchor import anchor_depth
    from e2l_segment.events import aperture_to_grip

    demo = DemoTrajectory.load(DataPaths.default().perception(demo_id))
    seg_cfg = load_stage_config(segment_config, set_, SegmentConfig)
    grip = aperture_to_grip(demo.t, demo.aperture, seg_cfg)
    demo = anchor_depth(demo, grip, seg_cfg)
    closed = grip == 1
    xyz, axis, n = estimate_grip(demo, closed)
    typer.echo(f"# {demo_id}: {n} closed-grip frames")
    typer.echo(
        f"T_hand_ee: {{xyz: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}], rotvec: [0.0, 0.0, 0.0]}}"
    )
    typer.echo(f"closing_axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
