"""`e2l generate ...`: RobotSegments -> many SimEpisodes."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Multiply demos into simulated episodes.", no_args_is_help=True)


@app.command()
def run(
    demo_ids: list[str] = typer.Argument(None, help="Source demos (default: all retargeted)."),
    run_id: str = typer.Option("run0", help="Output folder under data/generated/."),
    config: Path = config_option("generate"),
    sim_config: Path = typer.Option(Path("configs/sim.yaml"), help="Simulator config."),
    set_: list[str] = set_option(),
) -> None:
    """Generate episodes from data/robot_segments into data/generated/<run_id>."""
    from e2l_common.config import load_config
    from e2l_generate.run import generate

    cfg = load_stage_config(config, set_, GenerateConfig)
    paths = DataPaths.default()
    generate(
        demo_ids or paths.demo_ids("robot_segments"),
        run_id,
        cfg,
        load_config(sim_config, model=SimConfig),
        paths.root,
    )
