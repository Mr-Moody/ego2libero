"""`e2l generate ...`: RobotSegments -> many SimEpisodes."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.paths import DataPaths

app = typer.Typer(help="Multiply demos into simulated episodes.", no_args_is_help=True)

_COLUMNS = ("attempts", "successes", "failures", "placement_failed", "errors")


def yield_table(manifest: dict) -> Table:
    table = Table("demo", *_COLUMNS, "yield", title=f"generated/{manifest['run_id']}")
    for demo_id, counts in manifest["demos"].items():
        table.add_row(demo_id, *(str(counts[c]) for c in _COLUMNS), f"{counts['yield']:.0%}")
    return table


@app.command()
def run(
    demo_ids: list[str] = typer.Argument(None, help="Source demos (default: all retargeted)."),
    run_id: str = typer.Option("run0", help="Output folder under data/generated/."),
    overwrite: bool = typer.Option(False, help="Delete and replace an existing run folder."),
    config: Path = config_option("generate"),
    sim_config: Path = typer.Option(Path("configs/sim.yaml"), help="Simulator config."),
    set_: list[str] = set_option(),
) -> None:
    """Generate episodes from data/robot_segments into data/generated/<run_id>."""
    from e2l_common.config import load_config
    from e2l_generate.run import generate

    cfg = load_stage_config(config, set_, GenerateConfig)
    paths = DataPaths.default()
    demo_ids = demo_ids or paths.demo_ids("robot_segments")
    if not demo_ids:
        typer.echo(
            f"no demos in {paths.root / 'robot_segments'}; run `e2l retarget run` "
            "or `python scripts/scripted_segments.py` first"
        )
        raise typer.Exit(1)
    manifest = generate(
        demo_ids,
        run_id,
        cfg,
        load_config(sim_config, model=SimConfig),
        paths.root,
        overwrite=overwrite,
    )
    Console().print(yield_table(manifest))
