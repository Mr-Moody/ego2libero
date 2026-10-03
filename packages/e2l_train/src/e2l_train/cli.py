"""`e2l train ...`: dataset export and SmolVLA fine-tuning."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import TrainConfig

app = typer.Typer(help="LeRobot export and SmolVLA fine-tuning.", no_args_is_help=True)


@app.command()
def export(
    run_ids: list[str] = typer.Argument(None, help="Generated runs to include (default: all)."),
    config: Path = config_option("train/smolvla"),
    set_: list[str] = set_option(),
) -> None:
    """Export data/generated/<run_id>/ episodes to a LeRobot dataset."""
    from e2l_common.paths import DataPaths
    from e2l_train.export import export_dataset

    cfg = load_stage_config(config, set_, TrainConfig)
    paths = DataPaths.default()
    runs = run_ids or sorted(p.name for p in (paths.root / "generated").glob("*") if p.is_dir())
    episodes = [p for r in runs for p in sorted(paths.generated(r).glob("*.npz"))]
    export_dataset(episodes, cfg)


@app.command()
def finetune(
    config: Path = config_option("train/smolvla"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without running."),
    set_: list[str] = set_option(),
) -> None:
    """Fine-tune SmolVLA with lerobot-train."""
    from e2l_train.finetune import launch

    launch(load_stage_config(config, set_, TrainConfig), dry_run=dry_run)
