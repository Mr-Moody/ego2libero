"""`e2l eval ...`: rollouts, metrics, ablations, figures."""

from pathlib import Path

import typer

from e2l_common.cliutils import config_option, load_stage_config, set_option
from e2l_common.config import EvalConfig, SimConfig

app = typer.Typer(help="Policy evaluation and reporting.", no_args_is_help=True)

SIM_CONFIG = typer.Option(Path("configs/sim.yaml"), help="Simulator config.")


@app.command()
def baseline(
    episodes: int = typer.Option(5, help="Seeded episodes to run."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    out_dir: Path = typer.Option(Path("outputs/eval/baseline")),
    set_: list[str] = set_option(),
) -> None:
    """Pretrained smolvla_libero on the chosen task: success rate and one video."""
    from e2l_common.config import load_config
    from e2l_eval.baseline import run_baseline

    cfg = load_stage_config(config, set_, EvalConfig)
    run_baseline(cfg, load_config(sim_config, model=SimConfig), episodes, out_dir)


@app.command()
def rollout(
    policy: str = typer.Option(None, help="Checkpoint path or hub id (default: config)."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    set_: list[str] = set_option(),
) -> None:
    """Evaluate a policy for n_episodes seeded episodes."""
    from e2l_common.config import load_config
    from e2l_eval.rollout import rollout as run_rollout

    cfg = load_stage_config(config, set_, EvalConfig)
    sim_cfg = load_config(sim_config, model=SimConfig)
    run_rollout(policy or cfg.policy_path, cfg, sim_cfg, cfg.output_dir)


@app.command()
def ablate(config: Path = config_option("eval"), set_: list[str] = set_option()) -> None:
    """Sweep demo count x augmentation."""
    from e2l_eval.ablation import run_ablation

    run_ablation(load_stage_config(config, set_, EvalConfig))


@app.command()
def report(
    results: Path = typer.Option(Path("outputs/eval/ablation.json")),
    out_dir: Path = typer.Option(Path("docs/figures")),
) -> None:
    """Plots, tables and video grids into docs/figures/."""
    from e2l_eval.report import make_report

    make_report(results, out_dir)
