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


def resolve_policy(cfg: EvalConfig, run: str | None, policy: str | None) -> tuple[str, str]:
    """(run label, checkpoint): an explicit `policy` wins; else `run` is looked up in cfg.runs;
    no run means label `custom` with `policy` or cfg.policy_path."""
    if run is None:
        return "custom", policy or cfg.policy_path
    if policy:
        return run, policy
    if run not in cfg.runs:
        labels = ", ".join(sorted(cfg.runs))
        raise typer.BadParameter(
            f"unknown run {run!r}; eval.yaml runs: {labels}", param_hint="--run"
        )
    return run, cfg.runs[run]


@app.command()
def rollout(
    policy: str = typer.Option(None, help="Checkpoint path or hub id (overrides --run)."),
    run: str = typer.Option(None, help="Run label from eval.yaml `runs` (A, B, reference)."),
    setting: str = typer.Option("standard", help="standard | wide"),
    videos: int = typer.Option(3, help="Episodes to save as mp4."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    set_: list[str] = set_option(),
) -> None:
    """Evaluate a policy for n_episodes seeded episodes into output_dir/<run>/<setting>/."""
    from e2l_common.config import load_config
    from e2l_eval import rollout as rollout_module

    if setting not in rollout_module.SETTINGS:
        choices = ", ".join(rollout_module.SETTINGS)
        raise typer.BadParameter(f"{setting!r} is not one of {choices}", param_hint="--setting")
    cfg = load_stage_config(config, set_, EvalConfig)
    label, policy_path = resolve_policy(cfg, run, policy)
    sim_cfg = load_config(sim_config, model=SimConfig)
    out_dir = cfg.output_dir / label / setting
    rollout_module.rollout(
        policy_path, cfg, sim_cfg, out_dir, max_videos=videos, setting=setting, run=label
    )


ablate_app = typer.Typer(
    help="Demo count x augmentation sweep, trained on the DGX Spark via the Hub.",
    no_args_is_help=True,
)
app.add_typer(ablate_app, name="ablate")


@ablate_app.command("prepare")
def ablate_prepare(
    cells: str = typer.Option(None, help="Comma-separated cell labels (default: every cell)."),
    config: Path = config_option("eval"),
    train_config: Path = typer.Option(Path("configs/train/smolvla.yaml"), help="Train config."),
    set_: list[str] = set_option(),
) -> None:
    """Export + push each cell's dataset; write and push plan.json and spark.sh."""
    from e2l_common.config import TrainConfig, load_config
    from e2l_eval.ablation import prepare

    cfg = load_stage_config(config, set_, EvalConfig)
    only = [c.strip() for c in cells.split(",")] if cells else None
    plan = prepare(cfg, load_config(train_config, model=TrainConfig), only)
    typer.echo(f"wrote {plan} and {plan.parent / 'spark.sh'}")
    typer.echo((plan.parent / "spark.sh").read_text().split("set -euo")[0])


@ablate_app.command("evaluate")
def ablate_evaluate(
    plan: str = typer.Argument(..., help="Plan name: outputs/ablation/<plan>/plan.json."),
    videos: int = typer.Option(3, help="Episodes per setting to save as mp4."),
    config: Path = config_option("eval"),
    sim_config: Path = SIM_CONFIG,
    set_: list[str] = set_option(),
) -> None:
    """Evaluate every trained cell of a plan on standard and wide; rerun as cells finish."""
    from e2l_common.config import load_config
    from e2l_eval.ablation import PLAN_ROOT, evaluate

    plan_path = PLAN_ROOT / plan / "plan.json"
    if not plan_path.is_file():
        raise typer.BadParameter(f"no {plan_path}; run `e2l eval ablate prepare`")
    cfg = load_stage_config(config, set_, EvalConfig)
    status = evaluate(cfg, load_config(sim_config, model=SimConfig), plan_path, videos)
    for label, state in status.items():
        typer.echo(f"{label:10s} {state}")


@app.command()
def report(
    results: Path = typer.Option(
        Path("outputs/eval"), help="Folder of <run>/<setting>/results.json."
    ),
    out_dir: Path = typer.Option(Path("docs/figures")),
    ablation_dir: Path = typer.Option(
        Path("outputs/ablation"), help="Folder of <plan>/plan.json (ablation charts)."
    ),
) -> None:
    """Results table, success-rate chart, rollout GIF and ablation charts into docs/figures/."""
    from e2l_eval.report import make_report

    for path in make_report(results, out_dir, ablation_dir):
        typer.echo(path)
