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


@app.command()
def ablate(config: Path = config_option("eval"), set_: list[str] = set_option()) -> None:
    """Sweep demo count x augmentation."""
    from e2l_eval.ablation import run_ablation

    run_ablation(load_stage_config(config, set_, EvalConfig))


@app.command()
def report(
    results: Path = typer.Option(
        Path("outputs/eval"), help="Folder of <run>/<setting>/results.json."
    ),
    out_dir: Path = typer.Option(Path("docs/figures")),
) -> None:
    """Results table, success-rate chart and rollout GIF into docs/figures/."""
    from e2l_eval.report import make_report

    for path in make_report(results, out_dir):
        typer.echo(path)
