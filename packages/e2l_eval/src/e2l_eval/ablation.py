"""Sweep number of human demos x augmentation."""

from e2l_common.config import EvalConfig
from e2l_common.stub import not_implemented


def run_ablation(cfg: EvalConfig) -> list[dict]:
    """For each demo count in cfg.demo_counts and augmentation flag, train (or load) a policy
    and evaluate it; returns rows with success rate and Wilson interval."""
    not_implemented("e2l_eval.ablation.run_ablation")
