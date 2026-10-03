"""`e2l eval baseline`: run the pretrained smolvla_libero checkpoint on the chosen task."""

from pathlib import Path

from e2l_common.config import EvalConfig, SimConfig
from e2l_common.stub import not_implemented


def run_baseline(cfg: EvalConfig, sim_cfg: SimConfig, episodes: int, out_dir: Path) -> float:
    not_implemented("e2l_eval.baseline.run_baseline")
