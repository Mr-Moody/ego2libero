"""Sample, transform, stitch, replay and keep successes."""

from pathlib import Path

from e2l_common.config import GenerateConfig, SimConfig
from e2l_common.stub import not_implemented


def generate(
    demo_ids: list[str], run_id: str, cfg: GenerateConfig, sim_cfg: SimConfig, data_root: Path
) -> dict:
    """For each source demo, generate `cfg.episodes_per_demo` attempts in parallel workers and
    save successful SimEpisodes to data/generated/<run_id>/. Writes manifest.json with yield
    per source demo and provenance; returns the manifest dict."""
    not_implemented("e2l_generate.run.generate")
