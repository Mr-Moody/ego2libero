from pathlib import Path

import pytest

from e2l_common.config import CONFIG_MODELS, load_config

CONFIGS = sorted((Path(__file__).parents[3] / "configs").rglob("*.yaml"))


def test_every_stage_has_a_config():
    assert {p.stem for p in CONFIGS} >= {
        "capture",
        "perception",
        "segment",
        "retarget",
        "sim",
        "generate",
        "smolvla",
        "eval",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, CONFIG_MODELS[path.stem])


def test_wide_eval_lies_outside_generation_range():
    """Held-out evaluation: wide placements start where generation sampling stops, and the
    standard and wide seed ranges never overlap."""
    configs = Path(__file__).parents[3] / "configs"
    gen = load_config(configs / "generate.yaml")
    ev = load_config(configs / "eval.yaml")
    assert ev.wide.min_offset >= gen.xy_range
    assert ev.wide.min_yaw >= gen.yaw_range
    assert ev.wide.seed_offset >= ev.n_episodes
