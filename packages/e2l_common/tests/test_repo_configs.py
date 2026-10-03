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
