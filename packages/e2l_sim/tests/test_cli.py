from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import e2l_sim.replay
from e2l_common.cli import app
from e2l_common.paths import DataPaths, repo_root
from e2l_sim.scripted import write_scripted

DEMO = "20261003_scripted_000"


@pytest.fixture
def fake_replay(tmp_path, monkeypatch):
    """Scripted demo on disk under a temp data root; replay records its call and saves stubs."""
    monkeypatch.setenv("E2L_DATA", str(tmp_path))
    write_scripted(DataPaths(tmp_path), count=1)
    calls = {}

    def save(path):
        calls["saved"] = path
        return path.with_suffix(".npz"), path.with_suffix(".json")

    def replay(segments, cfg, episode_index=0, **_):
        calls["episode_index"] = episode_index
        return SimpleNamespace(success=True, actions=np.zeros((5, 7)), save=save)

    monkeypatch.setattr(e2l_sim.replay, "replay", replay)
    return tmp_path, calls


def _invoke(*args: str):
    cfg = str(repo_root() / "configs" / "sim.yaml")
    return CliRunner().invoke(app, ["sim", "replay", DEMO, "-c", cfg, *args])


def test_replay_defaults_to_replay_folder(fake_replay):
    root, calls = fake_replay
    result = _invoke()
    assert result.exit_code == 0, result.output
    assert calls["saved"] == DataPaths(root).generated("replay", DEMO)
    assert calls["episode_index"] == 0


def test_replay_saves_under_run_id(fake_replay):
    root, calls = fake_replay
    result = _invoke("--run-id", "rehearsal_replay", "--episode-index", "3")
    assert result.exit_code == 0, result.output
    assert calls["saved"] == DataPaths(root).generated("rehearsal_replay", DEMO)
    assert calls["episode_index"] == 3


@pytest.mark.parametrize("run_id", ["../x", "a/b", ".hidden", ""])
def test_replay_rejects_run_id_with_separator(fake_replay, run_id):
    _, calls = fake_replay
    result = _invoke("--run-id", run_id)
    assert result.exit_code != 0
    assert "saved" not in calls and "episode_index" not in calls
