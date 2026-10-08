from pathlib import Path

import numpy as np
import pytest

from e2l_common.config import TrainConfig
from e2l_common.schemas import SimEpisode
from e2l_train.export import FPS, episode_frames, export_dataset

TASK = "put the bowl on the plate"


def _episode(T: int = 3, success: bool = True, seed: int = 0) -> SimEpisode:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (T, 256, 256, 3), dtype=np.uint8)
    return SimEpisode(
        task=TASK,
        success=success,
        source_demo_ids=["20261003_scripted_000"],
        images={"image": img, "image2": img[:, :, ::-1].copy()},
        state=rng.normal(size=(T, 8)),
        actions=rng.normal(size=(T, 7)),
        object_poses={"bowl": np.eye(4)},
    )


def _save(ep: SimEpisode, path: Path) -> Path:
    npz, _ = ep.save(path)
    return npz


def _cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(dataset_repo_id="local/e2l_test", dataset_root=tmp_path / "ds")


def _load(cfg: TrainConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)


def test_episode_frames_rotates_images_and_casts():
    ep = _episode()
    frames = episode_frames(ep)
    assert len(frames) == 3
    f = frames[1]
    np.testing.assert_array_equal(f["observation.images.image"], np.rot90(ep.images["image"][1], 2))
    np.testing.assert_array_equal(
        f["observation.images.image2"], np.rot90(ep.images["image2"][1], 2)
    )
    assert f["observation.state"].dtype == np.float32 and f["observation.state"].shape == (8,)
    assert f["action"].dtype == np.float32 and f["action"].shape == (7,)
    np.testing.assert_allclose(f["action"], ep.actions[1].astype(np.float32))
    assert f["task"] == TASK


def test_episode_frames_rejects_missing_camera():
    ep = _episode()
    ep.images.pop("image2")
    with pytest.raises(ValueError, match="image2"):
        episode_frames(ep)


def test_episode_frames_rejects_wrong_image_size():
    ep = _episode()
    ep.images["image"] = np.zeros((3, 128, 128, 3), np.uint8)
    with pytest.raises(ValueError, match="256"):
        episode_frames(ep)


def test_export_writes_loadable_dataset_of_successes_only(tmp_path):
    paths = [
        _save(_episode(T=3, seed=0), tmp_path / "ep" / "a"),
        _save(_episode(T=4, success=False, seed=1), tmp_path / "ep" / "b"),
        _save(_episode(T=2, seed=2), tmp_path / "ep" / "c"),
    ]
    cfg = _cfg(tmp_path)
    assert export_dataset(paths, cfg) == cfg.dataset_root

    ds = _load(cfg)
    assert ds.meta.total_episodes == 2
    assert len(ds) == 5
    assert ds.fps == FPS
    assert ds.meta.robot_type == "panda"
    item = ds[0]
    assert tuple(item["observation.images.image"].shape) == (3, 256, 256)
    assert tuple(item["observation.state"].shape) == (8,)
    assert tuple(item["action"].shape) == (7,)
    assert item["task"] == TASK
    # Round trip: stored image (CHW float in [0, 1]) is the 180-degree rotated original.
    stored = item["observation.images.image"].permute(1, 2, 0).numpy() * 255
    original = SimEpisode.load(paths[0]).images["image"][0]
    np.testing.assert_allclose(stored, np.rot90(original, 2), atol=1)


def test_export_max_episodes_caps_and_tolerates_too_large(tmp_path):
    paths = [_save(_episode(seed=s), tmp_path / "ep" / f"e{s}") for s in range(3)]
    cfg = _cfg(tmp_path)
    export_dataset(paths, cfg, max_episodes=2)
    assert _load(cfg).meta.total_episodes == 2

    cfg2 = TrainConfig(dataset_repo_id="local/e2l_test2", dataset_root=tmp_path / "ds2")
    export_dataset(paths, cfg2, max_episodes=10)
    assert _load(cfg2).meta.total_episodes == 3


def test_export_refuses_existing_root(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.dataset_root.mkdir()
    with pytest.raises(FileExistsError):
        export_dataset([_save(_episode(), tmp_path / "ep" / "a")], cfg)


def test_export_without_successes_raises_before_writing(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="no successful"):
        export_dataset([_save(_episode(success=False), tmp_path / "ep" / "a")], cfg)
    assert not cfg.dataset_root.exists()
