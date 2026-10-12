"""SimEpisodes to a LeRobot dataset in the HuggingFaceVLA/libero format (what smolvla_libero was
trained on): same feature keys, dtypes, shapes, 10 fps label, one frame per control step."""

import json
from pathlib import Path

import numpy as np

from e2l_common.config import TrainConfig
from e2l_common.schemas import SimEpisode

FPS = 10
ROBOT_TYPE = "panda"
CAMERAS = ("image", "image2")
IMAGE_SHAPE = (256, 256, 3)
FEATURES = {
    **{
        f"observation.images.{cam}": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERAS
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def _check_cameras(ep: SimEpisode) -> None:
    for cam in CAMERAS:
        if cam not in ep.images:
            raise ValueError(f"episode has no camera {cam!r} (has {sorted(ep.images)})")
        if ep.images[cam].shape[1:] != IMAGE_SHAPE:
            raise ValueError(f"camera {cam!r} images are {ep.images[cam].shape[1:]}, want 256x256")


def episode_frames(ep: SimEpisode) -> list[dict]:
    """One LeRobot frame per control step. Images are rotated 180 deg: HuggingFaceVLA/libero
    stores them that way, and LeRobot's LiberoProcessorStep rotates live observations to match."""
    _check_cameras(ep)
    return [
        {
            **{
                f"observation.images.{cam}": np.ascontiguousarray(ep.images[cam][i, ::-1, ::-1])
                for cam in CAMERAS
            },
            "observation.state": ep.state[i].astype(np.float32),
            "action": ep.actions[i].astype(np.float32),
            "task": ep.task,
        }
        for i in range(len(ep.actions))
    ]


def _succeeded(npz: Path) -> bool:
    return bool(json.loads(Path(npz).with_suffix(".json").read_text())["success"])


def export_dataset(
    episode_paths: list[Path], cfg: TrainConfig, max_episodes: int | None = None
) -> Path:
    """Write the successful episodes among `episode_paths` (at most `max_episodes`) to a new
    LeRobot dataset at `cfg.dataset_root`. Refuses to touch an existing directory."""
    if cfg.dataset_root is None:
        raise ValueError("export needs a local dataset_root to write to")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if cfg.dataset_root.exists():
        raise FileExistsError(f"{cfg.dataset_root} exists; choose a new dataset_root")
    keep = [p for p in episode_paths if _succeeded(p)][:max_episodes]
    if not keep:
        raise ValueError(f"no successful episodes among {len(episode_paths)} files")
    for path in keep:  # fail before creating anything, naming the offending file
        try:
            _check_cameras(SimEpisode.load(path))
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e

    dataset = LeRobotDataset.create(
        cfg.dataset_repo_id,
        FPS,
        FEATURES,
        root=cfg.dataset_root,
        robot_type=ROBOT_TYPE,
        use_videos=False,
        image_writer_threads=4,
    )
    for path in keep:
        for frame in episode_frames(SimEpisode.load(path)):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return cfg.dataset_root
