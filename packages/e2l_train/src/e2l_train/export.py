"""SimEpisodes to a LeRobot dataset matching the smolvla_libero checkpoint."""

from pathlib import Path

from e2l_common.config import TrainConfig
from e2l_common.stub import not_implemented


def export_dataset(episode_paths: list[Path], cfg: TrainConfig) -> Path:
    """Write successful SimEpisodes to a LeRobot dataset at `cfg.dataset_root` with the feature
    keys, image size, state/action layout and task-string format that
    HuggingFaceVLA/smolvla_libero expects (check against HuggingFaceVLA/libero first)."""
    not_implemented("e2l_train.export.export_dataset")
