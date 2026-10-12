"""Hugging Face Hub transfers between the laptop and the DGX Spark. Imports are lazy so tests
and `--help` work offline; needs `hf auth login` (write token) for anything but namespace()."""

from pathlib import Path


def namespace(configured: str | None) -> str:
    """The configured Hub namespace, else the logged-in user."""
    if configured:
        return configured
    from huggingface_hub import whoami

    return whoami()["name"]


def repo_exists(repo_id: str, repo_type: str = "model") -> bool:
    from huggingface_hub import HfApi

    return HfApi().repo_exists(repo_id, repo_type=repo_type)


def push_dataset(root: Path, repo_id: str, private: bool) -> None:
    """Upload the LeRobot dataset at `root` (images, no videos) to `repo_id`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LeRobotDataset(repo_id, root=root).push_to_hub(private=private, push_videos=False)


def push_folder(folder: Path, repo_id: str, private: bool) -> None:
    """Upload a small folder (plan.json, spark.sh) as a dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="dataset")


def download_policy(repo_id: str) -> Path:
    """Local snapshot of a pushed checkpoint (config, weights, train_config, processors)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="model"))
