"""Data directory layout under `data/` and demo-id helpers. Demo ids look like
`20261004_bowlplate_007`: recording date, task slug, three-digit index."""

import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

__all__ = ["DataPaths", "DemoId", "repo_root"]

_DEMO_ID = re.compile(r"^(?P<date>\d{8})_(?P<task>[a-z0-9]+)_(?P<index>\d{3})$")


def repo_root(start: Path | None = None) -> Path:
    """Nearest ancestor holding the workspace `pyproject.toml` (falls back to cwd)."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        pyproject = d / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text():
            return d
    return here


@dataclass(frozen=True)
class DemoId:
    date: date
    task: str
    index: int

    def __str__(self) -> str:
        return f"{self.date:%Y%m%d}_{self.task}_{self.index:03d}"

    @classmethod
    def parse(cls, demo_id: str) -> "DemoId":
        m = _DEMO_ID.match(demo_id)
        if m is None:
            raise ValueError(f"bad demo id {demo_id!r}; expected YYYYMMDD_task_NNN")
        d = m["date"]
        return cls(date(int(d[:4]), int(d[4:6]), int(d[6:])), m["task"], int(m["index"]))


@dataclass(frozen=True)
class DataPaths:
    """Paths of every stage's files. `root` defaults to $E2L_DATA or `<repo>/data`.
    Schema paths have no suffix: `.npz` and `.json` are added by `save()` / `load()`."""

    root: Path

    @classmethod
    def default(cls) -> "DataPaths":
        return cls(Path(os.environ.get("E2L_DATA", repo_root() / "data")))

    def raw(self, session: str, demo_id: str) -> Path:
        return self.root / "raw" / session / f"{demo_id}.mp4"

    def normalised(self, demo_id: str) -> Path:
        return self.root / "normalised" / f"{demo_id}.mp4"

    def calib(self, device: str) -> Path:
        return self.root / "calib" / f"{device}.json"

    def perception(self, demo_id: str) -> Path:
        return self.root / "perception" / demo_id

    def segments(self, demo_id: str) -> Path:
        return self.root / "segments" / demo_id

    def robot_segments(self, demo_id: str) -> Path:
        return self.root / "robot_segments" / demo_id

    def generated(self, run_id: str, episode: str | None = None) -> Path:
        run = self.root / "generated" / run_id
        return run if episode is None else run / episode

    def manifest(self, run_id: str) -> Path:
        return self.generated(run_id) / "manifest.json"

    def lerobot(self, dataset_name: str) -> Path:
        return self.root / "lerobot" / dataset_name

    def demo_ids(self, stage: str = "normalised") -> list[str]:
        """Sorted demo ids present for a stage directory (by file stem)."""
        d = self.root / stage
        return sorted({p.stem for p in d.glob("*") if _DEMO_ID.match(p.stem)}) if d.is_dir() else []
