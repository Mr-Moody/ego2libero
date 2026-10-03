"""Non-interactive LIBERO bootstrap. Importing `libero.libero` prompts on stdin when its config
file is missing, so we point LIBERO_CONFIG_PATH at a project-local folder and write the default
config there first. Call `ensure_libero_config()` before any libero import."""

import importlib.util
import os
from pathlib import Path

import yaml

from e2l_common.paths import repo_root


def ensure_libero_config() -> Path:
    """Write `<repo>/.libero/config.yaml` with LIBERO's default paths if absent; return it."""
    config_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(repo_root() / ".libero")))
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        spec = importlib.util.find_spec("libero")
        if spec is None or not spec.submodule_search_locations:
            raise ModuleNotFoundError("libero is not installed; run `uv sync --group sim`")
        root = Path(next(iter(spec.submodule_search_locations))) / "libero"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(root),
                    "bddl_files": str(root / "bddl_files"),
                    "init_states": str(root / "init_files"),
                    "datasets": str(root.parent / "datasets"),
                    "assets": str(root / "assets"),
                }
            )
        )
    return config_file


def list_tasks(suite: str) -> list[tuple[int, str, str]]:
    """(task_id, name, language instruction) for every task in a LIBERO suite."""
    ensure_libero_config()
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[suite]()
    return [(i, bench.get_task(i).name, bench.get_task(i).language) for i in range(bench.n_tasks)]
