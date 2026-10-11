import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from e2l_eval.report import load_results, make_report, results_table, rollout_grid


def _write(root: Path, run: str, setting: str, successes, videos=0, placements=(), legacy=False):
    d = root / run / setting
    (d / "videos").mkdir(parents=True)
    paths = []
    for i in range(videos):
        p = d / "videos" / f"eval_episode_{i}.mp4"
        frames = np.zeros((12, 64, 64, 3), dtype=np.uint8)
        frames[:, :, :, i % 3] = (np.arange(12) * 20)[:, None, None]  # changes every frame
        imageio.mimwrite(p, frames, fps=20)
        # recorded on another machine: only the file name is still valid here
        paths.append(f"/other/machine/outputs/eval/{run}/{setting}/videos/{p.name}")
    data = {
        "successes": successes,
        "video_paths": paths,
        "video_successes": successes[:videos],
        "placements": list(placements),
    }
    if not legacy:
        data |= {"run": run, "setting": setting}
    (d / "results.json").write_text(json.dumps(data))


def test_load_results_falls_back_to_folders_and_skips_bad_files(tmp_path):
    _write(tmp_path, "A", "standard", [True, False], legacy=True)
    (tmp_path / "B" / "junk").mkdir(parents=True)
    (tmp_path / "B" / "junk" / "results.json").write_text(json.dumps({"successes": [True]}))
    rows = load_results(tmp_path)
    assert [(r["run"], r["setting"]) for r in rows] == [("A", "standard")]


def test_table_orders_runs_and_marks_missing_cells(tmp_path):
    _write(tmp_path, "reference", "standard", [True] * 8 + [False] * 2)
    _write(tmp_path, "B", "standard", [True] * 5 + [False] * 5)
    _write(
        tmp_path,
        "B",
        "wide",
        [True, False, False, False],
        placements=[{"placement_failed": True}, {"placement_failed": False}],
    )
    _write(tmp_path, "zeta", "wide", [False])
    table = results_table(load_results(tmp_path))
    lines = table.splitlines()
    assert lines[0].startswith("| Run | standard | wide |")
    body = [ln.split("|")[1].strip() for ln in lines[2:] if ln.startswith("| ")]
    assert body == ["B", "reference", "zeta"]
    assert "8/10 = 80% [49%–94%]" in table  # Wilson 95% for 8/10
    assert "1/4 = 25%" in table
    assert "| — |" in table  # reference has no wide result
    assert "1 of 2 wide episodes" in table  # placements recorded for 2 episodes


def test_grid_finds_videos_next_to_results_and_skips_missing_tiles(tmp_path):
    _write(tmp_path, "B", "standard", [True, True], videos=2)  # no failure video
    _write(tmp_path, "B", "wide", [False], videos=1)
    out = rollout_grid(load_results(tmp_path), tmp_path / "grid.gif", tile=64)
    assert out == tmp_path / "grid.gif"
    gif = Image.open(out)
    assert gif.size == (2 * 64, 64 + 16)  # two tiles (success, wide) plus label strip
    assert gif.n_frames >= 2


def test_make_report_without_videos_writes_table_and_chart_only(tmp_path):
    _write(tmp_path / "res", "reference", "standard", [True, False, True])
    written = make_report(tmp_path / "res", tmp_path / "fig")
    assert sorted(p.name for p in written) == ["results.md", "success_rates.png"]
    assert (tmp_path / "fig" / "success_rates.png").stat().st_size > 0
