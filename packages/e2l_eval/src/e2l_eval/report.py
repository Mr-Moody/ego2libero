"""Results table, success-rate chart and rollout GIF for the README, from
`<results_dir>/<run>/<setting>/results.json` files written by `e2l eval rollout`."""

import json
import logging
from pathlib import Path

import numpy as np

from e2l_eval.metrics import wilson_interval
from e2l_eval.rollout import SETTINGS

log = logging.getLogger(__name__)
RUN_ORDER = ("A", "B", "reference")
HEADLINE = "B"


def load_results(results_dir: Path) -> list[dict]:
    """Every `<run>/<setting>/results.json` under `results_dir`, with `dir` added; run and
    setting fall back to the folder names for files written before they were recorded."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/*/results.json")):
        data = json.loads(path.read_text())
        data.setdefault("run", path.parent.parent.name)
        data.setdefault("setting", path.parent.name)
        if not data.get("successes") or data["setting"] not in SETTINGS:
            log.warning("skipping %s: no successes or unknown setting", path)
            continue
        rows.append({**data, "dir": path.parent})
    return rows


def _runs(rows: list[dict]) -> list[str]:
    present = {r["run"] for r in rows}
    return [r for r in RUN_ORDER if r in present] + sorted(present - set(RUN_ORDER))


def _cell(row: dict | None) -> str:
    if row is None:
        return "—"
    k, n = sum(row["successes"]), len(row["successes"])
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def results_table(rows: list[dict]) -> str:
    """Markdown table: one row per run, one column per setting, `k/n = rate [95% Wilson]`."""
    by_key = {(r["run"], r["setting"]): r for r in rows}
    lines = ["| Run | " + " | ".join(SETTINGS) + " |", "|---" * (len(SETTINGS) + 1) + "|"]
    for run in _runs(rows):
        cells = [_cell(by_key.get((run, s))) for s in SETTINGS]
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    wide = [p for r in rows if r["setting"] == "wide" for p in r.get("placements", [])]
    failed = sum(bool(p.get("placement_failed")) for p in wide)
    if failed:
        lines += [
            "",
            f"{failed} of {len(wide)} wide episodes kept the init-state placement "
            "(no valid wide placement settled).",
        ]
    return "\n".join(lines) + "\n"


def plot_success_rates(rows: list[dict], path: Path) -> Path:
    """Grouped bars (x = run, one bar per setting) with 95% Wilson error bars and n on top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _runs(rows)
    by_key = {(r["run"], r["setting"]): r for r in rows}
    width = 0.8 / len(SETTINGS)
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(runs), 3.2))
    for j, setting in enumerate(SETTINGS):
        xs, rates, err = [], [], [[], []]
        for i, run in enumerate(runs):
            row = by_key.get((run, setting))
            if row is None:
                continue
            k, n = sum(row["successes"]), len(row["successes"])
            lo, hi = wilson_interval(k, n)
            xs.append(i + (j - (len(SETTINGS) - 1) / 2) * width)
            rates.append(k / n)
            err[0].append(k / n - lo)
            err[1].append(hi - k / n)
            ax.text(xs[-1], hi + 0.02, f"n={n}", ha="center", fontsize=7)
        ax.bar(xs, rates, width, yerr=err, capsize=3, label=setting)
    ax.set_xticks(range(len(runs)), runs)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_plans(ablation_dir: Path) -> list[dict]:
    """Every `<ablation_dir>/<plan>/plan.json` written by `e2l eval ablate prepare`."""
    return [json.loads(p.read_text()) for p in sorted(Path(ablation_dir).glob("*/plan.json"))]


def plot_ablation(plan: dict, rows: list[dict], path: Path) -> Path | None:
    """Success rate vs number of source demos, one line per arm (human only / + generated) with
    95% Wilson error bars, one panel per setting. Only cells with results are drawn; None (and
    no file) if the plan has none yet."""
    by_key = {(r["run"], r["setting"]): r for r in rows}
    points: dict[tuple[str, bool], list[tuple[int, int, int]]] = {}
    for cell in plan["cells"]:
        for setting in SETTINGS:
            row = by_key.get((f"{plan['name']}-{cell['label']}", setting))
            if row is not None:
                k, n = sum(row["successes"]), len(row["successes"])
                points.setdefault((setting, cell["augmented"]), []).append((cell["n_demos"], k, n))
    if not points:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(3.4 * len(SETTINGS), 3.0), sharey=True)
    for ax, setting in zip(axes, SETTINGS, strict=True):
        for augmented, name in ((False, "human only"), (True, "human + generated")):
            pts = sorted(points.get((setting, augmented), []))
            if not pts:
                continue
            xs = [x for x, _, _ in pts]
            rates = [k / n for _, k, n in pts]
            bounds = [wilson_interval(k, n) for _, k, n in pts]
            err = [
                [r - lo for r, (lo, _) in zip(rates, bounds, strict=True)],
                [hi - r for r, (_, hi) in zip(rates, bounds, strict=True)],
            ]
            ax.errorbar(xs, rates, yerr=err, marker="o", capsize=3, label=name)
        ax.set_title(setting)
        ax.set_xlabel("source demos")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"ablation: {plan['name']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _video(row: dict, index: int) -> Path | None:
    """Video `index` of a result: next to its results.json first (results copied from another
    machine keep their videos there, and recorded paths are relative to that machine's cwd),
    else the recorded path."""
    recorded = Path(row["video_paths"][index])
    for p in (row["dir"] / "videos" / recorded.name, recorded):
        if p.is_file():
            return p
    return None


def _tiles(rows: list[dict]) -> list[tuple[str, Path]]:
    """(label, video) for the headline run: first standard success, first standard failure,
    first wide video."""
    runs = _runs(rows)
    run = HEADLINE if HEADLINE in runs else (runs[0] if runs else None)
    by_key = {(r["run"], r["setting"]): r for r in rows}
    tiles = []
    for setting, want, label in (
        ("standard", True, "success"),
        ("standard", False, "failure"),
        ("wide", None, "wide pose"),
    ):
        row = by_key.get((run, setting))
        if row is None:
            continue
        for i, ok in enumerate(row.get("video_successes", [])):
            if want is None or ok == want:
                video = _video(row, i)
                if video is not None:
                    outcome = "" if want is not None else (" (ok)" if ok else " (fail)")
                    tiles.append((f"{run}: {label}{outcome}", video))
                    break
    return tiles


def _read_frames(path: Path, fps: int, tile: int, max_frames: int) -> list[np.ndarray]:
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(path)
    step = max(1, round(reader.get_meta_data().get("fps", fps) / fps))
    frames = []
    for i, frame in enumerate(reader):
        if i % step == 0:
            frames.append(np.asarray(Image.fromarray(frame).resize((tile, tile))))
        if len(frames) >= max_frames:
            break
    reader.close()
    return frames


def rollout_grid(
    rows: list[dict], path: Path, fps: int = 10, tile: int = 256, max_seconds: float = 15.0
) -> Path | None:
    """Labelled side-by-side GIF of up to three rollouts (see `_tiles`); None if no videos.
    Shorter clips hold their last frame."""
    from PIL import Image, ImageDraw

    tiles = _tiles(rows)
    if not tiles:
        return None
    clips = [_read_frames(v, fps, tile, int(max_seconds * fps)) for _, v in tiles]
    n = max(len(c) for c in clips)
    strip = 16
    frames = []
    for t in range(n):
        canvas = Image.new("RGB", (tile * len(clips), tile + strip), "white")
        draw = ImageDraw.Draw(canvas)
        for k, ((label, _), clip) in enumerate(zip(tiles, clips, strict=True)):
            canvas.paste(Image.fromarray(clip[min(t, len(clip) - 1)]), (k * tile, strip))
            draw.text((k * tile + 4, 2), label, fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
    return path


def make_report(
    results_dir: Path, out_dir: Path = Path("docs/figures"), ablation_dir: Path | None = None
) -> list[Path]:
    """`results.md`, `success_rates.png`, (if any videos) `rollouts.gif`, and for each plan in
    `ablation_dir` with evaluated cells an `ablation_<plan>.png`, in `out_dir`."""
    rows = load_results(results_dir)
    if not rows:
        raise FileNotFoundError(f"no <run>/<setting>/results.json under {results_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "results.md"
    table.write_text(results_table(rows))
    written = [table, plot_success_rates(rows, out_dir / "success_rates.png")]
    gif = rollout_grid(rows, out_dir / "rollouts.gif")
    written += [gif] if gif else []
    for plan in load_plans(ablation_dir) if ablation_dir else []:
        chart = plot_ablation(plan, rows, out_dir / f"ablation_{plan['name']}.png")
        written += [chart] if chart else []
    return written
