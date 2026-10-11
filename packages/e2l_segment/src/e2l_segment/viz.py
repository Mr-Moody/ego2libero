"""Segmentation plot: aperture with thresholds, binary grip, and segment boundaries."""

from pathlib import Path

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory
from e2l_segment.events import aperture_to_grip, grip_events
from e2l_segment.segment import segment_demo


def plot_segments(demo: DemoTrajectory, cfg: SegmentConfig, out: Path) -> Path:
    """Aperture (m) with close/open thresholds, grip shaded where closed, grasp/release lines
    labelled with each segment's reference object. Writes a PNG to `out`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grip = aperture_to_grip(demo.t, demo.aperture, cfg)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(demo.t, demo.aperture, color="black", lw=1.2, label="aperture")
    ax.axhline(cfg.close_threshold, color="tab:red", ls="--", lw=0.8, label="close")
    ax.axhline(cfg.open_threshold, color="tab:green", ls="--", lw=0.8, label="open")
    ax.fill_between(
        demo.t,
        0,
        1,
        where=grip == 1,
        transform=ax.get_xaxis_transform(),
        color="tab:orange",
        alpha=0.15,
        label="grip closed",
    )
    for e in grip_events(demo.t, grip):
        ax.axvline(e.t, color="tab:red" if e.kind == "grasp" else "tab:green", lw=1)
        ax.annotate(
            e.kind,
            (e.t, 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )
    title = f"{demo.demo_id} segmentation"
    try:
        segments = segment_demo(demo, cfg).segments
    except ValueError as err:
        segments, title = [], f"{title} (no segments: {err})"
    for s in segments:
        ax.annotate(
            s.ref_object,
            ((s.t_start + s.t_end) / 2, 0.02),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=8,
        )
    ax.set(xlabel="t (s)", ylabel="aperture (m)", title=title)
    ax.set_ylim(0, np.nanmax(np.r_[demo.aperture, cfg.open_threshold]) * 1.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
