"""Overlay videos and trajectory plots."""

from pathlib import Path

from e2l_common.stub import not_implemented


def render_overlay(video: Path, demo_id: str, out: Path) -> Path:
    """Draw landmarks, palm axes and marker axes on the normalised video; write mp4 to `out`."""
    not_implemented("e2l_perception.viz.render_overlay")


def plot_trajectory(demo_id: str, out: Path) -> Path:
    """Plot table-frame wrist position and aperture over time; write a PNG to `out`."""
    not_implemented("e2l_perception.viz.plot_trajectory")
