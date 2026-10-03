"""Write a hand-designed bowl-onto-plate `RobotSegments` file, to test replay without perception.

Segments follow the same cut as real demos (approach -> grasp, carry -> release, retreat) and
are expressed in the real object names of configs/sim.yaml `object_map`. The object frames are
LIBERO's body frames (origin at the base centre, z up), so the grasp is a top-down pinch on the
bowl rim and the release is just above the plate centre.

    uv run python scripts/scripted_segments.py
    uv run e2l sim replay 20261003_scripted_000 --video
"""

from pathlib import Path

import numpy as np
import typer

from e2l_common.geometry import make_T
from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegment, RobotSegments

FPS = 20.0
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])  # grip site z down, fingers closing along object x
GRASP = np.array([0.05, 0.0, 0.045])  # bowl rim (r 0.056 m, h 0.053 m), away from the cabinet


def line(p0, p1, seconds: float) -> np.ndarray:
    """Top-down grip-site poses from p0 to p1 (object frame), FPS samples per second."""
    s = np.linspace(0.0, 1.0, max(2, round(seconds * FPS)))[:, None]
    return make_T(R_TOP_DOWN, (1 - s) * np.asarray(p0) + s * np.asarray(p1))


def segment(ref: str, T: np.ndarray, closed: bool, flip_at_end: bool = False) -> RobotSegment:
    gripper = np.full(len(T), 1.0 if closed else -1.0)
    if flip_at_end:
        gripper[-1] = -gripper[-1]
    return RobotSegment(
        ref_object=ref, T_obj_ee=T, gripper=gripper, feasible=np.ones(len(T), dtype=bool)
    )


def scripted(demo_id: str) -> RobotSegments:
    up = np.array([0.0, 0.0, 0.10])
    place = GRASP + np.array([0.0, 0.0, 0.03])  # bowl base ~1.5 cm above the plate's base
    return RobotSegments(
        demo_id=demo_id,
        fps=FPS,
        segments=[
            segment("bowl", line(GRASP + up, GRASP, 1.5), closed=False, flip_at_end=True),
            segment("plate", line(place + 1.5 * up, place, 1.5), closed=True, flip_at_end=True),
            segment("plate", line(place, place + 1.5 * up, 1.0), closed=False),
        ],
    )


def main(demo_id: str = "20261003_scripted_000", root: Path | None = None) -> None:
    paths = DataPaths(root) if root else DataPaths.default()
    npz, _ = scripted(demo_id).save(paths.robot_segments(demo_id))
    print(f"wrote {npz}")


if __name__ == "__main__":
    typer.run(main)
