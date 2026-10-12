"""Hand-designed bowl-onto-plate `RobotSegments`: the nominal demo, or a jittered copy that
varies the way human demos do, to rehearse everything after retargeting without perception.

Segments follow the same cut as real demos (approach -> grasp, carry -> release, retreat) and
are expressed in the real object names of configs/sim.yaml `object_map`. The object frames are
LIBERO's body frames (origin at the base centre, z up), so the grasp is a top-down pinch on the
bowl rim and the release is just above the plate centre.
"""

from pathlib import Path

import numpy as np

from e2l_common.geometry import make_T
from e2l_common.paths import DataPaths
from e2l_common.schemas import RobotSegment, RobotSegments

FPS = 20.0
DEMO_PREFIX = "20261003_scripted"
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])  # grip site z down, fingers closing along object x
RIM_RADIUS = 0.05  # m; bowl rim r 0.056 m, h 0.053 m; angle 0 is away from the cabinet
GRASP_Z = 0.045
UP = 0.10  # m, approach height above the grasp; the place approach and retreat use 1.5x
PLACE_Z = 0.03  # m above the grasp height: bowl base ~1.5 cm above the plate's base
DURATIONS = (1.5, 1.5, 1.0)  # s: approach, carry, retreat
# Jitter, uniform: +/- bounds, and the per-segment duration scale range.
GRASP_ANGLE = 0.4  # rad around the rim
UP_JITTER = 0.02
PLACE_Z_JITTER = 0.01
PLACE_XY_JITTER = 0.01
DURATION_SCALE = (0.7, 1.3)


def scripted_demo_id(index: int) -> str:
    if not 0 <= index < 1000:
        raise ValueError(f"scripted demo index {index} is outside 0..999")
    return f"{DEMO_PREFIX}_{index:03d}"


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _line(R: np.ndarray, p0: np.ndarray, p1: np.ndarray, seconds: float) -> np.ndarray:
    """Poses with rotation R from p0 to p1 (object frame), FPS samples per second."""
    s = np.linspace(0.0, 1.0, max(2, round(seconds * FPS)))[:, None]
    return make_T(R, (1 - s) * p0 + s * p1)


def _segment(ref: str, T: np.ndarray, closed: bool, flip_at_end: bool = False) -> RobotSegment:
    gripper = np.full(len(T), 1.0 if closed else -1.0)
    if flip_at_end:
        gripper[-1] = -gripper[-1]
    return RobotSegment(
        ref_object=ref, T_obj_ee=T, gripper=gripper, feasible=np.ones(len(T), dtype=bool)
    )


def scripted_segments(demo_id: str, rng: np.random.Generator | None = None) -> RobotSegments:
    """The nominal demo when `rng` is None, else one draw of the jitter bounds above. The grip
    yaw turns with the grasp angle, so fingers always close across the rim, and the carry keeps
    the same grip-to-object pose relative to the plate, so the bowl lands centred on it."""
    angle, up, place_z, place_xy, scale = 0.0, UP, PLACE_Z, np.zeros(2), np.ones(3)
    if rng is not None:
        angle = rng.uniform(-GRASP_ANGLE, GRASP_ANGLE)
        up = UP + rng.uniform(-UP_JITTER, UP_JITTER)
        place_z = PLACE_Z + rng.uniform(-PLACE_Z_JITTER, PLACE_Z_JITTER)
        place_xy = rng.uniform(-PLACE_XY_JITTER, PLACE_XY_JITTER, size=2)
        scale = rng.uniform(*DURATION_SCALE, size=3)
    R = _rot_z(angle) @ R_TOP_DOWN
    grasp = np.array([RIM_RADIUS * np.cos(angle), RIM_RADIUS * np.sin(angle), GRASP_Z])
    place = grasp + np.array([place_xy[0], place_xy[1], place_z])
    lift = np.array([0.0, 0.0, up])
    t = np.asarray(DURATIONS) * scale
    return RobotSegments(
        demo_id=demo_id,
        fps=FPS,
        segments=[
            _segment("bowl", _line(R, grasp + lift, grasp, t[0]), closed=False, flip_at_end=True),
            _segment(
                "plate", _line(R, place + 1.5 * lift, place, t[1]), closed=True, flip_at_end=True
            ),
            _segment("plate", _line(R, place, place + 1.5 * lift, t[2]), closed=False),
        ],
    )


def write_scripted(paths: DataPaths, count: int, seed: int = 0) -> list[Path]:
    """Write demos 000..count-1 to robot_segments/: 000 nominal, i >= 1 jittered with
    rng([seed, i]). Returns the npz paths in index order."""
    if not 1 <= count <= 1000:
        raise ValueError(f"count must be in 1..1000, got {count}")
    out = []
    for i in range(count):
        demo_id = scripted_demo_id(i)
        rng = None if i == 0 else np.random.default_rng([seed, i])
        npz, _ = scripted_segments(demo_id, rng).save(paths.robot_segments(demo_id))
        out.append(npz)
    return out
