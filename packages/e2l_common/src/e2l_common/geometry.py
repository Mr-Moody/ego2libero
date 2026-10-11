"""SE(3) helpers. All poses are 4x4 float64 transforms named T_a_b (pose of frame b in frame a),
so p_a = T_a_b @ p_b. Units: metres, radians, seconds. Rotations are matrices or rotation
vectors; quaternions only appear at library call sites (see `quat_xyzw` naming)."""

from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "average_rotations",
    "compose",
    "delta_action",
    "interpolate_poses",
    "inv_T",
    "make_T",
    "matrix_to_rotvec",
    "perturb",
    "rotvec_to_matrix",
    "sample_wide_offset",
    "separated",
]


def make_T(R: np.ndarray | None = None, t: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 transform from rotation (3,3) and translation (3,). Batched inputs (...,3,3)
    and (...,3) give (...,4,4)."""
    R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
    t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
    batch = np.broadcast_shapes(R.shape[:-2], t.shape[:-1])
    T = np.zeros((*batch, 4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    """Invert rigid transform(s) (...,4,4) using R^T rather than a general inverse."""
    T = np.asarray(T, dtype=np.float64)
    R_inv = np.swapaxes(T[..., :3, :3], -1, -2)
    t_inv = -np.einsum("...ij,...j->...i", R_inv, T[..., :3, 3])
    return make_T(R_inv, t_inv)


def compose(*Ts: np.ndarray) -> np.ndarray:
    """Chain transforms left to right: compose(T_a_b, T_b_c) == T_a_c."""
    if not Ts:
        raise ValueError("compose needs at least one transform")
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (...,3) in radians to rotation matrices (...,3,3)."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*rotvec.shape[:-1], 3, 3)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (...,3,3) to axis-angle (...,3) in radians."""
    R = np.asarray(R, dtype=np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def interpolate_poses(t_src: np.ndarray, T_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Interpolate poses T_src (N,4,4) sampled at increasing times t_src (N,) onto t_query (M,):
    linear in translation, slerp in rotation. Query times are clamped to [t_src[0], t_src[-1]]."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_query = np.clip(np.asarray(t_query, dtype=np.float64), t_src[0], t_src[-1])
    T_src = np.asarray(T_src, dtype=np.float64)
    pos = np.stack([np.interp(t_query, t_src, T_src[:, i, 3]) for i in range(3)], axis=-1)
    if len(t_src) == 1:
        R = np.broadcast_to(T_src[0, :3, :3], (len(t_query), 3, 3))
    else:
        R = Slerp(t_src, Rotation.from_matrix(T_src[:, :3, :3]))(t_query).as_matrix()
    return make_T(R, pos)


def average_rotations(R: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (N,3,3), optionally weighted (N,). Returns (3,3)."""
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).mean(weights).as_matrix()


def delta_action(T_prev: np.ndarray, T_next: np.ndarray) -> np.ndarray:
    """6-D delta between two end-effector poses in the same base frame: (dpos (3), drot (3)).

    dpos = t_next - t_prev in the base frame; drot is the axis-angle of R_next @ R_prev^T, also
    in the base frame (robosuite OSC_POSE convention). Unscaled; `e2l_sim.control` applies the
    LIBERO action scaling and clipping."""
    T_prev = np.asarray(T_prev, dtype=np.float64)
    T_next = np.asarray(T_next, dtype=np.float64)
    dpos = T_next[..., :3, 3] - T_prev[..., :3, 3]
    dR = T_next[..., :3, :3] @ np.swapaxes(T_prev[..., :3, :3], -1, -2)
    return np.concatenate([dpos, matrix_to_rotvec(dR)], axis=-1)


def perturb(T_sim_obj: np.ndarray, d_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Shift an object pose (4,4) by `d_xy` in the sim xy plane and turn it by `yaw` about the
    sim z axis through its own origin; height and tilt are unchanged."""
    T = np.array(T_sim_obj, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, yaw])) @ T[:3, :3]
    T[:2, 3] += d_xy
    return T


def separated(poses: dict[str, np.ndarray], min_distance: float) -> bool:
    """True if every pair of poses is at least `min_distance` apart in xy."""
    return all(
        np.linalg.norm(a[:2, 3] - b[:2, 3]) >= min_distance
        for a, b in combinations(poses.values(), 2)
    )


def sample_wide_offset(
    rng: np.random.Generator, min_offset: float, max_offset: float, min_yaw: float, max_yaw: float
) -> tuple[np.ndarray, float]:
    """One (d_xy, yaw) draw: d_xy uniform over the disc of radius `max_offset`, rejecting draws
    inside the square +/- `min_offset` on both axes; |yaw| uniform in [min_yaw, max_yaw] with a
    random sign."""
    while True:
        d_xy = rng.uniform(-max_offset, max_offset, size=2)
        if np.linalg.norm(d_xy) <= max_offset and np.max(np.abs(d_xy)) > min_offset:
            break
    yaw = rng.uniform(min_yaw, max_yaw) * rng.choice([-1.0, 1.0])
    return d_xy, float(yaw)
