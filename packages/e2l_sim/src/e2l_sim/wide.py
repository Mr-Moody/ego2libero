"""Wide-pose evaluation: after LIBERO's reset, move the task objects to placements beyond the
generation range. The `LiberoEnv` subclass is built lazily so this module imports without
LIBERO (the stub checker imports every module)."""

from collections.abc import Iterable
from functools import cache

import numpy as np

from e2l_common.config import WidePoseConfig
from e2l_common.geometry import matrix_to_rotvec, perturb, sample_wide_offset


def placement_valid(
    moved: dict[str, np.ndarray], others: dict[str, np.ndarray], cfg: WidePoseConfig
) -> bool:
    """Moved objects stay inside cfg.table_xy and at least cfg.min_separation (xy) from every
    other object, moved or not. Pairs of unmoved objects are not checked."""
    (x0, x1), (y0, y1) = cfg.table_xy
    for name, T in moved.items():
        x, y = T[:2, 3]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        rest = [P for n, P in {**others, **moved}.items() if n != name]
        if any(np.linalg.norm(T[:2, 3] - P[:2, 3]) < cfg.min_separation for P in rest):
            return False
    return True


def sample_wide_placement(
    nominal: dict[str, np.ndarray],
    names: Iterable[str],
    cfg: WidePoseConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """One draw: each named nominal pose T_sim_obj (4,4) moved by `sample_wide_offset`, in
    sorted name order. None if the draw fails `placement_valid` against the other objects."""
    names = sorted(set(names))
    missing = [n for n in names if n not in nominal]
    if missing:
        raise KeyError(f"objects {missing} are not in the scene; have {sorted(nominal)}")
    moved = {}
    for n in names:
        d_xy, yaw = sample_wide_offset(
            rng, cfg.min_offset, cfg.max_offset, cfg.min_yaw, cfg.max_yaw
        )
        moved[n] = perturb(nominal[n], d_xy, yaw)
    others = {n: T for n, T in nominal.items() if n not in moved}
    return moved if placement_valid(moved, others, cfg) else None


def settled(actual: dict[str, np.ndarray], targets: dict[str, np.ndarray], tol: float) -> bool:
    """Every target object ended within `tol` of its target in xy and in height."""
    return all(
        np.linalg.norm(actual[n][:2, 3] - T[:2, 3]) <= tol and abs(actual[n][2, 3] - T[2, 3]) <= tol
        for n, T in targets.items()
    )


def _pose_record(T: np.ndarray) -> dict[str, list[float]]:
    return {"xyz": T[:3, 3].tolist(), "rotvec": matrix_to_rotvec(T[:3, :3]).tolist()}


@cache
def wide_pose_env_class() -> type:
    """`LiberoEnv` subclass whose reset moves `names` to a wide placement drawn from the reset
    seed. Construct with the usual LiberoEnv kwargs plus `wide: WidePoseConfig`, `names`."""
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    from e2l_sim.scene import _problem_env, object_poses, set_object_poses

    class WidePoseLiberoEnv(LiberoEnv):
        def __init__(self, *args, wide: WidePoseConfig, names: Iterable[str], **kwargs):
            super().__init__(*args, **kwargs)
            self.wide = wide
            self.names = sorted(names)
            self.placements: list[dict] = []

        def reset(self, seed=None, **kwargs):
            super().reset(seed=seed, **kwargs)  # init state, settle steps, control mode
            rng = np.random.default_rng(seed)
            saved = _problem_env(self).sim.get_state().flatten()
            nominal = object_poses(self)
            raw_obs, targets = None, None
            for _ in range(self.wide.max_tries):
                draw = sample_wide_placement(nominal, self.names, self.wide, rng)
                if draw is None:
                    continue
                self._env.regenerate_obs_from_state(saved)
                set_object_poses(self, draw)
                for _ in range(self.num_steps_wait):
                    raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())
                if settled(object_poses(self), {**nominal, **draw}, self.wide.settle_tol):
                    targets = draw
                    break
            if targets is None:  # run the episode from the unchanged init state
                raw_obs = self._env.regenerate_obs_from_state(saved)
            final = object_poses(self)
            self.placements.append(
                {
                    "seed": seed,
                    "placement_failed": targets is None,
                    "T_sim_obj": {n: _pose_record(final[n]) for n in self.names},
                }
            )
            return self._format_raw_obs(raw_obs), {"is_success": False}

    return WidePoseLiberoEnv
