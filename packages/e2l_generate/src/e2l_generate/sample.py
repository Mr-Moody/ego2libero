"""Sample new object placements around nominal (LIBERO init-state) poses."""

from collections.abc import Iterable

import numpy as np

from e2l_common.config import GenerateConfig
from e2l_common.geometry import perturb, separated


def sample_object_poses(
    nominal: dict[str, np.ndarray],
    names: Iterable[str],
    cfg: GenerateConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """Perturb the named nominal poses T_sim_obj (4,4) by a uniform xy offset within
    +/- cfg.xy_range and a yaw within +/- cfg.yaw_range. Draws that put two of them closer than
    cfg.min_separation are resampled, up to cfg.max_placement_tries times; then None. Names are
    processed in sorted order so a seeded rng gives the same placement for any input order."""
    names = sorted(set(names))
    missing = [n for n in names if n not in nominal]
    if missing:
        raise KeyError(f"objects {missing} are not in the scene; have {sorted(nominal)}")
    for _ in range(cfg.max_placement_tries):
        poses = {
            n: perturb(
                nominal[n],
                rng.uniform(-cfg.xy_range, cfg.xy_range, size=2),
                rng.uniform(-cfg.yaw_range, cfg.yaw_range),
            )
            for n in names
        }
        if separated(poses, cfg.min_separation):
            return poses
    return None
