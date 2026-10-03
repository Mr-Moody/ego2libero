"""Sample new object placements."""

import numpy as np

from e2l_common.config import GenerateConfig
from e2l_common.stub import not_implemented


def sample_object_poses(
    T_sim_obj_nominal: dict[str, np.ndarray], cfg: GenerateConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Perturb each nominal object pose T_sim_obj (4,4) by a uniform xy offset within
    +/- cfg.xy_range and a yaw within +/- cfg.yaw_range about the sim z axis."""
    not_implemented("e2l_generate.sample.sample_object_poses")
