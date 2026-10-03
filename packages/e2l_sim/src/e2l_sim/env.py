"""Create LIBERO environments from `configs/sim.yaml`."""

from e2l_common.config import SimConfig
from e2l_common.stub import not_implemented


def make_env(cfg: SimConfig):
    """LIBERO OffScreenRenderEnv for `cfg.suite` / `cfg.task_id`, seeded, with
    `cfg.camera_names` rendered at `cfg.image_size` and `cfg.control_freq` Hz."""
    not_implemented("e2l_sim.env.make_env")
