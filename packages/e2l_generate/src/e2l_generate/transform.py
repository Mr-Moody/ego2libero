"""MimicGen-style transform of object-centric segments to new object poses."""

import numpy as np

from e2l_common.stub import not_implemented


def transform_segment(T_obj_ee: np.ndarray, T_sim_obj_new: np.ndarray) -> np.ndarray:
    """T_sim_ee_new (M,4,4) = T_sim_obj_new (4,4) @ T_obj_ee (M,4,4)."""
    not_implemented("e2l_generate.transform.transform_segment")
