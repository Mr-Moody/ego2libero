import numpy as np
import pytest
from pydantic import ValidationError

from e2l_common.config import Pose, SegmentConfig, SimConfig, load_config


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_model_from_stem_and_overrides(tmp_path):
    p = write(
        tmp_path, "sim.yaml", "suite: libero_goal\ntask_id: 2\nT_sim_table:\n  xyz: [0.1, 0, 0.9]\n"
    )
    cfg = load_config(
        p, ["task_id=5", "T_sim_table.rotvec=[0, 0, 1.5]", "camera_names=[agentview]"]
    )
    assert isinstance(cfg, SimConfig)
    assert cfg.task_id == 5 and cfg.camera_names == ["agentview"]
    assert cfg.T_sim_table.xyz == (0.1, 0, 0.9) and cfg.T_sim_table.rotvec == (0, 0, 1.5)


def test_dict_overrides_and_explicit_model(tmp_path):
    p = write(tmp_path, "custom.yaml", "close_threshold: 0.03\n")
    cfg = load_config(p, {"min_dwell_s": 0.5}, model=SegmentConfig)
    assert cfg.close_threshold == 0.03 and cfg.min_dwell_s == 0.5
    with pytest.raises(ValueError, match="no config model"):
        load_config(p)


def test_unknown_keys_and_bad_overrides_fail(tmp_path):
    p = write(tmp_path, "segment.yaml", "close_threshold: 0.03\n")
    with pytest.raises(ValidationError):
        load_config(p, ["typo_key=1"])
    with pytest.raises(ValueError, match="key.sub=value"):
        load_config(p, ["no_equals"])


def test_pose_T():
    T = Pose(xyz=(1, 2, 3), rotvec=(0, 0, np.pi)).T
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [-1, 0, 0], atol=1e-12)
