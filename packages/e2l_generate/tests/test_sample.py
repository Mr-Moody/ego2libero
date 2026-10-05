import numpy as np
import pytest

from e2l_common.config import GenerateConfig
from e2l_common.geometry import make_T, matrix_to_rotvec, rotvec_to_matrix
from e2l_generate.sample import sample_object_poses

NOMINAL = {
    "bowl": make_T(rotvec_to_matrix(np.array([0.0, 0.0, 0.3])), [-0.1, 0.0, 0.9]),
    "plate": make_T(np.eye(3), [0.1, 0.0, 0.9]),
    "cheese": make_T(np.eye(3), [0.0, 0.2, 0.9]),
}


def test_offsets_and_yaw_stay_within_ranges():
    cfg = GenerateConfig(xy_range=0.05, yaw_range=0.4, min_separation=0.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        poses = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, rng)
        assert set(poses) == {"bowl", "plate"}  # unreferenced objects are not returned
        for name, T in poses.items():
            d = T[:3, 3] - NOMINAL[name][:3, 3]
            assert np.all(np.abs(d[:2]) <= 0.05) and d[2] == 0.0
            rv = matrix_to_rotvec(T[:3, :3] @ NOMINAL[name][:3, :3].T)
            assert abs(rv[2]) <= 0.4 + 1e-12
            np.testing.assert_allclose(rv[:2], 0.0, atol=1e-12)  # yaw only: stays upright
            np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_zero_noise_returns_nominal():
    cfg = GenerateConfig(xy_range=0.0, yaw_range=0.0, min_separation=0.0)
    poses = sample_object_poses(NOMINAL, ["bowl"], cfg, np.random.default_rng(1))
    np.testing.assert_allclose(poses["bowl"], NOMINAL["bowl"])


def test_seeded_and_independent_of_name_order():
    cfg = GenerateConfig()
    a = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, np.random.default_rng(7))
    b = sample_object_poses(NOMINAL, ["plate", "bowl"], cfg, np.random.default_rng(7))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_accepted_placements_respect_min_separation():
    # Nominal xy distance 0.2 m, noise +/-0.08 m each: many draws fall below 0.19 m.
    cfg = GenerateConfig(xy_range=0.08, yaw_range=0.0, min_separation=0.19)
    rng = np.random.default_rng(3)
    accepted = 0
    for _ in range(100):
        poses = sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, rng)
        if poses is not None:
            accepted += 1
            assert np.linalg.norm(poses["bowl"][:2, 3] - poses["plate"][:2, 3]) >= 0.19
    assert accepted > 0


def test_returns_none_when_no_placement_fits():
    cfg = GenerateConfig(xy_range=0.01, min_separation=1.0, max_placement_tries=5)
    assert sample_object_poses(NOMINAL, ["bowl", "plate"], cfg, np.random.default_rng(0)) is None


def test_unknown_object_is_named():
    with pytest.raises(KeyError, match="bowl_2"):
        sample_object_poses(NOMINAL, ["bowl_2"], GenerateConfig(), np.random.default_rng(0))
