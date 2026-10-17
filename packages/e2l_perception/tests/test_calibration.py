import cv2
import numpy as np

from e2l_perception.calibration import board_object_points, calibrate_from_corners

SIZE = (1920, 1080)
K_TRUE = np.array([[1690.0, 0.0, 970.0], [0.0, 1690.0, 540.0], [0.0, 0.0, 1.0]])


def synthetic_views(n: int, noise_px: float, seed: int = 0) -> list[np.ndarray]:
    """Checkerboard corners (N, 1, 2) seen by K_TRUE (no distortion) from varied poses."""
    rng = np.random.default_rng(seed)
    obj = board_object_points()
    obj = obj - obj.mean(axis=0)
    views = []
    for _ in range(n):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.7)])
        img, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, np.zeros(5))
        views.append((img + rng.normal(0, noise_px, img.shape)).astype(np.float32))
    return views


def test_clean_views_recover_intrinsics():
    intr, kept = calibrate_from_corners(synthetic_views(20, 0.1), SIZE)
    assert kept.all()
    assert intr.rms < 0.3
    np.testing.assert_allclose(intr.K[[0, 1], [0, 1]], 1690.0, rtol=0.01)
    assert intr.dist[4] == 0.0  # k3 fixed


def test_corrupted_views_are_dropped():
    views = synthetic_views(20, 0.1)
    rng = np.random.default_rng(1)
    for i in (3, 9, 15):  # e.g. motion blur / rolling-shutter skew
        views[i] = (views[i] + rng.normal(0, 4.0, views[i].shape)).astype(np.float32)
    intr, kept = calibrate_from_corners(views, SIZE)
    assert set(np.flatnonzero(~kept)) == {3, 9, 15}
    assert intr.rms < 0.3
    np.testing.assert_allclose(intr.K[[0, 1], [0, 1]], 1690.0, rtol=0.01)
