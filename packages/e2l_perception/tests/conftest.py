from types import SimpleNamespace

import numpy as np
import pytest

from e2l_perception.hand.detector import HandDetection

# A plausible right hand in MediaPipe world coordinates (metres, wrist-centred, palm facing -z).
_FINGER_X = {1: -0.03, 5: -0.02, 9: 0.0, 13: 0.02, 17: 0.035}


def canonical_hand() -> np.ndarray:
    pts = np.zeros((21, 3))
    for base, x in _FINGER_X.items():
        for j in range(4):
            pts[base + j] = [x * (1 + 0.1 * j), -0.03 - 0.025 * j - (0.03 if base != 1 else 0), 0]
    pts[[1, 2, 3, 4], 2] = [-0.01, -0.015, -0.02, -0.025]  # thumb out of the palm plane
    return pts


@pytest.fixture
def hand_pts() -> np.ndarray:
    return canonical_hand()


def _make_detection(
    handedness: str, wrist_px: tuple[float, float], score: float = 0.9
) -> HandDetection:
    lm2d = np.tile(np.asarray(wrist_px, np.float32), (21, 1))
    lm2d[1:] += np.random.default_rng(0).normal(scale=20, size=(20, 2)).astype(np.float32)
    return HandDetection(
        landmarks_2d=lm2d,
        landmarks_3d=canonical_hand().astype(np.float32),  # wrist at origin for every hand
        visible_mask=np.ones(21, bool),
        handedness=handedness,
        handedness_score=score,
    )


def _fake_mediapipe_hand(pts3d: np.ndarray, label: str, score: float = 0.95):
    lm2d = [SimpleNamespace(x=0.5 + p[0], y=0.5 + p[1], z=p[2]) for p in pts3d]
    lm3d = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in pts3d]
    return lm2d, lm3d, SimpleNamespace(category_name=label, score=score)


@pytest.fixture
def make_detection():
    return _make_detection


@pytest.fixture
def fake_mediapipe_hand():
    return _fake_mediapipe_hand
