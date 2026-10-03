import numpy as np

from e2l_perception.hand.detector import to_detection
from e2l_perception.hand.palm import compute_palm_frame


def test_unmirrored_keeps_world_x(hand_pts, fake_mediapipe_hand):
    lm2d, lm3d, cat = fake_mediapipe_hand(hand_pts, "Left")
    det = to_detection(lm2d, lm3d, cat, 640, 480, mirrored=False)
    np.testing.assert_allclose(det.landmarks_3d, hand_pts, atol=1e-6)
    np.testing.assert_allclose(det.landmarks_2d[0], [320, 240])


def test_mirrored_negates_world_x(hand_pts, fake_mediapipe_hand):
    lm2d, lm3d, cat = fake_mediapipe_hand(hand_pts, "Left")
    det = to_detection(lm2d, lm3d, cat, 640, 480, mirrored=True)
    np.testing.assert_allclose(det.landmarks_3d[:, 0], -hand_pts[:, 0], atol=1e-6)
    np.testing.assert_allclose(det.landmarks_3d[:, 1:], hand_pts[:, 1:], atol=1e-6)


def test_mirroring_flips_palm_chirality(hand_pts, fake_mediapipe_hand):
    """The x-negation reflects the hand, so the palm normal flips relative to the fingers."""
    lm2d, lm3d, cat = fake_mediapipe_hand(hand_pts, "Left")
    R_plain = compute_palm_frame(to_detection(lm2d, lm3d, cat, 1, 1, mirrored=False).landmarks_3d)
    R_mirror = compute_palm_frame(to_detection(lm2d, lm3d, cat, 1, 1, mirrored=True).landmarks_3d)
    assert np.isclose(np.linalg.det(R_plain), 1) and np.isclose(np.linalg.det(R_mirror), 1)
    assert not np.allclose(R_plain, R_mirror)


def test_handedness_label_and_score(hand_pts, fake_mediapipe_hand):
    lm2d, lm3d, cat = fake_mediapipe_hand(hand_pts, "Left", score=0.8)
    # MediaPipe labels assume a mirrored image: unmirrored input swaps the label.
    assert to_detection(lm2d, lm3d, cat, 1, 1, mirrored=False).handedness == "Right"
    det = to_detection(lm2d, lm3d, cat, 1, 1, mirrored=True)
    assert det.handedness == "Left" and np.isclose(det.handedness_score, 0.8)
