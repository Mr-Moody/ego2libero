import numpy as np

from e2l_perception.calibration import Intrinsics, load_intrinsics
from e2l_perception.hand.fusion import HandFusion


def test_fusion_tracks_static_hand_and_records_history(make_detection):
    fusion = HandFusion(fps=30, record=True)
    det = make_detection("Right", (100, 100))
    outs = [fusion.update([det]) for _ in range(10)]
    assert outs[0].shape == (21, 3)
    np.testing.assert_allclose(outs[-1], det.landmarks_3d, atol=1e-3)
    hist = fusion.ekf.history()
    assert hist["x_pred"].shape == (9, 126) and hist["P"].shape == (9, 126, 126)
    assert fusion.update([None]).shape == (21, 3)  # frozen output when the hand is lost


def test_fusion_without_recording_has_empty_history(make_detection):
    fusion = HandFusion()
    fusion.update([make_detection("Right", (1, 1))])
    fusion.update([make_detection("Right", (1, 1))])
    assert fusion.ekf.history()["x"].shape == (0,)


def test_intrinsics_round_trip(tmp_path, make_detection):
    K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])
    Intrinsics(K, np.zeros(5), (1920, 1080), 0.3).save(tmp_path / "phone.json")
    back = load_intrinsics(tmp_path / "phone.json")
    np.testing.assert_array_equal(back.K, K)
    assert back.size == (1920, 1080) and back.dist.shape == (5,)
