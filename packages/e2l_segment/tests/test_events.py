import numpy as np

from e2l_common.config import SegmentConfig
from e2l_segment.events import aperture_to_grip, grip_events

CFG = SegmentConfig(close_threshold=0.035, open_threshold=0.055, min_dwell_s=0.25)
FPS = 30.0


def _t(n: int) -> np.ndarray:
    return np.arange(n) / FPS


def test_pinch_closes_and_opens_at_thresholds():
    ap = np.r_[np.full(30, 0.08), np.full(30, 0.02), np.full(30, 0.08)]
    grip = aperture_to_grip(_t(90), ap, CFG)
    assert grip.dtype == np.int8
    np.testing.assert_array_equal(grip, np.r_[np.zeros(30), np.ones(30), np.zeros(30)])


def test_hysteresis_keeps_grip_closed_between_thresholds():
    ap = np.r_[np.full(20, 0.08), np.full(20, 0.02), np.tile([0.04, 0.05], 20), np.full(20, 0.08)]
    grip = aperture_to_grip(_t(len(ap)), ap, CFG)
    assert grip[20:80].all() and not grip[80:].any()


def test_short_blips_are_removed():
    ap = np.full(90, 0.08)
    ap[40:43] = 0.02  # 0.1 s < 0.25 s
    assert not aperture_to_grip(_t(90), ap, CFG).any()
    ap = np.full(90, 0.02)
    ap[40:43] = 0.08
    assert aperture_to_grip(_t(90), ap, CFG).all()


def test_nan_holds_state():
    ap = np.r_[np.full(20, 0.08), np.full(20, 0.02), np.full(30, np.nan), np.full(20, 0.08)]
    grip = aperture_to_grip(_t(len(ap)), ap, CFG)
    assert grip[20:70].all() and not grip[70:].any()


def test_all_nan_is_open():
    assert not aperture_to_grip(_t(10), np.full(10, np.nan), CFG).any()


def test_grip_events_kinds_indices_and_times():
    grip = np.r_[np.zeros(30), np.ones(30), np.zeros(30)].astype(np.int8)
    events = grip_events(_t(90), grip)
    assert [(e.kind, e.index) for e in events] == [("grasp", 30), ("release", 60)]
    assert np.isclose(events[1].t, 2.0)
    assert grip_events(_t(5), np.zeros(5, np.int8)) == []
