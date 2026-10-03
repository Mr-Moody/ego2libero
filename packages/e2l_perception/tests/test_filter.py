from e2l_perception.hand.filter import HandFilter


def test_world_wrists_are_identical_so_old_gate_could_not_separate_hands(make_detection):
    a, b = make_detection("Right", (100, 100)), make_detection("Left", (500, 100))
    assert (a.landmarks_3d[0] == b.landmarks_3d[0]).all()


def test_selects_requested_hand_even_when_other_is_listed_first(make_detection):
    f = HandFilter(hand="Right")
    right = make_detection("Right", (500, 300), score=0.6)
    out = f.select([make_detection("Left", (100, 300), score=0.99), right])
    assert out is right


def test_keeps_tracked_hand_and_follows_nearest_wrist(make_detection):
    f = HandFilter(hand="Right", max_jump_px=50)
    f.select([make_detection("Right", (100, 100))])
    near = make_detection("Right", (120, 110))
    far = make_detection("Right", (400, 400), score=0.99)
    assert f.select([far, near]) is near
    assert f.tracked_hand_age == 2


def test_rejects_jumps_and_reacquires_after_memory(make_detection):
    f = HandFilter(hand="Right", max_jump_px=50, memory_frames=2)
    f.select([make_detection("Right", (100, 100))])
    jump = make_detection("Right", (600, 600))
    assert f.select([jump]) is None  # miss 1: gate 100 px
    assert f.select([jump]) is None  # miss 2: gate 150 px
    assert f.select([jump]) is jump  # memory expired: reacquire
    assert f.frames_since_lost == 0 and f.tracked_hand_age == 1


def test_wrong_hand_only_is_a_miss(make_detection):
    f = HandFilter(hand="Right")
    f.select([make_detection("Right", (100, 100))])
    assert f.select([make_detection("Left", (100, 100))]) is None
    assert f.frames_since_lost == 1


def test_either_hand_when_unset(make_detection):
    f = HandFilter(hand=None)
    left = make_detection("Left", (100, 100))
    assert f.select([left]) is left
