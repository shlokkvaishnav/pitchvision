import numpy as np
import pytest

from pitchvision.camera import PITCH_WIDTH, Footprint, camera_centre, visible_mask

FP = Footprint(near_width=30.0, far_width=50.0, pan_smoothing=0.0)


def test_player_at_the_ball_is_visible_and_far_player_is_hidden():
    pos = np.array([[[10.0, 0.0], [-45.0, 0.0]]])  # (T=1, N=2, 2)
    mask = visible_mask(pos, np.array([10.0]), FP)
    assert mask.tolist() == [[True, False]]


def test_far_side_is_wider_than_near_side():
    # 20 m from the camera centre: outside the near half-width (15 m), inside the far half-width (25 m)
    near = [20.0, -PITCH_WIDTH / 2]
    far = [20.0, PITCH_WIDTH / 2]
    mask = visible_mask(np.array([[near, far]]), np.array([0.0]), FP)
    assert mask.tolist() == [[False, True]]


def test_nan_positions_are_not_visible():
    pos = np.array([[[np.nan, 0.0], [0.0, np.nan]]])
    assert not visible_mask(pos, np.array([0.0]), FP).any()


def test_pan_is_smooth_and_holds_on_missing_ball():
    fp = Footprint(near_width=30.0, far_width=50.0, pan_smoothing=0.5)
    c = camera_centre(np.array([0.0, 20.0, np.nan]), fp)
    assert c.tolist() == [0.0, 10.0, 10.0]


def test_centre_stays_inside_the_pitch():
    c = camera_centre(np.array([52.0]), FP)
    assert c[0] == pytest.approx(52.5 - 25.0)


def test_bad_footprint_raises():
    with pytest.raises(ValueError):
        Footprint(near_width=50.0, far_width=30.0)
    with pytest.raises(ValueError):
        visible_mask(np.zeros((3, 2)), np.zeros(3), FP)


def test_polygon_mask_uses_the_real_footprint():
    from pitchvision.camera import polygon_mask

    square = np.array([[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]])
    pos = np.array([[[0.0, 0.0], [11.0, 0.0], [np.nan, 0.0]]])
    assert polygon_mask(pos, square).tolist() == [[True, False, False]]
    assert not polygon_mask(pos, np.full((1, 4, 2), np.nan)).any()


def test_custom_pitch_length_moves_the_clip():
    c = camera_centre(np.array([60.0]), FP, pitch_length=120.0)
    assert c[0] == pytest.approx(60.0 - 25.0)


def test_replay_moves_donor_footprints_with_the_target_ball():
    from pitchvision.camera import footprint_sequences, replay

    square = np.array([[-10.0, -30.0], [10.0, -30.0], [20.0, 39.0], [-20.0, 39.0]])
    corners = np.stack([square + [5.0, 0.0], square + [6.0, 0.0], np.full((4, 2), np.nan)])  # frame 3 has no corners
    donor = footprint_sequences(np.ones(3, int), np.array([5.0, 6.0, 7.0]), corners)
    assert np.allclose(donor[1][0], square) and np.allclose(donor[1][2], square - [1.0, 0.0])  # ffill, then relative
    out = replay(donor, np.array([1, 1, 1, 1]), np.array([0.0, 40.0, np.nan, 1.0]))
    assert np.allclose(out[1], square + [40.0, 0.0])
    assert np.allclose(out[2, :, 0], square[:, 0] - 1.0 + 40.0)  # target ball NaN: keep the last ball x
    assert np.allclose(out[3], square + [1.0, 0.0])  # donor is shorter: repeat it
