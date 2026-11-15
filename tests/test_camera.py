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
