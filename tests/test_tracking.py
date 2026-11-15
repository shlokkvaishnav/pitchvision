import json

import numpy as np
import pytest

from pitchvision.tracking import Tracking, load_skillcorner


def _write_fixture(tmp_path):
    match = {
        "id": 99,
        "home_team": {"id": 1},
        "pitch_length": 104,
        "pitch_width": 68,
        "players": [{"id": 20, "team_id": 2}, {"id": 10, "team_id": 1}],
    }
    corners = {
        "x_top_left": -50,
        "y_top_left": 39,
        "x_bottom_left": -20,
        "y_bottom_left": -37,
        "x_bottom_right": 20,
        "y_bottom_right": -37,
        "x_top_right": 50,
        "y_top_right": 39,
    }
    frames = [
        {"frame": 1, "period": None, "player_data": []},
        {
            "frame": 2,
            "period": 1,
            "ball_data": {"x": 1.0, "y": 2.0},
            "image_corners_projection": corners,
            "player_data": [
                {"x": 5.0, "y": 6.0, "player_id": 10, "is_detected": True},
                {"x": -5.0, "y": 1.0, "player_id": 20, "is_detected": False},
            ],
        },
    ]
    (tmp_path / "match.json").write_text(json.dumps(match))
    (tmp_path / "tracking.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    return tmp_path / "tracking.jsonl", tmp_path / "match.json"


def test_load_skillcorner_fixture(tmp_path):
    tr = load_skillcorner(*_write_fixture(tmp_path))
    assert tr.players.shape == (1, 2, 2)
    assert tr.player_ids == ["10", "20"] and tr.teams == ["home", "away"]
    assert tr.players[0, 0].tolist() == [5.0, 6.0]
    assert tr.detected.tolist() == [[True, False]]
    assert tr.corners[0, 0].tolist() == [-20.0, -37.0]  # bottom-left first
    assert tr.pitch == (104.0, 68.0)


def test_tracking_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Tracking("m", 25.0, np.zeros(3), np.zeros((3, 2)), np.zeros((3, 2, 2)), ["a"], ["home"], (105.0, 68.0))
