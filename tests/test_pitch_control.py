import numpy as np

from pitchvision.pitch_control import control_map, grid, home_control


def test_lone_player_controls_nearby_and_far_cells_go_to_the_closer_team():
    pos = np.array([[-30.0, 0.0], [30.0, 0.0]])
    vel = np.zeros((2, 2))
    team = np.array([0, 1])
    ball = np.array([0.0, 0.0])
    assert home_control(np.array([-30.0, 0.0]), pos, vel, team, ball) > 0.99
    assert home_control(np.array([30.0, 0.0]), pos, vel, team, ball) < 0.01


def test_mirror_symmetry_gives_one_half_on_the_middle_line():
    rng = np.random.default_rng(0)
    home = rng.uniform(-40, -5, (5, 2))
    away = home * np.array([-1, 1])
    pos, team = np.vstack([home, away]), np.array([0] * 5 + [1] * 5)
    vel = np.zeros_like(pos)
    for y in (-20.0, 0.0, 15.0):
        assert abs(home_control(np.array([0.0, y]), pos, vel, team, np.array([0.0, 0.0])) - 0.5) < 1e-9


def test_running_towards_a_cell_increases_control():
    pos, team, ball = np.array([[0.0, 0.0], [6.0, 0.0]]), np.array([0, 1]), np.array([0.0, 10.0])
    cell = np.array([3.0, 10.0])
    still = home_control(cell, pos, np.zeros((2, 2)), team, ball)
    running = home_control(cell, pos, np.array([[3.0, 5.0], [0.0, 0.0]]), team, ball)
    assert running > still


def test_map_shape():
    pos = np.array([[-10.0, 0.0], [10.0, 0.0]])
    m = control_map(pos, np.zeros((2, 2)), np.array([0, 1]), np.array([0.0, 0.0]), nx=16)
    assert m.shape == grid(16).shape[:2] and np.all((m >= 0) & (m <= 1))
