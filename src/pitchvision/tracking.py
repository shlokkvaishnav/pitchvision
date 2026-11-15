"""Standard tracking format and loaders (see specs/01-data-and-camera.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SKILLCORNER_MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
_CORNER_KEYS = ("bottom_left", "bottom_right", "top_right", "top_left")  # counter-clockwise


@dataclass
class Tracking:
    match_id: str
    fps: float
    period: np.ndarray  # (T,)
    ball: np.ndarray  # (T, 2)
    players: np.ndarray  # (T, N, 2), NaN = no data
    player_ids: list[str]
    teams: list[str]  # "home" or "away"
    pitch: tuple[float, float]  # (length, width) in metres
    detected: np.ndarray | None = None  # (T, N) bool, broadcast data only
    corners: np.ndarray | None = None  # (T, 4, 2), broadcast data only

    def __post_init__(self) -> None:
        t, n = self.players.shape[:2]
        if self.players.shape != (t, n, 2) or self.ball.shape != (t, 2) or self.period.shape != (t,):
            raise ValueError("inconsistent shapes")
        if len(self.player_ids) != n or len(self.teams) != n:
            raise ValueError("player_ids and teams must have one entry per player")
        if self.detected is not None and self.detected.shape != (t, n):
            raise ValueError("detected must have shape (T, N)")
        if self.corners is not None and self.corners.shape != (t, 4, 2):
            raise ValueError("corners must have shape (T, 4, 2)")

    def select(self, rows: np.ndarray) -> Tracking:
        """Return a copy with only the given frames."""
        return Tracking(
            match_id=self.match_id,
            fps=self.fps,
            period=self.period[rows],
            ball=self.ball[rows],
            players=self.players[rows],
            player_ids=self.player_ids,
            teams=self.teams,
            pitch=self.pitch,
            detected=None if self.detected is None else self.detected[rows],
            corners=None if self.corners is None else self.corners[rows],
        )


def from_kloppy(dataset, match_id: str) -> Tracking:
    """Convert a kloppy TrackingDataset in metric, centred coordinates ("secondspectrum")."""
    dims = dataset.metadata.pitch_dimensions
    if dims.x_dim.min >= 0 or dims.y_dim.min >= 0:
        raise ValueError("load the dataset with centred metric coordinates, for example coordinates='secondspectrum'")
    players = [p for team in dataset.metadata.teams for p in team.players]
    index = {p.player_id: i for i, p in enumerate(players)}
    t = len(dataset.frames)
    pos = np.full((t, len(players), 2), np.nan)
    ball = np.full((t, 2), np.nan)
    period = np.zeros(t, dtype=int)
    for k, frame in enumerate(dataset.frames):
        period[k] = frame.period.id
        if frame.ball_coordinates is not None:
            ball[k] = frame.ball_coordinates.x, frame.ball_coordinates.y
        for player, data in frame.players_data.items():
            if data.coordinates is not None:
                pos[k, index[player.player_id]] = data.coordinates.x, data.coordinates.y
    return Tracking(
        match_id=str(match_id),
        fps=float(dataset.metadata.frame_rate),
        period=period,
        ball=ball,
        players=pos,
        player_ids=[str(p.player_id) for p in players],
        teams=[p.team.ground.value for p in players],
        pitch=(dims.pitch_length, dims.pitch_width),
    )


def load_skillcorner(tracking_path: str | Path, match_path: str | Path) -> Tracking:
    """Parse a SkillCorner open data match. Keeps frames that have player data."""
    meta = json.loads(Path(match_path).read_text())
    home_id = meta["home_team"]["id"]
    people = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    index = {p["id"]: i for i, p in enumerate(people)}

    rows = []
    with open(tracking_path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("player_data") and r.get("period") is not None:
                rows.append(r)
    t, n = len(rows), len(people)
    pos = np.full((t, n, 2), np.nan)
    det = np.zeros((t, n), dtype=bool)
    ball = np.full((t, 2), np.nan)
    corners = np.full((t, 4, 2), np.nan)
    period = np.zeros(t, dtype=int)
    for k, r in enumerate(rows):
        period[k] = r["period"]
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            ball[k] = b["x"], b["y"]
        c = r.get("image_corners_projection")
        if c:
            corners[k] = [[c[f"x_{key}"], c[f"y_{key}"]] for key in _CORNER_KEYS]
        for p in r["player_data"]:
            i = index.get(p["player_id"])
            if i is not None:
                pos[k, i] = p["x"], p["y"]
                det[k, i] = bool(p.get("is_detected"))
    return Tracking(
        match_id=str(meta["id"]),
        fps=10.0,
        period=period,
        ball=ball,
        players=pos,
        player_ids=[str(p["id"]) for p in people],
        teams=["home" if p["team_id"] == home_id else "away" for p in people],
        pitch=(float(meta["pitch_length"]), float(meta["pitch_width"])),
        detected=det,
        corners=corners,
    )
