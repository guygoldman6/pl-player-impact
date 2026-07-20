"""Parse raw Understat match JSON into typed records.

Understat conventions (verified against real matches):
- Roster entries with ``position == "Sub"`` came off the bench; their
  ``roster_out`` field holds the roster id of the player they replaced.
- A replaced player's ``time`` is the minute of the substitution, so the
  sub's entry minute equals the replaced player's exit minute.
- A sent-off player has ``red_card == "1"`` and exits at entry + time with
  no replacement (his ``roster_in`` stays "0").
- The clock caps regulation at 90, but stoppage-time subs chain past it
  (e.g. starter time=90 replaced by a sub with time=1 -> match length 91).
"""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class Appearance:
    roster_id: str
    player_id: str
    player: str
    side: str  # "h" or "a"
    position: str
    time: int
    roster_in: str  # roster id of the player who came ON for this player, "0" if none
    roster_out: str  # roster id of the player this sub replaced, "0" for starters
    red_card: bool


@dataclass(frozen=True)
class Shot:
    minute: int
    side: str  # shooter's side, "h" or "a"
    player_id: str
    player: str
    result: str  # Goal, OwnGoal, MissedShots, SavedShot, BlockedShot, ShotOnPost
    xg: float
    situation: str


@dataclass(frozen=True)
class MatchMeta:
    match_id: str
    season: int
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


def parse_match(payload: dict, season: int) -> tuple[MatchMeta, list[Appearance], list[Shot]]:
    m = payload["match"]
    meta = MatchMeta(
        match_id=str(m["id"]),
        season=season,
        date=m["datetime"],
        home_team=html.unescape(m["h"]["title"]),
        away_team=html.unescape(m["a"]["title"]),
        home_goals=int(m["goals"]["h"]),
        away_goals=int(m["goals"]["a"]),
    )

    appearances = []
    for side in ("h", "a"):
        for entry in payload["roster"][side].values():
            time_played = int(entry["time"])
            is_sub = entry["position"] == "Sub"
            # skip unused bench players (present in some payloads with time 0)
            if is_sub and entry["roster_out"] == "0" and time_played == 0:
                continue
            appearances.append(
                Appearance(
                    roster_id=str(entry["id"]),
                    player_id=str(entry["player_id"]),
                    player=html.unescape(entry["player"]),
                    side=side,
                    position=entry["position"],
                    time=time_played,
                    roster_in=str(entry["roster_in"]),
                    roster_out=str(entry["roster_out"]),
                    red_card=str(entry["red_card"]) == "1",
                )
            )

    shots = []
    for side in ("h", "a"):
        for s in payload["shots"].get(side, []):
            shots.append(
                Shot(
                    minute=int(s["minute"]),
                    side=s["h_a"],
                    player_id=str(s["player_id"]),
                    player=html.unescape(s["player"]),
                    result=s["result"],
                    xg=float(s["xG"]),
                    situation=s.get("situation", ""),
                )
            )
    shots.sort(key=lambda s: s.minute)
    return meta, appearances, shots
