"""Transform raw Understat JSON into tidy parquet tables under data/processed/.

Tables:
- matches:      one row per match (teams, date, final score)
- appearances:  one row per player-match (entry/exit minutes on the match clock)
- shots:        one row per shot (minute, side, result, xG)
- stints:       one row per stint (rosters as player-id lists, goals, xG)
- players:      one row per player (canonical name, latest team, total minutes)
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from .config import Config, load_config
from .parse import parse_match
from .scrape import season_dir
from .stints import build_stints, compute_intervals

log = logging.getLogger(__name__)


def build_all(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    match_rows, app_rows, shot_rows, stint_rows = [], [], [], []

    for season in cfg.seasons:
        files = sorted(season_dir(cfg, season).glob("*.json"))
        log.info("season %s: building from %d match files", season, len(files))
        for f in files:
            payload = json.loads(f.read_text())
            meta, apps, shots = parse_match(payload, season)
            intervals, match_length = compute_intervals(apps)
            stints = build_stints(apps, shots)

            match_rows.append(
                {
                    "match_id": meta.match_id, "season": season, "date": meta.date,
                    "home_team": meta.home_team, "away_team": meta.away_team,
                    "home_goals": meta.home_goals, "away_goals": meta.away_goals,
                    "match_length": match_length,
                }
            )
            for iv in intervals:
                a = iv.appearance
                team = meta.home_team if a.side == "h" else meta.away_team
                app_rows.append(
                    {
                        "match_id": meta.match_id, "season": season, "date": meta.date,
                        "side": a.side, "team": team,
                        "player_id": a.player_id, "player": a.player,
                        "position": a.position, "entry": iv.entry, "exit": iv.exit,
                        "minutes": iv.exit - iv.entry,
                        "understat_time": a.time, "red_card": a.red_card,
                    }
                )
            for s in shots:
                shot_rows.append(
                    {
                        "match_id": meta.match_id, "season": season,
                        "minute": s.minute, "side": s.side,
                        "player_id": s.player_id, "player": s.player,
                        "result": s.result, "xg": s.xg, "situation": s.situation,
                    }
                )
            for i, st in enumerate(stints):
                stint_rows.append(
                    {
                        "match_id": meta.match_id, "season": season, "stint_idx": i,
                        "start": st.start, "end": st.end, "duration": st.duration,
                        "h_players": sorted(st.h_players),
                        "a_players": sorted(st.a_players),
                        "h_goals": st.h_goals, "a_goals": st.a_goals,
                        "h_xg": st.h_xg, "a_xg": st.a_xg,
                    }
                )

    tables = {
        "matches": pd.DataFrame(match_rows),
        "appearances": pd.DataFrame(app_rows),
        "shots": pd.DataFrame(shot_rows),
        "stints": pd.DataFrame(stint_rows),
    }
    tables["players"] = _player_table(tables["appearances"])

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(cfg.processed_dir / f"{name}.parquet", index=False)
        log.info("wrote %s: %d rows", name, len(df))
    return tables


def _position_group(pos: str) -> str | None:
    """Map an understat position string to GK/DEF/MID/FWD.

    'Sub' rows carry no positional information and are excluded.
    """
    if pos == "Sub":
        return None
    if pos == "GK":
        return "GK"
    if pos in {"DC", "DR", "DL"}:
        return "DEF"
    if pos.startswith("FW") or pos in {"AML", "AMR"}:
        return "FWD"
    return "MID"


def _player_table(appearances: pd.DataFrame) -> pd.DataFrame:
    latest = appearances.sort_values("date").groupby("player_id").last()
    totals = appearances.groupby("player_id")["minutes"].sum()

    pos = appearances.assign(group=appearances["position"].map(_position_group))
    pos = pos.dropna(subset=["group"])
    primary = (
        pos.groupby(["player_id", "group"])["minutes"].sum()
        .reset_index()
        .sort_values("minutes")
        .groupby("player_id")["group"].last()
    )
    return pd.DataFrame(
        {
            "player": latest["player"],
            "latest_team": latest["team"],
            "position": primary.reindex(totals.index).fillna("MID"),
            "total_minutes": totals,
        }
    ).reset_index()
