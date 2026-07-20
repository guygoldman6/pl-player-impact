"""Naive on/off plus-minus: the intuitive baseline (and its cautionary tale).

For each (player, team): the team's goal difference per 90 while the player is
on the pitch, minus the same while they are off (including full matches missed).
Computed for goals and for xG. No adjustment for teammates, opponents, or
substitution context — that is the point; RAPM fixes it.
"""

from __future__ import annotations

import pandas as pd

from .config import Config


def stint_team_view(matches: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per (stint, team-perspective): goal/xG diff from that team's view."""
    st = stints.merge(matches[["match_id", "home_team", "away_team"]], on="match_id")
    home = st.assign(
        team=st["home_team"],
        players=st["h_players"],
        gd=st["h_goals"] - st["a_goals"],
        xgd=st["h_xg"] - st["a_xg"],
    )
    away = st.assign(
        team=st["away_team"],
        players=st["a_players"],
        gd=st["a_goals"] - st["h_goals"],
        xgd=st["a_xg"] - st["h_xg"],
    )
    cols = ["match_id", "season", "stint_idx", "duration", "team", "players", "gd", "xgd"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def naive_plus_minus(
    cfg: Config, matches: pd.DataFrame, stints: pd.DataFrame
) -> pd.DataFrame:
    """Per (player, team): on/off goal and xG differential per 90."""
    team_view = stint_team_view(matches, stints)

    on = team_view.explode("players").rename(columns={"players": "player_id"})
    on_agg = on.groupby(["player_id", "team"]).agg(
        on_minutes=("duration", "sum"), on_gd=("gd", "sum"), on_xgd=("xgd", "sum")
    )

    team_totals = team_view.groupby("team").agg(
        team_minutes=("duration", "sum"), team_gd=("gd", "sum"), team_xgd=("xgd", "sum")
    )

    df = on_agg.join(team_totals, on="team").reset_index()
    df["off_minutes"] = df["team_minutes"] - df["on_minutes"]
    df["off_gd"] = df["team_gd"] - df["on_gd"]
    df["off_xgd"] = df["team_xgd"] - df["on_xgd"]

    s = cfg.per90_scale
    df["on_gd90"] = df["on_gd"] / df["on_minutes"] * s
    df["off_gd90"] = (df["off_gd"] / df["off_minutes"] * s).where(df["off_minutes"] > 0)
    df["naive_gd90"] = df["on_gd90"] - df["off_gd90"]
    df["on_xgd90"] = df["on_xgd"] / df["on_minutes"] * s
    df["off_xgd90"] = (df["off_xgd"] / df["off_minutes"] * s).where(df["off_minutes"] > 0)
    df["naive_xgd90"] = df["on_xgd90"] - df["off_xgd90"]
    return df
