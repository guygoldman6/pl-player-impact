"""Reconciliation gates for the processed data.

Hard gates (must hold for every match):
1. Sum of stint goals == understat's final score.
2. Kickoff stint has 11 players per side; counts only ever drop (red cards).
3. Per-player stint minutes equal the appearance interval exactly.

Cross-source gate:
4. Understat final scores match football-data.co.uk's independent CSVs.

Soft check:
5. Appearance minutes vs understat's own `time` field (allowed to differ by
   the stoppage overrun for players who were on at the final whistle).
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)

# understat name -> football-data.co.uk name, where they differ
FOOTBALLDATA_TEAMS = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield United",
}


def load_tables(cfg: Config) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(cfg.processed_dir / f"{name}.parquet")
        for name in ("matches", "appearances", "shots", "stints")
    }


def check_scores(matches: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Gate 1: stint goals must sum to the final score for every match."""
    sums = stints.groupby("match_id")[["h_goals", "a_goals"]].sum()
    merged = matches.set_index("match_id").join(sums, how="left")
    bad = merged[
        (merged["home_goals"] != merged["h_goals"])
        | (merged["away_goals"] != merged["a_goals"])
    ]
    return bad.reset_index()


def check_kickoff_eleven(stints: pd.DataFrame) -> pd.DataFrame:
    """Gate 2: every match starts 11v11 and player counts never increase."""
    first = stints[stints["stint_idx"] == 0]
    bad_start = first[
        (first["h_players"].str.len() != 11) | (first["a_players"].str.len() != 11)
    ]
    counts = stints.assign(
        h_n=stints["h_players"].str.len(), a_n=stints["a_players"].str.len()
    )
    increased = counts.groupby("match_id").filter(
        lambda g: (g.sort_values("stint_idx")[["h_n", "a_n"]].diff().max().max() or 0) > 0
    )
    return pd.concat([bad_start, increased]).drop_duplicates(subset=["match_id", "stint_idx"])


def check_player_minutes(appearances: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Gate 3: per player-match, summed stint durations == interval length."""
    long = stints.explode("h_players").rename(columns={"h_players": "player_id"})[
        ["match_id", "player_id", "duration"]
    ]
    long_a = stints.explode("a_players").rename(columns={"a_players": "player_id"})[
        ["match_id", "player_id", "duration"]
    ]
    stint_minutes = (
        pd.concat([long, long_a]).groupby(["match_id", "player_id"])["duration"].sum()
    )
    merged = appearances.merge(
        stint_minutes.rename("stint_minutes"),
        on=["match_id", "player_id"],
        how="left",
    ).fillna({"stint_minutes": 0})
    return merged[merged["minutes"] != merged["stint_minutes"]]


def check_footballdata_scores(cfg: Config, matches: pd.DataFrame) -> pd.DataFrame:
    """Gate 4: understat final scores vs football-data.co.uk's CSVs."""
    frames = []
    for season in cfg.seasons:
        code = cfg.footballdata_codes[season]
        path = cfg.raw_dir / "footballdata" / f"E0_{code}.csv"
        if not path.exists():
            log.warning("missing football-data CSV for season %s", season)
            continue
        fd = pd.read_csv(path)
        fd["fd_date"] = pd.to_datetime(fd["Date"], format="%d/%m/%Y").dt.date
        frames.append(fd[["fd_date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
    if not frames:
        return pd.DataFrame()
    fd_all = pd.concat(frames)

    us = matches.copy()
    us["fd_home"] = us["home_team"].map(lambda t: FOOTBALLDATA_TEAMS.get(t, t))
    us["fd_away"] = us["away_team"].map(lambda t: FOOTBALLDATA_TEAMS.get(t, t))
    us["fd_date"] = pd.to_datetime(us["date"]).dt.date

    merged = us.merge(
        fd_all,
        left_on=["fd_date", "fd_home", "fd_away"],
        right_on=["fd_date", "HomeTeam", "AwayTeam"],
        how="left",
    )
    unmatched = merged[merged["FTHG"].isna()]
    if len(unmatched):
        log.warning(
            "%d matches not found in football-data (name/date mismatch?): %s",
            len(unmatched),
            unmatched[["date", "home_team", "away_team"]].head(10).to_dict("records"),
        )
    matched = merged.dropna(subset=["FTHG"])
    return matched[
        (matched["home_goals"] != matched["FTHG"])
        | (matched["away_goals"] != matched["FTAG"])
    ]


def run_all_checks(cfg: Config | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    t = load_tables(cfg)
    failures = {
        "scores": check_scores(t["matches"], t["stints"]),
        "kickoff_eleven": check_kickoff_eleven(t["stints"]),
        "player_minutes": check_player_minutes(t["appearances"], t["stints"]),
        "footballdata_scores": check_footballdata_scores(cfg, t["matches"]),
    }
    for name, bad in failures.items():
        status = "OK" if bad.empty else f"{len(bad)} FAILURES"
        log.info("check %-22s %s", name, status)
    return failures
