"""Out-of-sample validation: do lineup ratings predict future matches?

Train the npxG-RAPM on all matches before ``holdout_cutoff``, then predict each
held-out match's npxG differential from its kickoff XIs:

    pred = home_advantage + sum(ratings of home XI) - sum(ratings of away XI)

with unseen/low-minute players at replacement level. Compared against two
baselines: (a) team strength (mean train npxG diff per 90 for/against) + home
advantage, and (b) home advantage alone.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from . import rapm
from .config import Config, load_config
from .validate import load_tables

log = logging.getLogger(__name__)


def run_holdout(cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    t = load_tables(cfg)
    matches, appearances, stints = t["matches"], t["appearances"], t["stints"]

    dates = pd.to_datetime(matches["date"])
    cutoff = pd.Timestamp(cfg.holdout_cutoff)
    train_ids = set(matches.loc[dates < cutoff, "match_id"])
    test_ids = set(matches.loc[dates >= cutoff, "match_id"])
    log.info("holdout: %d train / %d test matches (cutoff %s)",
             len(train_ids), len(test_ids), cfg.holdout_cutoff)

    train_apps = appearances[appearances["match_id"].isin(train_ids)]
    train_stints = stints[stints["match_id"].isin(train_ids)].reset_index(drop=True)
    design = rapm.build_design(cfg, train_apps, train_stints)

    with open(cfg.processed_dir / "model_meta.json") as f:
        lam = json.load(f)["xg"]["lambda"]  # reuse the full-sample lambda
    model = Ridge(alpha=lam, fit_intercept=True)
    y = design.y_xg
    model.fit(design.X, y, sample_weight=design.weights)
    rating = dict(zip(design.columns, model.coef_))
    replacement = rating[rapm.REPLACEMENT]
    home_adv = float(model.intercept_)

    # actual per-match npxG and goal differentials
    per_match = stints.groupby("match_id").agg(
        npxg_diff=("h_npxg", "sum"), a_npxg=("a_npxg", "sum"),
        goal_diff=("h_goals", "sum"), a_goals=("a_goals", "sum"),
    )
    per_match["npxg_diff"] -= per_match["a_npxg"]
    per_match["goal_diff"] -= per_match["a_goals"]

    # baseline (a): team strength = mean signed npxG diff over train matches
    m_train = matches[matches["match_id"].isin(train_ids)].set_index("match_id")
    m_train = m_train.join(per_match[["npxg_diff"]])
    for_team: dict[str, list] = {}
    for r in m_train.itertuples():
        for_team.setdefault(r.home_team, []).append(r.npxg_diff)
        for_team.setdefault(r.away_team, []).append(-r.npxg_diff)
    strength = {team: float(np.mean(v)) for team, v in for_team.items()}
    mean_home_adv = float(m_train["npxg_diff"].mean())

    # predictions per test match
    kickoff = appearances[(appearances["match_id"].isin(test_ids))
                          & (appearances["entry"] == 0)]
    rows = []
    m_test = matches[matches["match_id"].isin(test_ids)].set_index("match_id")
    for mid, xi in kickoff.groupby("match_id"):
        sides = {
            side: sum(rating.get(pid, replacement)
                      for pid in xi.loc[xi["side"] == side, "player_id"])
            for side in ("h", "a")
        }
        meta_row = m_test.loc[mid]
        rows.append({
            "match_id": mid,
            "pred_rapm": home_adv + sides["h"] - sides["a"],
            "pred_team": mean_home_adv
            + strength.get(meta_row["home_team"], 0.0)
            - strength.get(meta_row["away_team"], 0.0),
            "pred_home": mean_home_adv,
            "actual_npxg": per_match.loc[mid, "npxg_diff"],
            "actual_goals": per_match.loc[mid, "goal_diff"],
        })
    df = pd.DataFrame(rows)

    results: dict = {"n_test": len(df), "cutoff": cfg.holdout_cutoff, "lambda": lam}
    for target in ("actual_npxg", "actual_goals"):
        for pred in ("pred_rapm", "pred_team", "pred_home"):
            mse = float(((df[target] - df[pred]) ** 2).mean())
            rho = float(spearmanr(df[target], df[pred]).statistic) if pred != "pred_home" else float("nan")
            results[f"{pred}__{target}"] = {"mse": round(mse, 4), "spearman": round(rho, 4)}
    with open(cfg.processed_dir / "holdout.json", "w") as f:
        json.dump(results, f, indent=2)
    df.to_parquet(cfg.processed_dir / "holdout_predictions.parquet", index=False)
    for k, v in results.items():
        if isinstance(v, dict):
            log.info("%-28s mse=%.4f rho=%s", k, v["mse"], v["spearman"])
    return results
