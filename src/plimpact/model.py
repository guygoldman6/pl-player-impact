"""Orchestrate the modeling ladder: naive plus-minus -> RAPM -> xG-RAPM.

Writes:
- processed/ratings.parquet   one row per qualified player with all ratings
- processed/naive.parquet     per (player, team) naive on/off splits
- processed/cv_curves.parquet lambda-vs-CV-error for both responses
- processed/model_meta.json   chosen lambdas, home advantage, controls
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from . import rapm
from .config import Config, load_config
from .finishing import finishing_table
from .naive import naive_plus_minus
from .validate import load_tables

log = logging.getLogger(__name__)


def run_all(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    t = load_tables(cfg)
    matches, appearances, stints = t["matches"], t["appearances"], t["stints"]
    players = pd.read_parquet(cfg.processed_dir / "players.parquet")

    log.info("naive plus-minus")
    naive = naive_plus_minus(cfg, matches, stints)
    naive = naive.merge(players[["player_id", "player"]], on="player_id", how="left")
    naive.to_parquet(cfg.processed_dir / "naive.parquet", index=False)

    design = rapm.build_design(cfg, appearances, stints)
    log.info(
        "design: %d stints x %d columns (%d qualified players)",
        design.X.shape[0], design.X.shape[1], len(design.columns) - 2,
    )

    meta: dict = {}
    ratings = players.set_index("player_id")
    curves = []
    for name, y in (("goals", design.y_goals), ("xg", design.y_xg)):
        log.info("cross-validating lambda for %s response", name)
        lam, curve = rapm.cv_lambda(cfg, design, y)
        curves.append(curve.assign(response=name))
        coefs, fit_meta = rapm.fit_rapm(design, y, lam)
        meta[name] = fit_meta
        log.info("%s: lambda=%g home_adv=%.3f", name, lam, fit_meta["home_advantage"])

        log.info("bootstrapping %s CIs (%d iters)", name, cfg.bootstrap_iters)
        ci = rapm.bootstrap_ci(design, y, lam, cfg.bootstrap_iters)
        col = f"rapm_{name}"
        ratings[col] = coefs
        ratings[f"{col}_lo"] = ci["ci_lo"]
        ratings[f"{col}_hi"] = ci["ci_hi"]
        ratings[f"{col}_se"] = ci["se"]

    # attach minutes-weighted naive rating per player for comparison
    naive_player = (
        naive.assign(w=naive["on_minutes"])
        .groupby("player_id")
        .apply(
            lambda g: pd.Series(
                {
                    "naive_gd90": (g["naive_gd90"] * g["w"]).sum() / g["w"].sum(),
                    "naive_xgd90": (g["naive_xgd90"] * g["w"]).sum() / g["w"].sum(),
                }
            ),
            include_groups=False,
        )
    )
    ratings = ratings.join(naive_player)

    # individual finishing overlay -> headline impact90
    log.info("finishing overlay")
    fin = finishing_table(cfg, t["shots"], appearances).set_index("player_id")
    meta["finishing_tau2"] = fin.attrs["tau2"]
    ratings["shots"] = fin["shots"]
    ratings["finishing_per90"] = fin["finishing_per90"]
    ratings["finishing_per90"] = ratings["finishing_per90"].fillna(0.0)
    ratings["impact90"] = ratings["rapm_xg"] + ratings["finishing_per90"]
    ratings["impact90_lo"] = ratings["rapm_xg_lo"] + ratings["finishing_per90"]
    ratings["impact90_hi"] = ratings["rapm_xg_hi"] + ratings["finishing_per90"]

    ratings = ratings.dropna(subset=["rapm_goals"]).reset_index()
    ratings.to_parquet(cfg.processed_dir / "ratings.parquet", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(
        cfg.processed_dir / "cv_curves.parquet", index=False
    )
    with open(cfg.processed_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote ratings for %d players", len(ratings))
    return ratings
