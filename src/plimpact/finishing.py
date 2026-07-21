"""Individual finishing skill via empirical-Bayes shrinkage.

RAPM measures collective on-pitch impact and (deliberately) credits chance
*creation*, not conversion. Finishing is an individual skill, measured here
from each player's own shots: goals minus xG per shot, shrunk toward zero
because raw over/under-performance is mostly luck in small samples.

Model: player p's per-shot conversion edge d_p = (G_p - sum xG_p) / n_p with
sampling variance se2_p = sum xG_i(1 - xG_i) / n_p^2 (Bernoulli). True skill
s_p ~ N(0, tau^2); tau^2 estimated by method of moments across players
(observed variance of d_p minus mean sampling variance, floored at 0).
Posterior mean: shrunk_p = d_p * tau^2 / (tau^2 + se2_p).

The overlay is expressed per 90 through the player's own shot volume:
finishing_per90 = shrunk_p * shots_per90. Penalties are included — conversion
from the spot is part of finishing (their open-play xG was already stripped
from the RAPM response).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

MIN_SHOTS_FOR_TAU = 30  # players used to estimate the prior variance


def finishing_table(
    cfg: Config, shots: pd.DataFrame, appearances: pd.DataFrame
) -> pd.DataFrame:
    """Per player: shots, raw and shrunk finishing per shot, finishing per 90."""
    own = shots[shots["result"] != "OwnGoal"].copy()
    own["goal"] = (own["result"] == "Goal").astype(float)
    own["var"] = own["xg"] * (1.0 - own["xg"])

    g = own.groupby("player_id").agg(
        shots=("goal", "size"),
        goals=("goal", "sum"),
        xg_sum=("xg", "sum"),
        var_sum=("var", "sum"),
    )
    g["d"] = (g["goals"] - g["xg_sum"]) / g["shots"]
    g["se2"] = g["var_sum"] / g["shots"] ** 2

    est = g[g["shots"] >= MIN_SHOTS_FOR_TAU]
    tau2 = max(0.0, float(est["d"].var() - est["se2"].mean()))
    log.info(
        "finishing prior: tau^2=%.5f (tau=%.4f per shot) from %d players",
        tau2, np.sqrt(tau2), len(est),
    )

    g["shrunk"] = g["d"] * tau2 / (tau2 + g["se2"]) if tau2 > 0 else 0.0

    minutes = appearances.groupby("player_id")["minutes"].sum()
    g["minutes"] = minutes.reindex(g.index)
    g["shots_per90"] = g["shots"] / g["minutes"] * cfg.per90_scale
    g["finishing_per90"] = g["shrunk"] * g["shots_per90"]
    g.attrs["tau2"] = tau2
    return g.reset_index()
