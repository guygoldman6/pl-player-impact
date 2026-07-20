"""Regularized Adjusted Plus-Minus (RAPM) via ridge regression on stints.

Design matrix: one row per stint, one column per qualified player (+1 on pitch
for the home side, -1 for the away side), a pooled "replacement" column for
players under the minutes threshold, and a man-advantage control (home minus
away player count, non-zero after red cards). The unpenalized intercept
captures home advantage. Response: stint goal (or xG) differential scaled to
per-90; rows weighted by stint duration.

A player's coefficient reads as: goals (or xG) per 90 the player adds over a
replacement-level player, holding teammates and opponents fixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .config import Config

log = logging.getLogger(__name__)

REPLACEMENT = "__replacement__"
MAN_DIFF = "__man_diff__"


@dataclass
class StintDesign:
    X: sparse.csr_matrix
    y_goals: np.ndarray
    y_xg: np.ndarray
    weights: np.ndarray          # stint durations in minutes
    match_ids: np.ndarray
    columns: list[str]           # player_id per column, plus REPLACEMENT and MAN_DIFF


def qualified_players(appearances: pd.DataFrame, min_minutes: int) -> list[str]:
    totals = appearances.groupby("player_id")["minutes"].sum()
    return sorted(totals[totals >= min_minutes].index)


def build_design(cfg: Config, appearances: pd.DataFrame, stints: pd.DataFrame) -> StintDesign:
    players = qualified_players(appearances, cfg.min_minutes)
    col_of = {p: i for i, p in enumerate(players)}
    rep_col = len(players)
    man_col = len(players) + 1

    rows, cols, vals = [], [], []
    for i, stint in enumerate(stints.itertuples()):
        for side_players, sign in ((stint.h_players, 1.0), (stint.a_players, -1.0)):
            for pid in side_players:
                j = col_of.get(pid, rep_col)
                rows.append(i)
                cols.append(j)
                vals.append(sign)
        man = len(stint.h_players) - len(stint.a_players)
        if man:
            rows.append(i)
            cols.append(man_col)
            vals.append(float(man))

    n = len(stints)
    X = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, len(players) + 2)
    )
    # duplicate (row, col) pairs sum automatically, which is what we want for
    # multiple replacement-level players on the same pitch
    X.sum_duplicates()

    duration = stints["duration"].to_numpy(dtype=float)
    scale = cfg.per90_scale / duration
    return StintDesign(
        X=X,
        y_goals=(stints["h_goals"] - stints["a_goals"]).to_numpy(dtype=float) * scale,
        y_xg=(stints["h_xg"] - stints["a_xg"]).to_numpy(dtype=float) * scale,
        weights=duration,
        match_ids=stints["match_id"].to_numpy(),
        columns=[*players, REPLACEMENT, MAN_DIFF],
    )


def cv_lambda(cfg: Config, design: StintDesign, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Pick ridge lambda by grouped CV (folds never split a match across sets)."""
    gkf = GroupKFold(n_splits=cfg.cv_folds)
    records = []
    for lam in cfg.ridge_lambdas:
        errors = []
        for train, test in gkf.split(design.X, y, groups=design.match_ids):
            model = Ridge(alpha=lam, fit_intercept=True)
            model.fit(design.X[train], y[train], sample_weight=design.weights[train])
            pred = model.predict(design.X[test])
            errors.append(
                np.average((y[test] - pred) ** 2, weights=design.weights[test])
            )
        records.append({"lambda": lam, "cv_mse": float(np.mean(errors))})
        log.info("lambda=%-7g cv_mse=%.6f", lam, records[-1]["cv_mse"])
    curve = pd.DataFrame(records)
    best = float(curve.loc[curve["cv_mse"].idxmin(), "lambda"])
    return best, curve


def fit_rapm(design: StintDesign, y: np.ndarray, lam: float) -> tuple[pd.Series, dict]:
    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(design.X, y, sample_weight=design.weights)
    coefs = pd.Series(model.coef_, index=design.columns)
    meta = {
        "lambda": lam,
        "home_advantage": float(model.intercept_),
        "man_diff_coef": float(coefs[MAN_DIFF]),
        "replacement_coef": float(coefs[REPLACEMENT]),
    }
    return coefs.drop([MAN_DIFF]), meta


def split_half_ratings(
    design: StintDesign, y: np.ndarray, lam: float, seed: int = 13
) -> pd.DataFrame:
    """Reliability check: fit on two random halves of the matches, return both
    coefficient vectors so their correlation can be inspected."""
    rng = np.random.default_rng(seed)
    matches = np.unique(design.match_ids)
    half = rng.permutation(matches)[: len(matches) // 2]
    in_a = np.isin(design.match_ids, half)
    out = {}
    for name, mask in (("half_a", in_a), ("half_b", ~in_a)):
        model = Ridge(alpha=lam, fit_intercept=True)
        model.fit(design.X[mask], y[mask], sample_weight=design.weights[mask])
        out[name] = pd.Series(model.coef_, index=design.columns)
    return pd.DataFrame(out).drop([MAN_DIFF, REPLACEMENT])


def bootstrap_ci(
    design: StintDesign, y: np.ndarray, lam: float, iters: int, seed: int = 7
) -> pd.DataFrame:
    """Cluster bootstrap by match: resample matches, refit, take percentile CIs."""
    rng = np.random.default_rng(seed)
    unique_matches = np.unique(design.match_ids)
    row_idx_of_match = {
        m: np.flatnonzero(design.match_ids == m) for m in unique_matches
    }
    samples = np.empty((iters, design.X.shape[1]))
    for b in range(iters):
        chosen = rng.choice(unique_matches, size=len(unique_matches), replace=True)
        idx = np.concatenate([row_idx_of_match[m] for m in chosen])
        model = Ridge(alpha=lam, fit_intercept=True)
        model.fit(design.X[idx], y[idx], sample_weight=design.weights[idx])
        samples[b] = model.coef_
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    return pd.DataFrame(
        {"ci_lo": lo, "ci_hi": hi, "se": samples.std(axis=0)}, index=design.columns
    ).drop([MAN_DIFF])
