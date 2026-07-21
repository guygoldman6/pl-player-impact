"""Unit tests for the empirical-Bayes finishing overlay."""

import numpy as np
import pandas as pd
import pytest

from plimpact.config import load_config
from plimpact.finishing import finishing_table


def make_shots(player_id, n, goals, xg_each):
    results = ["Goal"] * goals + ["SavedShot"] * (n - goals)
    return pd.DataFrame(
        {
            "player_id": player_id,
            "result": results,
            "xg": xg_each,
        }
    )


def make_apps(players, minutes):
    return pd.DataFrame({"player_id": players, "minutes": minutes})


@pytest.fixture()
def cfg():
    return load_config()


def test_small_sample_overperformer_shrinks_hard(cfg):
    # a large population of neutral finishers pins tau^2 near zero-ish;
    # 5 shots with 3 goals at 0.1 xG each is huge raw overperformance
    frames = [make_shots(f"n{i}", 60, 6, 0.1) for i in range(40)]
    frames.append(make_shots("lucky", 5, 3, 0.1))
    shots = pd.concat(frames, ignore_index=True)
    apps = make_apps([f"n{i}" for i in range(40)] + ["lucky"], [3000] * 41)

    table = finishing_table(cfg, shots, apps).set_index("player_id")
    raw = table.loc["lucky", "d"]
    shrunk = table.loc["lucky", "shrunk"]
    assert raw == pytest.approx(0.5)
    assert abs(shrunk) < 0.25 * raw  # at least 75% of the raw edge removed


def test_large_sample_retains_more_signal_than_small(cfg):
    rng = np.random.default_rng(3)
    frames = []
    names, mins = [], []
    for i in range(40):  # heterogeneous population -> tau^2 > 0
        n = 80
        goals = int(rng.binomial(n, 0.10 + 0.06 * rng.random()))
        frames.append(make_shots(f"n{i}", n, goals, 0.1))
        names.append(f"n{i}")
        mins.append(4000)
    frames.append(make_shots("big", 300, 45, 0.1))    # +0.05/shot on 300 shots
    frames.append(make_shots("small", 20, 3, 0.1))    # +0.05/shot on 20 shots
    names += ["big", "small"]
    mins += [9000, 1000]
    shots = pd.concat(frames, ignore_index=True)

    table = finishing_table(cfg, shots, make_apps(names, mins)).set_index("player_id")
    keep_big = table.loc["big", "shrunk"] / table.loc["big", "d"]
    keep_small = table.loc["small", "shrunk"] / table.loc["small", "d"]
    # retention must grow with sample size, and a 300-shot sample must keep a
    # substantial share of its raw signal (exact value depends on tau^2)
    assert keep_big > 2 * keep_small
    assert keep_big > 0.25


def test_homogeneous_population_gives_zero_tau(cfg):
    # every player finishes exactly at expectation -> no skill variance
    frames = [make_shots(f"n{i}", 50, 5, 0.1) for i in range(35)]
    shots = pd.concat(frames, ignore_index=True)
    apps = make_apps([f"n{i}" for i in range(35)], [3000] * 35)
    table = finishing_table(cfg, shots, apps)
    assert (table["shrunk"] == 0).all()
    assert (table["finishing_per90"] == 0).all()


def test_own_goals_excluded(cfg):
    shots = pd.concat(
        [
            make_shots("a", 40, 4, 0.1),
            pd.DataFrame({"player_id": ["a"], "result": ["OwnGoal"], "xg": [0.0]}),
        ],
        ignore_index=True,
    )
    table = finishing_table(cfg, shots, make_apps(["a"], [2000]))
    assert int(table.loc[0, "shots"]) == 40
