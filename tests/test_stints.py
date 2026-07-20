"""Unit tests for stint segmentation on synthetic match fixtures.

Sides are kept small (3 players) for readability; the builder does not
require 11 — the 11-at-kickoff invariant is checked against real data in
validate.py instead.
"""

import pytest

from plimpact.parse import Appearance, Shot
from plimpact.stints import StintError, build_stints, compute_intervals


def apc(rid, player_id, side, position="MC", time=90, roster_in="0",
        roster_out="0", red_card=False):
    return Appearance(
        roster_id=rid, player_id=player_id, player=f"P{player_id}", side=side,
        position=position, time=time, roster_in=roster_in,
        roster_out=roster_out, red_card=red_card,
    )


def shot(minute, side, result="MissedShots", xg=0.1, player_id="x"):
    return Shot(minute=minute, side=side, player_id=player_id,
                player=f"P{player_id}", result=result, xg=xg, situation="OpenPlay")


def starters(side, n=3, prefix=""):
    return [apc(f"{side}{prefix}{i}", f"{side}{prefix}{i}", side) for i in range(n)]


def test_no_subs_single_stint():
    apps = starters("h") + starters("a")
    stints = build_stints(apps, [shot(10, "h", "Goal")])
    assert len(stints) == 1
    s = stints[0]
    assert (s.start, s.end, s.duration) == (0, 90, 90)
    assert len(s.h_players) == 3 and len(s.a_players) == 3
    assert (s.h_goals, s.a_goals) == (1, 0)


def test_sub_at_60_splits_and_goal_lands_in_second_stint():
    out = apc("h0", "h0", "h", time=60, roster_in="h9")
    sub = apc("h9", "h9", "h", position="Sub", time=30, roster_out="h0")
    apps = [out, apc("h1", "h1", "h"), sub] + starters("a")
    stints = build_stints(apps, [shot(75, "a", "Goal")])
    assert [(s.start, s.end) for s in stints] == [(0, 60), (60, 90)]
    assert "h0" in stints[0].h_players and "h9" not in stints[0].h_players
    assert "h9" in stints[1].h_players and "h0" not in stints[1].h_players
    assert stints[0].a_goals == 0 and stints[1].a_goals == 1


def test_red_card_removes_player_without_replacement():
    sent_off = apc("h0", "h0", "h", time=30, red_card=True)
    apps = [sent_off, apc("h1", "h1", "h"), apc("h2", "h2", "h")] + starters("a")
    stints = build_stints(apps, [])
    assert [(s.start, s.end) for s in stints] == [(0, 30), (30, 90)]
    assert len(stints[0].h_players) == 3
    assert len(stints[1].h_players) == 2
    assert len(stints[1].a_players) == 3


def test_same_minute_double_sub_single_boundary():
    o1 = apc("h0", "h0", "h", time=46, roster_in="h8")
    o2 = apc("h1", "h1", "h", time=46, roster_in="h9")
    s1 = apc("h8", "h8", "h", position="Sub", time=44, roster_out="h0")
    s2 = apc("h9", "h9", "h", position="Sub", time=44, roster_out="h1")
    apps = [o1, o2, apc("h2", "h2", "h"), s1, s2] + starters("a")
    stints = build_stints(apps, [])
    assert [(s.start, s.end) for s in stints] == [(0, 46), (46, 90)]
    assert stints[1].h_players == frozenset({"h8", "h9", "h2"})


def test_stoppage_time_sub_extends_match_length():
    out = apc("h0", "h0", "h", time=90, roster_in="h9")
    sub = apc("h9", "h9", "h", position="Sub", time=2, roster_out="h0")
    apps = [out, apc("h1", "h1", "h"), sub] + starters("a")
    intervals, match_length = compute_intervals(apps)
    assert match_length == 92
    stints = build_stints(apps, [])
    assert [(s.start, s.end) for s in stints] == [(0, 90), (90, 92)]
    assert "h9" in stints[1].h_players


def test_sub_of_sub_chain():
    a = apc("h0", "h0", "h", time=30, roster_in="h8")
    b = apc("h8", "h8", "h", position="Sub", time=30, roster_in="h9", roster_out="h0")
    c = apc("h9", "h9", "h", position="Sub", time=30, roster_out="h8")
    apps = [a, b, c, apc("h1", "h1", "h")] + starters("a")
    by_rid = {iv.appearance.roster_id: iv for iv in compute_intervals(apps)[0]}
    assert (by_rid["h0"].entry, by_rid["h0"].exit) == (0, 30)
    assert (by_rid["h8"].entry, by_rid["h8"].exit) == (30, 60)
    assert (by_rid["h9"].entry, by_rid["h9"].exit) == (60, 90)
    stints = build_stints(apps, [])
    assert [(s.start, s.end) for s in stints] == [(0, 30), (30, 60), (60, 90)]


def test_own_goal_counts_for_opposing_side():
    apps = starters("h") + starters("a")
    stints = build_stints(apps, [shot(20, "h", "OwnGoal")])
    assert stints[0].h_goals == 0
    assert stints[0].a_goals == 1


def test_stoppage_shot_clamped_into_last_stint():
    out = apc("h0", "h0", "h", time=60, roster_in="h9")
    sub = apc("h9", "h9", "h", position="Sub", time=30, roster_out="h0")
    apps = [out, apc("h1", "h1", "h"), sub] + starters("a")
    stints = build_stints(apps, [shot(93, "h", "Goal", xg=0.7)])
    assert stints[-1].h_goals == 1
    assert stints[-1].h_xg == pytest.approx(0.7)
    assert stints[0].h_goals == 0


def test_boundary_minute_shot_goes_to_new_stint():
    out = apc("h0", "h0", "h", time=60, roster_in="h9")
    sub = apc("h9", "h9", "h", position="Sub", time=30, roster_out="h0")
    apps = [out, apc("h1", "h1", "h"), sub] + starters("a")
    stints = build_stints(apps, [shot(60, "h", "Goal")])
    assert stints[0].h_goals == 0
    assert stints[1].h_goals == 1


def test_player_minutes_reconcile_with_interval_lengths():
    out = apc("h0", "h0", "h", time=25, roster_in="h9")
    sub = apc("h9", "h9", "h", position="Sub", time=65, roster_out="h0")
    apps = [out, apc("h1", "h1", "h"), sub] + starters("a")
    stints = build_stints(apps, [])
    for pid, expected in [("h0", 25), ("h9", 65), ("h1", 90), ("a0", 90)]:
        played = sum(s.duration for s in stints if pid in s.h_players | s.a_players)
        assert played == expected, pid


def test_dangling_roster_out_raises():
    sub = apc("h9", "h9", "h", position="Sub", time=30, roster_out="missing")
    with pytest.raises(StintError):
        compute_intervals(starters("h") + [sub] + starters("a"))
