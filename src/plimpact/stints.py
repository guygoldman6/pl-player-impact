"""Segment a match into stints: maximal intervals with an unchanged set of players.

Boundaries occur at substitutions and red cards. Each stint records who was on
the pitch for each side plus the goals and xG generated inside the interval.
This is the core data structure for every plus-minus model downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parse import Appearance, Shot


@dataclass(frozen=True)
class Interval:
    """A player's time on the pitch, [entry, exit) on understat's match clock."""

    appearance: Appearance
    entry: int
    exit: int


@dataclass(frozen=True)
class Stint:
    start: int
    end: int
    h_players: frozenset[str]  # player_ids on pitch for the home side
    a_players: frozenset[str]
    h_goals: int
    a_goals: int
    h_xg: float
    a_xg: float

    @property
    def duration(self) -> int:
        return self.end - self.start


class StintError(ValueError):
    """Raised when a match's roster data cannot be consistently segmented."""


def compute_intervals(appearances: list[Appearance]) -> tuple[list[Interval], int]:
    """Resolve entry/exit minutes for every appearance; return intervals + match length.

    Entry minutes chain through substitutions: a sub enters at the exit minute
    of the player referenced by his ``roster_out``. A player exits early when he
    was replaced (``roster_in != "0"``) or sent off; otherwise he plays to the end.
    """
    by_roster_id = {a.roster_id: a for a in appearances}
    entries: dict[str, int] = {
        a.roster_id: 0 for a in appearances if a.position != "Sub"
    }

    def exits_early(a: Appearance) -> bool:
        return a.roster_in != "0" or a.red_card

    # resolve sub entry minutes by chaining through roster_out references
    unresolved = [a for a in appearances if a.position == "Sub"]
    while unresolved:
        progressed = False
        remaining = []
        for a in unresolved:
            replaced = by_roster_id.get(a.roster_out)
            if replaced is None:
                raise StintError(
                    f"sub {a.player} has roster_out={a.roster_out} not in roster"
                )
            if replaced.roster_id in entries:
                entries[a.roster_id] = entries[replaced.roster_id] + replaced.time
                progressed = True
            else:
                remaining.append(a)
        if not progressed:
            raise StintError("circular substitution chain")
        unresolved = remaining

    # every player's implied exit is entry + time; the match runs at least to
    # the latest of these (stoppage-time subs push it past 90)
    match_length = max([90, *(entries[a.roster_id] + a.time for a in appearances)])

    intervals = [
        Interval(
            appearance=a,
            entry=entries[a.roster_id],
            exit=(entries[a.roster_id] + a.time) if exits_early(a) else match_length,
        )
        for a in appearances
    ]
    return intervals, match_length


def goal_side(shot: Shot) -> str | None:
    """Which side a shot scored for, or None if it did not score."""
    if shot.result == "Goal":
        return shot.side
    if shot.result == "OwnGoal":
        return "a" if shot.side == "h" else "h"
    return None


def build_stints(appearances: list[Appearance], shots: list[Shot]) -> list[Stint]:
    intervals, match_length = compute_intervals(appearances)

    boundaries = sorted(
        {0, match_length}
        | {iv.entry for iv in intervals}
        | {iv.exit for iv in intervals}
    )
    boundaries = [b for b in boundaries if 0 <= b <= match_length]

    stints = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        on_pitch = [iv for iv in intervals if iv.entry <= start and iv.exit >= end]
        # a shot at minute m belongs to the stint starting at m when m is a
        # boundary (integer minutes make sub-vs-shot ordering ambiguous; this
        # convention is applied uniformly and noted in the writeup)
        in_stint = [
            s for s in shots if start <= min(s.minute, match_length - 1) < end
        ]
        goals = {"h": 0, "a": 0}
        xg = {"h": 0.0, "a": 0.0}
        for s in in_stint:
            xg[s.side] += s.xg
            scored_for = goal_side(s)
            if scored_for is not None:
                goals[scored_for] += 1
        stints.append(
            Stint(
                start=start,
                end=end,
                h_players=frozenset(
                    iv.appearance.player_id for iv in on_pitch if iv.appearance.side == "h"
                ),
                a_players=frozenset(
                    iv.appearance.player_id for iv in on_pitch if iv.appearance.side == "a"
                ),
                h_goals=goals["h"],
                a_goals=goals["a"],
                h_xg=xg["h"],
                a_xg=xg["a"],
            )
        )
    return stints
