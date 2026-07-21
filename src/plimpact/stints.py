"""Segment a match into stints: maximal intervals with an unchanged set of players
AND an unchanged score.

Boundaries occur at substitutions, red cards, and goals. Each stint records who
was on the pitch for each side, the score when the stint began (for game-state
controls), and the goals/xG generated inside the interval. This is the core data
structure for every plus-minus model downstream.

Minute conventions (understat gives integer minutes, so sub-vs-shot ordering
within a minute is ambiguous; these conventions are applied uniformly and noted
in the writeup):
- Within a minute, goals come before roster changes (goal -> reaction sub), so a
  goal at a roster-boundary minute belongs to the stint *ending* there.
- Non-goal shots at a roster-boundary minute belong to the stint *starting* there.
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
    h_npxg: float  # xG excluding penalties
    a_npxg: float
    score_h: int  # score when the stint began
    score_a: int

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


def _roster_segments(intervals: list[Interval], match_length: int) -> list[tuple[int, int]]:
    boundaries = sorted(
        {0, match_length}
        | {iv.entry for iv in intervals}
        | {iv.exit for iv in intervals}
    )
    boundaries = [b for b in boundaries if 0 <= b <= match_length]
    return [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]


def build_stints(appearances: list[Appearance], shots: list[Shot]) -> list[Stint]:
    intervals, match_length = compute_intervals(appearances)
    segments = _roster_segments(intervals, match_length)

    # assign each shot to a roster segment (goals pre-boundary, others post)
    goals_in_seg: dict[int, list[Shot]] = {i: [] for i in range(len(segments))}
    others_in_seg: dict[int, list[Shot]] = {i: [] for i in range(len(segments))}
    for s in shots:
        if goal_side(s) is not None:
            m = min(s.minute, match_length)
            idx = next(
                (i for i, (a, b) in enumerate(segments) if a <= m <= b and (m > a or a == 0)),
                len(segments) - 1,
            )
            goals_in_seg[idx].append(s)
        else:
            m = min(s.minute, match_length - 1)
            idx = next(
                (i for i, (a, b) in enumerate(segments) if a <= m < b),
                len(segments) - 1,
            )
            others_in_seg[idx].append(s)

    stints: list[Stint] = []
    score = {"h": 0, "a": 0}
    for i, (seg_start, seg_end) in enumerate(segments):
        on_pitch = [
            iv for iv in intervals if iv.entry <= seg_start and iv.exit >= seg_end
        ]
        rosters = {
            side: frozenset(
                iv.appearance.player_id for iv in on_pitch if iv.appearance.side == side
            )
            for side in ("h", "a")
        }

        # split this segment at interior goal minutes; a goal belongs to the
        # sub-stint ending at its minute (goals at the segment start, only
        # possible at minute 0, stay in the first sub-stint)
        goals = goals_in_seg[i]
        cuts = sorted({s.minute for s in goals if seg_start < s.minute < seg_end})
        edges = [seg_start, *cuts, seg_end]
        sub_stints = list(zip(edges[:-1], edges[1:]))

        for j, (a, b) in enumerate(sub_stints):
            stint_goals = [
                s for s in goals
                if (a < min(s.minute, match_length) <= b) or (min(s.minute, match_length) == a == 0 and j == 0)
                or (j == len(sub_stints) - 1 and min(s.minute, match_length) > b)
            ]
            stint_others = [
                s for s in others_in_seg[i] if a <= min(s.minute, match_length - 1) < b
            ] if len(sub_stints) > 1 else others_in_seg[i]

            g = {"h": 0, "a": 0}
            xg = {"h": 0.0, "a": 0.0}
            npxg = {"h": 0.0, "a": 0.0}
            for s in stint_goals + stint_others:
                xg[s.side] += s.xg
                if s.situation != "Penalty":
                    npxg[s.side] += s.xg
            for s in stint_goals:
                g[goal_side(s)] += 1

            stints.append(
                Stint(
                    start=a, end=b,
                    h_players=rosters["h"], a_players=rosters["a"],
                    h_goals=g["h"], a_goals=g["a"],
                    h_xg=xg["h"], a_xg=xg["a"],
                    h_npxg=npxg["h"], a_npxg=npxg["a"],
                    score_h=score["h"], score_a=score["a"],
                )
            )
            score["h"] += g["h"]
            score["a"] += g["a"]
    return stints
