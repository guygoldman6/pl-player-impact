"""Publication charts for the README, rendered with a consistent style.

Static PNGs on the light surface. Identity is carried by position (sorted
axes, direct labels), not color: ratings are one measure, so charts stay in
a single hue with recessive chrome.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from .config import Config, load_config

log = logging.getLogger(__name__)

# reference palette (dataviz skill): light-mode slots + chrome
BLUE = "#2a78d6"
RED = "#e34948"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def fig_top20(ratings: pd.DataFrame, path, metric: str = "impact90") -> None:
    """Top 20 players: dot with 90% bootstrap interval, sorted, direct-labeled."""
    top = ratings.nlargest(20, metric).sort_values(metric)
    fig, ax = _fig(9, 8)
    y = range(len(top))
    ax.hlines(y, top[f"{metric}_lo"], top[f"{metric}_hi"],
              color=BLUE, alpha=0.35, linewidth=2)
    ax.scatter(top[metric], y, color=BLUE, s=42, zorder=3)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.annotate(f'{r[metric]:+.2f}', (r[f"{metric}_hi"], i),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    labels = [f'{r["player"]}  ·  {r["latest_team"]}' for _, r in top.iterrows()]
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("impact per 90 vs replacement: npxG-RAPM + finishing (90% bootstrap interval)")
    ax.set_title("Premier League player impact — top 20", loc="left",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_naive_vs_rapm(ratings: pd.DataFrame, path) -> None:
    """The correction, visualized: naive on/off rating vs adjusted rating."""
    df = ratings.dropna(subset=["naive_gd90", "rapm_xg"])
    fig, ax = _fig(8, 6.5)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.scatter(df["naive_gd90"], df["rapm_xg"], s=22, color=BLUE, alpha=0.55,
               edgecolors=SURFACE, linewidths=0.5)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.axvline(0, color=BASELINE, linewidth=1)
    # annotate the most extreme disagreements
    df = df.assign(gap=(df["naive_gd90"].rank(pct=True) - df["rapm_xg"].rank(pct=True)).abs())
    for _, r in df.nlargest(5, "gap").iterrows():
        ax.annotate(r["player"], (r["naive_gd90"], r["rapm_xg"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("Naive on/off goal diff per 90 (raw plus-minus)")
    ax.set_ylabel("xG-RAPM per 90 (adjusted)")
    ax.set_title("What adjustment does: naive plus-minus vs RAPM", loc="left",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_rating_vs_minutes(ratings: pd.DataFrame, path) -> None:
    """Shrinkage funnel: estimates tighten as minutes accumulate."""
    fig, ax = _fig(8, 6)
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.scatter(ratings["total_minutes"], ratings["impact90"], s=22, color=BLUE,
               alpha=0.55, edgecolors=SURFACE, linewidths=0.5)
    ax.axhline(0, color=BASELINE, linewidth=1)
    for _, r in ratings.nlargest(3, "impact90").iterrows():
        ax.annotate(r["player"], (r["total_minutes"], r["impact90"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    ax.set_xlabel("Minutes played (three seasons)")
    ax.set_ylabel("Impact per 90 (npxG-RAPM + finishing)")
    ax.set_title("Impact rating vs playing time", loc="left",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_cv_curves(cv: pd.DataFrame, path) -> None:
    """Lambda selection: CV error vs ridge penalty, one panel per response."""
    responses = list(cv["response"].unique())
    fig, axes = plt.subplots(1, len(responses), figsize=(9, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, resp in zip(axes, responses):
        _style(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.8)
        sub = cv[cv["response"] == resp]
        ax.plot(sub["lambda"], sub["cv_mse"], color=BLUE, linewidth=2,
                marker="o", markersize=5)
        best = sub.loc[sub["cv_mse"].idxmin()]
        ax.scatter([best["lambda"]], [best["cv_mse"]], color=RED, s=45, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("ridge λ (log scale)")
        ax.set_title(f"{resp} response", loc="left", fontsize=10, color=INK_2)
    axes[0].set_ylabel("grouped-CV weighted MSE")
    fig.suptitle("Penalty selection by match-grouped cross-validation",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def make_outputs(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_parquet(cfg.processed_dir / "ratings.parquet")
    cv = pd.read_parquet(cfg.processed_dir / "cv_curves.parquet")

    cols = ["player", "latest_team", "position", "total_minutes",
            "impact90", "impact90_lo", "impact90_hi",
            "rapm_xg", "rapm_xg_lo", "rapm_xg_hi", "finishing_per90", "shots",
            "rapm_goals", "rapm_goals_lo", "rapm_goals_hi",
            "naive_gd90", "naive_xgd90"]
    (ratings.sort_values("impact90", ascending=False)[cols]
            .round(4)
            .to_csv(cfg.outputs_dir / "rankings.csv", index=False))

    fig_top20(ratings, cfg.outputs_dir / "top20_xg_rapm.png")
    fig_naive_vs_rapm(ratings, cfg.outputs_dir / "naive_vs_rapm.png")
    fig_rating_vs_minutes(ratings, cfg.outputs_dir / "rating_vs_minutes.png")
    fig_cv_curves(cv, cfg.outputs_dir / "cv_curves.png")
    log.info("outputs written to %s", cfg.outputs_dir)
