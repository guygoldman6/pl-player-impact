"""Load project configuration from config.yaml at the repo root."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    league: str
    seasons: tuple[int, ...]
    footballdata_codes: dict[int, str]
    min_minutes: int
    per90_scale: float
    ridge_lambdas: tuple[float, ...]
    cv_folds: int
    bootstrap_iters: int
    holdout_cutoff: str
    raw_dir: Path
    processed_dir: Path
    outputs_dir: Path


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    path = path or REPO_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        league=raw["league"],
        seasons=tuple(raw["seasons"]),
        footballdata_codes=dict(raw["footballdata_codes"]),
        min_minutes=raw["min_minutes"],
        per90_scale=raw["per90_scale"],
        ridge_lambdas=tuple(raw["ridge_lambdas"]),
        cv_folds=raw["cv_folds"],
        bootstrap_iters=raw["bootstrap_iters"],
        holdout_cutoff=raw["holdout_cutoff"],
        raw_dir=REPO_ROOT / raw["raw_dir"],
        processed_dir=REPO_ROOT / raw["processed_dir"],
        outputs_dir=REPO_ROOT / raw["outputs_dir"],
    )
