"""Pull raw match data from Understat (rosters + shots) and football-data.co.uk (results).

All responses are cached as JSON/CSV under data/raw/. Re-running skips anything
already on disk, so an interrupted pull resumes where it left off.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import requests
from tqdm import tqdm
from understatapi import UnderstatClient

from .config import Config, load_config

log = logging.getLogger(__name__)

REQUEST_DELAY_S = 1.0
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 30
FOOTBALLDATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def season_dir(cfg: Config, season: int) -> Path:
    return cfg.raw_dir / "understat" / str(season)


def matches_path(cfg: Config, season: int) -> Path:
    return cfg.raw_dir / "understat" / f"matches_{season}.json"


def fetch_match_list(cfg: Config, season: int, refresh: bool = False) -> list[dict]:
    """League match list (teams, date, final score) for one season."""
    path = matches_path(cfg, season)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    with UnderstatClient() as client:
        matches = client.league(league=cfg.league).get_match_data(season=str(season))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matches))
    return matches


def fetch_season(cfg: Config, season: int) -> None:
    """Roster + shot data for every finished match of a season, one JSON per match."""
    matches = fetch_match_list(cfg, season)
    finished = [m for m in matches if m["isResult"]]
    out_dir = season_dir(cfg, season)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [m for m in finished if not (out_dir / f"{m['id']}.json").exists()]
    log.info("season %s: %d matches, %d to fetch", season, len(finished), len(todo))
    if not todo:
        return
    client = UnderstatClient()
    try:
        for m in tqdm(todo, desc=f"understat {season}"):
            mid = m["id"]
            data = _fetch_match_with_retry(client, mid)
            if data is None:
                raise RuntimeError(f"giving up on match {mid} after {MAX_ATTEMPTS} attempts")
            client, payload = data
            (out_dir / f"{mid}.json").write_text(
                json.dumps({"match": m, "roster": payload["rosters"], "shots": payload["shots"]})
            )
            time.sleep(REQUEST_DELAY_S + random.uniform(0, 0.5))
    finally:
        client.session.close()


def _fetch_match_with_retry(
    client: UnderstatClient, match_id: str
) -> tuple[UnderstatClient, dict] | None:
    """Fetch one match's {rosters, shots} payload, reconnecting on dropped connections.

    ``_get_data`` returns rosters + shots in ONE request; the public
    get_roster_data/get_shot_data methods would each refetch the same payload.
    Understat rate-limits by abruptly closing connections, so on failure we wait
    with increasing backoff and recreate the client session.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client, client.match(match=match_id)._get_data()
        except (requests.exceptions.RequestException, ConnectionError) as err:
            if attempt == MAX_ATTEMPTS:
                return None
            wait = BACKOFF_BASE_S * attempt
            log.warning(
                "match %s attempt %d failed (%s); reconnecting in %ds",
                match_id, attempt, type(err).__name__, wait,
            )
            time.sleep(wait)
            try:
                client.session.close()
            except Exception:
                pass
            client = UnderstatClient()
    return None


def fetch_footballdata(cfg: Config, season: int) -> Path:
    """Independent final-score CSV from football-data.co.uk (plain download)."""
    code = cfg.footballdata_codes[season]
    path = cfg.raw_dir / "footballdata" / f"E0_{code}.csv"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(FOOTBALLDATA_URL.format(code=code), timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def fetch_all(cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    for season in cfg.seasons:
        fetch_footballdata(cfg, season)
        fetch_season(cfg, season)
