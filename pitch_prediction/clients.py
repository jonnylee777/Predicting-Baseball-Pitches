"""HTTP clients for MLB probable starters and Baseball Savant CSV exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MLB_STATS_API = "https://statsapi.mlb.com/api/v1"
SAVANT_EXPORT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {"User-Agent": "Predicting-Baseball-Pitches/1.0 (daily research pipeline)"}
    )
    return session


@dataclass(frozen=True)
class Starter:
    game_pk: int
    official_date: str
    game_time_utc: str
    game_status: str
    side: str
    team_id: int
    team_name: str
    opponent_id: int
    opponent_name: str
    pitcher_id: int
    pitcher_name: str


@dataclass(frozen=True)
class MissingProbable:
    game_pk: int
    official_date: str
    side: str
    team_id: int
    team_name: str


@dataclass(frozen=True)
class ScheduleResult:
    game_count: int
    starters: list[Starter]
    missing_probables: list[MissingProbable]


class MlbStatsClient:
    """Small wrapper around MLB's public schedule and people endpoints."""

    def __init__(
        self, session: requests.Session | None = None, timeout_seconds: float = 30
    ) -> None:
        self.session = session or _session()
        self.timeout_seconds = timeout_seconds

    def probable_starters(self, game_date: date) -> ScheduleResult:
        response = self.session.get(
            f"{MLB_STATS_API}/schedule",
            params={
                "sportId": 1,
                "date": game_date.isoformat(),
                "hydrate": "probablePitcher,team",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        starters: list[Starter] = []
        missing: list[MissingProbable] = []
        games = [game for day in payload.get("dates", []) for game in day["games"]]

        for game in games:
            for side, opponent_side in (("away", "home"), ("home", "away")):
                appearance = game["teams"][side]
                team = appearance["team"]
                opponent = game["teams"][opponent_side]["team"]
                pitcher = appearance.get("probablePitcher")
                if not pitcher:
                    missing.append(
                        MissingProbable(
                            game_pk=int(game["gamePk"]),
                            official_date=game["officialDate"],
                            side=side,
                            team_id=int(team["id"]),
                            team_name=team["name"],
                        )
                    )
                    continue

                starters.append(
                    Starter(
                        game_pk=int(game["gamePk"]),
                        official_date=game["officialDate"],
                        game_time_utc=game["gameDate"],
                        game_status=game["status"]["detailedState"],
                        side=side,
                        team_id=int(team["id"]),
                        team_name=team["name"],
                        opponent_id=int(opponent["id"]),
                        opponent_name=opponent["name"],
                        pitcher_id=int(pitcher["id"]),
                        pitcher_name=pitcher["fullName"],
                    )
                )

        return ScheduleResult(len(games), starters, missing)

    def mlb_debut_date(self, player_id: int) -> date:
        response = self.session.get(
            f"{MLB_STATS_API}/people/{player_id}", timeout=self.timeout_seconds
        )
        response.raise_for_status()
        person = response.json()["people"][0]
        debut = person.get("mlbDebutDate")
        if not debut:
            raise ValueError(f"MLB did not return a debut date for player {player_id}")
        return date.fromisoformat(debut)


class BaseballSavantClient:
    """Download the same pitch-detail CSV exposed by Savant's Graphs view."""

    def __init__(
        self, session: requests.Session | None = None, timeout_seconds: float = 120
    ) -> None:
        self.session = session or _session()
        self.timeout_seconds = timeout_seconds

    def pitcher_pitches(
        self, player_id: int, start_date: date, end_date: date
    ) -> pd.DataFrame:
        if end_date < start_date:
            raise ValueError("end_date must not be earlier than start_date")

        # These parameters mirror the regular-season pitch-detail export used
        # for the Kevin Gausman source CSV in this repository.
        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfBBT": "",
            "hfPR": "",
            "hfZ": "",
            "stadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfGT": "R|",
            "hfSea": "",
            "hfSit": "",
            "player_type": "pitcher",
            "hfOuts": "",
            "opponent": "",
            "pitcher_throws": "",
            "batter_stands": "",
            "hfSA": "",
            "game_date_gt": start_date.isoformat(),
            "game_date_lt": end_date.isoformat(),
            "pitchers_lookup[]": str(player_id),
            "team": "",
            "position": "",
            "hfRO": "",
            "home_road": "",
            "hfFlag": "",
            "metric_1": "",
            "hfInn": "",
            "min_pitches": "0",
            "min_results": "0",
            "group_by": "name",
            "sort_col": "pitches",
            "player_event_sort": "h_launch_speed",
            "sort_order": "desc",
            "min_abs": "0",
            "type": "details",
        }
        response = self.session.get(
            SAVANT_EXPORT_URL, params=params, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        text = response.content.decode("utf-8-sig")
        if text.lstrip().lower().startswith("<!doctype html"):
            raise RuntimeError("Baseball Savant returned HTML instead of a CSV export")
        try:
            frame = pd.read_csv(StringIO(text), low_memory=False)
        except pd.errors.EmptyDataError as exc:
            raise RuntimeError("Baseball Savant returned an empty response") from exc
        frame.columns = [str(column).strip().strip('"') for column in frame.columns]
        return frame
