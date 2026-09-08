"""Client for MLB's GUMBO live game feed.

``statsapi`` exposes a game's complete state at
``/api/v1.1/game/{game_pk}/feed/live``. Two properties make it usable for
live prediction:

* A pitch appears in the feed with its Statcast measurements attached
  (``pitchData.startSpeed``, ``zone``, ``strikeZoneTop``) and its classified
  type (``details.type.code``). The state needed to predict pitch N+1 is
  therefore published as soon as pitch N is processed.
* The ``?timecode=`` parameter returns the feed exactly as it existed at that
  moment, so a completed game can be replayed as a sequence of live snapshots.
  This makes offline validation of live behaviour possible without waiting for
  a game to start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


STATS_API = "https://statsapi.mlb.com/api/v1"
STATS_API_V11 = "https://statsapi.mlb.com/api/v1.1"


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=16))
    session.headers.update(
        {"User-Agent": "Predicting-Baseball-Pitches/1.0 (live pitch prediction)"}
    )
    return session


@dataclass(frozen=True)
class GumboSnapshot:
    """One observation of a game's live state."""

    game_pk: int
    payload: dict[str, Any]

    @property
    def timestamp(self) -> str | None:
        return self.payload.get("metaData", {}).get("timeStamp")

    @property
    def detailed_state(self) -> str:
        return (
            self.payload.get("gameData", {})
            .get("status", {})
            .get("detailedState", "Unknown")
        )

    @property
    def abstract_state(self) -> str:
        return (
            self.payload.get("gameData", {})
            .get("status", {})
            .get("abstractGameState", "Unknown")
        )

    @property
    def is_live(self) -> bool:
        return self.abstract_state == "Live"

    @property
    def is_final(self) -> bool:
        return self.abstract_state == "Final"

    @property
    def official_date(self) -> str | None:
        return self.payload.get("gameData", {}).get("datetime", {}).get("officialDate")

    @property
    def all_plays(self) -> list[dict[str, Any]]:
        return self.payload.get("liveData", {}).get("plays", {}).get("allPlays", [])

    @property
    def current_play(self) -> dict[str, Any] | None:
        return self.payload.get("liveData", {}).get("plays", {}).get("currentPlay")

    def player(self, player_id: int) -> dict[str, Any]:
        return self.payload.get("gameData", {}).get("players", {}).get(
            f"ID{player_id}", {}
        )

    def birth_year(self, player_id: int) -> int | None:
        birth_date = self.player(player_id).get("birthDate")
        if not birth_date:
            return None
        try:
            return int(str(birth_date)[:4])
        except ValueError:
            return None

    @property
    def team_abbreviations(self) -> tuple[str | None, str | None]:
        """Return ``(home, away)`` abbreviations, which match Savant's codes."""

        teams = self.payload.get("gameData", {}).get("teams", {})
        return (
            teams.get("home", {}).get("abbreviation"),
            teams.get("away", {}).get("abbreviation"),
        )

    @property
    def game_type(self) -> str:
        return self.payload.get("gameData", {}).get("game", {}).get("type", "R")


class GumboClient:
    """Read-only access to the live feed, timestamps, and win probability."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout_seconds: float = 20,
    ) -> None:
        self.session = session or _session()
        self.timeout_seconds = timeout_seconds

    def feed_live(
        self, game_pk: int, timecode: str | None = None
    ) -> GumboSnapshot:
        """Fetch the current feed, or its state at ``timecode``."""

        params = {"timecode": timecode} if timecode else None
        response = self.session.get(
            f"{STATS_API_V11}/game/{game_pk}/feed/live",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return GumboSnapshot(game_pk=game_pk, payload=response.json())

    def timestamps(self, game_pk: int) -> list[str]:
        """Every archived snapshot timecode for a game, oldest first.

        Used to replay a completed game as the sequence of states a live
        client would have observed.
        """

        response = self.session.get(
            f"{STATS_API_V11}/game/{game_pk}/feed/live/timestamps",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return sorted(response.json())

    def win_probability(self, game_pk: int) -> dict[int, float]:
        """Map ``atBatIndex`` to the batting team's win probability.

        Savant's ``bat_win_exp`` varies pitch by pitch because it accounts for
        the count; this endpoint publishes one value per plate appearance. The
        result is therefore an approximation of that feature, not a
        reproduction of it.
        """

        response = self.session.get(
            f"{STATS_API}/game/{game_pk}/winProbability",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        entries = response.json()

        probabilities: dict[int, float] = {}
        for entry in entries:
            at_bat_index = entry.get("atBatIndex")
            if at_bat_index is None:
                continue
            is_top = entry.get("about", {}).get("isTopInning")
            home = entry.get("homeTeamWinProbability")
            away = entry.get("awayTeamWinProbability")
            if home is None or away is None:
                continue
            # The batting team is the away team in the top half.
            batting = away if is_top else home
            probabilities[int(at_bat_index)] = float(batting) / 100.0
        return probabilities

    def schedule(self, game_date: str) -> list[dict[str, Any]]:
        response = self.session.get(
            f"{STATS_API}/schedule",
            params={
                "sportId": 1,
                "date": game_date,
                "hydrate": "probablePitcher,team,linescore",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return [game for day in payload.get("dates", []) for game in day["games"]]
