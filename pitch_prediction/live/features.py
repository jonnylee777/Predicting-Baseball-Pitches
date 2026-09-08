"""Build KG4 model features from the GUMBO live feed.

The production models are trained on Savant's KG4 feature frame, so live
prediction requires reconstructing that exact frame from a different source.
This module walks a feed snapshot and emits one KG4-shaped row per pitch,
containing only information available *before* that pitch was thrown.

Three feed semantics matter and were each verified against Savant ground truth:

* ``playEvents[].count`` is the count *after* the pitch, while Savant's
  ``balls``/``strikes`` are the count *before* it. The pre-pitch count is
  therefore carried from the preceding event.
* ``pitchData.strikeZoneTop``/``Bottom`` are constant for a batter within a
  game and match Savant's ``sz_top``/``sz_bot``. They are attached to a thrown
  pitch, so a batter's first plate appearance uses the pre-game estimate and
  every later pitch reuses the observed value.
* Savant's ``at_bat_number`` counts plate appearances faced by *this pitcher*,
  not the game's at-bat index, so it is tracked per pitcher.

Three KG4 columns are left missing for the model's imputer:
``if_fielding_alignment`` and ``of_fielding_alignment``, which the feed does
not carry, and ``bat_win_exp``, whose only live source is published once per
plate appearance while Savant's value changes with the count. Supplying that
approximation measured worse than supplying nothing, so ``win_probability`` is
opt-in.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..feature_engineering import KG4_COLUMNS, ROLLING_PITCH_TYPES
from .context import PregameContext
from .gumbo import GumboSnapshot
from .mapping import (
    PLATE_APPEARANCE_EVENTS,
    TranslationWarnings,
    batted_ball_type,
    hit_location,
    launch_speed_angle,
    pitch_description,
    pitch_type_code,
    plate_appearance_event,
)


# Columns the live feed cannot provide. Kept in the frame as missing values so
# the trained pipeline's imputer handles them exactly as it would a Savant row
# with a gap. Combined Random Forest importance is under 0.5%.
UNAVAILABLE_LIVE_COLUMNS = ("if_fielding_alignment", "of_fielding_alignment")

# Metadata carried alongside the features for logging and evaluation. These are
# never passed to the model.
META_COLUMNS = (
    "gumbo_at_bat_index",
    "gumbo_pitch_number",
    "actual_pitch_type",
    "is_pending",
    "sz_source",
)


@dataclass
class LiveFeatureResult:
    features: pd.DataFrame
    meta: pd.DataFrame
    warnings: TranslationWarnings

    @property
    def pending_index(self) -> int | None:
        """Row index of the pitch that has not been thrown yet, if any."""

        pending = self.meta.index[self.meta["is_pending"]]
        return int(pending[-1]) if len(pending) else None


@dataclass
class _GameState:
    """Mutable state advanced as the feed is walked in order."""

    pitcher_pitch_number: int = 0
    pitcher_at_bat_number: int = 0

    previous_pitch: dict[str, Any] | None = None
    previous_at_bat: dict[str, Any] | None = None

    recent_pitch_types: deque[str] = field(default_factory=lambda: deque(maxlen=3))
    observed_zones: dict[int, tuple[float, float]] = field(default_factory=dict)
    career_pa_vs_batter: dict[int, int] = field(default_factory=dict)
    game_pa_by_batter: dict[int, int] = field(default_factory=dict)

    # Set when a play by the target pitcher begins.
    current_at_bat_index: int | None = None


class LiveFeatureBuilder:
    """Translate a GUMBO snapshot into KG4 rows for one pitcher."""

    def __init__(
        self,
        context: PregameContext,
        win_probability: dict[int, float] | None = None,
    ) -> None:
        self.context = context
        self.win_probability = win_probability or {}

    # --------------------------------------------------------------
    # PUBLIC
    # --------------------------------------------------------------

    def build(
        self,
        snapshot: GumboSnapshot,
        *,
        include_pending: bool = True,
    ) -> LiveFeatureResult:
        """Emit one KG4 row per pitch thrown by the pitcher.

        When ``include_pending`` is set and the pitcher is mid-at-bat, a final
        row is appended for the pitch that has not been thrown yet. That row is
        the live prediction target; its ``actual_pitch_type`` is unknown.
        """

        warnings = TranslationWarnings()
        rows: list[dict[str, Any]] = []
        metas: list[dict[str, Any]] = []

        state = _GameState(
            career_pa_vs_batter=dict(self.context.career_pa_vs_batter),
        )
        state.recent_pitch_types.extend(self.context.seed_pitch_types)

        for record in self._walk(snapshot, state, warnings, include_pending):
            rows.append(record[0])
            metas.append(record[1])

        features = pd.DataFrame(rows, columns=list(KG4_COLUMNS))
        meta = pd.DataFrame(metas, columns=list(META_COLUMNS))
        return LiveFeatureResult(
            features=features.reset_index(drop=True),
            meta=meta.reset_index(drop=True),
            warnings=warnings,
        )

    # --------------------------------------------------------------
    # FEED WALK
    # --------------------------------------------------------------

    def _walk(
        self,
        snapshot: GumboSnapshot,
        state: _GameState,
        warnings: TranslationWarnings,
        include_pending: bool,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        pitcher_id = self.context.pitcher_id
        home_team, away_team = snapshot.team_abbreviations
        plays = snapshot.all_plays

        # Game state carried across every play, not just the pitcher's.
        outs = 0
        runners: dict[str, int | None] = {"1B": None, "2B": None, "3B": None}
        away_score = 0
        home_score = 0
        half_inning_key: tuple[int, str] | None = None

        for play in plays:
            about = play.get("about", {})
            inning = int(about.get("inning", 0))
            half = str(about.get("halfInning", "top"))
            key = (inning, half)

            if half_inning_key is not None and key != half_inning_key:
                outs = 0
                runners = {"1B": None, "2B": None, "3B": None}
            half_inning_key = key

            is_target = play.get("matchup", {}).get("pitcher", {}).get("id") == pitcher_id

            if is_target:
                yield from self._walk_play(
                    play=play,
                    snapshot=snapshot,
                    state=state,
                    warnings=warnings,
                    include_pending=include_pending,
                    pre_outs=outs,
                    pre_runners=dict(runners),
                    pre_away_score=away_score,
                    pre_home_score=home_score,
                    home_team=home_team,
                    away_team=away_team,
                )

            # Savant's n_priorpa_thisgame_player_at_bat counts a batter's plate
            # appearances against every pitcher in the game, so this counter is
            # advanced for all plays rather than only the target pitcher's.
            self._record_game_plate_appearance(state, play)

            # Advance to this play's post-state for the next iteration.
            outs = int(play.get("count", {}).get("outs", outs))
            matchup = play.get("matchup", {})
            runners = {
                "1B": _runner_id(matchup.get("postOnFirst")),
                "2B": _runner_id(matchup.get("postOnSecond")),
                "3B": _runner_id(matchup.get("postOnThird")),
            }
            result = play.get("result", {})
            away_score = int(result.get("awayScore", away_score))
            home_score = int(result.get("homeScore", home_score))

    def _walk_play(
        self,
        *,
        play: dict[str, Any],
        snapshot: GumboSnapshot,
        state: _GameState,
        warnings: TranslationWarnings,
        include_pending: bool,
        pre_outs: int,
        pre_runners: dict[str, int | None],
        pre_away_score: int,
        pre_home_score: int,
        home_team: str | None,
        away_team: str | None,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        at_bat_index = int(about.get("atBatIndex", -1))

        # A new plate appearance for this pitcher.
        if state.current_at_bat_index != at_bat_index:
            state.current_at_bat_index = at_bat_index
            state.pitcher_at_bat_number += 1
            self._start_at_bat(state, matchup)

        batter_id = int(matchup.get("batter", {}).get("id", 0))
        is_top = bool(about.get("isTopInning", True))

        pitch_events = [
            event for event in play.get("playEvents", []) if event.get("isPitch")
        ]

        pre_balls, pre_strikes = 0, 0
        for event in pitch_events:
            row, meta = self._build_row(
                snapshot=snapshot,
                state=state,
                warnings=warnings,
                play=play,
                event=event,
                batter_id=batter_id,
                at_bat_index=at_bat_index,
                balls=pre_balls,
                strikes=pre_strikes,
                outs=pre_outs,
                runners=pre_runners,
                away_score=pre_away_score,
                home_score=pre_home_score,
                home_team=home_team,
                away_team=away_team,
                is_top=is_top,
                pending=False,
            )
            yield row, meta

            self._advance_pitch(state, event, batter_id, warnings)
            count = event.get("count", {})
            pre_balls = int(count.get("balls", pre_balls))
            pre_strikes = int(count.get("strikes", pre_strikes))

        # The pitcher is mid-at-bat: the next pitch is the live prediction.
        if include_pending and not about.get("isComplete", True):
            row, meta = self._build_row(
                snapshot=snapshot,
                state=state,
                warnings=warnings,
                play=play,
                event=None,
                batter_id=batter_id,
                at_bat_index=at_bat_index,
                balls=pre_balls,
                strikes=pre_strikes,
                outs=pre_outs,
                runners=pre_runners,
                away_score=pre_away_score,
                home_score=pre_home_score,
                home_team=home_team,
                away_team=away_team,
                is_top=is_top,
                pending=True,
            )
            yield row, meta

        if about.get("isComplete", False):
            self._finish_at_bat(state, play, warnings)

    # --------------------------------------------------------------
    # STATE TRANSITIONS
    # --------------------------------------------------------------

    def _start_at_bat(self, state: _GameState, matchup: dict[str, Any]) -> None:
        batter_id = int(matchup.get("batter", {}).get("id", 0))
        state.game_pa_by_batter.setdefault(batter_id, 0)

    def _advance_pitch(
        self,
        state: _GameState,
        event: dict[str, Any],
        batter_id: int,
        warnings: TranslationWarnings,
    ) -> None:
        """Record a thrown pitch as history for the pitches that follow."""

        details = event.get("details", {})
        pitch_data = event.get("pitchData", {})
        hit_data = event.get("hitData", {})

        pitch_type = (details.get("type") or {}).get("code")
        description = pitch_description(
            (details.get("call") or {}).get("code"), warnings
        )

        state.pitcher_pitch_number += 1
        state.previous_pitch = {
            "release_speed": _as_float(pitch_data.get("startSpeed")),
            "description": description,
            "zone": _as_float(pitch_data.get("zone")),
            "type": pitch_type_code(description),
            "hit_distance_sc": _as_float(hit_data.get("totalDistance")),
            "launch_speed": _as_float(hit_data.get("launchSpeed")),
            "launch_angle": _as_float(hit_data.get("launchAngle")),
            "pitch_type": pitch_type,
        }
        if pitch_type:
            state.recent_pitch_types.append(str(pitch_type))

        # A batter's zone is constant within a game, so the first measurement
        # is reused for every later pitch to that batter.
        top = _as_float(pitch_data.get("strikeZoneTop"))
        bottom = _as_float(pitch_data.get("strikeZoneBottom"))
        if top is not None and bottom is not None and batter_id not in state.observed_zones:
            state.observed_zones[batter_id] = (top, bottom)

    def _finish_at_bat(
        self, state: _GameState, play: dict[str, Any], warnings: TranslationWarnings
    ) -> None:
        """Store this plate appearance's result for the next one to reference."""

        result = play.get("result", {})
        matchup = play.get("matchup", {})
        batter_id = int(matchup.get("batter", {}).get("id", 0))

        # The batted-ball detail comes from the final pitch of the plate
        # appearance, matching how Savant records the at-bat's outcome.
        hit_data: dict[str, Any] = {}
        for event in play.get("playEvents", []):
            if event.get("isPitch") and event.get("hitData"):
                hit_data = event["hitData"]

        speed = _as_float(hit_data.get("launchSpeed"))
        angle = _as_float(hit_data.get("launchAngle"))

        state.previous_at_bat = {
            "events": plate_appearance_event(result.get("eventType"), warnings),
            "hit_location": hit_location(hit_data.get("location"), warnings),
            "bb_type": batted_ball_type(hit_data.get("trajectory"), warnings),
            "launch_speed_angle": launch_speed_angle(speed, angle),
        }

        # Savant derives prior_pa_vs_pitcher_career from distinct at-bat
        # numbers, and an interrupted trip to the plate still receives its own
        # at-bat number, so this counter advances on every completed at-bat.
        # n_priorpa_thisgame_player_at_bat behaves differently and is gated on
        # a real plate-appearance result in _record_game_plate_appearance.
        state.career_pa_vs_batter[batter_id] = (
            state.career_pa_vs_batter.get(batter_id, 0) + 1
        )

    @staticmethod
    def _record_game_plate_appearance(
        state: _GameState, play: dict[str, Any]
    ) -> None:
        if not _completes_plate_appearance(play):
            return
        batter_id = int(play.get("matchup", {}).get("batter", {}).get("id", 0))
        state.game_pa_by_batter[batter_id] = (
            state.game_pa_by_batter.get(batter_id, 0) + 1
        )

    # --------------------------------------------------------------
    # ROW ASSEMBLY
    # --------------------------------------------------------------

    def _build_row(
        self,
        *,
        snapshot: GumboSnapshot,
        state: _GameState,
        warnings: TranslationWarnings,
        play: dict[str, Any],
        event: dict[str, Any] | None,
        batter_id: int,
        at_bat_index: int,
        balls: int,
        strikes: int,
        outs: int,
        runners: dict[str, int | None],
        away_score: int,
        home_score: int,
        home_team: str | None,
        away_team: str | None,
        is_top: bool,
        pending: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        context = self.context
        matchup = play.get("matchup", {})
        about = play.get("about", {})

        pitch_number_of_ab = (
            int(event.get("pitchNumber", 0)) if event is not None else balls + strikes + 1
        )

        sz_top, sz_bot, sz_source = self._strike_zone(state, batter_id)

        bat_score = away_score if is_top else home_score
        fld_score = home_score if is_top else away_score

        previous_pitch = state.previous_pitch or {}
        previous_at_bat = state.previous_at_bat or {}

        at_bat_number = state.pitcher_at_bat_number
        row: dict[str, Any] = {
            "game_date": context.game_date.isoformat(),
            "pitch_number_of_ab": pitch_number_of_ab,
            "at_bat_number_of_game": at_bat_number,
            "events_of_prev_ab": previous_at_bat.get("events"),
            "hit_location_of_prev_ab": previous_at_bat.get("hit_location"),
            "bb_type_of_prev_ab": previous_at_bat.get("bb_type"),
            "launch_speed_angle_of_prev_ab": previous_at_bat.get("launch_speed_angle"),
            "inning": int(about.get("inning", 0)),
            "n_thruorder_pitcher": (at_bat_number - 1) // 9 + 1,
            "pitch_type": None,
            "batter": batter_id,
            "pitcher": context.pitcher_id,
            "game_type": snapshot.game_type,
            "stand": (matchup.get("batSide") or {}).get("code"),
            "p_throws": (matchup.get("pitchHand") or {}).get("code"),
            "home_team": home_team,
            "away_team": away_team,
            "balls": balls,
            "strikes": strikes,
            "on_3b": runners.get("3B"),
            "on_2b": runners.get("2B"),
            "on_1b": runners.get("1B"),
            "outs_when_up": outs,
            "inning_topbot": "Top" if is_top else "Bot",
            "sz_top": sz_top,
            "sz_bot": sz_bot,
            "game_pk": snapshot.game_pk,
            "bat_score": bat_score,
            "fld_score": fld_score,
            "if_fielding_alignment": None,
            "of_fielding_alignment": None,
            "bat_win_exp": self.win_probability.get(at_bat_index),
            "age_pit": self._age(snapshot, context.pitcher_id),
            "age_bat": self._age(snapshot, batter_id),
            "n_priorpa_thisgame_player_at_bat": state.game_pa_by_batter.get(
                batter_id, 0
            ),
            "pitcher_days_since_prev_game": context.pitcher_days_since_prev_game,
            "release_speed_of_prev_pitch": previous_pitch.get("release_speed"),
            "description_of_prev_pitch": previous_pitch.get("description"),
            "zone_of_prev_pitch": previous_pitch.get("zone"),
            "type_of_prev_pitch": previous_pitch.get("type"),
            "hit_distance_sc_of_prev_pitch": previous_pitch.get("hit_distance_sc"),
            "launch_speed_of_prev_pitch": previous_pitch.get("launch_speed"),
            "launch_angle_of_prev_pitch": previous_pitch.get("launch_angle"),
            "pitch_type_of_prev_pitch": previous_pitch.get("pitch_type"),
            "count": f"{balls}-{strikes}",
            "count_state": _count_state(balls, strikes),
            "pitch_number_of_game": state.pitcher_pitch_number + 1,
            "prior_pa_vs_pitcher_career": state.career_pa_vs_batter.get(batter_id, 0),
            "pitcher_team_score_diff": fld_score - bat_score,
            "strike_zone_height": (
                sz_top - sz_bot if sz_top is not None and sz_bot is not None else None
            ),
        }
        row.update(self._rolling_rates(state))

        actual = None
        if event is not None:
            actual = ((event.get("details", {}).get("type")) or {}).get("code")

        meta = {
            "gumbo_at_bat_index": at_bat_index,
            "gumbo_pitch_number": pitch_number_of_ab,
            "actual_pitch_type": actual,
            "is_pending": pending,
            "sz_source": sz_source,
        }
        return row, meta

    def _strike_zone(
        self, state: _GameState, batter_id: int
    ) -> tuple[float | None, float | None, str]:
        """Resolve the batter's strike zone, preferring in-game measurement."""

        observed = state.observed_zones.get(batter_id)
        if observed is not None:
            return observed[0], observed[1], "observed_in_game"
        top, bottom = self.context.batter_zone(batter_id)
        return top, bottom, self.context.batter_zone_source(batter_id)

    def _rolling_rates(self, state: _GameState) -> dict[str, float | None]:
        """prev3 pitch-type rates over the pitcher's last three pitches."""

        recent = list(state.recent_pitch_types)
        if not recent:
            return {f"prev3_pitch_rate_{name}": None for name in ROLLING_PITCH_TYPES}

        known = set(ROLLING_PITCH_TYPES[:-1])
        normalized = [value if value in known else "OTHER" for value in recent]
        total = len(normalized)
        return {
            f"prev3_pitch_rate_{name}": normalized.count(name) / total
            for name in ROLLING_PITCH_TYPES
        }

    def _age(self, snapshot: GumboSnapshot, player_id: int) -> int | None:
        """Savant reports age as of the season, i.e. season minus birth year."""

        birth_year = snapshot.birth_year(player_id)
        if birth_year is None:
            return None
        return self.context.season - birth_year


def _completes_plate_appearance(play: dict[str, Any]) -> bool:
    """Whether a play ended a plate appearance rather than a base-running out."""

    if not play.get("about", {}).get("isComplete", False):
        return False
    event_type = play.get("result", {}).get("eventType")
    return event_type in PLATE_APPEARANCE_EVENTS


def _runner_id(entry: Any) -> int | None:
    if isinstance(entry, dict) and "id" in entry:
        return int(entry["id"])
    return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _count_state(balls: int, strikes: int) -> str:
    if balls == 3 and strikes == 2:
        return "full_count"
    if strikes == 2 and balls < 3:
        return "put_away_count"
    if balls >= 2 and strikes <= 1:
        return "hitters_count"
    if strikes > balls:
        return "pitcher_ahead"
    if balls > strikes:
        return "pitcher_behind"
    return "even_count"
