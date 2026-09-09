"""Tests for the GUMBO live feature builder.

The fixture below is a hand-built feed that exercises the semantics verified
against Savant ground truth: post-pitch counts, the batter strike zone being
constant within a game, and a play that completes without completing a plate
appearance.
"""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from pitch_prediction.feature_engineering import KG4_COLUMNS
from pitch_prediction.live.context import LEAGUE_SZ_BOT, LEAGUE_SZ_TOP, PregameContext
from pitch_prediction.live.features import (
    NUMERIC_COLUMNS,
    TEXT_COLUMNS,
    LiveFeatureBuilder,
)
from pitch_prediction.live.gumbo import GumboSnapshot


PITCHER = 100001
BATTER_A = 200001
BATTER_B = 200002

GAME_PK = 999999


def _pitch(
    number: int,
    *,
    call: str,
    pitch_type: str,
    speed: float,
    zone: int,
    balls: int,
    strikes: int,
    sz_top: float = 3.4,
    sz_bot: float = 1.6,
    hit_data: dict | None = None,
) -> dict:
    """A pitch event. ``balls``/``strikes`` are the count AFTER the pitch."""

    event = {
        "isPitch": True,
        "pitchNumber": number,
        "details": {
            "call": {"code": call},
            "type": {"code": pitch_type},
        },
        "count": {"balls": balls, "strikes": strikes, "outs": 0},
        "pitchData": {
            "startSpeed": speed,
            "zone": zone,
            "strikeZoneTop": sz_top,
            "strikeZoneBottom": sz_bot,
        },
        "startTime": f"2026-08-20T18:0{number}:00.000Z",
        "endTime": f"2026-08-20T18:0{number}:03.000Z",
    }
    if hit_data:
        event["hitData"] = hit_data
    return event


def _play(
    at_bat_index: int,
    *,
    batter: int,
    events: list[dict],
    event_type: str | None,
    complete: bool = True,
    outs: int = 0,
    inning: int = 1,
) -> dict:
    return {
        "about": {
            "atBatIndex": at_bat_index,
            "inning": inning,
            "halfInning": "top",
            "isTopInning": True,
            "isComplete": complete,
        },
        "count": {"balls": 0, "strikes": 0, "outs": outs},
        "matchup": {
            "batter": {"id": batter, "fullName": f"Batter {batter}"},
            "batSide": {"code": "R"},
            "pitcher": {"id": PITCHER, "fullName": "Test Pitcher"},
            "pitchHand": {"code": "R"},
        },
        "result": {"eventType": event_type, "awayScore": 0, "homeScore": 0},
        "playEvents": events,
    }


def _snapshot(plays: list[dict]) -> GumboSnapshot:
    return GumboSnapshot(
        game_pk=GAME_PK,
        payload={
            "metaData": {"timeStamp": "20260820_180500"},
            "gameData": {
                "status": {"detailedState": "In Progress", "abstractGameState": "Live"},
                "datetime": {"officialDate": "2026-08-20"},
                "game": {"type": "R"},
                "teams": {
                    "home": {"abbreviation": "BAL"},
                    "away": {"abbreviation": "NYY"},
                },
                "players": {
                    f"ID{PITCHER}": {"fullName": "Test Pitcher", "birthDate": "1996-09-12"},
                    f"ID{BATTER_A}": {"fullName": "Batter A", "birthDate": "1999-02-22"},
                    f"ID{BATTER_B}": {"fullName": "Batter B", "birthDate": "2000-05-16"},
                },
            },
            "liveData": {"plays": {"allPlays": plays, "currentPlay": plays[-1]}},
        },
    )


def _context() -> PregameContext:
    return PregameContext(
        pitcher_id=PITCHER,
        season=2026,
        game_date=date(2026, 8, 20),
        pitcher_days_since_prev_game=5.0,
    )


class LiveFeatureBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plays = [
            _play(
                0,
                batter=BATTER_A,
                event_type="field_out",
                outs=1,
                events=[
                    _pitch(1, call="C", pitch_type="FF", speed=95.0, zone=5,
                           balls=0, strikes=1),
                    _pitch(2, call="X", pitch_type="SL", speed=85.0, zone=8,
                           balls=0, strikes=1,
                           hit_data={"launchSpeed": 90.0, "launchAngle": 10.0,
                                     "totalDistance": 150.0,
                                     "trajectory": "line_drive", "location": "6"}),
                ],
            ),
            _play(
                1,
                batter=BATTER_B,
                event_type=None,
                complete=False,
                outs=1,
                events=[
                    _pitch(1, call="B", pitch_type="CH", speed=84.0, zone=13,
                           balls=1, strikes=0, sz_top=3.2, sz_bot=1.5),
                ],
            ),
        ]
        self.result = LiveFeatureBuilder(_context()).build(_snapshot(self.plays))
        self.features = self.result.features

    def test_emits_kg4_schema_in_order(self) -> None:
        self.assertEqual(tuple(self.features.columns), KG4_COLUMNS)

    def test_row_per_thrown_pitch_plus_one_pending(self) -> None:
        # Three pitches thrown, and the pitcher is mid-at-bat, so a fourth row
        # is the prediction target for the pitch not yet thrown.
        self.assertEqual(len(self.features), 4)
        self.assertEqual(list(self.result.meta["is_pending"]), [False, False, False, True])
        self.assertEqual(self.result.pending_index, 3)

    def test_count_is_the_state_before_each_pitch(self) -> None:
        # The feed reports the count after a pitch; Savant reports it before.
        self.assertEqual(list(self.features["balls"]), [0, 0, 0, 1])
        self.assertEqual(list(self.features["strikes"]), [0, 1, 0, 0])
        self.assertEqual(list(self.features["count"]), ["0-0", "0-1", "0-0", "1-0"])
        self.assertEqual(
            list(self.features["count_state"]),
            ["even_count", "pitcher_ahead", "even_count", "pitcher_behind"],
        )

    def test_previous_pitch_features_come_from_the_prior_pitch(self) -> None:
        rows = self.features
        self.assertTrue(pd.isna(rows.loc[0, "pitch_type_of_prev_pitch"]))
        self.assertTrue(pd.isna(rows.loc[0, "release_speed_of_prev_pitch"]))

        self.assertEqual(rows.loc[1, "pitch_type_of_prev_pitch"], "FF")
        self.assertEqual(rows.loc[1, "release_speed_of_prev_pitch"], 95.0)
        self.assertEqual(rows.loc[1, "description_of_prev_pitch"], "called_strike")
        self.assertEqual(rows.loc[1, "type_of_prev_pitch"], "S")
        self.assertEqual(rows.loc[1, "zone_of_prev_pitch"], 5.0)

        # The ball in play carries its batted-ball measurements forward.
        self.assertEqual(rows.loc[2, "pitch_type_of_prev_pitch"], "SL")
        self.assertEqual(rows.loc[2, "description_of_prev_pitch"], "hit_into_play")
        self.assertEqual(rows.loc[2, "launch_speed_of_prev_pitch"], 90.0)
        self.assertEqual(rows.loc[2, "hit_distance_sc_of_prev_pitch"], 150.0)

    def test_previous_at_bat_result_is_carried_into_the_next(self) -> None:
        rows = self.features
        self.assertTrue(pd.isna(rows.loc[0, "events_of_prev_ab"]))
        self.assertTrue(pd.isna(rows.loc[1, "events_of_prev_ab"]))
        self.assertEqual(rows.loc[2, "events_of_prev_ab"], "field_out")
        self.assertEqual(rows.loc[2, "bb_type_of_prev_ab"], "line_drive")
        self.assertEqual(rows.loc[2, "hit_location_of_prev_ab"], 6.0)
        self.assertIsNotNone(rows.loc[2, "launch_speed_angle_of_prev_ab"])

    def test_strike_zone_is_reused_once_observed_for_a_batter(self) -> None:
        # A batter's zone is constant within a game, so only their first pitch
        # needs an estimate.
        sources = list(self.result.meta["sz_source"])
        self.assertEqual(sources[0], "league_average")
        self.assertEqual(sources[1], "observed_in_game")
        self.assertEqual(self.features.loc[0, "sz_top"], LEAGUE_SZ_TOP)
        self.assertEqual(self.features.loc[1, "sz_top"], 3.4)
        self.assertEqual(self.features.loc[1, "sz_bot"], 1.6)
        self.assertAlmostEqual(self.features.loc[1, "strike_zone_height"], 1.8)
        # A different batter gets their own zone.
        self.assertEqual(self.features.loc[3, "sz_top"], 3.2)

    def test_counters_are_pitcher_relative(self) -> None:
        rows = self.features
        self.assertEqual(list(rows["at_bat_number_of_game"]), [1, 1, 2, 2])
        self.assertEqual(list(rows["pitch_number_of_game"]), [1, 2, 3, 4])
        self.assertEqual(list(rows["pitch_number_of_ab"]), [1, 2, 1, 2])
        self.assertEqual(list(rows["n_thruorder_pitcher"]), [1, 1, 1, 1])

    def test_ages_use_savant_season_minus_birth_year_convention(self) -> None:
        self.assertEqual(list(self.features["age_pit"].unique()), [30])
        self.assertEqual(self.features.loc[0, "age_bat"], 27)

    def test_numeric_columns_are_numeric_even_when_wholly_missing(self) -> None:
        """A column of Python ``None`` would reach the model as unfilled NaN.

        The model pipeline's imputer only treats ``numpy.nan`` as missing, so a
        column built from ``None`` -- no runner on third, or bat_win_exp with
        no live source -- must still be a numeric dtype or the Random Forest
        rejects the row.
        """

        for column in NUMERIC_COLUMNS:
            self.assertNotEqual(
                self.features[column].dtype, object, f"{column} is object dtype"
            )
            self.assertFalse(
                self.features[column].map(lambda value: value is None).any(),
                f"{column} still contains Python None",
            )

    def test_text_and_numeric_columns_together_cover_the_schema(self) -> None:
        self.assertEqual(
            set(TEXT_COLUMNS) | set(NUMERIC_COLUMNS), set(KG4_COLUMNS)
        )
        self.assertFalse(set(TEXT_COLUMNS) & set(NUMERIC_COLUMNS))

    def test_columns_absent_from_the_feed_are_left_missing(self) -> None:
        for column in ("if_fielding_alignment", "of_fielding_alignment", "bat_win_exp"):
            self.assertTrue(self.features[column].isna().all(), column)

    def test_rolling_rates_use_the_last_three_pitches(self) -> None:
        rows = self.features
        self.assertTrue(pd.isna(rows.loc[0, "prev3_pitch_rate_FF"]))
        self.assertEqual(rows.loc[1, "prev3_pitch_rate_FF"], 1.0)
        self.assertAlmostEqual(rows.loc[2, "prev3_pitch_rate_FF"], 0.5)
        self.assertAlmostEqual(rows.loc[2, "prev3_pitch_rate_SL"], 0.5)
        self.assertAlmostEqual(rows.loc[3, "prev3_pitch_rate_CH"], 1 / 3)

    def test_pregame_seed_primes_the_rolling_rates(self) -> None:
        context = _context()
        context.seed_pitch_types = ("SI", "SI", "SI")
        seeded = LiveFeatureBuilder(context).build(_snapshot(self.plays))
        # The first pitch of the game now has history from the prior start.
        self.assertEqual(seeded.features.loc[0, "prev3_pitch_rate_SI"], 1.0)

    def test_unknown_pitch_type_falls_into_the_other_bucket(self) -> None:
        plays = [
            _play(
                0,
                batter=BATTER_A,
                event_type=None,
                complete=False,
                events=[
                    _pitch(1, call="B", pitch_type="XX", speed=80.0, zone=1,
                           balls=1, strikes=0)
                ],
            )
        ]
        result = LiveFeatureBuilder(_context()).build(_snapshot(plays))
        self.assertEqual(result.features.loc[1, "prev3_pitch_rate_OTHER"], 1.0)


class InterruptedPlateAppearanceTests(unittest.TestCase):
    """A play can end without ending a plate appearance.

    When a runner is caught stealing for the final out, the batter leads off
    the next inning with a fresh count. Savant still assigns the interrupted
    trip its own at-bat number, so ``prior_pa_vs_pitcher_career`` counts it,
    but ``n_priorpa_thisgame_player_at_bat`` does not.
    """

    def setUp(self) -> None:
        self.plays = [
            _play(
                0,
                batter=BATTER_A,
                event_type="caught_stealing_2b",
                outs=3,
                events=[
                    _pitch(1, call="B", pitch_type="FF", speed=95.0, zone=1,
                           balls=1, strikes=0)
                ],
            ),
            _play(
                1,
                batter=BATTER_A,
                event_type=None,
                complete=False,
                inning=2,
                events=[
                    _pitch(1, call="C", pitch_type="SL", speed=86.0, zone=5,
                           balls=0, strikes=1)
                ],
            ),
        ]
        self.result = LiveFeatureBuilder(_context()).build(_snapshot(self.plays))

    def test_career_matchup_counter_includes_the_interrupted_trip(self) -> None:
        rows = self.result.features
        self.assertEqual(rows.loc[0, "prior_pa_vs_pitcher_career"], 0)
        self.assertEqual(rows.loc[1, "prior_pa_vs_pitcher_career"], 1)

    def test_in_game_plate_appearance_counter_excludes_it(self) -> None:
        rows = self.result.features
        self.assertEqual(rows.loc[0, "n_priorpa_thisgame_player_at_bat"], 0)
        self.assertEqual(rows.loc[1, "n_priorpa_thisgame_player_at_bat"], 0)

    def test_count_resets_for_the_resumed_plate_appearance(self) -> None:
        self.assertEqual(self.result.features.loc[1, "count"], "0-0")


if __name__ == "__main__":
    unittest.main()
