"""Pitcher-agnostic KG2, KG3, and KG4 feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import StatcastSchema


KG2_DROP_COLUMNS = (
    "vx0_of_prev_pitch",
    "vy0_of_prev_pitch",
    "vz0_of_prev_pitch",
    "ax_of_prev_pitch",
    "ay_of_prev_pitch",
    "az_of_prev_pitch",
    "plate_z_of_prev_pitch",
    "plate_x_of_prev_pitch",
    "pfx_z_of_prev_pitch",
    "pfx_x_of_prev_pitch",
    "api_break_z_with_gravity_of_prev_pitch",
    "api_break_x_arm_of_prev_pitch",
    "api_break_x_batter_in_of_prev_pitch",
    "attack_angle_of_prev_pitch",
    "hyper_speed_of_prev_pitch",
    "attack_direction_of_prev_pitch",
    "swing_length_of_prev_pitch",
    "intercept_ball_minus_batter_pos_x_inches_of_prev_pitch",
    "intercept_ball_minus_batter_pos_y_inches_of_prev_pitch",
    "swing_path_tilt_of_prev_pitch",
    "arm_angle_of_prev_pitch",
    "release_spin_rate_of_prev_pitch",
    "spin_axis_of_prev_pitch",
    "release_extension_of_prev_pitch",
    "release_pos_x_of_prev_pitch",
    "release_pos_y_of_prev_pitch",
    "release_pos_z_of_prev_pitch",
    "sv_id",
    "game_year",
    "tfs_deprecated",
    "tfs_zulu_deprecated",
    "spin_dir",
    "spin_rate_deprecated",
    "break_angle_deprecated",
    "break_length_deprecated",
    "des",
)

KG2_ADDED_COLUMNS = (
    "pitch_type_of_prev_pitch",
    "p/h_hand",
    "count",
    "count_state",
    "pitch_number_of_game",
    "prior_pa_vs_pitcher_career",
    "pitcher_team_score_diff",
    "strike_zone_height",
)

KG3_DROP_COLUMNS = (
    "effective_speed_of_prev_pitch",
    "delta_home_win_exp_of_prev_pitch",
    "home_win_exp",
    "age_bat_legacy",
    "post_home_score",
    "fielder_2",
    "fielder_3",
    "fielder_4",
    "fielder_5",
    "fielder_6",
    "fielder_7",
    "fielder_8",
    "fielder_9",
    "home_score",
    "away_score",
    "home_score_diff",
    "age_pit_legacy",
    "pitcher_days_until_next_game",
    "batter_days_until_next_game",
    "player_name",
    "batter_days_since_prev_game",
    "p/h_hand",
)

# Fixed categories avoid pitcher-dependent schemas. OTHER safely represents a
# future or legacy Savant code without changing the model input columns.
ROLLING_PITCH_TYPES = (
    "CH",
    "CS",
    "CU",
    "EP",
    "FA",
    "FC",
    "FF",
    "FO",
    "FS",
    "IN",
    "KC",
    "KN",
    "PO",
    "SC",
    "SI",
    "SL",
    "ST",
    "SV",
    "OTHER",
)
ROLLING_PITCH_COLUMNS = tuple(
    f"prev3_pitch_rate_{pitch_type}" for pitch_type in ROLLING_PITCH_TYPES
)

KG4_BASE_COLUMNS = (
    "game_date",
    "pitch_number_of_ab",
    "at_bat_number_of_game",
    "events_of_prev_ab",
    "hit_location_of_prev_ab",
    "bb_type_of_prev_ab",
    "launch_speed_angle_of_prev_ab",
    "inning",
    "n_thruorder_pitcher",
    "pitch_type",
    "batter",
    "pitcher",
    "game_type",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "balls",
    "strikes",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning_topbot",
    "sz_top",
    "sz_bot",
    "game_pk",
    "bat_score",
    "fld_score",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "bat_win_exp",
    "age_pit",
    "age_bat",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "release_speed_of_prev_pitch",
    "description_of_prev_pitch",
    "zone_of_prev_pitch",
    "type_of_prev_pitch",
    "hit_distance_sc_of_prev_pitch",
    "launch_speed_of_prev_pitch",
    "launch_angle_of_prev_pitch",
    "pitch_type_of_prev_pitch",
    "count",
    "count_state",
    "pitch_number_of_game",
    "prior_pa_vs_pitcher_career",
    "pitcher_team_score_diff",
    "strike_zone_height",
)
KG4_COLUMNS = (*KG4_BASE_COLUMNS, *ROLLING_PITCH_COLUMNS)


@dataclass(frozen=True)
class FeatureDatasets:
    kg2: pd.DataFrame
    kg3: pd.DataFrame
    kg4: pd.DataFrame


class PitchFeatureEngineer:
    """Apply the Model1 notebook's feature transformations consistently."""

    def __init__(self, cleaned_schema: StatcastSchema) -> None:
        self.cleaned_schema = cleaned_schema
        renamed = tuple(
            "pitch_number_of_ab"
            if column == "pitch_number"
            else "at_bat_number_of_game"
            if column == "at_bat_number"
            else column
            for column in cleaned_schema.columns
        )
        kg2_columns = tuple(
            column for column in renamed if column not in set(KG2_DROP_COLUMNS)
        ) + KG2_ADDED_COLUMNS
        kg3_columns = tuple(
            column for column in kg2_columns if column not in set(KG3_DROP_COLUMNS)
        ) + ROLLING_PITCH_COLUMNS
        self.kg2_schema = StatcastSchema(kg2_columns)
        self.kg3_schema = StatcastSchema(kg3_columns)
        self.kg4_schema = StatcastSchema(KG4_COLUMNS)

    def transform(
        self, cleaned: pd.DataFrame, source: str = "cleaned pitcher data"
    ) -> FeatureDatasets:
        cleaned = self.cleaned_schema.normalize(cleaned, source)
        kg2 = self._make_kg2(cleaned, source)
        kg3 = self._make_kg3(kg2, source)
        kg4 = self.kg4_schema.normalize(
            kg3.loc[:, KG4_COLUMNS].copy(), f"KG4 {source}"
        )
        return FeatureDatasets(kg2=kg2, kg3=kg3, kg4=kg4)

    def _make_kg2(self, cleaned: pd.DataFrame, source: str) -> pd.DataFrame:
        kg2 = cleaned.dropna(subset=["pitch_type"]).copy()
        kg2 = kg2.rename(
            columns={
                "pitch_number": "pitch_number_of_ab",
                "at_bat_number": "at_bat_number_of_game",
            }
        )
        kg2 = kg2.drop(columns=list(KG2_DROP_COLUMNS))
        sort_columns = [
            "game_date",
            "game_pk",
            "at_bat_number_of_game",
            "pitch_number_of_ab",
        ]
        kg2["pitch_type_of_prev_pitch"] = (
            kg2.sort_values(
                ["game_pk", "at_bat_number_of_game", "pitch_number_of_ab"]
            )
            .groupby("game_pk")["pitch_type"]
            .shift(1)
        )
        kg2["p/h_hand"] = (kg2["p_throws"] == kg2["stand"]).astype(int)
        kg2["count"] = (
            kg2["balls"].astype(int).astype(str)
            + "-"
            + kg2["strikes"].astype(int).astype(str)
        )
        kg2["count_state"] = self._count_state(kg2["balls"], kg2["strikes"])
        kg2["pitch_number_of_game"] = kg2.groupby("game_pk").cumcount() + 1
        kg2 = kg2.sort_values(sort_columns).reset_index(drop=True)

        plate_appearances = (
            kg2[
                [
                    "game_date",
                    "game_pk",
                    "at_bat_number_of_game",
                    "pitcher",
                    "batter",
                ]
            ]
            .drop_duplicates(subset=["game_pk", "at_bat_number_of_game"])
            .sort_values(["game_date", "game_pk", "at_bat_number_of_game"])
            .copy()
        )
        plate_appearances["prior_pa_vs_pitcher_career"] = plate_appearances.groupby(
            ["pitcher", "batter"]
        ).cumcount()
        kg2 = kg2.merge(
            plate_appearances[
                [
                    "game_pk",
                    "at_bat_number_of_game",
                    "prior_pa_vs_pitcher_career",
                ]
            ],
            on=["game_pk", "at_bat_number_of_game"],
            how="left",
        )
        kg2["pitcher_team_score_diff"] = kg2["fld_score"] - kg2["bat_score"]
        kg2["strike_zone_height"] = kg2["sz_top"] - kg2["sz_bot"]
        return self.kg2_schema.normalize(kg2, f"KG2 {source}")

    def _make_kg3(self, kg2: pd.DataFrame, source: str) -> pd.DataFrame:
        kg3 = kg2.drop(columns=list(KG3_DROP_COLUMNS)).copy()
        kg3 = kg3.sort_values(
            [
                "game_date",
                "game_pk",
                "at_bat_number_of_game",
                "pitch_number_of_ab",
            ]
        ).reset_index(drop=True)

        if kg3.empty:
            for column in ROLLING_PITCH_COLUMNS:
                kg3[column] = pd.Series(dtype=float)
        else:
            normalized_pitch_type = kg3["pitch_type"].where(
                kg3["pitch_type"].isin(ROLLING_PITCH_TYPES[:-1]), "OTHER"
            )
            categorical_pitch_type = pd.Categorical(
                normalized_pitch_type, categories=ROLLING_PITCH_TYPES
            )
            pitch_dummies = pd.get_dummies(
                categorical_pitch_type,
                prefix="prev3_pitch_rate",
            )
            pitch_dummies.index = kg3.index
            shifted = pitch_dummies.groupby(kg3["pitcher"]).shift(1)
            rolling = (
                shifted.groupby(kg3["pitcher"])
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            kg3 = pd.concat([kg3, rolling], axis=1)
        return self.kg3_schema.normalize(kg3, f"KG3 {source}")

    @staticmethod
    def _count_state(balls: pd.Series, strikes: pd.Series) -> pd.Series:
        conditions = (
            balls.eq(3) & strikes.eq(2),
            strikes.eq(2) & balls.lt(3),
            balls.ge(2) & strikes.le(1),
            strikes.gt(balls),
            balls.gt(strikes),
        )
        choices = (
            "full_count",
            "put_away_count",
            "hitters_count",
            "pitcher_ahead",
            "pitcher_behind",
        )
        return pd.Series(np.select(conditions, choices, default="even_count"), index=balls.index)
