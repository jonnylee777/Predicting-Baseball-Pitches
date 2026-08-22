"""Deterministic, pitcher-agnostic version of the cleaning notebook."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import StatcastSchema


PREVIOUS_PITCH_RESULT_COLUMNS = (
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "description",
    "zone",
    "type",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_y",
    "spin_axis",
    "delta_home_win_exp",
    "delta_run_exp",
    "bat_speed",
    "swing_length",
    "miss_distance",
    "estimated_slg_using_speedangle",
    "delta_pitcher_run_exp",
    "hyper_speed",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    "arm_angle",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
)

PREVIOUS_AT_BAT_RESULT_COLUMNS = (
    "events",
    "hit_location",
    "bb_type",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "launch_speed_angle",
)

REQUIRED_CLEANING_COLUMNS = {
    "pitch_type",
    "pitch_name",
    "game_date",
    "game_pk",
    "pitch_number",
    "at_bat_number",
    "inning",
    "n_thruorder_pitcher",
    *PREVIOUS_PITCH_RESULT_COLUMNS,
    *PREVIOUS_AT_BAT_RESULT_COLUMNS,
}


@dataclass(frozen=True)
class CleaningSummary:
    rows_before: int
    rows_after: int
    rows_dropped_without_pitch_type: int
    column_count: int


class PitchDataCleaner:
    """Transform a canonical Savant pitcher export into the notebook's kg1 form."""

    def __init__(
        self, raw_schema: StatcastSchema, cleaned_schema: StatcastSchema
    ) -> None:
        self.raw_schema = raw_schema
        self.cleaned_schema = cleaned_schema

    def transform(self, raw: pd.DataFrame, source: str = "raw pitcher data") -> pd.DataFrame:
        raw = self.raw_schema.normalize(raw, source)
        missing_required = sorted(REQUIRED_CLEANING_COLUMNS - set(raw.columns))
        if missing_required:
            raise ValueError(
                f"Cleaning input {source} is missing required columns: {missing_required}"
            )

        cleaned = raw.dropna(subset=["pitch_type"]).copy()

        cleaned = cleaned.sort_values(
            ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

        previous_pitch = (
            cleaned.groupby("game_pk")[list(PREVIOUS_PITCH_RESULT_COLUMNS)]
            .shift(1)
            .add_suffix("_of_prev_pitch")
        )
        cleaned = cleaned.drop(columns=list(PREVIOUS_PITCH_RESULT_COLUMNS))
        cleaned = pd.concat([cleaned, previous_pitch], axis=1)

        cleaned["_new_plate_appearance"] = cleaned["pitch_number"].eq(1)
        cleaned["at_bat_number"] = cleaned.groupby("game_pk")[
            "_new_plate_appearance"
        ].cumsum()
        cleaned = cleaned.drop(columns="_new_plate_appearance")

        cleaned["_row_order"] = range(len(cleaned))
        at_bat_results = (
            cleaned.groupby(["game_pk", "at_bat_number"], sort=False)[
                list(PREVIOUS_AT_BAT_RESULT_COLUMNS)
            ]
            .last()
            .reset_index()
        )
        at_bat_results[list(PREVIOUS_AT_BAT_RESULT_COLUMNS)] = at_bat_results.groupby(
            "game_pk", sort=False
        )[list(PREVIOUS_AT_BAT_RESULT_COLUMNS)].shift(1)
        at_bat_results = at_bat_results.rename(
            columns={
                column: f"{column}_of_prev_ab"
                for column in PREVIOUS_AT_BAT_RESULT_COLUMNS
            }
        )

        cleaned = cleaned.merge(
            at_bat_results,
            on=["game_pk", "at_bat_number"],
            how="left",
        )
        cleaned = (
            cleaned.sort_values("_row_order")
            .drop(columns="_row_order")
            .reset_index(drop=True)
        )
        cleaned = cleaned.drop(columns=list(PREVIOUS_AT_BAT_RESULT_COLUMNS))
        cleaned = cleaned.drop(columns="pitch_name")

        return self.cleaned_schema.normalize(cleaned, f"cleaned {source}")

    def summarize(self, raw: pd.DataFrame, cleaned: pd.DataFrame) -> CleaningSummary:
        return CleaningSummary(
            rows_before=len(raw),
            rows_after=len(cleaned),
            rows_dropped_without_pitch_type=int(raw["pitch_type"].isna().sum()),
            column_count=len(cleaned.columns),
        )
