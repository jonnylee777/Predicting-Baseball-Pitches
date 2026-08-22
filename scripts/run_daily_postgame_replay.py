"""Replay all completed starting pitchers for a day's MLB games.

Default behavior:

    process yesterday's games

Every successful pitcher-game is also written to:

    Data/daily_pipeline/performance_history.csv

The history file is cumulative across days and seasons.

Rerunning a game is safe because rows are uniquely identified by:

    game_pk + pitcher_id
"""

from __future__ import annotations

import argparse
import json

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

from pathlib import Path

import pandas as pd

from pitch_prediction.cleaning import (
    PitchDataCleaner,
)

from pitch_prediction.feature_engineering import (
    PitchFeatureEngineer,
)

from pitch_prediction.model import (
    PitchModelTrainer,
)

from pitch_prediction.performance_history import (
    record_performance,
)

from pitch_prediction.pipeline import (
    DailyStarterPipeline,
)

from pitch_prediction.postgame_replay import (
    PostgameReplayer,
)

from pitch_prediction.schema import (
    StatcastSchema,
)


# ============================================================
# PATHS
# ============================================================

OUTPUT_ROOT = Path(
    "Data/daily_pipeline"
)

PERFORMANCE_HISTORY_PATH = (
    OUTPUT_ROOT
    / "performance_history.csv"
)


# ============================================================
# HELPERS
# ============================================================


def is_final_game(
    game_status: str,
) -> bool:

    normalized = (
        str(
            game_status
        )
        .strip()
        .lower()
    )

    return (
        "final"
        in normalized
        or normalized
        in {
            "game over",
            "completed",
        }
    )


def count_true_values(
    series: pd.Series,
) -> int:

    if pd.api.types.is_bool_dtype(
        series
    ):

        return int(
            series.sum()
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return int(
        normalized
        .eq(
            "true"
        )
        .sum()
    )


def is_statcast_not_ready(
    exc: Exception,
) -> bool:

    message = str(
        exc
    ).lower()

    return (
        "completed game was not found"
        in message
        or
        "savant may not have published"
        in message
    )


def empty_summary_row(
    *,
    game_date: date,
    starter,
    opponent_name: str,
    game_status: str,
    replay_status: str,
    error: str | None = None,
) -> dict:

    return {
        "game_date":
            game_date.isoformat(),

        "game_pk":
            starter.game_pk,

        "pitcher_id":
            starter.pitcher_id,

        "pitcher_name":
            starter.pitcher_name,

        "opponent":
            opponent_name,

        "game_status":
            game_status,

        "replay_status":
            replay_status,

        "pitch_count":
            None,

        "model_correct":
            None,

        "baseline_correct":
            None,

        "model_accuracy":
            None,

        "baseline_accuracy":
            None,

        "accuracy_over_baseline":
            None,

        "accuracy_over_baseline_pp":
            None,

        "relative_improvement":
            None,

        "relative_improvement_percent":
            None,

        "baseline_strategy":
            None,

        "predictions_path":
            None,

        "summary_path":
            None,

        "error":
            error,
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Replay all completed MLB starting "
                "pitchers for one day."
            )
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Game date YYYY-MM-DD. "
            "If omitted, yesterday is used."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # TARGET DATE
    # ========================================================

    if args.date is None:

        game_date = (
            date.today()
            - timedelta(
                days=1
            )
        )

    else:

        game_date = (
            date.fromisoformat(
                args.date
            )
        )

    print()
    print("=" * 75)

    print(
        "DAILY POST-GAME REPLAY"
    )

    print("=" * 75)

    print(
        f"Target game date: "
        f"{game_date}"
    )

    if args.date is None:

        print(
            "(Automatically using yesterday)"
        )

    # ========================================================
    # PROJECT COMPONENTS
    # ========================================================

    print()
    print(
        "Loading project components..."
    )

    raw_schema = (
        StatcastSchema
        .from_file(
            Path(
                "config/"
                "statcast_columns.txt"
            )
        )
    )

    cleaned_schema = (
        StatcastSchema
        .from_file(
            Path(
                "config/"
                "cleaned_columns.txt"
            )
        )
    )

    cleaner = PitchDataCleaner(
        raw_schema,
        cleaned_schema,
    )

    feature_engineer = (
        PitchFeatureEngineer(
            cleaned_schema
        )
    )

    trainer = (
        PitchModelTrainer()
    )

    pipeline = (
        DailyStarterPipeline(
            output_root=(
                OUTPUT_ROOT
            ),

            schema=(
                raw_schema
            ),

            cleaner=(
                cleaner
            ),

            feature_engineer=(
                feature_engineer
            ),

            model_trainer=None,
        )
    )

    replayer = (
        PostgameReplayer(
            pipeline=(
                pipeline
            ),

            model_trainer=(
                trainer
            ),

            output_root=(
                OUTPUT_ROOT
            ),
        )
    )

    # ========================================================
    # STARTERS
    # ========================================================

    print()
    print(
        f"Getting MLB starters for "
        f"{game_date}..."
    )

    schedule = (
        pipeline
        .mlb
        .probable_starters(
            game_date
        )
    )

    starters = []
    seen = set()

    for starter in (
        schedule.starters
    ):

        key = (
            starter.game_pk,
            starter.pitcher_id,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        starters.append(
            starter
        )

    print(
        f"Found "
        f"{len(starters)} "
        f"starting-pitcher records."
    )

    if not starters:

        print(
            "No starters found."
        )

        return

    # ========================================================
    # DAILY OUTPUT DIRECTORY
    # ========================================================

    daily_dir = (
        OUTPUT_ROOT
        / "predictions"
        / "postgame"
        / game_date.isoformat()
    )

    daily_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    # ========================================================
    # PROCESS EACH STARTER
    # ========================================================

    for (
        index,
        starter,
    ) in enumerate(
        starters,
        start=1,
    ):

        print()
        print("-" * 75)

        print(
            f"[{index}/"
            f"{len(starters)}] "
            f"{starter.pitcher_name}"
        )

        opponent_name = getattr(
            starter,
            "opponent_name",
            "Unknown",
        )

        game_status = getattr(
            starter,
            "game_status",
            "Unknown",
        )

        print(
            f"Game PK: "
            f"{starter.game_pk}"
        )

        print(
            f"Opponent: "
            f"{opponent_name}"
        )

        print(
            f"Game status: "
            f"{game_status}"
        )

        # ====================================================
        # NON-FINAL GAME
        # ====================================================

        if not is_final_game(
            game_status
        ):

            print(
                "Skipping: "
                "game is not final."
            )

            summary_rows.append(
                empty_summary_row(
                    game_date=(
                        game_date
                    ),

                    starter=(
                        starter
                    ),

                    opponent_name=(
                        opponent_name
                    ),

                    game_status=(
                        game_status
                    ),

                    replay_status=(
                        "game_not_final"
                    ),
                )
            )

            continue

        # ====================================================
        # POST-GAME REPLAY
        # ====================================================

        try:

            result = (
                replayer.replay(
                    starter=(
                        starter
                    ),

                    game_date=(
                        game_date
                    ),
                )
            )

            prediction_log = (
                pd.read_csv(
                    result
                    .predictions_path
                )
            )

            model_correct_count = (
                count_true_values(
                    prediction_log[
                        "model_correct"
                    ]
                )
            )

            baseline_correct_count = (
                count_true_values(
                    prediction_log[
                        "baseline_correct"
                    ]
                )
            )

            evaluated_at_utc = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            # =================================================
            # DAILY SUMMARY ROW
            # =================================================

            daily_row = {
                "game_date":
                    game_date.isoformat(),

                "game_pk":
                    result.game_pk,

                "pitcher_id":
                    result.pitcher_id,

                "pitcher_name":
                    result.pitcher_name,

                "opponent":
                    opponent_name,

                "game_status":
                    game_status,

                "replay_status":
                    "success",

                "pitch_count":
                    result.pitch_count,

                "model_correct":
                    model_correct_count,

                "baseline_correct":
                    baseline_correct_count,

                "model_accuracy":
                    result.model_accuracy,

                "baseline_accuracy":
                    result.baseline_accuracy,

                "accuracy_over_baseline":
                    (
                        result
                        .accuracy_over_baseline
                    ),

                "accuracy_over_baseline_pp":
                    (
                        result
                        .accuracy_over_baseline
                        * 100
                    ),

                "relative_improvement":
                    (
                        result
                        .relative_improvement
                    ),

                "relative_improvement_percent":
                    (
                        result
                        .relative_improvement
                        * 100
                        if (
                            result
                            .relative_improvement
                            is not None
                        )
                        else None
                    ),

                "baseline_strategy":
                    (
                        result
                        .baseline_strategy
                    ),

                "predictions_path":
                    (
                        result
                        .predictions_path
                    ),

                "summary_path":
                    (
                        result
                        .summary_path
                    ),

                "error":
                    None,
            }

            summary_rows.append(
                daily_row
            )

            # =================================================
            # PERMANENT PERFORMANCE HISTORY
            # =================================================
            #
            # This is now the long-term source of truth.
            #
            # If this game is replayed again, the existing row is
            # replaced rather than duplicated.
            # =================================================

            history_row = {
                "game_date":
                    game_date.isoformat(),

                "season":
                    game_date.year,

                "game_pk":
                    result.game_pk,

                "pitcher_id":
                    result.pitcher_id,

                "pitcher_name":
                    result.pitcher_name,

                "opponent":
                    opponent_name,

                "pitch_count":
                    result.pitch_count,

                "model_correct":
                    model_correct_count,

                "baseline_correct":
                    baseline_correct_count,

                "model_accuracy":
                    result.model_accuracy,

                "baseline_accuracy":
                    result.baseline_accuracy,

                "accuracy_over_baseline":
                    (
                        result
                        .accuracy_over_baseline
                    ),

                "accuracy_over_baseline_pp":
                    (
                        result
                        .accuracy_over_baseline
                        * 100
                    ),

                "relative_improvement":
                    (
                        result
                        .relative_improvement
                    ),

                "relative_improvement_percent":
                    (
                        result
                        .relative_improvement
                        * 100
                        if (
                            result
                            .relative_improvement
                            is not None
                        )
                        else None
                    ),

                "baseline_strategy":
                    (
                        result
                        .baseline_strategy
                    ),

                "model_path":
                    (
                        result
                        .model_path
                    ),

                "predictions_path":
                    (
                        result
                        .predictions_path
                    ),

                "summary_path":
                    (
                        result
                        .summary_path
                    ),

                "evaluated_at_utc":
                    evaluated_at_utc,
            }

            record_performance(
                path=(
                    PERFORMANCE_HISTORY_PATH
                ),

                row=(
                    history_row
                ),
            )

            print()
            print(
                "SUCCESS"
            )

            print(
                f"Pitches: "
                f"{result.pitch_count}"
            )

            print(
                f"Model accuracy: "
                f"{result.model_accuracy:.2%}"
            )

            print(
                f"Stratified baseline: "
                f"{result.baseline_accuracy:.2%}"
            )

            print(
                f"Absolute lift: "
                f"{result.accuracy_over_baseline * 100:+.2f} pp"
            )

            if (
                result.relative_improvement
                is not None
            ):

                print(
                    f"Relative improvement: "
                    f"{result.relative_improvement:+.2%}"
                )

            print(
                "Saved to cumulative "
                "performance history."
            )

        # ====================================================
        # EXPECTED STATCAST DELAY
        # ====================================================

        except ValueError as exc:

            if is_statcast_not_ready(
                exc
            ):

                print(
                    "Skipping for now: "
                    "Statcast data is not ready."
                )

                summary_rows.append(
                    empty_summary_row(
                        game_date=(
                            game_date
                        ),

                        starter=(
                            starter
                        ),

                        opponent_name=(
                            opponent_name
                        ),

                        game_status=(
                            game_status
                        ),

                        replay_status=(
                            "statcast_not_ready"
                        ),

                        error=str(
                            exc
                        ),
                    )
                )

            else:

                print(
                    f"FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                summary_rows.append(
                    empty_summary_row(
                        game_date=(
                            game_date
                        ),

                        starter=(
                            starter
                        ),

                        opponent_name=(
                            opponent_name
                        ),

                        game_status=(
                            game_status
                        ),

                        replay_status=(
                            "failed"
                        ),

                        error=(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    )
                )

        # ====================================================
        # UNEXPECTED FAILURE
        # ====================================================

        except Exception as exc:

            print(
                f"FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            summary_rows.append(
                empty_summary_row(
                    game_date=(
                        game_date
                    ),

                    starter=(
                        starter
                    ),

                    opponent_name=(
                        opponent_name
                    ),

                    game_status=(
                        game_status
                    ),

                    replay_status=(
                        "failed"
                    ),

                    error=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

    # ========================================================
    # DAILY SUMMARY CSV
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    daily_csv_path = (
        daily_dir
        / "daily_summary.csv"
    )

    summary_df.to_csv(
        daily_csv_path,
        index=False,
    )

    successful = summary_df[
        summary_df[
            "replay_status"
        ]
        == "success"
    ].copy()

    successful_count = int(
        len(
            successful
        )
    )

    not_ready_count = int(
        (
            summary_df[
                "replay_status"
            ]
            == "statcast_not_ready"
        ).sum()
    )

    failed_count = int(
        (
            summary_df[
                "replay_status"
            ]
            == "failed"
        ).sum()
    )

    not_final_count = int(
        (
            summary_df[
                "replay_status"
            ]
            == "game_not_final"
        ).sum()
    )

    # ========================================================
    # DAILY AGGREGATES
    # ========================================================

    if successful_count > 0:

        total_pitches = int(
            successful[
                "pitch_count"
            ].sum()
        )

        total_model_correct = int(
            successful[
                "model_correct"
            ].sum()
        )

        total_baseline_correct = int(
            successful[
                "baseline_correct"
            ].sum()
        )

        overall_model_accuracy = (
            total_model_correct
            / total_pitches
        )

        overall_baseline_accuracy = (
            total_baseline_correct
            / total_pitches
        )

        overall_lift = (
            overall_model_accuracy
            - overall_baseline_accuracy
        )

        overall_relative_improvement = (
            trainer
            .relative_improvement(
                overall_model_accuracy,
                overall_baseline_accuracy,
            )
        )

    else:

        total_pitches = 0

        total_model_correct = 0

        total_baseline_correct = 0

        overall_model_accuracy = None

        overall_baseline_accuracy = None

        overall_lift = None

        overall_relative_improvement = None

    # ========================================================
    # DAILY JSON
    # ========================================================

    daily_json = {
        "game_date":
            game_date.isoformat(),

        "season":
            game_date.year,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "baseline_strategy":
            "stratified",

        "starter_records":
            int(
                len(
                    summary_df
                )
            ),

        "statuses": {
            "successful":
                successful_count,

            "statcast_not_ready":
                not_ready_count,

            "failed":
                failed_count,

            "game_not_final":
                not_final_count,
        },

        "aggregate_performance": {
            "total_pitches":
                total_pitches,

            "model_correct":
                total_model_correct,

            "baseline_correct":
                total_baseline_correct,

            "model_accuracy":
                overall_model_accuracy,

            "baseline_accuracy":
                overall_baseline_accuracy,

            "accuracy_over_baseline":
                overall_lift,

            "accuracy_over_baseline_pp":
                (
                    overall_lift
                    * 100
                    if overall_lift
                    is not None
                    else None
                ),

            "relative_improvement":
                overall_relative_improvement,

            "relative_improvement_percent":
                (
                    overall_relative_improvement
                    * 100
                    if (
                        overall_relative_improvement
                        is not None
                    )
                    else None
                ),
        },

        "files": {
            "daily_summary_csv":
                str(
                    daily_csv_path
                ),

            "performance_history":
                str(
                    PERFORMANCE_HISTORY_PATH
                ),
        },
    }

    daily_json_path = (
        daily_dir
        / "daily_summary.json"
    )

    daily_json_path.write_text(
        json.dumps(
            daily_json,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # PRINT DAILY REPORT
    # ========================================================

    print()
    print()
    print("=" * 75)

    print(
        "DAILY POST-GAME SUMMARY"
    )

    print("=" * 75)

    print(
        f"Game date: "
        f"{game_date}"
    )

    print()

    print(
        f"Successful replays: "
        f"{successful_count}"
    )

    print(
        f"Statcast not ready: "
        f"{not_ready_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print(
        f"Games not final: "
        f"{not_final_count}"
    )

    if successful_count > 0:

        print()

        print(
            f"Total pitches predicted: "
            f"{total_pitches}"
        )

        print(
            f"Overall model accuracy: "
            f"{overall_model_accuracy:.2%}"
        )

        print(
            f"Overall stratified baseline: "
            f"{overall_baseline_accuracy:.2%}"
        )

        print(
            f"Overall absolute lift: "
            f"{overall_lift * 100:+.2f} pp"
        )

        if (
            overall_relative_improvement
            is not None
        ):

            print(
                f"Overall relative improvement: "
                f"{overall_relative_improvement:+.2%}"
            )

    print()

    print(
        f"Daily CSV: "
        f"{daily_csv_path}"
    )

    print(
        f"Daily JSON: "
        f"{daily_json_path}"
    )

    print(
        f"Cumulative history: "
        f"{PERFORMANCE_HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()