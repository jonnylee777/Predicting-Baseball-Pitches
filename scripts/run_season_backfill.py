"""Walk-forward historical backtest for an MLB season.

For every completed starting-pitcher game in the requested date range:

1. Find that day's starters.
2. Build the pitcher's data through the target game.
3. Use ONLY games before the target game for training.
4. Train/load the pre-game Random Forest.
5. Predict every pitch in the target game.
6. Compare against the stratified baseline.
7. Save the pitcher-game result.
8. Add the result to cumulative performance history.

The script is resume-safe:
successful pitcher-games already recorded for the current backtest
version are skipped automatically.

Example:

    python -m scripts.run_season_backfill --season 2026

Or test a smaller range first:

    python -m scripts.run_season_backfill \
        --season 2026 \
        --start-date 2026-04-01 \
        --end-date 2026-04-07
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
# CONFIGURATION
# ============================================================

OUTPUT_ROOT = Path(
    "Data/daily_pipeline"
)

PERFORMANCE_HISTORY_PATH = (
    OUTPUT_ROOT
    / "performance_history.csv"
)

BACKFILL_STATUS_PATH = (
    OUTPUT_ROOT
    / "backfill_status.csv"
)

BACKTEST_VERSION = (
    "walkforward_rf_stratified_v1"
)


# ============================================================
# DATE HELPERS
# ============================================================


def default_season_start(
    season: int,
) -> date:
    """
    Return the default start date for a season.

    The 2026 MLB regular season began March 25.

    For another season, explicitly passing --start-date is
    recommended.
    """

    if season == 2026:
        return date(
            2026,
            3,
            25,
        )

    return date(
        season,
        1,
        1,
    )


def iter_dates(
    start_date: date,
    end_date: date,
):
    """Yield every calendar date in an inclusive range."""

    current = start_date

    while current <= end_date:

        yield current

        current += timedelta(
            days=1
        )


# ============================================================
# STATUS HELPERS
# ============================================================


def is_final_game(
    game_status: str,
) -> bool:

    normalized = (
        str(game_status)
        .strip()
        .lower()
    )

    return (
        "final" in normalized
        or normalized
        in {
            "game over",
            "completed",
        }
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


def is_insufficient_history(
    exc: Exception,
) -> bool:

    message = str(
        exc
    ).lower()

    phrases = [
        "no pre-game history",
        "at least two games are required",
        "no usable pitches remain",
    ]

    return any(
        phrase in message
        for phrase in phrases
    )


# ============================================================
# CSV HELPERS
# ============================================================


def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


def upsert_row(
    dataframe: pd.DataFrame,
    row: dict,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    """
    Insert or replace one row.

    Example unique key:

        game_pk + pitcher_id

    This makes the backfill safe to rerun without generating
    duplicate pitcher-games.
    """

    new_row = pd.DataFrame(
        [row]
    )

    if dataframe.empty:
        return new_row

    missing_keys = [
        column
        for column in key_columns
        if column not in dataframe.columns
    ]

    if missing_keys:
        return pd.concat(
            [
                dataframe,
                new_row,
            ],
            ignore_index=True,
        )

    mask = pd.Series(
        True,
        index=dataframe.index,
    )

    for column in key_columns:

        existing_values = (
            dataframe[column]
            .astype(str)
        )

        new_value = str(
            row[column]
        )

        mask &= (
            existing_values
            == new_value
        )

    remaining = dataframe.loc[
        ~mask
    ].copy()

    if remaining.empty:
        return new_row

    return pd.concat(
        [
            remaining,
            new_row,
        ],
        ignore_index=True,
    )


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    working = dataframe.copy()

    sort_columns = [
        column
        for column in [
            "game_date",
            "game_pk",
            "pitcher_name",
        ]
        if column
        in working.columns
    ]

    if sort_columns:

        working = (
            working
            .sort_values(
                sort_columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    working.to_csv(
        path,
        index=False,
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
        normalized.eq(
            "true"
        ).sum()
    )


# ============================================================
# RESUME LOGIC
# ============================================================


def already_completed(
    history: pd.DataFrame,
    *,
    game_pk: int,
    pitcher_id: int,
) -> bool:
    """
    Return True if this pitcher-game was already successfully
    evaluated with the current backtest version.
    """

    if history.empty:
        return False

    required = {
        "game_pk",
        "pitcher_id",
        "backtest_version",
        "baseline_strategy",
    }

    if not required.issubset(
        history.columns
    ):
        return False

    game_ids = pd.to_numeric(
        history["game_pk"],
        errors="coerce",
    )

    pitcher_ids = pd.to_numeric(
        history["pitcher_id"],
        errors="coerce",
    )

    mask = (
        (game_ids == game_pk)
        &
        (pitcher_ids == pitcher_id)
        &
        (
            history[
                "backtest_version"
            ].astype(str)
            == BACKTEST_VERSION
        )
        &
        (
            history[
                "baseline_strategy"
            ].astype(str)
            == "stratified"
        )
    )

    return bool(
        mask.any()
    )


# ============================================================
# FORCE-RERUN SUPPORT
# ============================================================


def remove_existing_artifacts(
    *,
    game_date: date,
    game_pk: int,
    pitcher_id: int,
) -> None:
    """
    Delete old model and replay artifacts for one pitcher-game.

    Used only with --force.
    """

    model_dir = (
        OUTPUT_ROOT
        / "models"
        / game_date.isoformat()
        / "pitchers"
    )

    replay_dir = (
        OUTPUT_ROOT
        / "predictions"
        / "postgame"
        / game_date.isoformat()
    )

    paths = [
        model_dir
        / f"{pitcher_id}.joblib",

        model_dir
        / f"{pitcher_id}.json",

        replay_dir
        / f"{game_pk}_{pitcher_id}.csv",

        replay_dir
        / f"{game_pk}_{pitcher_id}.json",
    ]

    for path in paths:

        if path.exists():

            path.unlink()


# ============================================================
# RELATIVE IMPROVEMENT
# ============================================================


def relative_improvement(
    model_accuracy: float,
    baseline_accuracy: float,
) -> float | None:

    if baseline_accuracy <= 0:
        return None

    return (
        model_accuracy
        - baseline_accuracy
    ) / baseline_accuracy


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-safe walk-forward "
            "pitch-prediction backtest."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="MLB season to backfill.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Optional YYYY-MM-DD start date. "
            "For 2026 the default is 2026-03-25."
        ),
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "Optional YYYY-MM-DD end date. "
            "Default is yesterday."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Retrain and replay games even if "
            "they already exist in performance history."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # RESOLVE DATE RANGE
    # ========================================================

    if args.start_date:

        start_date = (
            date.fromisoformat(
                args.start_date
            )
        )

    else:

        start_date = (
            default_season_start(
                args.season
            )
        )

    yesterday = (
        date.today()
        - timedelta(days=1)
    )

    if args.end_date:

        end_date = (
            date.fromisoformat(
                args.end_date
            )
        )

    else:

        end_date = yesterday

    season_end = date(
        args.season,
        12,
        31,
    )

    end_date = min(
        end_date,
        yesterday,
        season_end,
    )

    if start_date > end_date:

        raise ValueError(
            "start_date must be before "
            "or equal to end_date."
        )

    if (
        start_date.year
        != args.season
        or end_date.year
        != args.season
    ):

        raise ValueError(
            "The requested date range must "
            "remain inside the selected season."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 78)
    print(
        "MLB PITCH PREDICTION — "
        "WALK-FORWARD SEASON BACKTEST"
    )
    print("=" * 78)

    print(
        f"Season: {args.season}"
    )

    print(
        f"Start date: {start_date}"
    )

    print(
        f"End date: {end_date}"
    )

    print(
        f"Backtest version: "
        f"{BACKTEST_VERSION}"
    )

    print(
        "Baseline: stratified"
    )

    print(
        "Resume existing results: "
        f"{'NO' if args.force else 'YES'}"
    )

    # ========================================================
    # PROJECT COMPONENTS
    # ========================================================

    print()
    print(
        "Loading project components..."
    )

    raw_schema = (
        StatcastSchema.from_file(
            Path(
                "config/statcast_columns.txt"
            )
        )
    )

    cleaned_schema = (
        StatcastSchema.from_file(
            Path(
                "config/cleaned_columns.txt"
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

    trainer = PitchModelTrainer()

    pipeline = DailyStarterPipeline(
        output_root=OUTPUT_ROOT,
        schema=raw_schema,
        cleaner=cleaner,
        feature_engineer=(
            feature_engineer
        ),
        model_trainer=None,
    )

    replayer = PostgameReplayer(
        pipeline=pipeline,
        model_trainer=trainer,
        output_root=OUTPUT_ROOT,
    )

    # ========================================================
    # LOAD EXISTING HISTORY
    # ========================================================

    history = load_csv(
        PERFORMANCE_HISTORY_PATH
    )

    status_history = load_csv(
        BACKFILL_STATUS_PATH
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    successful_this_run = 0
    resumed_this_run = 0
    insufficient_this_run = 0
    not_ready_this_run = 0
    failed_this_run = 0
    nonfinal_this_run = 0

    # ========================================================
    # WALK FORWARD ONE DAY AT A TIME
    # ========================================================

    dates = list(
        iter_dates(
            start_date,
            end_date,
        )
    )

    for day_number, game_date in enumerate(
        dates,
        start=1,
    ):

        print()
        print()
        print("=" * 78)

        print(
            f"DATE "
            f"[{day_number}/{len(dates)}]: "
            f"{game_date}"
        )

        print("=" * 78)

        # ====================================================
        # GET THAT DAY'S STARTERS
        # ====================================================

        try:

            schedule = (
                pipeline.mlb
                .probable_starters(
                    game_date
                )
            )

        except Exception as exc:

            print(
                "Could not retrieve MLB schedule:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            continue

        starters = []

        seen = set()

        for starter in schedule.starters:

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
            f"Starter records: "
            f"{len(starters)}"
        )

        if not starters:
            continue

        # ====================================================
        # EACH STARTER = ONE WALK-FORWARD TEST
        # ====================================================

        for starter_number, starter in enumerate(
            starters,
            start=1,
        ):

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

            print()
            print(
                f"  [{starter_number}/"
                f"{len(starters)}] "
                f"{starter.pitcher_name}"
            )

            print(
                f"      Game PK: "
                f"{starter.game_pk}"
            )

            print(
                f"      Opponent: "
                f"{opponent_name}"
            )

            # =================================================
            # SKIP NON-FINAL
            # =================================================

            if not is_final_game(
                game_status
            ):

                print(
                    "      SKIP — game not final"
                )

                nonfinal_this_run += 1

                status_row = {
                    "game_date":
                        game_date.isoformat(),

                    "season":
                        args.season,

                    "game_pk":
                        starter.game_pk,

                    "pitcher_id":
                        starter.pitcher_id,

                    "pitcher_name":
                        starter.pitcher_name,

                    "opponent":
                        opponent_name,

                    "status":
                        "game_not_final",

                    "error":
                        None,

                    "backtest_version":
                        BACKTEST_VERSION,

                    "checked_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }

                status_history = upsert_row(
                    status_history,
                    status_row,
                    key_columns=[
                        "game_pk",
                        "pitcher_id",
                    ],
                )

                save_dataframe(
                    status_history,
                    BACKFILL_STATUS_PATH,
                )

                continue

            # =================================================
            # RESUME
            # =================================================

            if (
                not args.force
                and already_completed(
                    history,
                    game_pk=(
                        starter.game_pk
                    ),
                    pitcher_id=(
                        starter.pitcher_id
                    ),
                )
            ):

                print(
                    "      DONE — "
                    "already in performance history"
                )

                resumed_this_run += 1

                continue

            # =================================================
            # FORCE OLD ARTIFACT REMOVAL
            # =================================================

            if args.force:

                remove_existing_artifacts(
                    game_date=game_date,
                    game_pk=(
                        starter.game_pk
                    ),
                    pitcher_id=(
                        starter.pitcher_id
                    ),
                )

            # =================================================
            # REPLAY
            # =================================================

            try:

                result = replayer.replay(
                    starter=starter,
                    game_date=game_date,
                )

                prediction_log = (
                    pd.read_csv(
                        result.predictions_path
                    )
                )

                model_correct = (
                    count_true_values(
                        prediction_log[
                            "model_correct"
                        ]
                    )
                )

                baseline_correct = (
                    count_true_values(
                        prediction_log[
                            "baseline_correct"
                        ]
                    )
                )

                # =============================================
                # SUCCESSFUL PERFORMANCE ROW
                # =============================================

                performance_row = {
                    "game_date":
                        game_date.isoformat(),

                    "season":
                        args.season,

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
                        model_correct,

                    "baseline_correct":
                        baseline_correct,

                    "model_accuracy":
                        result.model_accuracy,

                    "baseline_accuracy":
                        result.baseline_accuracy,

                    "accuracy_over_baseline":
                        result.accuracy_over_baseline,

                    "accuracy_over_baseline_pp":
                        (
                            result
                            .accuracy_over_baseline
                            * 100
                        ),

                    "relative_improvement":
                        result.relative_improvement,

                    "relative_improvement_percent":
                        (
                            result.relative_improvement
                            * 100
                            if result.relative_improvement
                            is not None
                            else None
                        ),

                    "baseline_strategy":
                        result.baseline_strategy,

                    "training_cutoff":
                        (
                            game_date
                            - timedelta(days=1)
                        ).isoformat(),

                    "model_path":
                        result.model_path,

                    "predictions_path":
                        result.predictions_path,

                    "summary_path":
                        result.summary_path,

                    "backtest_version":
                        BACKTEST_VERSION,

                    "evaluated_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }

                history = upsert_row(
                    history,
                    performance_row,
                    key_columns=[
                        "game_pk",
                        "pitcher_id",
                    ],
                )

                save_dataframe(
                    history,
                    PERFORMANCE_HISTORY_PATH,
                )

                # =============================================
                # SUCCESS STATUS ROW
                # =============================================

                status_row = {
                    "game_date":
                        game_date.isoformat(),

                    "season":
                        args.season,

                    "game_pk":
                        result.game_pk,

                    "pitcher_id":
                        result.pitcher_id,

                    "pitcher_name":
                        result.pitcher_name,

                    "opponent":
                        opponent_name,

                    "status":
                        "success",

                    "error":
                        None,

                    "backtest_version":
                        BACKTEST_VERSION,

                    "checked_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }

                status_history = upsert_row(
                    status_history,
                    status_row,
                    key_columns=[
                        "game_pk",
                        "pitcher_id",
                    ],
                )

                save_dataframe(
                    status_history,
                    BACKFILL_STATUS_PATH,
                )

                successful_this_run += 1

                print(
                    "      SUCCESS"
                )

                print(
                    f"      Pitches: "
                    f"{result.pitch_count}"
                )

                print(
                    f"      Model: "
                    f"{result.model_accuracy:.2%}"
                )

                print(
                    f"      Stratified baseline: "
                    f"{result.baseline_accuracy:.2%}"
                )

                print(
                    f"      Absolute lift: "
                    f"{result.accuracy_over_baseline * 100:+.2f} pp"
                )

                if (
                    result.relative_improvement
                    is not None
                ):

                    print(
                        f"      Relative improvement: "
                        f"{result.relative_improvement:+.2%}"
                    )

            # =================================================
            # EXPECTED DATA-LIMITATION CASES
            # =================================================

            except ValueError as exc:

                if is_statcast_not_ready(
                    exc
                ):

                    replay_status = (
                        "statcast_not_ready"
                    )

                    not_ready_this_run += 1

                elif is_insufficient_history(
                    exc
                ):

                    replay_status = (
                        "insufficient_history"
                    )

                    insufficient_this_run += 1

                else:

                    replay_status = (
                        "failed"
                    )

                    failed_this_run += 1

                print(
                    f"      {replay_status.upper()}"
                )

                print(
                    f"      {exc}"
                )

                status_row = {
                    "game_date":
                        game_date.isoformat(),

                    "season":
                        args.season,

                    "game_pk":
                        starter.game_pk,

                    "pitcher_id":
                        starter.pitcher_id,

                    "pitcher_name":
                        starter.pitcher_name,

                    "opponent":
                        opponent_name,

                    "status":
                        replay_status,

                    "error":
                        str(exc),

                    "backtest_version":
                        BACKTEST_VERSION,

                    "checked_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }

                status_history = upsert_row(
                    status_history,
                    status_row,
                    key_columns=[
                        "game_pk",
                        "pitcher_id",
                    ],
                )

                save_dataframe(
                    status_history,
                    BACKFILL_STATUS_PATH,
                )

            # =================================================
            # UNEXPECTED ERROR
            # =================================================

            except Exception as exc:

                failed_this_run += 1

                print(
                    "      FAILED"
                )

                print(
                    f"      "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                status_row = {
                    "game_date":
                        game_date.isoformat(),

                    "season":
                        args.season,

                    "game_pk":
                        starter.game_pk,

                    "pitcher_id":
                        starter.pitcher_id,

                    "pitcher_name":
                        starter.pitcher_name,

                    "opponent":
                        opponent_name,

                    "status":
                        "failed",

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),

                    "backtest_version":
                        BACKTEST_VERSION,

                    "checked_at_utc":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),
                }

                status_history = upsert_row(
                    status_history,
                    status_row,
                    key_columns=[
                        "game_pk",
                        "pitcher_id",
                    ],
                )

                save_dataframe(
                    status_history,
                    BACKFILL_STATUS_PATH,
                )

    # ========================================================
    # BUILD SEASON SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 78)
    print(
        "BUILDING CUMULATIVE SEASON SUMMARY"
    )
    print("=" * 78)

    if history.empty:

        season_history = pd.DataFrame()

    else:

        history_dates = pd.to_datetime(
            history["game_date"],
            errors="coerce",
        )

        history_seasons = pd.to_numeric(
            history["season"],
            errors="coerce",
        )

        season_history = history[
            (history_seasons == args.season)
            &
            (
                history_dates.dt.date
                >= start_date
            )
            &
            (
                history_dates.dt.date
                <= end_date
            )
            &
            (
                history[
                    "backtest_version"
                ].astype(str)
                == BACKTEST_VERSION
            )
        ].copy()

    backtest_dir = (
        OUTPUT_ROOT
        / "backtests"
        / str(
            args.season
        )
    )

    backtest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    season_history_path = (
        backtest_dir
        / "performance_history.csv"
    )

    save_dataframe(
        season_history,
        season_history_path,
    )

    # ========================================================
    # AGGREGATE PERFORMANCE
    # ========================================================

    if not season_history.empty:

        total_starts = int(
            len(
                season_history
            )
        )

        unique_pitchers = int(
            season_history[
                "pitcher_id"
            ].nunique()
        )

        total_pitches = int(
            pd.to_numeric(
                season_history[
                    "pitch_count"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        total_model_correct = int(
            pd.to_numeric(
                season_history[
                    "model_correct"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        total_baseline_correct = int(
            pd.to_numeric(
                season_history[
                    "baseline_correct"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        if total_pitches > 0:

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

            overall_relative = (
                relative_improvement(
                    overall_model_accuracy,
                    overall_baseline_accuracy,
                )
            )

        else:

            overall_model_accuracy = None
            overall_baseline_accuracy = None
            overall_lift = None
            overall_relative = None

        model_accuracies = pd.to_numeric(
            season_history[
                "model_accuracy"
            ],
            errors="coerce",
        )

        baseline_accuracies = pd.to_numeric(
            season_history[
                "baseline_accuracy"
            ],
            errors="coerce",
        )

        lifts = pd.to_numeric(
            season_history[
                "accuracy_over_baseline_pp"
            ],
            errors="coerce",
        )

        relative_values = pd.to_numeric(
            season_history[
                "relative_improvement"
            ],
            errors="coerce",
        )

        mean_start_model_accuracy = float(
            model_accuracies.mean()
        )

        mean_start_baseline_accuracy = float(
            baseline_accuracies.mean()
        )

        mean_start_lift_pp = float(
            lifts.mean()
        )

        valid_relative = (
            relative_values
            .dropna()
        )

        mean_start_relative = (
            float(
                valid_relative.mean()
            )
            if not valid_relative.empty
            else None
        )

    else:

        total_starts = 0
        unique_pitchers = 0
        total_pitches = 0
        total_model_correct = 0
        total_baseline_correct = 0

        overall_model_accuracy = None
        overall_baseline_accuracy = None
        overall_lift = None
        overall_relative = None

        mean_start_model_accuracy = None
        mean_start_baseline_accuracy = None
        mean_start_lift_pp = None
        mean_start_relative = None

    # ========================================================
    # STATUS COUNTS
    # ========================================================

    if status_history.empty:

        status_counts = {}

    else:

        status_dates = pd.to_datetime(
            status_history[
                "game_date"
            ],
            errors="coerce",
        )

        status_seasons = pd.to_numeric(
            status_history[
                "season"
            ],
            errors="coerce",
        )

        relevant_status = (
            status_history[
                (status_seasons == args.season)
                &
                (
                    status_dates.dt.date
                    >= start_date
                )
                &
                (
                    status_dates.dt.date
                    <= end_date
                )
                &
                (
                    status_history[
                        "backtest_version"
                    ].astype(str)
                    == BACKTEST_VERSION
                )
            ]
        )

        status_counts = {
            str(status):
                int(count)
            for status, count
            in relevant_status[
                "status"
            ]
            .value_counts()
            .items()
        }

    # ========================================================
    # SAVE SEASON JSON
    # ========================================================

    season_summary = {
        "season":
            args.season,

        "start_date":
            start_date.isoformat(),

        "end_date":
            end_date.isoformat(),

        "backtest_version":
            BACKTEST_VERSION,

        "evaluation_method":
            "walk_forward",

        "baseline_strategy":
            "stratified",

        "leakage_rule":
            (
                "Each pitcher-game model is trained "
                "only on pitches dated before the "
                "target game."
            ),

        "coverage": {
            "pitcher_games":
                total_starts,

            "unique_pitchers":
                unique_pitchers,

            "total_pitches":
                total_pitches,

            "status_counts":
                status_counts,
        },

        "aggregate_performance": {
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
                overall_relative,

            "relative_improvement_percent":
                (
                    overall_relative
                    * 100
                    if overall_relative
                    is not None
                    else None
                ),
        },

        "mean_pitcher_game_performance": {
            "model_accuracy":
                mean_start_model_accuracy,

            "baseline_accuracy":
                mean_start_baseline_accuracy,

            "accuracy_over_baseline_pp":
                mean_start_lift_pp,

            "relative_improvement":
                mean_start_relative,

            "relative_improvement_percent":
                (
                    mean_start_relative
                    * 100
                    if mean_start_relative
                    is not None
                    else None
                ),
        },

        "files": {
            "cumulative_history":
                str(
                    PERFORMANCE_HISTORY_PATH
                ),

            "backfill_status":
                str(
                    BACKFILL_STATUS_PATH
                ),

            "season_history":
                str(
                    season_history_path
                ),
        },

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    season_summary_path = (
        backtest_dir
        / "season_summary.json"
    )

    season_summary_path.write_text(
        json.dumps(
            season_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        f"{args.season} WALK-FORWARD BACKTEST SUMMARY"
    )
    print("=" * 78)

    print(
        f"Pitcher-games evaluated: "
        f"{total_starts}"
    )

    print(
        f"Unique pitchers: "
        f"{unique_pitchers}"
    )

    print(
        f"Total pitches evaluated: "
        f"{total_pitches}"
    )

    if (
        overall_model_accuracy
        is not None
    ):

        print()

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

        if overall_relative is not None:

            print(
                f"Overall relative improvement: "
                f"{overall_relative:+.2%}"
            )

    print()
    print(
        "This run:"
    )

    print(
        f"  New successful: "
        f"{successful_this_run}"
    )

    print(
        f"  Already completed: "
        f"{resumed_this_run}"
    )

    print(
        f"  Insufficient history: "
        f"{insufficient_this_run}"
    )

    print(
        f"  Statcast not ready: "
        f"{not_ready_this_run}"
    )

    print(
        f"  Game not final: "
        f"{nonfinal_this_run}"
    )

    print(
        f"  Failed: "
        f"{failed_this_run}"
    )

    print()
    print(
        f"Cumulative history: "
        f"{PERFORMANCE_HISTORY_PATH}"
    )

    print(
        f"Backfill status: "
        f"{BACKFILL_STATUS_PATH}"
    )

    print(
        f"2026 history: "
        f"{season_history_path}"
    )

    print(
        f"Season summary: "
        f"{season_summary_path}"
    )


if __name__ == "__main__":
    main()